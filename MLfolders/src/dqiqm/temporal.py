"""Temporal-correlation features of detection events and raw readouts, for the InfoRecord's ``derived/``.

Every function takes plain arrays: ``dets`` (shots x detectors, 0/1), and per detector its round
``t`` (Stim's time coordinate: 0-based, the final data detectors at t = R) and its stabilizer
index ``stab``.

* :func:`pij`: the pairwise error probability of every detector pair (Google Quantum AI, Nature
  2021, Supplementary eq. for p_ij), from the shots alone:

      p_ij = 1/2 − 1/2 · sqrt(1 − 4 (<x_i x_j> − <x_i><x_j>) / (1 − 2<x_i> − 2<x_j> + 4<x_i x_j>)).

  An error that fires exactly the pair (i, j) with probability p gives p_ij ≈ p.
* :func:`lag_table`: p_ij averaged per (stabilizer, stabilizer, lag), lag = t_j − t_i in {0, 1, 2}.
  Without reset, a record flip (the bit is misread, the ancilla is not flipped) fires the same
  stabilizer at lags 0 and 2 (D_k, D_(k+2)), and a flip of the ancilla's state fires it at lag 1
  (D_(k+1), D_(k+2)); so same-stabilizer lag 2 measures the record flips, lag 1 the state flips
  and the measurement-side gate errors.
* :func:`rate_by_round`, :func:`rate_by_bucket`: firing rates per detector, by round and by
  block of consecutive shots (drift).
* :func:`syndrome_weight`: detection events per round, per shot.
* :func:`streaks`: runs of consecutive rounds in which one stabilizer's detector fires, the
  signature of a leaked qubit (its checks fire about half the time, round after round).
* :func:`assignment`, :func:`readout_correlation`, :func:`readout_flips`: readout figures from
  E3: the assignment matrix, correlated misassignment, and record flips separated from state
  flips in repeated readouts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pij(dets: np.ndarray, *, chunk: int = 1 << 16) -> np.ndarray:
    """The p_ij matrix (float64, detectors x detectors); its diagonal is each detector's firing rate."""
    dets = np.asarray(dets)
    n, m = dets.shape
    if n == 0:
        raise ValueError("no shots")
    s1 = np.zeros(m)
    s2 = np.zeros((m, m))
    for lo in range(0, n, chunk):
        x = dets[lo : lo + chunk].astype(np.float32)
        s1 += x.sum(0, dtype=np.float64)
        s2 += (x.T @ x).astype(np.float64)
    xi = s1 / n
    xij = s2 / n
    num = xij - np.outer(xi, xi)
    den = 1 - 2 * xi[:, None] - 2 * xi[None, :] + 4 * xij
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = 1 - 4 * num / den
        p = 0.5 - 0.5 * np.sqrt(np.clip(inner, 0.0, None))
    p = np.where(np.isfinite(p), p, 0.0)
    np.fill_diagonal(p, xi)
    return p


def lag_table(p: np.ndarray, t: Sequence[int], stab: Sequence[int], lags: Sequence[int] = (0, 1, 2)) -> list[dict]:
    """Mean p_ij per (stab_i, stab_j, lag) over every pair with t_j − t_i = lag (i ≠ j), and the pair count."""
    t = np.asarray(t)
    stab = np.asarray(stab)
    rows = []
    n_stab = int(stab.max()) + 1 if len(stab) else 0
    for lag in lags:
        ii, jj = np.nonzero((t[None, :] - t[:, None]) == lag)
        keep = ii != jj
        if lag == 0:
            keep &= ii < jj  # each unordered pair once
        ii, jj = ii[keep], jj[keep]
        si, sj = stab[ii], stab[jj]
        if lag == 0:
            si, sj = np.minimum(si, sj), np.maximum(si, sj)
        vals = p[ii, jj]
        for a in range(n_stab):
            for b in range(n_stab):
                sel = (si == a) & (sj == b)
                if sel.any():
                    rows.append({"stab_i": a, "stab_j": b, "lag": lag, "pij_mean": float(vals[sel].mean()),
                                 "pij_max": float(vals[sel].max()), "pairs": int(sel.sum())})
    return rows


def lag_summary(p: np.ndarray, t: Sequence[int], stab: Sequence[int]) -> dict[str, float]:
    """Same-stabilizer p_ij at lags 1 and 2, and different-stabilizer at lags 0 and 1, averaged over pairs."""
    t = np.asarray(t)
    stab = np.asarray(stab)
    dt = t[None, :] - t[:, None]
    same = stab[None, :] == stab[:, None]
    out = {}
    for name, sel in (
        ("same_stab_lag1", same & (dt == 1)),
        ("same_stab_lag2", same & (dt == 2)),
        ("same_stab_lag3", same & (dt == 3)),
        ("cross_stab_lag0", ~same & (dt == 0)),
        ("cross_stab_lag1", ~same & (dt == 1)),
    ):
        out[name] = float(p[sel].mean()) if sel.any() else float("nan")
    return out


def rate_by_round(dets: np.ndarray, t: Sequence[int], stab: Sequence[int]) -> list[dict]:
    rate = np.asarray(dets).mean(0)
    return [{"detector": i, "stab": int(s), "round": int(r), "rate": float(rate[i]), "shots": len(dets)}
            for i, (s, r) in enumerate(zip(stab, t))]


def buckets(shots: int, n_buckets: int = 10) -> np.ndarray:
    """Each shot's block of consecutive shots: shot · n_buckets // shots (n_buckets blocks, in shot order)."""
    return (np.arange(shots) * min(n_buckets, max(shots, 1))) // max(shots, 1)


def rate_by_bucket(dets: np.ndarray, bucket: np.ndarray, stab: Sequence[int]) -> list[dict]:
    """Per block of shots and per stabilizer: the mean firing rate of its detectors."""
    dets = np.asarray(dets)
    stab = np.asarray(stab)
    rows = []
    for b in np.unique(bucket):
        sel = bucket == b
        idx = np.nonzero(sel)[0]
        per_det = dets[sel].mean(0)
        for s in np.unique(stab):
            rows.append({"bucket": int(b), "shot_lo": int(idx.min()), "shot_hi": int(idx.max()), "stab": int(s),
                         "rate": float(per_det[stab == s].mean()), "shots": int(sel.sum())})
    return rows


def syndrome_weight(dets: np.ndarray, t: Sequence[int]) -> tuple[np.ndarray, list[int]]:
    """(shots x rounds) detection events per round, and the rounds (sorted t values) of its columns."""
    dets = np.asarray(dets)
    t = np.asarray(t)
    rounds = sorted(int(r) for r in np.unique(t))
    w = np.stack([dets[:, t == r].sum(1) for r in rounds], axis=1) if rounds else np.zeros((len(dets), 0), int)
    return w.astype(np.int32), rounds


def weight_distribution(w: np.ndarray, rounds: Sequence[int]) -> list[dict]:
    rows = []
    for j, r in enumerate(rounds):
        vals, counts = np.unique(w[:, j], return_counts=True)
        rows += [{"round": int(r), "weight": int(v), "count": int(c)} for v, c in zip(vals, counts)]
    return rows


def streaks(dets: np.ndarray, t: Sequence[int], stab: Sequence[int]) -> tuple[np.ndarray, list[dict]]:
    """Per shot, the longest run of consecutive rounds in which one stabilizer fires; and the histogram of runs.

    A stabilizer's detectors are taken in order of t; a round where it has no detector (the first
    round of a check that starts random, or the final round of one the data readout does not
    give) ends a run. Returns (longest per shot, rows of (stab, length, count) over all shots).
    """
    dets = np.asarray(dets).astype(np.int32)
    t = np.asarray(t)
    stab = np.asarray(stab)
    rounds = sorted(int(r) for r in np.unique(t))
    stabs = sorted(int(s) for s in np.unique(stab))
    col = {r: j for j, r in enumerate(rounds)}
    grid = np.zeros((len(dets), len(stabs), len(rounds)), np.int32)
    si = {s: i for i, s in enumerate(stabs)}
    for d, (s, r) in enumerate(zip(stab, t)):
        grid[:, si[int(s)], col[int(r)]] = dets[:, d]
    run = np.zeros(grid.shape[:2], np.int32)
    longest = np.zeros(grid.shape[:2], np.int32)
    hist: dict[tuple[int, int], int] = {}

    def close(ended: np.ndarray):
        for (shot, s) in zip(*np.nonzero(ended)):
            key = (stabs[s], int(run[shot, s]))
            hist[key] = hist.get(key, 0) + 1

    for j in range(len(rounds)):
        fire = grid[:, :, j] > 0
        close((~fire) & (run > 0))
        run = np.where(fire, run + 1, 0)
        longest = np.maximum(longest, run)
    close(run > 0)
    rows = [{"stab": s, "length": length, "count": c} for (s, length), c in sorted(hist.items())]
    return longest.max(1) if longest.size else np.zeros(len(dets), np.int32), rows


# ---------------------------------------------------------------------------
# Readout (E3).
# ---------------------------------------------------------------------------


def assignment(raw: np.ndarray, prep: Sequence[int], n_qubits: int, k: int = 0) -> list[dict]:
    """Per qubit, from readout layer ``k`` (0-based; qubit q's bit is column k·n_qubits + q): P(misread | prep)."""
    raw = np.asarray(raw)
    rows = []
    for q in range(n_qubits):
        bits = raw[:, k * n_qubits + q]
        wrong = bits != prep[q]
        rows.append({"qubit": q, "prep": int(prep[q]), "layer": k, "p_wrong": float(wrong.mean()),
                     "p_one": float(bits.mean()), "shots": len(raw)})
    return rows


def readout_correlation(raw: np.ndarray, prep: Sequence[int], n_qubits: int, k: int = 0) -> np.ndarray:
    """The Pearson correlation of misassignment indicators between qubits (readout layer ``k``)."""
    raw = np.asarray(raw)
    err = (raw[:, k * n_qubits : (k + 1) * n_qubits] != np.asarray(prep)[None, :]).astype(np.float64)
    sd = err.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(err, rowvar=False)
    c = np.where(np.outer(sd, sd) > 0, c, 0.0)
    np.fill_diagonal(c, 1.0)
    return c


def readout_flips(dets: np.ndarray, raw: np.ndarray, prep: Sequence[int], n_qubits: int, repeats: int) -> list[dict]:
    """Per qubit, in a repeated readout: record flips and state flips, separated.

    The detectors are D_1 = m_1 + prep and D_k = m_k + m_(k−1), column (k−1)·n_qubits + q. A record
    flip at readout k fires D_k and D_(k+1); a state flip between readouts k−1 and k fires D_k alone.
    With r the record-flip and s the state-flip probability per readout (both small):

    * p_ij(D_k, D_(k+1)) ≈ r, averaged over k = 1 .. K−1: ``record_flip``;
    * P(D_k) ≈ 2r + s for 2 ≤ k ≤ K−1: ``state_flip`` = mean P(D_k) − 2·record_flip.

    Also the misread fraction at the first and the last readout, against the prepared bit: a
    state that decays during the train shows a last readout worse than the first.
    """
    dets = np.asarray(dets)
    raw = np.asarray(raw)
    rows = []
    for q in range(n_qubits):
        cols = [k * n_qubits + q for k in range(repeats)]
        d = dets[:, cols]
        p = pij(d)
        rec = float(np.mean([p[k, k + 1] for k in range(repeats - 1)])) if repeats > 1 else float("nan")
        bulk = float(d[:, 1 : repeats - 1].mean()) if repeats > 2 else float("nan")
        bits = raw[:, cols]
        rows.append({
            "qubit": q,
            "prep": int(prep[q]),
            "repeats": repeats,
            "p_det_bulk": bulk,
            "record_flip": rec,
            "state_flip": bulk - 2 * rec if repeats > 2 else float("nan"),
            "p_wrong_first": float((bits[:, 0] != prep[q]).mean()),
            "p_wrong_last": float((bits[:, -1] != prep[q]).mean()),
            "shots": len(raw),
        })
    return rows
