"""The hardware plan for the characterization: jobs, QPU time, the pilot and its stop/go bands.

Every job runs only after Sal's yes, through ``dqiqm.hw.submit``, one hardware worker at a time.
QPU time is estimated as garnet-design's plan does it (``dqiqm.experiments.suite.estimate``):
shots x (circuit duration + the 298 us ``reset_wait``) + circuits x 0.1 s, with a pessimistic
column at Garnet's published CLOPS (1.92 ms per shot). A circuit's duration is its gate layers
(each as long as its longest gate; barriers synchronise) plus the terminal readout window
(``measure_fidelity``, about 1.04 us on the patch), the compiled figure, not the probe length.

The pilot and the no-merging check
    A circuit job returns no compiled duration (only status timestamps), so the brief's "compiled
    duration against N x the gate duration" cannot be read from the server. A correct merge of
    repeated gates also leaves the ideal state unchanged. What a merge cannot fake is error
    accumulation, so the pilot's stop/go rests on it:

    * **C1:** P1 after 42 ``prx(pi/2)`` (ideally |1>) is below P1 after 2, by the decay of 40
      pulses. The band's floor is half the *T1/T2-only* prediction (no gate error at all), pooled
      over the 17 qubits. A merged sequence (42 -> one X) gives no difference.
    * **C2:** the conditional Ramsey's contrast after 17 CZs (a pilot-only chain) is below that
      after 1, by 16 CZs' worth of IRB error. The band's ceiling on the pooled ratio is halfway between the IRB
      prediction and 1 (a merged or cancelled sequence).

    Before submission, offline: ``count_ops()`` and the serialized sequence of every circuit
    (:func:`dqiqm.characterization.circuits.check`) and IQM's ``validate_circuit`` against the
    live backend, and ``native_sha256`` equal to the record's. With read access (no allowance
    needed), the hw worker can also compile the exact circuits with Pulla's standard compiler, as
    ``scripts/pulla_check.py`` does, and count N pulses per qubit (16 ns each) and 2N flux pulses
    per CZ chain. That shows what IQM's compiler does with our circuits, not what the server ran.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dqiqm.characterization import model as Mo
from dqiqm.characterization.circuits import C2_PSI, Design, c1_sequence, duration_ns
from dqiqm.characterization.patch import DEVICE, Patch, Priors
from dqiqm.experiments import device as dev

#: garnet-design's plan constants (``dqiqm.experiments.suite``), copied so the plan stands alone:
#: a fixed cost per circuit, and Garnet's published CLOPS of 2,600 as a pessimistic per-shot cost.
PER_CIRCUIT_OVERHEAD_S = 0.1
PESSIMISTIC_PER_SHOT_S = 100 * 10 * 100 * 5 / 2600 / 1e5

#: Shots per circuit, per job group.
SHOTS = {"P0": 1000, "C1": 1000, "C2": 500, "C4-C1": 500, "C4-C2": 250, "C5": 500}
PILOT_C1_N = (0, 2, 41, 42, 43)
PILOT_C2 = {"layer": 1, "orientation": "tanc", "s": 1, "n": (1, 17)}
C5_C1 = (("amp_x", 0, "pos"), ("amp_x", 2, "pos"), ("amp_x", 41, "pos"), ("amp_x", 43, "pos"),
         ("tilt", 16, "pos"), ("tilt", 16, "neg"))
C5_C2_N = (1, 9)


@dataclass(frozen=True)
class Job:
    label: str
    phase: str  # "pilot", "must", "should", "if time", "blocked"
    experiment: str
    circuits: tuple[str, ...]
    shots: int
    note: str


def jobs(designs: Mapping[str, Design], patch: Patch) -> list[Job]:
    """The plan, in order. ``designs`` maps name -> design for every C1/C2/C4 circuit."""
    names = list(designs)
    c1 = [n for n in names if designs[n].experiment == "C1"]
    out = [Job("P0", "pilot", "C1+C2", tuple(pilot_circuits(designs)), SHOTS["P0"],
               "pilot: readout anchors, the C1 and C2 no-merging sentinels, ε·N and the C2 phase; stop/go before C1")]
    out.append(Job("C1", "must", "C1", tuple(c1), SHOTS["C1"], "PRX amplification, all 17 qubits in parallel"))
    for li in range(1, len(patch.layers) + 1):
        c2 = [n for n in names if designs[n].experiment == "C2" and designs[n].params["layer"] == li
              and not designs[n].params.get("pilot_only")]
        out.append(Job(f"C2-L{li}", "must", "C2", tuple(c2), SHOTS["C2"],
                       f"conditional Ramsey, CZ layer {li}'s 6 couplers in parallel, both orientations"))
    c4_1 = [n for n in names if designs[n].experiment == "C4" and designs[n].params["family"] == "prx_amplification"]
    out.append(Job("C4-C1", "should", "C4", tuple(c4_1), SHOTS["C4-C1"], "C1 isolated: one qubit driven at a time"))
    for li in range(1, len(patch.layers) + 1):
        c4_2 = [n for n in names if designs[n].experiment == "C4" and designs[n].params["family"] == "conditional_ramsey"
                and designs[n].params["layer"] == li]
        out.append(Job(f"C4-C2-L{li}", "should", "C4", tuple(c4_2), SHOTS["C4-C2"],
                       f"C2 isolated: layer {li}'s couplers one at a time, target = ancilla"))
    c5 = [f"C1_{k}_" + (f"p{n:02d}_{s}" if k == "tilt" else f"n{n:02d}") for k, n, s in C5_C1]
    c5 += [n for n in names if designs[n].experiment == "C2" and designs[n].params["orientation"] == "tanc"
           and designs[n].params["n"] in C5_C2_N and not designs[n].params.get("pilot_only")]
    out.append(Job("C5", "if time", "C1+C2", tuple(c5), SHOTS["C5"],
                   "drift subset: repeat at each batch boundary, next to the E6 snapshots"))
    return out


def pilot_circuits(designs: Mapping[str, Design]) -> list[str]:
    c1 = [f"C1_amp_x_n{n:02d}" for n in PILOT_C1_N]
    c2 = [f"C2_L{PILOT_C2['layer']}_{PILOT_C2['orientation']}_s{PILOT_C2['s']}_n{n:02d}_q{k}"
          for n in PILOT_C2["n"] for k in range(len(C2_PSI))]
    missing = [n for n in c1 + c2 if n not in designs]
    if missing:
        raise KeyError(f"pilot circuits not built: {missing}")
    return c1 + c2


def estimate(job: Job, designs: Mapping[str, Design], priors: Priors, durations: Mapping[str, float],
             per_shot_s: float) -> dict[str, Any]:
    durs = [duration_ns(designs[n], durations, priors.terminal_window_ns) for n in job.circuits]
    shots_s = sum(job.shots * (d * 1e-9 + per_shot_s) for d in durs)
    over = PER_CIRCUIT_OVERHEAD_S * len(durs)
    return {
        "label": job.label, "phase": job.phase, "experiment": job.experiment, "device": DEVICE,
        "circuits": len(job.circuits), "shots": job.shots, "total_shots": job.shots * len(job.circuits),
        "circuit_us_min": min(durs) / 1e3, "circuit_us_max": max(durs) / 1e3, "circuit_us_mean": float(np.mean(durs)) / 1e3,
        "qpu_s_shots": shots_s, "qpu_s_per_circuit_overhead": over, "qpu_s": shots_s + over,
        "qpu_s_pessimistic": sum(job.shots * (d * 1e-9 + PESSIMISTIC_PER_SHOT_S) for d in durs) + over,
        "per_shot_overhead_s": per_shot_s, "note": job.note, "order": list(job.circuits),
    }


# ---------------------------------------------------------------------------
# The pilot's bands, from the calibration's priors.
# ---------------------------------------------------------------------------


def _p_read1(q: str, priors: Priors, n: int, *, lam: float, relax: bool) -> float:
    pq = priors.qubits[q]
    m = Mo.QubitModel(lam=lam, a0=pq["p1_given_0"], a1=pq["p0_given_1"],
                      t1_ns=pq["t1_ns"] if relax else math.inf, t2_ns=pq["t2_ns"] if relax else math.inf,
                      gate_ns=pq["prx_ns"])
    return Mo.p_read1(c1_sequence("amp_x", n), m)


def pilot_bands(patch: Patch, priors: Priors, shots: int = SHOTS["P0"]) -> dict[str, Any]:
    """The pilot's stop/go bands and the predictions they come from (see the module docstring)."""
    c1 = {}
    d_t1, d_rb, var = [], [], []
    for q in patch.qubits:
        pq = priors.qubits[q]
        relax_only = {n: _p_read1(q, priors, n, lam=0.0, relax=True) for n in (0, 2, 42)}
        rb_only = {n: _p_read1(q, priors, n, lam=2 * pq["error_1q"], relax=False) for n in (2, 42)}
        both = {n: _p_read1(q, priors, n, lam=2 * pq["error_1q"], relax=True) for n in (2, 42)}
        d_t1.append(relax_only[2] - relax_only[42])
        d_rb.append(rb_only[2] - rb_only[42])
        var.append((both[2] * (1 - both[2]) + both[42] * (1 - both[42])) / shots)
        c1[q] = {"p1_n00": relax_only[0], "p1_n02": both[2], "p1_n42": both[42],
                 "decay_t1t2_only": relax_only[2] - relax_only[42], "decay_rb_only": rb_only[2] - rb_only[42],
                 "decay_both": both[2] - both[42]}
    d_floor_pred = float(np.mean(d_t1))
    d_sigma = float(math.sqrt(sum(var)) / len(var))
    c2 = {}
    ratios, rvar = [], []
    layer = patch.layers[PILOT_C2["layer"] - 1]
    n_lo, n_hi = PILOT_C2["n"]
    for d, a in layer:
        t = a if PILOT_C2["orientation"] == "tanc" else d
        r = priors.couplers[dev.pair_key(d, a)]["cz_error"]
        pt = priors.qubits[t]
        ratio = (1 - 4 * r / 3) ** (n_hi - n_lo)
        c0 = (1 - pt["p1_given_0"] - pt["p0_given_1"])  # the readout's scale on the contrast at n = 1
        var_c = 0.5 / shots  # each of sin and cos is a difference of two P's near 1/2
        ratios.append(ratio)
        rvar.append((var_c + ratio ** 2 * var_c) / c0 ** 2)
        c2[f"{t}<-{d if t == a else a}"] = {"cz_irb_error": r, "contrast_ratio_irb": ratio}
    r_pred = float(np.mean(ratios))
    r_sigma = float(math.sqrt(sum(rvar)) / len(rvar))
    return {
        "circuits": None,  # filled by the caller with the job's order
        "shots": shots,
        "columns": len(patch.qubits),
        "column_qubits": list(patch.qubits),
        "readout": {
            "p1_n00_max": 0.10,
            "p1_n02_min": 0.75,
            "note": "every qubit: P(read 1) after C1_amp_x_n00 (|0>) and C1_amp_x_n02 (an X, |1>); the terminal "
                    "readout is measure_fidelity, whose worst P(0|1) on the patch is QB20's 0.090",
        },
        "c1_no_merge": {
            "statistic": "D = mean over the 17 qubits of [P1(C1_amp_x_n02) - P1(C1_amp_x_n42)]",
            "predicted_t1t2_only": d_floor_pred,
            "predicted_rb_only": float(np.mean(d_rb)),
            "predicted_t1t2_and_rb": float(np.mean([v["decay_both"] for v in c1.values()])),
            "sigma_pooled": d_sigma,
            "stop_if_below": d_floor_pred / 2,
            "merged_expectation": 0.0,
            "per_qubit": c1,
        },
        "c2_no_merge": {
            "statistic": f"R = mean over layer {PILOT_C2['layer']}'s 6 couplers (target = ancilla, s = 1) of "
                         f"contrast(n={n_hi}) / contrast(n={n_lo}), contrast = hypot(P(pi)-P(0), P(pi/2)-P(3pi/2))",
            "predicted_irb": r_pred,
            "sigma_pooled": r_sigma,
            "stop_if_above": 1 - (1 - r_pred) / 2,
            "merged_expectation": 1.0,
            "per_coupler": c2,
        },
        "info_only": [
            "eps per qubit from C1_amp_x_n41 and _n43 (P1 ~ (1 +- sin(N eps))/2): reported, not a stop/go, since a "
            "well-calibrated gate has eps near 0",
            f"the C2 phase per CZ, Phi(n={n_hi}) - Phi(n={n_lo}) = {n_hi - n_lo}(phi_t + delta) mod 2pi: reported",
            "the job timeline (status timestamps): the per-shot overhead, to re-estimate the plan",
        ],
        "stop_rules": [
            "memory.npz has one (1000, 17) array per circuit, in the plan's order, and meta.json the calibration_set_id",
            "the readout bands hold on every qubit",
            "D >= c1_no_merge.stop_if_below",
            "R <= c2_no_merge.stop_if_above",
            "any failure: stop and report; never spend more shots chasing it",
        ],
    }


def pilot_stats(arrays: Sequence[np.ndarray], order: Sequence[str], patch: Patch) -> dict[str, Any]:
    """The pilot's statistics from its records (the hw worker's stop/go, and the dry run's)."""
    idx = {n: i for i, n in enumerate(order)}
    p = {n: np.asarray(arrays[i]).mean(axis=0) for n, i in idx.items()}
    d = float(np.mean(p["C1_amp_x_n02"] - p["C1_amp_x_n42"]))
    layer = patch.layers[PILOT_C2["layer"] - 1]
    cols = [patch.index(a if PILOT_C2["orientation"] == "tanc" else dd) for dd, a in layer]
    n_lo, n_hi = PILOT_C2["n"]

    def contrast(n: int) -> np.ndarray:
        q = [p[f"C2_L{PILOT_C2['layer']}_{PILOT_C2['orientation']}_s{PILOT_C2['s']}_n{n:02d}_q{k}"][cols] for k in range(4)]
        return np.hypot(q[2] - q[0], q[1] - q[3])

    ratio = contrast(n_hi) / contrast(n_lo)
    return {"p1_n00": p["C1_amp_x_n00"].tolist(), "p1_n02": p["C1_amp_x_n02"].tolist(), "D": d,
            "R": float(np.mean(ratio)), "R_per_coupler": ratio.tolist()}
