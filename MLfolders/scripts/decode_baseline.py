"""The simulation baseline, end to end, in hardware format: records -> m2d -> PyMatching -> fit of ε_L.

For each basis (Z, X) and each round count, it samples **measurement records** (not detection
events) from the stand-in circuit with Stim's ``compile_sampler``, stores them as one batched
"job" in ``dqiqm.hw.save_raw``'s format (``memory.npz`` + ``meta.json``, in a scratch directory),
loads them back with ``dqiqm.decode.load_memory``, and runs ``detection_events`` -> ``decode`` ->
``logical_error_probability`` -> ``dqiqm.stats.fit_eps``.

It checks that against a direct estimate on the same circuits, from independent samples:
``compile_detector_sampler`` decoded by PyMatching straight from the circuit's model, with no
``dqiqm.decode`` in the path, fitted the same way; and each round count's own ε_L.

The stand-in (the M1 contract) is Stim's generated with-reset d=3 rotated memory circuit with
uniform circuit-level noise p, as in ``challenge/raw_stim_d3_surface_code.py``. Its X basis uses
RX/MX, which our hardware circuit will not. The decoding and the fit do not depend on reset;
``dqiqm.surface`` replaces the stand-in once it lands.

Writes, under ``data/sim/decode-fit/``: ``baseline_points.csv`` (one row per basis, round count
and path), ``baseline_fit.json`` (the checks, every fit and the comparisons) and two plots.

Run:
    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
        uv run python scripts/decode_baseline.py
"""

from __future__ import annotations

import os

# The fits do tiny linear algebra: threaded BLAS only spins, on a shared machine. Set before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from dqiqm import decode, stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "sim" / "decode-fit"
SEED = 2026
BASES = ("Z", "X")


def standin(basis: str, rounds: int, p: float, distance: int = 3) -> stim.Circuit:
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def seed_for(path: str, basis: str, rounds: int) -> int:
    """A fixed seed per (path, basis, round count), so every sample is reproducible and independent."""
    return SEED + {"e2e": 0, "direct": 100_000, "noiseless": 200_000}[path] + 1_000 * BASES.index(basis) + rounds


def check_circuit(circuit: stim.Circuit, basis: str, rounds: int, distance: int) -> dict:
    """Noiseless (through the record path), counts, DEM and distance."""
    noiseless = circuit.without_noise()
    rec = noiseless.compile_sampler(seed=seed_for("noiseless", basis, rounds)).sample(1000).astype(np.uint8)
    dets, obs = decode.detection_events(noiseless, rec)
    dem = circuit.detector_error_model(decompose_errors=True)
    out = {
        "basis": basis,
        "rounds": rounds,
        "num_measurements": circuit.num_measurements,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "noiseless_shots": len(rec),
        "noiseless_detection_events": int(dets.sum()),
        "noiseless_observable_flips": int(obs.sum()),
        "dem_errors": sum(1 for i in dem.flattened() if i.type == "error"),
        "graphlike_distance": len(circuit.shortest_graphlike_error()),
    }
    ok = (
        out["noiseless_detection_events"] == 0
        and out["noiseless_observable_flips"] == 0
        and out["num_detectors"] == (distance**2 - 1) * rounds
        and out["num_observables"] == 1
        and out["num_measurements"] == (distance**2 - 1) * rounds + distance**2
        and out["graphlike_distance"] == distance
    )
    out["ok"] = bool(ok)
    return out


def write_job(job_dir: Path, records: list[np.ndarray], shots: int, names: list[str]) -> None:
    """Records in dqiqm.hw.save_raw's layout: memory.npz (c<i>, uint8, column j = measurement j) + meta.json."""
    job_dir.mkdir(parents=True)
    np.savez_compressed(job_dir / "memory.npz", **{f"c{i}": r for i, r in enumerate(records)})
    meta = {
        "job_id": f"sim-{job_dir.name}",
        "task": "decode-fit",
        "device": "stim",
        "shots": shots,
        "circuits": len(records),
        "circuit_names": names,
        "success": True,
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def compare(a: stats.EpsFit, b: stats.EpsFit) -> dict:
    sigma = float(np.hypot(a.eps_se, b.eps_se))
    diff = a.eps - b.eps
    return {
        "eps_e2e": a.eps,
        "eps_direct": b.eps,
        "diff": diff,
        "sigma_diff": sigma,
        "z": diff / sigma,
        "e2e_in_direct_bootstrap_ci": bool(b.eps_ci_bootstrap[0] <= a.eps <= b.eps_ci_bootstrap[1]),
        "direct_in_e2e_bootstrap_ci": bool(a.eps_ci_bootstrap[0] <= b.eps <= a.eps_ci_bootstrap[1]),
        "agree": bool(abs(diff) <= stats.Z95 * sigma),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p", type=float, default=0.003, help="uniform circuit-level noise (default 0.003)")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--shots", type=int, default=100_000, help="shots per round count and path")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    checks, points = [], []
    counts = {(b, path): ([], [], []) for b in BASES for path in ("e2e", "direct")}
    with tempfile.TemporaryDirectory(prefix="decode-fit-") as scratch:
        for basis in BASES:
            circuits = [standin(basis, r, args.p, args.distance) for r in args.rounds]
            for r, c in zip(args.rounds, circuits):
                checks.append(check_circuit(c, basis, r, args.distance))

            # End to end, in hardware format: records -> memory.npz -> load_memory -> m2d -> decode.
            records = [
                c.compile_sampler(seed=seed_for("e2e", basis, r)).sample(args.shots).astype(np.uint8)
                for r, c in zip(args.rounds, circuits)
            ]
            job = Path(scratch) / f"{basis.lower()}-basis"
            write_job(job, records, args.shots, [f"d{args.distance}-{basis}-r{r}" for r in args.rounds])
            del records
            for i, (r, c) in enumerate(zip(args.rounds, circuits)):
                rec, _ = decode.load_memory(job, i, circuit=c)
                dets, obs = decode.detection_events(c, rec)
                le = decode.logical_error_probability(decode.decode(c, dets), obs)
                for col, v in zip(counts[(basis, "e2e")], (r, le.k, le.n)):
                    col.append(v)

            # Direct: detection events straight from Stim, PyMatching straight from the circuit's model.
            for r, c in zip(args.rounds, circuits):
                matcher = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
                dets, obs = c.compile_detector_sampler(seed=seed_for("direct", basis, r)).sample(
                    args.shots, separate_observables=True
                )
                k = int(np.any(matcher.decode_batch(dets) != obs, axis=1).sum())
                for col, v in zip(counts[(basis, "direct")], (r, k, args.shots)):
                    col.append(v)

    fits, comparisons = {}, {}
    for (basis, path), (r, k, n) in counts.items():
        e, elo, ehi = stats.eps_single(k, n, r)
        lo, hi = stats.wilson(np.array(k), np.array(n))
        for i in range(len(r)):
            points.append({
                "basis": basis, "path": path, "rounds": r[i], "shots": n[i], "k": k[i], "p_l": k[i] / n[i],
                "wilson_lo": float(lo[i]), "wilson_hi": float(hi[i]),
                "eps_single": float(e[i]), "eps_single_lo": float(elo[i]), "eps_single_hi": float(ehi[i]),
            })
        for free in (False, True):
            fits[(basis, path, free)] = stats.fit_eps(r, k, n, free_amplitude=free, bootstrap=args.bootstrap, seed=SEED)
    for basis in BASES:
        for free in (False, True):
            comparisons[f"{basis}, {'free A' if free else 'A = 1'}"] = compare(
                fits[(basis, "e2e", free)], fits[(basis, "direct", free)]
            )
        # per point: two independent estimates of the same P_L
        (r, ke, n), (_, kd, _) = counts[(basis, "e2e")], counts[(basis, "direct")]
        pe, pd = np.array(ke) / np.array(n), np.array(kd) / np.array(n)
        pool = (np.array(ke) + np.array(kd)) / (2 * np.array(n))
        z = (pe - pd) / np.sqrt(pool * (1 - pool) * 2 / np.array(n))
        comparisons[f"{basis}, per point"] = {"rounds": r, "z": z.tolist(), "max_abs_z": float(np.abs(z).max())}

    with open(args.out / "baseline_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(points[0]), lineterminator="\n")
        w.writeheader()
        for row in points:
            w.writerow({kk: (f"{vv:.6g}" if isinstance(vv, float) else vv) for kk, vv in row.items()})
    summary = {
        "what": "decode-fit simulation baseline: Stim stand-in (with reset), end to end from measurement records "
        "vs a direct detector-sampler estimate",
        "circuit": f"stim surface_code:rotated_memory_{{z,x}}, distance {args.distance}, uniform p = {args.p} "
        "(after_clifford_depolarization, before_measure_flip_probability, after_reset_flip_probability, "
        "before_round_data_depolarization)",
        "decoder": "pymatching 2.4.0, MWPM, from the circuit's own detector error model (decompose_errors=True)",
        "shots_per_point": args.shots,
        "rounds": args.rounds,
        "seed": SEED,
        "bootstrap": args.bootstrap,
        "stim": stim.__version__,
        "pymatching": pymatching.__version__,
        "checks": checks,
        "fits": {f"{b}, {path}, {'free A' if free else 'A = 1'}": f.to_dict() for (b, path, free), f in fits.items()},
        "comparisons": comparisons,
    }
    (args.out / "baseline_fit.json").write_text(json.dumps(summary, indent=2) + "\n")

    title = f"d = {args.distance} rotated memory, Stim stand-in (with reset), uniform p = {args.p}"
    series = {f"{b} basis": [fits[(b, "e2e", True)], fits[(b, "e2e", False)]] for b in BASES}
    src = f"data/sim/decode-fit/baseline_points.csv (path = e2e), {args.shots:,} shots per point"
    stats.plot_fidelity_decay(series, title=title, source=src, path=args.out / "baseline_decay.png")
    stats.plot_fidelity_decay(series, quantity="F", title=title, source=src, path=args.out / "baseline_decay_logF.png")

    # The report, for the PR body.
    ok = all(c["ok"] for c in checks)
    print(f"checks: {len(checks)} circuits, all noiseless/counts/DEM/distance ok: {ok}")
    for basis in BASES:
        print(f"\n{basis} basis, {args.shots:,} shots per point")
        print("| rounds | path | k | P_L | Wilson 95% | eps (this n alone) | 95% |")
        print("|---|---|---|---|---|---|---|")
        for row in points:
            if row["basis"] == basis:
                print(
                    f"| {row['rounds']} | {row['path']} | {row['k']} | {row['p_l']:.5f} | "
                    f"[{row['wilson_lo']:.5f}, {row['wilson_hi']:.5f}] | {row['eps_single']:.5f} | "
                    f"[{row['eps_single_lo']:.5f}, {row['eps_single_hi']:.5f}] |"
                )
    print("\n| basis | path | form | eps_L | 95% curvature | 95% bootstrap | A | chi2/dof | max abs resid |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (b, path, free), f in fits.items():
        a = f"{f.amplitude:.4f} [{f.amplitude_ci_bootstrap[0]:.4f}, {f.amplitude_ci_bootstrap[1]:.4f}]" if free else "1"
        print(
            f"| {b} | {path} | {'free A' if free else 'A = 1'} | {f.eps:.6f} | "
            f"[{f.eps_ci_curvature[0]:.6f}, {f.eps_ci_curvature[1]:.6f}] | "
            f"[{f.eps_ci_bootstrap[0]:.6f}, {f.eps_ci_bootstrap[1]:.6f}] | {a} | "
            f"{f.chi2:.1f}/{f.dof} | {np.abs(f.residuals).max():.2f} |"
        )
    print("\n| comparison | eps e2e | eps direct | diff | sigma | z | agree (|z| < 1.96) |")
    print("|---|---|---|---|---|---|---|")
    agree = True
    for name, c in comparisons.items():
        if "per point" in name:
            print(f"| {name} | | | | | max abs z = {c['max_abs_z']:.2f} | |")
            continue
        agree &= c["agree"]
        print(
            f"| {name} | {c['eps_e2e']:.6f} | {c['eps_direct']:.6f} | {c['diff']:+.6f} | "
            f"{c['sigma_diff']:.6f} | {c['z']:+.2f} | {c['agree']} |"
        )
    print(f"\nwrote {args.out}/baseline_points.csv, baseline_fit.json, baseline_decay.png, baseline_decay_logF.png")
    print(f"elapsed {time.time() - t0:.1f} s")
    return 0 if ok and agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
