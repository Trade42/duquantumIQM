"""The closure test: simulate C1 and C2 with known coherent errors and check the analysis recovers them.

    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
        uv run python scripts/garnet_calib_closure.py              # ~25 min; writes InfoRecord/characterization/sim/
    ... scripts/garnet_calib_closure.py --replicas 2 --none 1 --out <scratch dir>   # a quick look

Everything here is SIMULATED, on the stand-in (IQMFakeGarnet with the live calibration's noise,
``dqiqm.experiments.device.fake_backend``), seeded. Each world runs the plan's jobs (C1, then C2
layer by layer) at the plan's shots, through the real path: the placed and checked circuits,
their twins with the injected errors (``characterization.inject``), Aer MPS, ``hw.save_raw`` into
``sim/raw/``, and the analysis reading ``memory.npz`` back (``characterization.analysis``).

* ``noiseless``: no noise, nothing injected. Every estimate must be about 0, and every circuit's
  counts must agree with the exact ideal probabilities.
* ``none``: the calibrated noise, nothing injected. Every estimate must be about 0.
* ``injected``: the calibrated noise plus a fresh seeded set of errors per replica (per qubit
  eps, eta in +-0.03 rad; per coupler delta, phi_a, phi_b in +-0.08 rad). Every estimate must
  hold the injected value within its 95% interval about 95% of the time.

Then the pilot, dry: its 13 circuits on the stand-in must give GO, and a mock of a merged run (the
N = 2 circuit's data in place of N = 42, n = 1's in place of n = 17) must give STOP.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dqiqm import hw  # noqa: E402
from dqiqm.characterization import analysis as A  # noqa: E402
from dqiqm.characterization import circuits as CC  # noqa: E402
from dqiqm.characterization import fit as F  # noqa: E402
from dqiqm.characterization import inject as IJ  # noqa: E402
from dqiqm.characterization import model as Mo  # noqa: E402
from dqiqm.characterization import patch as P  # noqa: E402
from dqiqm.characterization import plan as PL  # noqa: E402
from dqiqm.characterization import table as T  # noqa: E402
from dqiqm.experiments import device as dev  # noqa: E402

TASK = "garnet-calib-closure"
SEED = 2026
THREADS = 4
SIM = REPO / "InfoRecord" / "characterization" / "sim"
LABEL = "SIMULATED"


def job_seed(label: str) -> int:
    return SEED + zlib.crc32(label.encode()) % 1_000_000


def stand_in_name(cal) -> str:
    return f"IQMFakeGarnet+profile({cal.calibration_set_id[:8]})"


def run_job(label: str, placed: list, be, errors: IJ.Errors | None, shots: int, *, noisy: bool, out: Path, cal,
            patch, day: str, purpose: str) -> Path:
    circuits = [IJ.twin(p, errors or IJ.Errors(), be) for p in placed]
    seed = job_seed(label)
    res = IJ.aer(be, seed=seed, noisy=noisy, threads=THREADS).run(circuits, shots=shots, memory=True).result()
    if not res.success:
        raise RuntimeError(f"{label}: Aer failed: {res.status}")
    res.job_id = f"sim-{label}"
    mapping = {i: q for i, q in enumerate(patch.qubits)}
    return hw.save_raw(res, task=TASK, device=stand_in_name(cal), qubit_mapping=[mapping] * len(placed), shots=shots,
                       physical_qubits=[list(patch.qubits)] * len(placed),
                       purpose=f"{LABEL} closure, job {label}, Aer seed {seed}: {purpose}", day=day, data_root=out / "raw")


def exact_ideal(design: CC.Design, patch) -> np.ndarray:
    """Ideal P1 of every column (noise-free, no errors), rounded to 1e-12 so 0, 1/2 and 1 are exact."""
    p = np.zeros(len(patch.qubits))
    if design.params["family"] == "prx_amplification":
        seq = tuple((float(a), int(b)) for a, b in design.params["sequence"])
        v = Mo.p_read1(seq, Mo.QubitModel())
        for q in design.params["driven"]:
            p[patch.index(q)] = v
    else:
        for t, c in zip(design.params["targets"], design.params["controls"]):
            r = Mo.simulate_design(design.layers, (t, c))
            p[patch.index(t)], p[patch.index(c)] = r[t], r[c]
    return np.round(p, 12)


def noiseless_check(designs, arrays, patch) -> dict:
    """Every column's count against its exact ideal probability: deterministic columns exactly, the
    others by z = (P_hat - p) / sqrt(p (1 - p) / shots)."""
    zs = []
    for des, arr in zip(designs, arrays):
        p_hat = arr.mean(axis=0)
        p = exact_ideal(des, patch)
        det = (p == 0) | (p == 1)
        if np.any(p_hat[det] != p[det]):
            raise AssertionError(f"{des.name}: a deterministic outcome was not deterministic")
        zs.append((p_hat[~det] - p[~det]) / np.sqrt(p[~det] * (1 - p[~det]) / arr.shape[0]))
    z = np.concatenate(zs)
    return {"deterministic_columns": int(len(designs) * len(patch.qubits) - z.size),
            "random_columns": int(z.size), "mean_z": float(z.mean()), "std_z": float(z.std()),
            "max_abs_z": float(np.abs(z).max()), "deterministic_columns_exact": True}


def world(name: str, r: int, *, be, cal, patch, pri, placed_jobs, noisy: bool, errors, out: Path, day: str, log):
    t0 = time.time()
    tag = f"{name}-r{r:02d}"
    dirs, designs, arrays = [], [], []
    for label, placed, shots in placed_jobs:
        d = run_job(f"{tag}-{label}", placed, be, errors, shots, noisy=noisy, out=out, cal=cal, patch=patch, day=day,
                    purpose=f"world {name}, replica {r}")
        meta, arr = A.load_job(d)
        dirs.append(str(d.relative_to(out)))
        designs += [p.design for p in placed]
        arrays += arr
    t_sim = time.time() - t0
    c1 = A.c1_fits(A.c1_data(designs, arrays, patch, pri), pri)
    c2f = A.c2_fits(A.c2_points(designs, arrays, patch))
    cp = A.couplers(c2f)
    common = A.common_fields(pri, timestamp_utc=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), simulated=True,
                             seed=job_seed(f"{tag}-C1"), status=f"{LABEL} closure ({name})")
    inj_prx = {} if errors is None else errors.prx
    inj_cz = {} if errors is None else errors.cz
    rows = A.c1_rows(c1, pri, experiment="C1", context="parallel (all 17)", common=common,
                     injected={q: inj_prx.get(q, (0.0, 0.0)) for q in patch.qubits})
    rows += A.c2_rows(cp, pri, experiment="C2", context="CZ layer (6 couplers in parallel)", common=common,
                      injected={p: inj_cz.get(p, (0.0, 0.0, 0.0)) for p in patch.couplers})
    d = out / "derived" / name / f"r{r:02d}"
    d.mkdir(parents=True, exist_ok=True)
    T.write(rows, d / "table.csv")
    extra = {}
    if name == "noiseless":
        extra["ideal_vs_counts"] = noiseless_check(designs, arrays, patch)
    summary = {"label": LABEL, "world": name, "replica": r, "simulator": {
        "backend": stand_in_name(cal), "aer_method": "matrix_product_state", "noisy": noisy, "threads": THREADS,
        "noise": "IQMFakeGarnet error profile from the live snapshot (dqiqm.experiments.device.error_profile, Ramsey T2; "
                 "readout errors are measure's, not measure_fidelity's)"},
        "injected": None if errors is None else errors.to_json(), "raw": dirs,
        "seconds_simulating": t_sim, "coverage": A.coverage(rows), **extra,
        "c1": {q: f.to_json() for q, f in c1.items()},
        "c2": {f"{o}:{t}<-{c}": f.to_json() for (o, t, c), f in c2f.items()}}
    (d / "fits.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    log(f"{tag}: simulated in {t_sim:.0f} s, analysed in {time.time() - t0 - t_sim:.0f} s; coverage "
        + ", ".join(f"{k} {v['covered']}/{v['n']}" for k, v in summary["coverage"].items()))
    return rows, summary


def pilot_dry(be, cal, patch, placed_by_name, out: Path, day: str, plan: dict, log) -> dict:
    order = plan["pilot"]["circuits"]
    placed = [placed_by_name[n] for n in order]
    d = run_job("pilot-P0", placed, be, None, plan["pilot"]["shots"], noisy=True, out=out, cal=cal, patch=patch, day=day,
                purpose="the pilot P0 on the stand-in")
    _, arrays = A.load_job(d)
    stats = PL.pilot_stats(arrays, order, patch)
    merged = list(arrays)
    merged[order.index("C1_amp_x_n42")] = arrays[order.index("C1_amp_x_n02")]
    for k in range(4):
        merged[order.index(f"C2_L1_tanc_s1_n17_q{k}")] = arrays[order.index(f"C2_L1_tanc_s1_n01_q{k}")]
    mstats = PL.pilot_stats(merged, order, patch)
    b = plan["pilot"]

    def verdict(s):
        checks = {
            "shape": all(a.shape == (b["shots"], b["columns"]) for a in arrays),
            "readout_n00": max(s["p1_n00"]) <= b["readout"]["p1_n00_max"],
            "readout_n02": min(s["p1_n02"]) >= b["readout"]["p1_n02_min"],
            "c1_no_merge": s["D"] >= b["c1_no_merge"]["stop_if_below"],
            "c2_no_merge": s["R"] <= b["c2_no_merge"]["stop_if_above"],
        }
        return {"checks": checks, "verdict": "GO" if all(checks.values()) else "STOP"}

    res = {"label": LABEL, "raw": str(d.relative_to(out)), "stand_in": {**stats, **verdict(stats)},
           "mock_merged": {**mstats, **verdict(mstats),
                           "how": "C1_amp_x_n02's record in place of n42, and n01's in place of n17: what the device "
                                  "would return if the server collapsed each chain"}}
    log(f"pilot dry run: stand-in D={stats['D']:.4f} R={stats['R']:.4f} -> {res['stand_in']['verdict']}; "
        f"mock merged D={mstats['D']:.4f} R={mstats['R']:.4f} -> {res['mock_merged']['verdict']}")
    return res


def pooled(summaries: list[dict]) -> dict:
    out: dict = {}
    for s in summaries:
        for k, v in s["coverage"].items():
            o = out.setdefault(k, {"n": 0, "covered": 0, "max_abs_z": 0.0})
            o["n"] += v["n"]
            o["covered"] += v["covered"]
            o["max_abs_z"] = max(o["max_abs_z"], v["max_abs_z"])
    for o in out.values():
        o["fraction"] = o["covered"] / o["n"]
        # the binomial 95% range of the covered count if the intervals were exact
        sd = math.sqrt(o["n"] * 0.95 * 0.05)
        o["expected_range"] = [0.95 * o["n"] - 1.96 * sd, 0.95 * o["n"] + 1.96 * sd]
    return out


def zero_summary(rows_by_world: dict[str, list[list[dict]]]) -> dict:
    """For the worlds with nothing injected: the largest |estimate| / se and |estimate| per kind."""
    out = {}
    for w, reps in rows_by_world.items():
        o: dict = {}
        for rows in reps:
            for r in rows:
                kind = "phi" if str(r["parameter"]).startswith("phi_") else str(r["parameter"])
                v, se = float(r["fit_value"]), float(r["se"])
                k = o.setdefault(kind, {"n": 0, "covered": 0, "max_abs": 0.0, "max_abs_z": 0.0})
                k["n"] += 1
                k["covered"] += int(float(r["fit_ci_low"]) <= 0.0 <= float(r["fit_ci_high"]))
                k["max_abs"] = max(k["max_abs"], abs(v))
                k["max_abs_z"] = max(k["max_abs_z"], abs(v) / se if se > 0 else 0.0)
        out[w] = o
    return out


def recheck_noiseless(out: Path) -> int:
    """Re-run :func:`noiseless_check` on the saved noiseless records and update the two files that hold it."""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    patch = P.pilot_patch(IQMFakeGarnet())
    by_name = {d.name: d for d in CC.c1_designs(patch) + CC.c2_designs(patch)}
    fits_path = out / "derived" / "noiseless" / "r01" / "fits.json"
    fits = json.loads(fits_path.read_text())
    plan = json.loads((REPO / "InfoRecord" / "characterization" / "job_plan.json").read_text())
    order = {j["label"]: j["order"] for j in plan["jobs"]}
    designs, arrays = [], []
    for rel in fits["raw"]:
        meta, arr = A.load_job(out / rel)
        label = rel.rsplit("sim-noiseless-r01-", 1)[1]  # Aer's results carry no circuit names: the order is the plan's
        designs += [by_name[n] for n in order[label]]
        arrays += arr
    check = noiseless_check(designs, arrays, patch)
    fits["ideal_vs_counts"] = check
    fits_path.write_text(json.dumps(fits, indent=1, default=float) + "\n")
    closure_path = out / "closure.json"
    closure = json.loads(closure_path.read_text())
    closure["noiseless_ideal_vs_counts"] = check
    closure_path.write_text(json.dumps(closure, indent=1, default=float) + "\n")
    print(json.dumps(check))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replicas", type=int, default=10, help="injected worlds")
    ap.add_argument("--none", type=int, default=5, help="calibrated-noise worlds with nothing injected")
    ap.add_argument("--out", type=Path, default=SIM)
    ap.add_argument("--clean", action="store_true", help="delete --out first")
    ap.add_argument("--recheck-noiseless", action="store_true",
                    help="redo only the noiseless world's count check from its saved raw records (no simulation)")
    a = ap.parse_args()
    if a.recheck_noiseless:
        return recheck_noiseless(a.out)
    out = a.out
    if a.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "closure.log").open("w")

    def log(msg: str) -> None:
        line = f"{dt.datetime.now(dt.UTC):%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    day = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    cal = dev.live_calibration()
    be = dev.fake_backend(cal)
    patch = P.pilot_patch(be, cal)
    pri = P.priors(patch, cal)
    plan = json.loads((REPO / "InfoRecord" / "characterization" / "job_plan.json").read_text())
    designs = CC.c1_designs(patch) + CC.c2_designs(patch) + CC.c2_pilot_designs(patch)
    placed_by_name = {}
    from dqiqm import layout as L

    arch = L.Architecture.from_backend(be)
    for d in designs:
        p = CC.place(d, patch, be)
        CC.check(p, arch, patch.couplers)
        placed_by_name[d.name] = p
    jobs = [(j["label"], [placed_by_name[n] for n in j["order"]], j["shots"]) for j in plan["jobs"] if j["phase"] == "must"]
    log(f"{LABEL}: stand-in {stand_in_name(cal)}; jobs " + ", ".join(f"{l} {len(p)}x{s}" for l, p, s in jobs))

    summaries: dict[str, list[dict]] = {"noiseless": [], "none": [], "injected": []}
    rows_by_world: dict[str, list[list[dict]]] = {"noiseless": [], "none": []}
    kw = dict(be=be, cal=cal, patch=patch, pri=pri, placed_jobs=jobs, out=out, day=day, log=log)
    rows, s = world("noiseless", 1, noisy=False, errors=None, **kw)
    summaries["noiseless"].append(s)
    rows_by_world["noiseless"].append(rows)
    for r in range(1, a.none + 1):
        rows, s = world("none", r, noisy=True, errors=None, **kw)
        summaries["none"].append(s)
        rows_by_world["none"].append(rows)
    for r in range(1, a.replicas + 1):
        errors = IJ.random_errors(patch.qubits, patch.couplers, seed=1000 + r)
        _, s = world("injected", r, noisy=True, errors=errors, **kw)
        summaries["injected"].append(s)
    pilot = pilot_dry(be, cal, patch, placed_by_name, out, day, plan, log)
    result = {
        "label": LABEL, "device": P.DEVICE, "calibration_set_id": pri.calibration_set_id, "stand_in": stand_in_name(cal),
        "jobs": [{"label": l, "circuits": len(p), "shots": s} for l, p, s in jobs],
        "worlds": {"noiseless": 1, "none": a.none, "injected": a.replicas},
        "injected_ranges_rad": {"eps": 0.03, "eta": 0.03, "delta": 0.08, "phi": 0.08},
        "coverage_injected": pooled(summaries["injected"]),
        "nothing_injected": zero_summary(rows_by_world),
        "noiseless_ideal_vs_counts": summaries["noiseless"][0].get("ideal_vs_counts"),
        "pilot": pilot,
        "seeds": {"aer": f"{SEED} + crc32(job label) % 1e6", "errors": "1000 + replica"},
    }
    (out / "closure.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    log("coverage (injected): " + ", ".join(f"{k} {v['covered']}/{v['n']} ({v['fraction']:.3f})"
                                             for k, v in result["coverage_injected"].items()))
    for w, o in result["nothing_injected"].items():
        log(f"nothing injected, {w}: " + ", ".join(f"{k} max |est| {v['max_abs']:.2e} rad, {v['covered']}/{v['n']} hold 0"
                                                     for k, v in o.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
