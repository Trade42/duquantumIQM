"""The Emerald dry run: E1 (and E4) on the stand-in Emerald, written as a full InfoRecord under InfoRecord/emerald/.

    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
        uv run python scripts/emerald_dryrun.py [--clean]

The stand-in is ``IQMFakeAphrodite().copy_with_error_profile(...)`` with the live Emerald calibration
of 2026-10-24 (set 23e12f6b), the 5 CZ-less couplers padded (``dqiqm.experiments.emerald``),
simulated by Aer's matrix-product-state method, seeded, on at most 4 threads. Every job of the
Emerald plan is simulated at small shot counts, in its planned circuit order, through the hardware
path: an Aer ``Result`` saved by ``dqiqm.hw.save_raw`` into ``InfoRecord/emerald/sim/raw/``, read
back by ``dqiqm.decode.load_memory``, then written as ``datasets/``, ``derived/`` and ``ler/`` by
``dqiqm.inforecord``. Its functions are copied from ``scripts/garnet_dryrun.py`` (not importable:
module globals) with Emerald's paths and labels.

It writes, outside ``sim/``: the planned hardware circuits (``circuits/<experiment>/``, not
``job_plan.json``, which only ``scripts/emerald_plan.py`` writes), ``architecture/emerald.json``,
and the calibration package (``calibration/emerald/``). In ``sim/checks/``:

* ``structure.json``: per circuit, counts, the DEM of its noisy copy (``decompose_errors=True``),
  Stim's graphlike distance, and the live-coupler check;
* ``noiseless.json``: every circuit run noiselessly through the same path: 0 events, 0 flips;
* ``decode.json``: E1 decoded by PyMatching (noisy copy of the circuit that ran) against "always 0";
* ``crosscheck.json``: E1 and E4 detectors against the team's ``make_stim_circuit`` (read-only);
* ``d5.json``: the d=5 feasibility answer;
* ``summary.json`` and ``log.txt``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from dqiqm import decode as D  # noqa: E402
from dqiqm import hw  # noqa: E402
from dqiqm import inforecord as IR  # noqa: E402
from dqiqm import stats  # noqa: E402
from dqiqm.experiments import device as dev  # noqa: E402
from dqiqm.experiments import emerald as EM  # noqa: E402
from dqiqm.experiments import suite as SU  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORD = EM.RECORD
SIM = RECORD / "sim"
TASK = "emerald-dryrun"
THREADS = 4
SEED = 2026
TEAM_MAIN = Path("/home/sal/duquantumIQM/worktrees/team-main")

#: Dry-run shots per circuit (the hardware plan's are in dqiqm.experiments.emerald.SHOTS).
DRY_SHOTS = {"P0": 1000, "E1": 96, "E4": 48}

DECODER_NOTE = ("dqiqm.experiments.emerald.calibrated_noise(live Emerald calibration, idle_ns=0) = "
                "device.calibrated_noise(idle_ns=0): the stand-in's own noise (gate depolarizing + relaxation over gate "
                "durations, mean readout assignment error as record flips)")
STAND_IN_NOISE = ("IQMFakeAphrodite error profile from the live snapshot (dqiqm.experiments.emerald.error_profile: "
                  "device.error_profile, Ramsey T2, the 5 CZ-less couplers padded at depolarizing 1.0 and never used); "
                  "no noise on delay or idle qubits")


def aer(backend, *, noisy: bool = True, method: str = "matrix_product_state", seed: int = SEED):
    from qiskit_aer import AerSimulator

    return AerSimulator(method=method, noise_model=backend.noise_model if noisy else None, seed_simulator=seed,
                        max_parallel_threads=THREADS)


def job_seed(label: str) -> int:
    return SEED + zlib.crc32(label.encode()) % 1_000_000


def decoder_for(spec, cal):
    if spec.noisy is None or not spec.stim.num_observables:
        return None
    return spec.noisy(EM.calibrated_noise(cal, spec, idle_ns=0.0))


def _nan_to_none(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _nan_to_none(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_nan_to_none(x) for x in v]
    return v


def tag_json(path: Path, **keys) -> None:
    """Add ``keys`` at the top of a JSON object file (NaN written as null)."""
    body = json.loads(path.read_text())
    path.write_text(json.dumps(_nan_to_none({**keys, **body}), indent=1) + "\n")


def tag_placements(root: Path) -> None:
    """``device`` on every circuit of every ``circuits/<experiment>/placement.json``."""
    for p in sorted((root / "circuits").glob("*/placement.json")):
        body = json.loads(p.read_text())
        p.write_text(json.dumps({k: {"device": EM.DEVICE, **v} for k, v in body.items()}, indent=1, sort_keys=True) + "\n")


def write_architecture(root: Path, snap, arch, *, source: str) -> Path:
    """``inforecord.write_architecture``, as ``architecture/emerald.json`` with ``device``."""
    garnet_named = IR.write_architecture(root, snap, arch, source=source)
    out = garnet_named.with_name(f"{EM.DEVICE}.json")
    body = json.loads(garnet_named.read_text())
    garnet_named.unlink()
    out.write_text(json.dumps({"device": EM.DEVICE, **body,
                               "cz_less_couplers": [list(p) for p in EM.CZ_LESS],
                               "note": "couplers: the 85 with a CZ in this calibration set; the static architecture has 90"},
                              indent=1) + "\n")
    return out


def calibration_readme(cal, label: str | None = None) -> str:
    return f"""# Emerald calibration{f' ({label})' if label else ''}: the pre-training package

Device **emerald**. Calibration set `{cal.calibration_set_id}`, fetched {cal.fetched_utc} UTC, read from
`{cal.source}` (`dqiqm.experiments.device.calibration_from_snapshot`). The snapshot itself,
`20261024T191321Z-{cal.calibration_set_id}.json`, is a byte copy of the resonance-check task's
`data/calibration/emerald/{cal.calibration_set_id}.20261024T191321Z.json`.

| File | What |
|---|---|
| `qubits.csv` | per qubit (54): T1, T2 Ramsey, T2 echo, readout P(1&#124;0) and P(0&#124;1), the 1Q error, readout leakage, durations; times in ns |
| `couplers.csv` | per calibrated coupler (85 of the 90 static ones): CZ error (interleaved RB), CZ error (Clifford RB), CZ duration |
| `durations.json` | prx, CZ and measure durations per qubit or coupler (ns), and `reset_wait_ns`, the passive-reset wait between shots |
| `placement.json` | E1's two patches eA and eB: Stim qubit -> device qubit, its role (data, X_ancilla, Z_ancilla), check and Stim coordinates; the full ranking |

## Where each figure comes from

| Column | Snapshot field |
|---|---|
| `t1_ns` | `characterization.model.<q>.t1_time` |
| `t2_ramsey_ns` | `characterization.model.<q>.t2_time` (QB3 has none: `device.t2_ns` falls back to the echo T2 there) |
| `t2_echo_ns` | `characterization.model.<q>.t2_echo_time` |
| `p1_given_0`, `p0_given_1` | `metrics.ssro.measure.constant.<q>.error_0_to_1` / `error_1_to_0`: the default `measure` circuits use, **not** the heralded `measure_fidelity` figure of `hw.calibration_fidelities` (optimistic) |
| `readout_error` | the mean of the two |
| `error_1q` | 1 − `metrics.rb.prx.drag_crf_sx.<q>.fidelity` |
| `readout_leakage` | `metrics.rrc.measure.constant.<q>.leakage` |
| `prx_ns`, `measure_ns` | `gates.prx.drag_crf_sx.<q>.duration`, `gates.measure.constant.<q>.duration` (seconds in the snapshot) |
| `cz_error_irb` | 1 − `metrics.irb.cz.<impl>.<a>__<b>.fidelity` (either order; Clifford RB if missing) |
| `cz_error_clifford_rb` | 1 − `metrics.rb.clifford.uz_cz.<a>__<b>.fidelity` |
| `cz_ns` | `gates.cz.<impl>.<a>__<b>.duration` |
| `reset_wait_ns` | the largest `gates.reset_wait.reset_wait.<q>.duration`: {cal.reset_wait_ns / 1e3:.0f} µs |

**The 5 CZ-less couplers** (QB6–QB7, QB15–QB23, QB24–QB32, QB49–QB50, QB50–QB54) are not in
`couplers.csv`; no planned circuit uses them.

## Pre-training: read this first

- **Pre-train on noise built from this package, with idle terms**: the T1/T2 decay of every qubit
  over the ancilla readout window (the slowest ancilla's `measure_ns`, about 360 ns here) plus any
  delay, on top of the gate and readout errors. The data qubits idle through every ancilla
  readout; without reset, a decayed ancilla also fires detectors two rounds apart.
- **Do not pre-train on `sim/`.** The stand-in there has no idle noise at all and roughly doubles
  the gate errors (Aer adds relaxation on top of RB figures that already include it), so its
  shots are not Emerald's.
- **Choose the placement by each shot's `patch`** (`eA` or `eB`, in `shots.csv` and the manifest's
  `spec.patch`): the two patches are different qubit sets, so a noise model built for one does not
  fit the other.
- The ML split's `validation` is the fine-tuning side's held-out 20%: load it as finetune or
  validation, **never as test**. Only batch 3 is test.

Regenerate offline from the committed snapshot: `uv run python scripts/emerald_plan.py` (it also
rewrites the job plan). Every later snapshot (before and after each batch) gets its own package.
"""


def run_jobs(suite, jobs, cal, *, day: str, log, files) -> list[dict]:
    """Simulate every job, save its raw result, and write its datasets and derived features."""
    runs = []
    cal_rel = {"before": CAL_REL, "after": CAL_REL}
    for job in jobs:
        specs = [suite[n] for n in job.circuits]
        seed = job_seed(job.label)
        shots = DRY_SHOTS["P0" if job.label == "EP0" else job.experiment]
        t0 = time.time()
        res = aer(suite.backend, seed=seed).run([s.hardware.circuit for s in specs], shots=shots, memory=True).result()
        if not res.success:
            raise RuntimeError(f"{job.label}: Aer failed: {res.status}")
        res.job_id = f"sim-{job.label}"
        mappings = [{i: s.physical[q] for i, q in enumerate(s.hardware.stim_qubits)} for s in specs]
        sent = [sorted(set(m.values()), key=dev.qubit_sort_key) for m in mappings]
        job_dir = hw.save_raw(res, task=TASK, device=EM.stand_in_name(cal), qubit_mapping=mappings, shots=shots,
                              physical_qubits=sent, purpose=f"SIMULATED Emerald dry run, job {job.label}: {job.note}",
                              day=day, data_root=SIM / "raw")
        log(f"{job.label}: {len(specs)} circuits x {shots} shots in {time.time() - t0:.1f} s -> {IR._rel(job_dir, REPO)}")
        for i, s in enumerate(specs):
            raw, meta = D.load_memory(job_dir, i, circuit=s.stim)
            run_id = (f"b{job.batch}_" if job.batch else f"{job.label.lower()}_") + s.name
            dec = decoder_for(s, cal)
            labels = SU.split(job.label, job.batch or 0, shots)
            ds, numbers = IR.write_dataset(
                SIM, s, raw, run_id=run_id, batch=job.batch or 0, job_id=res.job_id, position_in_job=i, arch=suite.arch,
                cal=cal, durations=suite.durations, circuit_files=files[s.name],
                raw_ref={"path": str((job_dir / "memory.npz").relative_to(SIM)), "key": f"c{i}",
                         "meta": str((job_dir / "meta.json").relative_to(SIM))},
                calibration_paths=cal_rel, decoder_circuit=dec, decoder_note=DECODER_NOTE if dec is not None else "",
                split_labels=labels, split_info=SU.split_rule(job.label, job.batch or 0, labels), simulated=True,
                extra={"device": EM.DEVICE, "job_timeline": meta.get("timeline"),
                       "seeds": {"aer_seed_simulator": seed, "circuit_index": i},
                       "simulator": {"backend": EM.stand_in_name(cal), "aer_method": "matrix_product_state",
                                     "noise": STAND_IN_NOISE, "threads": THREADS}},
            )
            der, summary = IR.write_derived(SIM, s, run_id, ds)
            tag_json(der / "summary.json", label="SIMULATED", device=EM.DEVICE, stand_in=EM.stand_in_name(cal))
            runs.append({"job": job.label, "batch": job.batch or 0, "spec": s, "run_id": run_id, "dataset": ds,
                         "derived": der, "numbers": numbers, "summary": summary})
    return runs


def ler_records(runs, cal, log) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        s, p = r["spec"], r["spec"].params
        if r["numbers"]["pm_failures"] is None or not r["batch"]:
            continue
        if s.experiment == "E1":
            key = ("E1", f"{p['basis']}_{p['patch']}_b{r['batch']}",
                   f"Emerald E1 d=3 {p['basis']} basis, patch {p['patch']}, batch {r['batch']}")
        elif s.experiment == "E4":
            key = ("E4", f"{p['basis']}_{p['patch']}_t{int(p['delay_ns'])}ns_b{r['batch']}",
                   f"Emerald E4 d=3 {p['basis']}, {p['delay_ns'] / 1e3:g} us idle between rounds, batch {r['batch']}")
        else:
            continue
        groups[key].append(r)
    out = []
    for (exp, run_id, title), rs in sorted(groups.items()):
        points = [{"circuit": r["spec"].name, "dataset": str(r["dataset"].relative_to(SIM)), "rounds": r["spec"].rounds,
                   "shots": r["numbers"]["shots"], "failures": r["numbers"]["pm_failures"],
                   "raw_failures": r["numbers"]["obs_flips"]} for r in rs]
        path, res = IR.write_ler(SIM, exp, run_id, points, title=f"SIMULATED: {title}", label=run_id)
        tag_json(path / "ler_fits.json", label="SIMULATED", device=EM.DEVICE, stand_in=EM.stand_in_name(cal),
                 decoder="pymatching on the noisy copy of the circuit that ran", noise=DECODER_NOTE,
                 zero_failure_points=[x["rounds"] for x in points if x["failures"] == 0])
        out.append({"experiment": exp, "run_id": run_id, "points": points, "fits": res})
        if "free_A" in res:
            log(f"LER {exp} {run_id}: eps(A=1) {res['fixed_A']['eps']:.4f}, eps(free A) {res['free_A']['eps']:.4f}, "
                f"A {res['free_A']['amplitude']:.3f}")
    return out


def structure(suite, cal) -> list[dict]:
    rows = []
    for s in suite.specs:
        row = {"circuit": s.name, "experiment": s.experiment, "rounds": s.rounds, "detectors": s.stim.num_detectors,
               "measurements": s.stim.num_measurements, "observables": s.stim.num_observables,
               "swaps": s.checks.get("swaps"), "cz": int(s.hardware.circuit.count_ops().get("cz", 0)),
               "live_couplers": EM.check_live_couplers(s, cal),
               "qubits": sorted(set(s.physical.values()), key=dev.qubit_sort_key)}
        dec = decoder_for(s, cal)
        if dec is not None:
            row["dem_errors"] = dec.detector_error_model(decompose_errors=True).num_errors
            row["graphlike_distance"] = len(dec.shortest_graphlike_error())
            row["same_detectors"] = dec.num_detectors == s.stim.num_detectors and dec.num_measurements == s.stim.num_measurements
        rows.append(row)
    return rows


def noiseless(suite, shots: int = 32) -> list[dict]:
    rows = []
    sim = aer(suite.backend, noisy=False, seed=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        for k, s in enumerate(suite.specs):
            res = sim.run(s.hardware.circuit, shots=shots, memory=True).result()
            res.job_id = f"noiseless-{k}"
            d = hw.save_raw(res, task="noiseless", device="aer-noiseless", qubit_mapping=[{}], shots=shots, data_root=Path(tmp))
            raw, _ = D.load_memory(d, 0, circuit=s.stim)
            dets, obs = D.detection_events(s.stim, raw)
            rows.append({"circuit": s.name, "shots": shots, "detection_events": int(dets.sum()),
                         "observable_flips": int(obs.sum()), "random_columns": int(((raw.mean(0) > 0) & (raw.mean(0) < 1)).sum())})
    return rows


def decode_check(runs) -> dict:
    by: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in runs:
        if r["spec"].experiment != "E1" or r["numbers"]["pm_failures"] is None:
            continue
        key = (r["spec"].params["basis"], r["spec"].params["patch"], r["spec"].rounds)
        by[key][0] += r["numbers"]["shots"]
        by[key][1] += r["numbers"]["pm_failures"]
        by[key][2] += r["numbers"]["obs_flips"]
    rows = []
    tot = [0, 0, 0]
    for (basis, patch, rounds), (n, k_pm, k_raw) in sorted(by.items()):
        lo, hi = stats.wilson(k_pm, n)
        rlo, rhi = stats.wilson(k_raw, n)
        rows.append({"basis": basis, "patch": patch, "rounds": rounds, "shots": n, "pm_failures": k_pm, "p_pm": k_pm / n,
                     "p_pm_ci": [lo, hi], "raw_failures": k_raw, "p_raw": k_raw / n, "p_raw_ci": [rlo, rhi]})
        tot = [tot[0] + n, tot[1] + k_pm, tot[2] + k_raw]
    n, k_pm, k_raw = tot
    lo, hi = stats.wilson(k_pm, n)
    rlo, rhi = stats.wilson(k_raw, n)
    return {"label": "SIMULATED", "device": EM.DEVICE, "decoder": DECODER_NOTE, "rows": rows,
            "pooled": {"shots": n, "pm_failures": k_pm, "p_pm": k_pm / n, "p_pm_ci": [lo, hi], "raw_failures": k_raw,
                       "p_raw": k_raw / n, "p_raw_ci": [rlo, rhi], "better_than_always_0": hi < rlo}}


def _record_sets(c):
    """Each DETECTOR and OBSERVABLE_INCLUDE as a frozenset of absolute measurement indices (crosscheck.py's method)."""
    dets, obs, m = [], {}, 0
    for inst in c.flattened():
        if inst.name in ("M", "MR", "MX", "MRX", "MY", "MRY", "MZ", "MRZ"):
            m += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            s: set[int] = set()
            for t in inst.targets_copy():
                s ^= {m + t.value}
            dets.append(frozenset(s))
        elif inst.name == "OBSERVABLE_INCLUDE":
            s = obs.setdefault(int(inst.gate_args_copy()[0]), set())
            for t in inst.targets_copy():
                s ^= {m + t.value}
    return dets, {k: frozenset(v) for k, v in obs.items()}


def crosscheck(suite, team_main: Path) -> dict:
    """Every E1/E4 circuit's detectors against the team's make_stim_circuit: in order, records and coordinates."""
    path = team_main / "challenge" / "surface_code.py"
    if not path.exists():
        return {"skipped": f"{path} not found"}
    import subprocess
    import types

    sys.dont_write_bytecode = True  # read-only: no __pycache__ in the team's worktree
    # The two files only (surface_code and its relative import internal_helpers, both import only
    # numpy and stim), as a package of our own name; the team's challenge/__init__.py is not run.
    pkg = types.ModuleType("team_challenge")
    pkg.__path__ = [str(path.parent)]
    sys.modules["team_challenge"] = pkg
    mod = None
    for name in ("internal_helpers", "surface_code"):
        spec = importlib.util.spec_from_file_location(f"team_challenge.{name}", path.parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    head = subprocess.run(["git", "-C", str(team_main), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for s in suite.specs:
        if s.params.get("family") != "surface_memory":
            continue
        team = mod.make_stim_circuit(3, s.rounds, basis=s.params["basis"], no_reset=True, noise=None)
        td, to = _record_sets(team)
        od, oo = _record_sets(s.stim)
        tc, oc = team.get_detector_coordinates(), s.stim.get_detector_coordinates()
        row = {"circuit": s.name, "detectors": f"{team.num_detectors}/{s.stim.num_detectors}",
               "measurements": f"{team.num_measurements}/{s.stim.num_measurements}",
               "same_detectors_in_order": td == od, "same_coordinates": tc == oc, "same_observable": to == oo}
        row["pass"] = row["same_detectors_in_order"] and row["same_coordinates"] and row["same_observable"] and \
            team.num_measurements == s.stim.num_measurements
        rows.append(row)
    return {"team_generator": str(path), "team_head": head, "method": "orchestration/scratch/gen-crosscheck/crosscheck.py "
            "record sets, compared per detector in order, plus get_detector_coordinates() and the observable",
            "circuits": len(rows), "pass": all(r["pass"] for r in rows), "rows": rows}


def sim_readme(cal, summary) -> str:
    return f"""# SIMULATED: the Emerald dry run

Everything under `sim/` is **simulated**: the stand-in `{EM.stand_in_name(cal)}` (IQMFakeAphrodite,
Emerald's 54 qubits and 90 couplers, with the live noise of calibration set `{cal.calibration_set_id}`;
the 5 CZ-less couplers padded and never used), Aer matrix-product-state method, seeded. It has
**no idle noise**: Aer puts none on `delay` or on idle qubits, so E4 is checked for structure only.

It is the full record's layout at small shot counts ({json.dumps(DRY_SHOTS)} per circuit), so every
reader and every analysis can be built before the hardware data exist. Do not mix it with hardware
data, and do not pre-train on it (see `../calibration/emerald/README.md`): every manifest says
`"label": "SIMULATED"`, every shot `simulated = 1`. The manifests' `device` is `emerald` (the
device the circuits are for); `simulator.backend` and the raw `meta.json`'s `device` name the
stand-in.

Regenerate it (about {summary['wall_s'] / 60:.0f} min wall on 4 cores):

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
    uv run python scripts/emerald_dryrun.py --clean
```

`checks/` holds the run's own checks: `structure.json`, `noiseless.json`, `decode.json`,
`crosscheck.json`, `d5.json`, `summary.json` and the log.
"""


CAL_REL = ""


def main() -> int:
    global CAL_REL, RECORD, SIM
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean", action="store_true", help="remove InfoRecord/emerald/sim/ first (it is regenerable)")
    ap.add_argument("--day", default=None, help="YYYYMMDD for raw/ (default: today, UTC)")
    ap.add_argument("--team-main", default=str(TEAM_MAIN), help="the team's checkout, for the detector cross-check (read only)")
    ap.add_argument("--smoke", metavar="DIR", default=None, help="a small suite at a few shots, into DIR (a pipeline check)")
    args = ap.parse_args()
    if args.smoke:
        RECORD = Path(args.smoke).resolve()
        SIM = RECORD / "sim"
        for k in DRY_SHOTS:
            DRY_SHOTS[k] = 8
    t_start = time.time()
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    if SIM.exists():
        if not args.clean:
            print(f"{SIM} exists; rerun with --clean to regenerate it", file=sys.stderr)
            return 2
        shutil.rmtree(SIM)
    SIM.mkdir(parents=True)
    day = args.day or dt.datetime.now(dt.UTC).strftime("%Y%m%d")

    # The live snapshot, byte-copied into the record once.
    snap_path = EM.DEFAULT_SNAPSHOT
    if not snap_path.exists():
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EM.SNAPSHOT_SOURCE, snap_path)
    if snap_path.read_bytes() != EM.SNAPSHOT_SOURCE.read_bytes():
        raise RuntimeError(f"{snap_path} is not a byte copy of {EM.SNAPSHOT_SOURCE}")
    snap = dev.load_snapshot(snap_path)
    cal = EM.live_calibration(snap_path)
    t0 = time.time()
    if args.smoke:
        suite = EM.build(cal, e1_rounds=(2, 3, 10), e4_rounds=(2,), e4_delays_ns=(1000.0,), rank_shots=2000)
    else:
        suite = EM.build(cal)
    log(f"suite: {len(suite.specs)} circuits built and checked in {time.time() - t0:.1f} s; calibration {cal.calibration_set_id}")
    for exp in ("E1", "E4"):
        log(f"  {exp}: {len(suite.of(exp))} circuits")
    for k, v in suite.choices["E1"]["patches"].items():
        log(f"  patch {k}: {', '.join(v)}")

    hw_jobs = EM.jobs(suite)
    src = str(snap_path.relative_to(REPO))
    IR.write_circuits(RECORD, suite.specs, suite.durations)
    tag_placements(RECORD)
    write_architecture(RECORD, snap, suite.arch, source=src)
    pkg = RECORD / "calibration" / EM.DEVICE
    IR.write_calibration_package(pkg, cal, suite)
    tag_json(pkg / "placement.json", device=EM.DEVICE)
    (pkg / "README.md").write_text(calibration_readme(cal))
    files = IR.write_circuits(SIM, suite.specs, suite.durations)
    tag_placements(SIM)
    write_architecture(SIM, snap, suite.arch, source=src)
    cal_dir = SIM / "calibration" / EM.DEVICE
    cal_dir.mkdir(parents=True)
    cal_file = cal_dir / f"20261024T191321Z-{cal.calibration_set_id}.json"
    ep = EM.error_profile(cal)
    cal_file.write_text(json.dumps({
        "label": "SIMULATED: the noise of the dry run's stand-in", "device": EM.DEVICE, "stand_in": EM.stand_in_name(cal),
        "built_from": src, **cal.to_json(),
        "error_profile": {"t1s": ep.t1s, "t2s": ep.t2s, "prx": ep.single_qubit_gate_depolarizing_error_parameters["prx"],
                          "cz": {f"{a}-{b}": v for (a, b), v in ep.two_qubit_gate_depolarizing_error_parameters["cz"].items()},
                          "readout_errors": ep.readout_errors,
                          "durations_ns": {"prx": ep.single_qubit_gate_durations["prx"], "cz": ep.two_qubit_gate_durations["cz"]},
                          "cz_less_sentinel": {"couplers": [list(p) for p in EM.CZ_LESS], "depolarizing": EM.CZ_LESS_SENTINEL}},
    }, indent=1) + "\n")
    CAL_REL = str(cal_file.relative_to(SIM))
    IR.write_calibration_package(cal_dir, cal, suite, label="copied into the SIMULATED record")
    tag_json(cal_dir / "placement.json", device=EM.DEVICE)
    (cal_dir / "README.md").write_text(calibration_readme(cal, "copied into the SIMULATED record: the stand-in's noise is this calibration"))

    runs = run_jobs(suite, hw_jobs, cal, day=day, log=log, files=files)
    ler = ler_records(runs, cal, log)

    checks = SIM / "checks"
    checks.mkdir()
    head = {"label": "SIMULATED", "device": EM.DEVICE, "stand_in": EM.stand_in_name(cal)}
    t0 = time.time()
    st = structure(suite, cal)
    (checks / "structure.json").write_text(json.dumps({**head, "rows": st}, indent=1) + "\n")
    log(f"structure: {len(st)} circuits in {time.time() - t0:.1f} s; distances "
        f"{sorted({r.get('graphlike_distance') for r in st if 'graphlike_distance' in r})}; "
        f"swaps {sorted({r['swaps'] for r in st})}; cz on live couplers: all")
    t0 = time.time()
    nl = noiseless(suite)
    (checks / "noiseless.json").write_text(json.dumps({**head, "rows": nl}, indent=1) + "\n")
    bad = [r for r in nl if r["detection_events"] or r["observable_flips"]]
    log(f"noiseless: {len(nl)} circuits, {sum(r['detection_events'] for r in nl)} detection events, "
        f"{sum(r['observable_flips'] for r in nl)} observable flips, up to {max(r['random_columns'] for r in nl)} random columns "
        f"({time.time() - t0:.1f} s)")
    dc = decode_check(runs)
    (checks / "decode.json").write_text(json.dumps(dc, indent=1) + "\n")
    pooled = dc["pooled"]
    log(f"decode E1 pooled: PyMatching {pooled['p_pm']:.4f} [{pooled['p_pm_ci'][0]:.4f}, {pooled['p_pm_ci'][1]:.4f}] vs "
        f"always-0 {pooled['p_raw']:.4f} [{pooled['p_raw_ci'][0]:.4f}, {pooled['p_raw_ci'][1]:.4f}] over {pooled['shots']} shots, "
        f"better: {pooled['better_than_always_0']}")
    cc = crosscheck(suite, Path(args.team_main))
    (checks / "crosscheck.json").write_text(json.dumps(cc, indent=1) + "\n")
    log(f"crosscheck against the team's make_stim_circuit: {cc.get('circuits', 0)} circuits, pass {cc.get('pass', cc.get('skipped'))}")
    d5 = EM.d5_feasibility(cal)
    (checks / "d5.json").write_text(json.dumps({"device": EM.DEVICE, **d5}, indent=1) + "\n")
    log(f"d=5: {d5['live_placements']} zero-SWAP placements on the calibrated couplers; {d5['all_90_placements']} on all 90, "
        f"needing {d5['all_90_qubit_sets'][0]['orientations'][0]['needs_cz_less'] if d5['all_90_qubit_sets'] else []}")
    summary = {
        **head, "calibration_set_id": cal.calibration_set_id, "dry_shots": DRY_SHOTS, "jobs": len(hw_jobs),
        "runs": len(runs), "ler_records": len(ler), "noiseless_failures": len(bad),
        "crosscheck_pass": cc.get("pass"), "wall_s": round(time.time() - t_start, 1),
        "derived_means": {exp: {k: float(np.nanmean([r["summary"][k] for r in runs if r["spec"].experiment == exp]))
                                for k in ("det_rate_mean", "same_stab_lag1", "same_stab_lag2", "cross_stab_lag0")}
                          for exp in ("E1", "E4") if any(r["spec"].experiment == exp for r in runs)},
        "ler": [{"experiment": x["experiment"], "run_id": x["run_id"], "shots": sum(p["shots"] for p in x["points"]),
                 **({"eps_fixed_A": x["fits"]["fixed_A"]["eps"], "eps_fixed_A_ci_bootstrap": x["fits"]["fixed_A"]["eps_ci_bootstrap"],
                     "eps_free_A": x["fits"]["free_A"]["eps"], "amplitude": x["fits"]["free_A"]["amplitude"]}
                    if "free_A" in x["fits"] else {})} for x in ler],
    }
    (checks / "summary.json").write_text(json.dumps(_nan_to_none(summary), indent=1) + "\n")
    (SIM / "README.md").write_text(sim_readme(cal, summary))
    tree = IR.tree_manifest(SIM)
    (SIM / "FILES.json").write_text(json.dumps(tree, indent=1) + "\n")
    log(f"sim/: {tree['files']} files, {tree['bytes'] / 1e6:.1f} MB")
    log(f"done in {time.time() - t_start:.0f} s: {len(runs)} runs, {len(ler)} LER records")
    (checks / "log.txt").write_text("\n".join(lines) + "\n")
    ok = not bad and pooled["better_than_always_0"] and cc.get("pass", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
