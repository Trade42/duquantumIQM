"""The Garnet dry run: every experiment of the suite on the stand-in Garnet, written as a full InfoRecord.

    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
        uv run python scripts/garnet_dryrun.py [--clean]

The stand-in is ``IQMFakeGarnet().copy_with_error_profile(...)`` with the live calibration of
2026-10-24 (set a18a947f), simulated by Aer's matrix-product-state method (exact here; the
checks compare it with the statevector method), seeded, on at most 4 threads. Every job of the
hardware plan is simulated at small shot counts, in its planned circuit order, and goes through
the hardware path: an Aer ``Result`` saved by ``dqiqm.hw.save_raw`` into ``InfoRecord/sim/raw/``,
read back by ``dqiqm.decode.load_memory``, then written as ``datasets/``, ``derived/`` and ``ler/``.

It also writes ``InfoRecord/sim/checks/``:

* ``structure.json``: per circuit, the detector, measurement and observable counts, the DEM of
  its noisy copy (``decompose_errors=True``) and Stim's graphlike distance;
* ``noiseless.json``: every hardware circuit run noiselessly through the same path: no detection
  event and no observable flip on any shot;
* ``mps_vs_statevector.json``: one circuit by both Aer methods at one seed;
* ``decode.json``: E1 decoded by PyMatching (noisy copy of the circuit that ran) against "always 0";
* ``features.json``: Stim runs with injected record flips and state flips, written and analysed
  by the same writer, so the derived features are checked against known rates;
* ``sizes.json``: bytes per shot of every file, and what a million shots cost;
* ``summary.md``: the tables of all of the above.

And, outside ``sim/``, the planned hardware circuits (``InfoRecord/circuits/``, with the job plan
``circuits/job_plan.json``), the architecture and the README with the column dictionary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
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
from dqiqm import surface as S  # noqa: E402
from dqiqm.experiments import device as dev  # noqa: E402
from dqiqm.experiments import readout as RO  # noqa: E402
from dqiqm.experiments import suite as SU  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "InfoRecord"
SIM = RECORD / "sim"
TASK = "garnet-dryrun"
THREADS = 4
SEED = 2026

#: Dry-run shots per circuit, per job group (the hardware plan's are in dqiqm.experiments.suite.SHOTS).
DRY_SHOTS = {"P0": 1000, "P1": 256, "E1": 128, "E2": 64, "E3": 256, "E4": 128, "E5": 256}


def aer(backend, *, noisy: bool = True, method: str = "matrix_product_state", seed: int = SEED):
    from qiskit_aer import AerSimulator

    return AerSimulator(method=method, noise_model=backend.noise_model if noisy else None, seed_simulator=seed,
                        max_parallel_threads=THREADS)


def job_seed(label: str) -> int:
    return SEED + zlib.crc32(label.encode()) % 1_000_000


def stand_in_name(cal: dev.Calibration) -> str:
    return f"IQMFakeGarnet+profile({cal.calibration_set_id[:8]})"


def decoder_for(spec, cal):
    if spec.noisy is None or not spec.stim.num_observables:
        return None
    nm = dev.calibrated_noise(cal, spec.physical, spec.roles, pairs=spec.cx_pairs, idle_ns=0.0, x_ancillas=spec.x_ancillas)
    return spec.noisy(nm)


DECODER_NOTE = ("dqiqm.experiments.device.calibrated_noise(live calibration, idle_ns=0): the stand-in's own noise "
                "(gate depolarizing + relaxation over gate durations, mean readout assignment error as record flips)")


def run_jobs(suite, jobs, cal, *, day: str, log) -> list[dict]:
    """Simulate every job, save its raw result, and write its datasets and derived features."""
    runs = []
    files = {s: f for s, f in WRITTEN_CIRCUITS.items()}
    cal_rel = {"before": CAL_REL, "after": CAL_REL}
    for job in jobs:
        specs = [suite[n] for n in job.circuits]
        seed = job_seed(job.label)
        shots = DRY_SHOTS["P0" if job.label == "P0" else "P1" if job.label == "P1" else job.experiment]
        t0 = time.time()
        res = aer(suite.backend, seed=seed).run([s.hardware.circuit for s in specs], shots=shots, memory=True).result()
        if not res.success:
            raise RuntimeError(f"{job.label}: Aer failed: {res.status}")
        res.job_id = f"sim-{job.label}"
        mappings = [{i: s.physical[q] for i, q in enumerate(s.hardware.stim_qubits)} for s in specs]
        sent = [sorted(set(m.values()), key=dev.qubit_sort_key) for m in mappings]
        job_dir = hw.save_raw(res, task=TASK, device=stand_in_name(cal), qubit_mapping=mappings, shots=shots,
                              physical_qubits=sent, purpose=f"SIMULATED dry run, job {job.label}: {job.note}", day=day,
                              data_root=SIM / "raw")
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
                extra={"device": stand_in_name(cal), "job_timeline": meta.get("timeline"),
                       "seeds": {"aer_seed_simulator": seed, "circuit_index": i},
                       "simulator": {"backend": stand_in_name(cal), "aer_method": "matrix_product_state",
                                     "noise": "IQMFakeGarnet error profile from the live snapshot (dqiqm.experiments.device.error_profile, Ramsey T2)",
                                     "threads": THREADS}},
            )
            der, summary = IR.write_derived(SIM, s, run_id, ds)
            runs.append({"job": job.label, "batch": job.batch or 0, "spec": s, "run_id": run_id, "dataset": ds,
                         "derived": der, "numbers": numbers, "summary": summary})
    return runs


def ler_records(runs, log) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        s = r["spec"]
        p = s.params
        if r["numbers"]["pm_failures"] is None or not r["batch"]:
            continue
        if s.experiment == "E1":
            key = ("E1", f"{p['basis']}_{p['patch']}_b{r['batch']}", f"E1 d=3 {p['basis']} basis, patch {p['patch']}, batch {r['batch']}")
        elif s.experiment == "E4":
            key = ("E4", f"{p['basis']}_{p['patch']}_t{int(p['delay_ns'])}ns_b{r['batch']}",
                   f"E4 d=3 {p['basis']}, {p['delay_ns'] / 1e3:g} us idle between rounds, batch {r['batch']}")
        elif s.experiment == "E2":
            key = ("E2", f"{p['kind']}_{'p' if p['prep'] == '+' else p['prep']}_b{r['batch']}",
                   f"E2 repetition d=10, {p['kind']}-flip, |{p['prep']}>, batch {r['batch']}")
        elif s.experiment == "E5":
            key = ("E5", f"{p['kind']}_b{r['batch']}", f"E5 stability ({p['kind']} edge checks), batch {r['batch']}")
        else:
            continue
        groups[key].append(r)
    out = []
    for (exp, run_id, title), rs in sorted(groups.items()):
        points = [{"circuit": r["spec"].name, "dataset": str(r["dataset"].relative_to(SIM)), "rounds": r["spec"].rounds,
                   "shots": r["numbers"]["shots"], "failures": r["numbers"]["pm_failures"],
                   "raw_failures": r["numbers"]["obs_flips"]} for r in rs]
        path, res = IR.write_ler(SIM, exp, run_id, points, title=f"SIMULATED: {title}", label=run_id, fit=exp != "E5")
        out.append({"experiment": exp, "run_id": run_id, "points": points, "fits": res})
        if "free_A" in res:
            log(f"LER {exp} {run_id}: eps(A=1) {res['fixed_A']['eps']:.4f}, eps(free A) {res['free_A']['eps']:.4f}, A {res['free_A']['amplitude']:.3f}")
    return out


def structure(suite, cal) -> list[dict]:
    rows = []
    for s in suite.specs:
        row = {"circuit": s.name, "experiment": s.experiment, "rounds": s.rounds, "detectors": s.stim.num_detectors,
               "measurements": s.stim.num_measurements, "observables": s.stim.num_observables,
               "swaps": s.checks.get("swaps"), "cz": int(s.hardware.circuit.count_ops().get("cz", 0))}
        dec = decoder_for(s, cal)
        if dec is not None:
            dem = dec.detector_error_model(decompose_errors=True)
            row["dem_errors"] = dem.num_errors
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


def mps_vs_statevector(suite) -> dict:
    s = suite["E1_d3_Z_r02_pA"]
    out = {"circuit": s.name, "shots": 200, "seed": SEED}
    mem = {}
    for method in ("matrix_product_state", "statevector"):
        t0 = time.time()
        res = aer(suite.backend, method=method).run(s.hardware.circuit, shots=200, memory=True).result()
        mem[method] = hw._memory_bits(res.get_memory(0))  # noqa: SLF001 -- save_raw's own reversal
        dets, _ = D.detection_events(s.stim, mem[method])
        out[method] = {"seconds": round(time.time() - t0, 2), "detection_fraction": float(dets.mean())}
    out["identical_records"] = bool(np.array_equal(mem["matrix_product_state"], mem["statevector"]))
    return out


def decode_check(runs) -> dict:
    by: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in runs:
        if r["spec"].experiment != "E1" or r["numbers"]["pm_failures"] is None:
            continue
        key = (r["spec"].params["basis"], r["spec"].rounds)
        by[key][0] += r["numbers"]["shots"]
        by[key][1] += r["numbers"]["pm_failures"]
        by[key][2] += r["numbers"]["obs_flips"]
    rows = []
    tot = [0, 0, 0]
    for (basis, rounds), (n, k_pm, k_raw) in sorted(by.items()):
        lo, hi = stats.wilson(k_pm, n)
        rlo, rhi = stats.wilson(k_raw, n)
        rows.append({"basis": basis, "rounds": rounds, "shots": n, "pm_failures": k_pm, "p_pm": k_pm / n, "p_pm_ci": [lo, hi],
                     "raw_failures": k_raw, "p_raw": k_raw / n, "p_raw_ci": [rlo, rhi]})
        tot = [tot[0] + n, tot[1] + k_pm, tot[2] + k_raw]
    n, k_pm, k_raw = tot
    lo, hi = stats.wilson(k_pm, n)
    rlo, rhi = stats.wilson(k_raw, n)
    return {"rows": rows, "pooled": {"shots": n, "pm_failures": k_pm, "p_pm": k_pm / n, "p_pm_ci": [lo, hi],
                                     "raw_failures": k_raw, "p_raw": k_raw / n, "p_raw_ci": [rlo, rhi],
                                     "better_than_always_0": hi < rlo}}


def features_check(suite, cal) -> dict:
    """Stim runs with known injected errors, written and analysed by the record's own writer."""
    out = {}
    shots = 40_000
    s1 = suite["E1_d3_Z_r10_pA"]
    cases = {
        "record_flips_0.03": S.NoiseModel(record_flip_probability=0.03),
        "ancilla_state_flips_0.03": S.NoiseModel(qubit_rates={a: {"idle_during_measurement": 0.03} for a in s1.stabilizers}),
        "none": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for k, (name, nm) in enumerate(cases.items()):
            noisy = S.memory_circuit(3, 10, "Z", no_reset=True, noise=nm) if nm is not None else s1.stim
            raw = noisy.compile_sampler(seed=SEED + k).sample(shots).astype(np.uint8)
            ds, _ = IR.write_dataset(root, s1, raw, run_id=f"inj_{name}", batch=0, job_id="stim", position_in_job=0,
                                     arch=suite.arch, cal=cal, durations=suite.durations,
                                     circuit_files=WRITTEN_CIRCUITS[s1.name], raw_ref={}, calibration_paths={},
                                     simulated=True)
            _, summary = IR.write_derived(root, s1, f"inj_{name}", ds)
            out[f"E1 Z R=10, {name}"] = {k2: summary[k2] for k2 in ("same_stab_lag1", "same_stab_lag2", "same_stab_lag3",
                                                                   "cross_stab_lag0", "det_rate_mean")} | {"shots": shots}
        s3 = suite["E3_repeat_all1_k15"]
        coords = [suite.arch.coords[s3.physical[q]] for q in range(len(s3.physical))]
        for k, (rec, st) in enumerate(((0.02, 0.0), (0.0, 0.01), (0.02, 0.01))):
            noisy = RO.circuit(coords, [1] * len(coords), 15, record_flip=rec, state_flip=st)
            raw = noisy.compile_sampler(seed=SEED + 10 + k).sample(20_000).astype(np.uint8)
            name = f"rec{rec}_state{st}"
            ds, _ = IR.write_dataset(root, s3, raw, run_id=f"inj_{name}", batch=0, job_id="stim", position_in_job=0,
                                     arch=suite.arch, cal=cal, durations=suite.durations,
                                     circuit_files=WRITTEN_CIRCUITS[s3.name], raw_ref={}, calibration_paths={}, simulated=True)
            _, summary = IR.write_derived(root, s3, f"inj_{name}", ds)
            out[f"E3 repeat |1>, injected record {rec}, state {st}"] = {
                "record_flip_mean": summary["record_flip_mean"], "state_flip_mean": summary["state_flip_mean"], "shots": 20_000}
    return out


def sizes(runs) -> dict:
    """Bytes per shot of each file of the biggest E1 run, scaled to 10^6 shots, unpacked and packed."""
    out = {}
    for name in ("E1_d3_Z_r20_pA", "E1_d3_Z_r03_pA"):
        r = next(x for x in runs if x["spec"].name == name)
        ds = r["dataset"]
        n = r["numbers"]["shots"]
        m = json.loads((ds / "manifest.json").read_text())
        per = {}
        for f in ("raw_meas.npy", "det_events.npy", "obs_flips.npy"):
            cols = m["array_info"][f]["shape"][1]
            per[f] = {"uint8_bytes_per_shot": cols, "packed_bytes_per_shot": (cols + 7) // 8}
        csv_bytes = (ds / "shots.csv").stat().st_size
        gz = len(gzip.compress((ds / "shots.csv").read_bytes(), 9))
        per["shots.csv"] = {"bytes_per_shot": csv_bytes / n, "gzip_bytes_per_shot": gz / n}
        million = 1_000_000
        out[name] = {
            "measurements": m["num_measurements"], "detectors": m["num_detectors"], "per_shot": per,
            "per_million_shots_MB": {
                "uint8 npy (raw + dets + obs)": sum(v["uint8_bytes_per_shot"] for k, v in per.items() if k.endswith(".npy")) * million / 1e6,
                "bit-packed npy (raw + dets + obs)": sum(v["packed_bytes_per_shot"] for k, v in per.items() if k.endswith(".npy")) * million / 1e6,
                "shots.csv": per["shots.csv"]["bytes_per_shot"] * million / 1e6,
                "shots.csv.gz": per["shots.csv"]["gzip_bytes_per_shot"] * million / 1e6,
                "parquet": "not available: pyarrow is not in uv.lock (the brief forbids adding it)",
            },
        }
    return out


def dictionary_md() -> str:
    """The column dictionary, escaped for markdown: the ``|`` of P(1|0) would split a table cell, and a
    placeholder such as ``<q>`` outside code renders as an HTML tag.

    The escaping belongs in ``inforecord.dictionary_markdown``; it sits here until that module next changes.
    """
    lines = []
    for line in IR.dictionary_markdown().replace("P(1|0)", "P(1\\|0)").replace("P(0|1)", "P(0\\|1)").split("\n"):
        parts = line.split("`")
        parts[::2] = [re.sub(r"<(\w+)>", r"&lt;\1&gt;", x) for x in parts[::2]]
        lines.append("`".join(parts))
    return "\n".join(lines)


def readme(cal, suite, plan, files, files_sha, summary) -> str:
    """InfoRecord/README.md: what the record is, its layout and conventions, and the column dictionary."""
    total_shots = sum(j["total_shots"] for j in plan)
    total_s = sum(j["qpu_s"] for j in plan)
    return f"""# InfoRecord: everything we collect from IQM Garnet

Format `{IR.FORMAT}`, written by `scripts/garnet_dryrun.py` and `dqiqm.inforecord` (the
garnet-design task, 2026-10-24). Sal's request: collect all the data Garnet gives, from circuits
designed to carry as much information as possible, especially temporal correlations, with every
column useful to an ML decoder. The experiments and the hardware job plan are in
`docs/garnet-design.md`.

## Devices

This root, `InfoRecord/`, is IQM Garnet's record. Emerald's will be `InfoRecord/emerald/`, with the
identical layout inside it (another task builds it). Every run's `manifest.json` names its
`device` (in `sim/`, the stand-in's). The circuits' `placement.json` files and the `ler/` records
don't carry a `device` key yet: their device is the root they sit in. Adding the key is on the
backlog, as it regenerates `circuits/` and `sim/`.

## What is real and what is simulated

| Path | What | Status |
|---|---|---|
| `architecture/garnet.json` | the live dynamic and static architecture, grid coordinates | **hardware** (calibration set `{cal.calibration_set_id}`) |
| `calibration/garnet/` | the live snapshot the circuits and noise were built from (E6 adds one before and after each batch), and its pre-training package: `qubits.csv`, `couplers.csv`, `durations.json` (prx, measure and CZ durations per qubit or coupler, in ns, and `reset_wait_ns`: the passive-reset wait between shots, {cal.reset_wait_ns / 1e3:.0f} µs, as there is no active reset), `placement.json` (`scripts/garnet_calibration_package.py`, offline) | **hardware** |
| `circuits/` | the planned hardware circuits (`target="hardware"` only, zero SWAPs), `placement.json`, `job_plan.json` | planned, not yet run |
| `raw/`, `datasets/`, `derived/`, `ler/` | hardware results | **empty until the hardware jobs run** |
| `sim/` | the dry run on the stand-in, in exactly this layout | **SIMULATED** (`"label": "SIMULATED"` in every manifest, `simulated = 1` in every shot) |

**What git holds of `sim/`.** The full dry run is {files['files']:,} files, {files['bytes'] / 1e6:.1f} MB. Git holds its
README, `FILES.json`, `checks/`, `ler/`, `calibration/`, `architecture/`, `raw/` and a few representative runs
(`InfoRecord/.gitignore` lists them); the rest stays on disk. Regenerate the whole tree, seeded, in about
{summary['wall_s'] / 60:.0f} min under the lock:

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/garnet_dryrun.py --clean
```

It keeps `circuits/job_plan.json`'s `pilot` block (the pilot's sanity bands) while the calibration
set is the same. Otherwise run `scripts/garnet_plan.py` after it, which writes the block.

`sim/FILES.json` (sha256 `{files_sha}`) lists every file the run writes with its size, sha256 and a canonical
sha256 that ignores wall-clock fields; `dqiqm.inforecord.compare_trees` checks a regenerated tree against it.
Hardware data is different: its `raw/` is committed in full, always (it cannot be regenerated).

The stand-in is `IQMFakeGarnet().copy_with_error_profile(...)` with the live calibration's noise
(`dqiqm.experiments.device`): per-qubit T1, Ramsey T2, 1Q and CZ errors, and the asymmetric
readout. It puts no noise on `delay` or on idle qubits, so E4's idles are checked for structure
only in `sim/`.

## Layout

```
architecture/garnet.json
calibration/garnet/<UTC>-<calset>.json
circuits/<experiment>/<name>.stim       the circuit's meaning: detectors, observables, QUBIT_COORDS (noise-free)
circuits/<experiment>/<name>.qasm       exactly what runs (OpenQASM 3 of the transpiled circuit)
circuits/<experiment>/placement.json    per circuit: Stim qubit -> device qubit, roles, checks, gate counts, hashes
circuits/job_plan.json                  the jobs, their circuit order, shots and estimated QPU time
raw/<YYYYMMDD>-<task>/<job id>/         dqiqm.hw.save_raw: meta.json + memory.npz (c<i>: uint8 shots x clbits); never edited
datasets/<experiment>/<run id>/         one circuit in one batch, ML-ready (run id: b<batch>_<circuit>, p0_/p1_ for the pilot)
    raw_meas.npy det_events.npy obs_flips.npy shots.csv detectors.csv measurements.csv qubits.csv manifest.json
derived/<experiment>/<run id>/          pij.npy pij_lags.csv rate_by_round.csv rate_by_bucket.csv streaks.csv
                                        syndrome_weight.csv summary.json; E3 also assignment.csv readout_corr.npy readout_flips.csv
ler/<experiment>/<run id>/              ler_points.csv ler_fits.json ler.png, one per (basis or kind, patch, batch)
sim/                                    all of the above, SIMULATED; sim/checks/ holds the dry run's own checks
```

## Conventions

- `round` is Stim's detector time coordinate t: 0-based, with the final data-qubit detectors at
  t = R. A measurement's `round` is the same t: round k's readout at t = k − 1, the final data
  readout at t = R.
- `x`, `y` are Stim's detector coordinates, exactly as `circuit.get_detector_coordinates()` gives
  them. The rows of `detectors.csv` and the columns of `det_events.npy` are in the circuit's
  detector order, which for E1 is that of the team's `make_stim_circuit`.
- Column j of `raw_meas.npy` is Stim measurement j = clbit j = column j of the raw `memory.npz`.
- Arrays are `uint8` 0/1. One with more than {IR.PACK_THRESHOLD:,} entries is bit-packed along its
  columns (`np.packbits(a, axis=1, bitorder="little")`): the manifest's `array_info` says
  `"packed": true` and gives the unpacked shape. `dqiqm.inforecord.load_array(run_dir, "dets")`
  reads either. The dry run's arrays are all unpacked.
- Detection events and observable flips are against the circuit's noiseless reference sample
  (Stim's m2d converter): a noiseless shot is all zeros.
- `shots.csv` is CSV: `pyarrow` is not in `uv.lock` (and the brief forbids adding it), so no
  Parquet. `sim/checks/sizes.json` has what a million shots cost per file.
- The decoder column `pm_pred` is PyMatching on a noisy copy of the circuit that ran
  (`dqiqm.decode`), with the noise named in the manifest's `decoder`.
- Hashes, in `placement.json` and each manifest: `qasm_sha256` is the sha256 of the `.qasm` file;
  `stim_sha256` is the sha256 of the Stim text **without the file's trailing newline**
  (`str(circuit)`), so hashing the `.stim` file as stored doesn't match; `native_sha256` is
  `dqiqm.experiments.circuits.native_sha256` of the hardware circuit.
- Submitting: the `.qasm` files document what runs, but they can't be loaded here
  (`qiskit.qasm3.loads` needs `qiskit_qasm3_import`, which is not in `uv.lock`). The hardware
  worker rebuilds the circuits with `dqiqm.experiments.suite.build(device.live_calibration(<the
  snapshot>))` and, before it submits, checks each circuit's `circuits.native_sha256` against its
  `placement.json` entry. A mismatch stops the job.

## Pre-training the ML decoder

Pre-train on synthetic shots built from the calibration package (`calibration/garnet/`) **with
idle terms**: the T1/T2 decay over each readout window, plus any `delay` (E4). In Stim,
`dqiqm.experiments.device.calibrated_noise(cal, ..., idle_ns=<readout window + delay>)` builds it;
like the stand-in, it adds relaxation over each gate on top of the RB and IRB figures, so its gate
noise is on the high side (on the backlog). **Don't pre-train on `sim/`:** the stand-in has no idle
noise, and its gate noise is about twice the benchmarked figures (median ×2.05 for 1Q gates, ×1.82
for CZ), so its shots are not Garnet's. `calibration/garnet/placement.json` holds E1's two
placements, pA and pB: choose the placement by each shot's `patch` (`shots.csv`).

## The ML split (fixed before any data)

Batch 3, whole, is the **test** set and is never fine-tuned on. In every other job (the pilot phase
included) a seeded draw of 20% of the shot indices, the same for every circuit of the job, is
**validation**, the rest **finetune** (`dqiqm.experiments.suite.split`). Each shot's split is in
`shots.csv`'s `split`; the rule, the run's role and the counts in the manifest's `split`.

`validation` is the fine-tuning side's held-out 20%. A loader with only finetune and test loads it
as finetune (or as its own validation set), **never as test**: test is batch 3 only.

P1, the barrier A/B, runs the pilot circuit with and without barriers, and both arms are in
finetune and validation. The barrier-off run (`p1_E1_d3_Z_r03_pA_nobar`) has a `.stim`
byte-identical to the barrier-on one's; only `circuit`, `barriers`, `native_sha256` and the timing
differ. **Filter on `barriers`** (`shots.csv`, or the manifest's `spec.barriers`): fine-tune and
validate on the runs whose `barriers` matches the batches' setting, which P1 decides, and leave
P1's other arm out.

## Reading a run

```python
import json, numpy as np, csv
from dqiqm.inforecord import load_array
run = "InfoRecord/sim/datasets/E1/b1_E1_d3_Z_r10_pA"
m = json.load(open(f"{{run}}/manifest.json"))
dets, obs, raw = (load_array(run, k) for k in ("dets", "obs", "raw_meas"))
detectors = list(csv.DictReader(open(f"{{run}}/detectors.csv")))
```

## The plan in one line

{len(plan)} jobs, {total_shots:,} shots, about {total_s / 60:.0f} min of QPU time at the assumed overheads
(`circuits/job_plan.json`, `docs/garnet-design.md`).

## Column dictionary

Printed from `dqiqm.inforecord.COLUMNS`, which also writes and checks every file.

{dictionary_md()}"""


def sim_readme(cal, summary) -> str:
    return f"""# SIMULATED: the Garnet dry run

Everything under `sim/` is **simulated**: the stand-in `{stand_in_name(cal)}` (IQMFakeGarnet with the
live noise of calibration set `{cal.calibration_set_id}`), Aer matrix-product-state method, seeded.
It is the full record's layout, at small shot counts ({json.dumps(DRY_SHOTS)} per circuit), so
every reader and every analysis can be built before the hardware data exist. Do not mix it with
hardware data: every manifest says `"label": "SIMULATED"`, every shot `simulated = 1`.

Regenerate it (about {summary['wall_s'] / 60:.0f} min wall on 4 cores):

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
    uv run python scripts/garnet_dryrun.py --clean
```

`checks/` holds the run's own checks: `structure.json`, `noiseless.json`, `mps_vs_statevector.json`,
`decode.json`, `features.json`, `sizes.json`, `summary.json` and the log.
"""


WRITTEN_CIRCUITS: dict = {}
CAL_REL = ""


def main() -> int:
    global WRITTEN_CIRCUITS, CAL_REL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean", action="store_true", help="remove InfoRecord/sim/ first (it is regenerable)")
    ap.add_argument("--day", default=None, help="YYYYMMDD for raw/ (default: today, UTC)")
    ap.add_argument("--smoke", metavar="DIR", default=None,
                    help="a quick run of a small suite at a few shots, into DIR instead of InfoRecord/ (a pipeline check)")
    args = ap.parse_args()
    global RECORD, SIM
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
    import datetime as dt

    day = args.day or dt.datetime.now(dt.UTC).strftime("%Y%m%d")

    snap_path = dev.DEFAULT_SNAPSHOT
    snap = dev.load_snapshot(snap_path)
    cal = dev.live_calibration(snap_path)
    t0 = time.time()
    if args.smoke:
        suite = SU.build(cal, e1_rounds=(2, 3, 10, 20), e2_rounds=(1, 3), e4_rounds=(2, 3), e4_delays_ns=(1000.0,),
                         e5_rounds=(2, 3), e3_random=1, rank_shots=2000)
    else:
        suite = SU.build(cal)
    log(f"suite: {len(suite.specs)} circuits built and checked in {time.time() - t0:.1f} s; calibration {cal.calibration_set_id}")
    for exp in ("E1", "E2", "E3", "E4", "E5"):
        log(f"  {exp}: {len(suite.of(exp))} circuits")

    # The planned hardware circuits and the plan (top level), then the same circuits under sim/.
    hw_jobs = SU.jobs(suite)
    plan = [SU.estimate(j, suite) for j in hw_jobs]
    plan_file = RECORD / "circuits" / "job_plan.json"
    old_plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    IR.write_circuits(RECORD, suite.specs, suite.durations)
    # scripts/garnet_plan.py adds the pilot's sanity bands; keep them while the calibration set is the same
    new_plan = SU.plan_json(suite, plan)
    if "pilot" in old_plan and old_plan.get("calibration_set_id") == cal.calibration_set_id:
        new_plan["pilot"] = old_plan["pilot"]
    else:
        log("job_plan.json has no pilot bands for this calibration set: run scripts/garnet_plan.py after this")
    plan_file.write_text(json.dumps(new_plan, indent=1, default=str) + "\n")
    IR.write_architecture(RECORD, snap, suite.arch, source=str(snap_path.relative_to(REPO)))
    WRITTEN_CIRCUITS = IR.write_circuits(SIM, suite.specs, suite.durations)
    IR.write_architecture(SIM, snap, suite.arch, source=str(snap_path.relative_to(REPO)))
    cal_dir = SIM / "calibration" / "garnet"
    cal_dir.mkdir(parents=True)
    cal_file = cal_dir / f"{snap_path.name.split('-', 1)[0]}-{cal.calibration_set_id}.json"
    ep = dev.error_profile(cal)
    cal_file.write_text(json.dumps({
        "label": "SIMULATED: the noise of the dry run's stand-in",
        "built_from": str(snap_path.relative_to(REPO)),
        **cal.to_json(),
        "error_profile": {"t1s": ep.t1s, "t2s": ep.t2s, "prx": ep.single_qubit_gate_depolarizing_error_parameters["prx"],
                          "cz": {f"{a}-{b}": v for (a, b), v in ep.two_qubit_gate_depolarizing_error_parameters["cz"].items()},
                          "readout_errors": ep.readout_errors, "durations_ns": {"prx": ep.single_qubit_gate_durations["prx"],
                                                                               "cz": ep.two_qubit_gate_durations["cz"]}},
    }, indent=1) + "\n")
    CAL_REL = str(cal_file.relative_to(SIM))
    IR.write_calibration_package(RECORD / "calibration" / "garnet", cal, suite)
    IR.write_calibration_package(cal_dir, cal, suite, label="copied into the SIMULATED record: the stand-in's noise is this calibration")

    # The jobs.
    runs = run_jobs(suite, hw_jobs, cal, day=day, log=log)
    ler = ler_records(runs, log)

    checks = SIM / "checks"
    checks.mkdir()
    t0 = time.time()
    st = structure(suite, cal)
    (checks / "structure.json").write_text(json.dumps(st, indent=1) + "\n")
    log(f"structure: {len(st)} circuits in {time.time() - t0:.1f} s")
    t0 = time.time()
    nl = noiseless(suite)
    (checks / "noiseless.json").write_text(json.dumps(nl, indent=1) + "\n")
    bad = [r for r in nl if r["detection_events"] or r["observable_flips"]]
    log(f"noiseless: {len(nl)} circuits, {sum(r['detection_events'] for r in nl)} detection events, "
        f"{sum(r['observable_flips'] for r in nl)} observable flips ({time.time() - t0:.1f} s)")
    mv = mps_vs_statevector(suite)
    (checks / "mps_vs_statevector.json").write_text(json.dumps(mv, indent=1) + "\n")
    log(f"MPS vs statevector: identical records {mv['identical_records']}")
    dc = decode_check(runs)
    (checks / "decode.json").write_text(json.dumps(dc, indent=1) + "\n")
    log(f"decode E1 pooled: PyMatching {dc['pooled']['p_pm']:.4f} vs always-0 {dc['pooled']['p_raw']:.4f}, "
        f"better: {dc['pooled']['better_than_always_0']}")
    ft = features_check(suite, cal)
    (checks / "features.json").write_text(json.dumps(ft, indent=1) + "\n")
    for k, v in ft.items():
        log(f"features {k}: {json.dumps({a: round(b, 5) if isinstance(b, float) else b for a, b in v.items()})}")
    sz = sizes(runs)
    (checks / "sizes.json").write_text(json.dumps(sz, indent=1) + "\n")
    summary = {
        "label": "SIMULATED",
        "calibration_set_id": cal.calibration_set_id,
        "stand_in": stand_in_name(cal),
        "dry_shots": DRY_SHOTS,
        "jobs": len(hw_jobs),
        "runs": len(runs),
        "ler_records": len(ler),
        "noiseless_failures": len(bad),
        "wall_s": round(time.time() - t_start, 1),
        "derived_means": {
            exp: {k: float(np.nanmean([r["summary"][k] for r in runs if r["spec"].experiment == exp]))
                  for k in ("det_rate_mean", "same_stab_lag1", "same_stab_lag2", "cross_stab_lag0")}
            for exp in ("E1", "E2", "E4", "E5")
        },
    }
    (checks / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (SIM / "README.md").write_text(sim_readme(cal, summary))
    files = IR.tree_manifest(SIM)
    (SIM / "FILES.json").write_text(json.dumps(files, indent=1) + "\n")
    files_sha = IR.sha256_file(SIM / "FILES.json")
    (RECORD / "README.md").write_text(readme(cal, suite, plan, files, files_sha, summary) + "\n")
    log(f"sim/: {files['files']} files, {files['bytes'] / 1e6:.1f} MB; FILES.json sha256 {files_sha}")
    (checks / "log.txt").write_text("\n".join(lines) + "\n")
    log(f"done in {time.time() - t_start:.0f} s: {len(runs)} runs, {len(ler)} LER records")
    return 1 if bad or not dc["pooled"]["better_than_always_0"] else 0


if __name__ == "__main__":
    sys.exit(main())
