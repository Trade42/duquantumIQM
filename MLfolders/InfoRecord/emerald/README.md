# InfoRecord/emerald: everything we collect from IQM Emerald

The Garnet design (`../README.md`, `docs/garnet-design.md`) applied to Emerald, the challenge's
own device: same format (`dqiqm.inforecord/1`), same layout, same writer (`dqiqm.inforecord`),
same column dictionary. This root holds Emerald only; Garnet's record is `InfoRecord/` itself,
frozen. Written by `scripts/emerald_dryrun.py` and `scripts/emerald_plan.py` (the emerald-design
task, 2026-10-24); the design, the placements and the job plan are in `docs/emerald-design.md`.

## The device convention

- Every dataset manifest, every `placement.json` entry, the calibration package's `placement.json`,
  `job_plan.json`, `architecture/emerald.json` and every LER record carry `"device": "emerald"`.
- The patches are **`eA`** and **`eB`** (Garnet's are `pA` and `pB`), so circuit names
  (`E1_d3_Z_r03_eA`) and run ids are unique across devices.
- Job labels start with `E` (`EP0`, `E-B1-1-E1`): they never collide with Garnet's in the ledger
  or in the split's seeds.

## What is real and what is simulated

| Path | What | Status |
|---|---|---|
| `architecture/emerald.json` | the live dynamic and static architecture, grid coordinates; `couplers` are the 85 with a CZ, `cz_less_couplers` the other 5 | **hardware** (calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`) |
| `calibration/emerald/` | the live snapshot (a byte copy of `data/calibration/emerald/23e12f6b-….20261024T191321Z.json`) and its pre-training package: `qubits.csv`, `couplers.csv`, `durations.json`, `placement.json`, `README.md` | **hardware** |
| `circuits/` | the planned hardware circuits (`target="hardware"` only, zero SWAPs, every CZ on a calibrated coupler), `placement.json`, `job_plan.json` | planned, not yet run |
| `raw/`, `datasets/`, `derived/`, `ler/` | hardware results | **empty until the hardware jobs run** |
| `sim/` | the dry run on the stand-in, in exactly this layout | **SIMULATED** (`"label": "SIMULATED"` in every manifest and LER record, `simulated = 1` in every shot) |

The stand-in is `IQMFakeAphrodite().copy_with_error_profile(...)` (Emerald's 54 qubit names and 90
couplers) with the live calibration's noise; the 5 CZ-less couplers are padded and never used, and
T2 is capped at T1 on 5 qubits (`dqiqm.experiments.emerald`). It has **no idle noise**: E4's idles
are checked for structure only in `sim/`. **Do not pre-train on `sim/`**: see the next section.

**What git holds of `sim/`:** its README, `FILES.json`, `checks/`, `ler/`, `calibration/`,
`architecture/`, `raw/` and five representative runs (`.gitignore` lists them); the rest stays on
disk. Regenerate it, seeded, under the lock:

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/emerald_dryrun.py --clean
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/emerald_plan.py
```

The dry run never writes `circuits/job_plan.json`; only the plan script does, so running the dry
run alone cannot drop the pilot's bands. `sim/FILES.json` lists every file with its size, sha256
and a canonical sha256 that ignores wall-clock fields; check a regenerated tree with
`dqiqm.inforecord.compare_trees` (the file's own sha256 changes on every run, so it is not quoted).

## For the ML decoder: read this first

- **Pre-training:** build noise from `calibration/emerald/` **with idle terms** (each qubit's
  T1/T2 decay over the ancilla readout window, about 360 ns, plus any delay), not from `sim/`,
  whose shots have no idle noise and about twice the benchmarked gate error.
- **Patches:** choose the placement by each shot's `patch` (`eA` or `eB`). They are different
  17-qubit sets (`calibration/emerald/placement.json`), unlike Garnet's pA/pB, which are one set in
  two orientations.
- **The split** (Garnet's, `dqiqm.experiments.suite.split`): batch 3, whole, is **test** and is
  never fine-tuned on. In every other job (the pilot included) a seeded 20% of shot indices is
  **validation**, the rest **finetune**. `validation` is the fine-tuning side's held-out part:
  load it as finetune or as validation, **never as test**. A loader with only finetune/test reads
  `validation` as finetune.
- There is no barrier-off circuit on Emerald (Garnet's P1 decides the barrier question), so every
  E1 shot is the same circuit family; E4 shots carry `delay_ns` > 0.

## Circuits and hashes

- `<name>.stim` is the circuit's meaning: detectors, observables, QUBIT_COORDS (noise-free). E1's
  and E4's detectors are identical, in order, records and coordinates, to the team's
  `make_stim_circuit` (`sim/checks/crosscheck.json`).
- `<name>.qasm` is OpenQASM 3 of the transpiled circuit: documentation only here, since
  `qiskit.qasm3.loads` needs `qiskit_qasm3_import`, which is not in the lock.
- In `placement.json`, `stim_sha256` is the sha256 of `str(circuit)` **without** the file's
  trailing newline; `qasm_sha256` is of the file; `native_sha256` is of the serialized instructions
  (`dqiqm.experiments.circuits.native_sha256`).
- **Before submitting,** the hardware worker rebuilds the suite with
  `dqiqm.experiments.emerald.build(<the "before" snapshot>)` and compares every circuit's
  `native_sha256` with `placement.json` (`scripts/emerald_plan.py` does exactly that), then
  submits the circuits in `job_plan.json` order, so `position_in_job` is right.

## Layout

The same as `../README.md`'s, with `emerald` for `garnet`:

```
architecture/emerald.json
calibration/emerald/<UTC>-<calset>.json  + qubits.csv couplers.csv durations.json placement.json README.md
circuits/<experiment>/<name>.stim .qasm, placement.json; circuits/job_plan.json
raw/<YYYYMMDD>-<task>/<job id>/          dqiqm.hw.save_raw: meta.json + memory.npz; never edited
datasets/<experiment>/<run id>/          raw_meas.npy det_events.npy obs_flips.npy shots.csv detectors.csv
                                         measurements.csv qubits.csv manifest.json (run id: b<batch>_<circuit>, ep0_ for the pilot)
derived/<experiment>/<run id>/           pij.npy pij_lags.csv rate_by_round.csv rate_by_bucket.csv streaks.csv
                                         syndrome_weight.csv summary.json
ler/<experiment>/<run id>/               ler_points.csv ler_fits.json ler.png, one per (basis, patch, batch)
sim/                                     all of the above, SIMULATED; sim/checks/ holds the dry run's own checks
```

The column dictionary, the conventions (`round` = Stim's t, clbit j = Stim measurement j,
bit-packing, detection events against the noiseless reference) and the reading example are in
`../README.md`; every file here is written and checked against the same `dqiqm.inforecord.COLUMNS`.
Beyond the dictionary, `ler_fits.json` and `derived/*/summary.json` here also carry `label`,
`device` and `stand_in` (SIMULATED marks), `ler_fits.json` the decoder and noise, and
`zero_failure_points` (the rounds with 0 failures); NaN is written as `null`.
