# emerald-pilot: the first Emerald job (EP0 on IQM Emerald)

emerald-pilot task, 2026-10-24. One job, one circuit, 1,000 shots, as Sal approved it (21:14 UTC):
EP0 = `E1_d3_Z_r03_eA`, the d = 3 rotated surface-code memory in the Z basis, R = 3 rounds, no reset, on
patch eA (QB13, QB14, QB19–QB22, QB27–QB31, QB36–QB39, QB44, QB45), with the plan's default barriers
(`docs/emerald-design.md`, the pilot section).

**Result: PASS.** Every stop/go check is inside its band, including the emerald-design review's
MINOR 3: the job ran on exactly eA's 17 qubits. PyMatching beats "always 0": P_L(R=3) = 0.129
[0.110, 0.151] against 0.336 [0.307, 0.366]. All numbers here are **HARDWARE**.

## The job

| | |
|---|---|
| job id | `01a17017-0aef-7503-a36d-6d3151635fdf` |
| device | `emerald` |
| calibration set | `23e12f6b-c23c-4736-8736-b0142989c0e6`: the design's own, and the server default before, at and after the job |
| circuits x shots | 1 x 1,000 (the whole allowance) |
| physical qubits | QB13, QB14, QB19–QB22, QB27–QB31, QB36–QB39, QB44, QB45 (eA's 17), zero SWAPs |
| submitted | 2026-10-24T22:45:20Z (the ledger row); completed 22:45:23.31Z |
| ledger | `orchestration/qpu-ledger.md`: exactly 2 rows, Garnet's pilot and this job's, `1 x 1000` |
| raw | `InfoRecord/emerald/raw/20261024-emerald-pilot/01a17017-0aef-7503-a36d-6d3151635fdf/` (`meta.json`, `memory.npz`), as `hw.save_raw` wrote it |
| code | `meta.json.commit` = `5ad091b`. The circuit (`emerald.build`), `hw.submit` and `save_raw` are main's (`376cd1c`). The driver `scripts/emerald_pilot.py` was committed on this branch (`5d98a1f`) before the job |

It was submitted once, through `hw.submit` only, armed for that one command with `DQ_QPU=run
DQ_QPU_TASK=emerald-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000`. The allowance counts this task's
ledger rows only (`hw.usage` filters by task), so the Garnet row did not count against it; `verify`
showed that before arming (1 row in the ledger, 0 `emerald-pilot` rows, the job fits). Nothing failed:
`save_raw` wrote the result itself, and the recovery path was not used. Re-pulled read-only by id
(`hw.retrieve`) into scratch, the memory, the calibration set, the qubit map, the physical qubits and
the timeline are identical to the committed raw data. `meta.json`'s `circuit_names` is `[null]`, as on
Garnet: that is what IQM's result header returned. The circuit is named in the dataset manifest and in
`pilot_checks.json`, and by its `native_sha256`.

## How to run it

```
uv sync --locked
uv run python scripts/emerald_pilot.py snapshot                 # live, read-only: resonance_check.py + patch check
uv run python scripts/emerald_pilot.py verify --snapshot <S> --live
DQ_QPU=run DQ_QPU_TASK=emerald-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \
    uv run python scripts/emerald_pilot.py submit --snapshot <S>   # done once; it now refuses (ledger row, raw dir)
uv run python scripts/emerald_pilot.py analyze --root <scratch> \
    --before data/calibration/emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261024T224404Z.json \
    --after  data/calibration/emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261024T224614Z.json
uv run pytest tests/test_emerald_pilot.py
```

`analyze` is offline and needs no `DQ_QPU`. It rebuilds the frozen EP0 (about 25 s), reads the
committed raw data and the ledger (for MINOR 3's check), and writes into fresh `datasets/`, `derived/`
and `ler/` directories under `InfoRecord/emerald/`. Those are committed, so to rerun it use `--root
<scratch>`. Nothing here needs the sim lock.

## Before the submission

**The freeze rule (emerald-design review, MINOR 1).** The circuit is frozen. EP0 was rebuilt from the
**committed** snapshot (`emerald.build(emerald.live_calibration(emerald.DEFAULT_SNAPSHOT))`, the
record's byte copy of the 23e12f6b snapshot of 19:13 UTC), never from a live one. Its `native_sha256`
equals `InfoRecord/emerald/circuits/E1/placement.json`'s. `InfoRecord/emerald/circuits/` was not
touched.

**Calibration.** The live snapshot (`snapshot`, 22:44:04Z) was calibration set `23e12f6b`, the design's
own, so the "calset changed" branch (check the frozen circuit against the new calibration, re-rank for
information only) did not apply. The frozen EP0 was checked against the live calibration anyway:
`prx` and `measure` (the implementation each locus actually runs) are served on all 17 qubits and `cz`
on all 24 couplers, and `emerald.check_live_couplers` with the live calibration passes (72 cz, 0
SWAPs). The job ran with `calibration_set_id=23e12f6b` passed explicitly, the before snapshot's.
`submit` checked that it was still the server default, and `IQMClient.create_run_request` validated
the circuit against the live dynamic architecture, before anything was sent.

**The circuit** (`verify`):

| check | value |
|---|---|
| `native_sha256` vs `placement.json` | `f70ccfd7…0fbd092`, equal |
| `stim_sha256`, `qasm_sha256` | equal to `placement.json` |
| ops (serialized) | prx 106, cz 72, measure 33, barrier 17: only {prx, cz, measure, barrier} |
| cz loci | 72, on 24 distinct couplers: all among the calibration's 85 CZ couplers, none of the 5 CZ-less ones (QB6–QB7, QB15–QB23, QB24–QB32, QB49–QB50, QB50–QB54), exactly the patch's |
| measurements | 33; measurement j in clbit j, on the device qubit its Stim qubit is placed on |
| SWAPs | 0 |
| qubits | exactly eA's 17 |

**Pinning the 17 qubits.** This is the method `scripts/garnet_pilot.py` established from iqm-client
35.0.3's code, re-run on Emerald:
- `hw.submit` calls `IQMBackend.create_run_request(circuits, shots, qubit_index_to_name=...)`, which
  only serializes: no transpile, no layout pass. Each qubit goes through EP0's layout to its physical
  index, then through `qubit_index_to_name`, the map EP0 was placed with, to a device name.
- `IQMClient.create_run_request` only validates those loci against the dynamic architecture, and sets
  `qubit_mapping=None`: the loci are physical names.
- `verify --live` built that very request on the read-only `hw.backend("emerald", 23e12f6b)`, whose
  `run` and `submit_*` refuse. Its instructions hash to the planned `native_sha256` (`f70ccfd7…`), and
  `hw._sent_qubits` is exactly eA's 17 qubits. Its options were heralding `none`, DD `disabled`,
  `active_reset_cycles=None` (reset by waiting) and `max_circuit_duration_over_t2=None`, plus the MOVE
  flags (`strict`, `full`), which do nothing on Emerald: it has no MOVE gate.
- Afterwards, the ledger row's qubits, `meta.json`'s `physical_qubits` and its `qubit_mapping` all equal
  the plan (MINOR 3, below).

**Rehearsal.** `analyze --simulated` on the dry run's sim-EP0 (`InfoRecord/emerald/sim/raw/…/sim-EP0`)
reproduces `job_plan.json`'s `pilot.stand_in` exactly (0.513, 0.05625, 0.08042, 1 all-zero row) and
writes `raw_meas`, `det_events` and `obs_flips` identical to the committed sim dataset. Its model
sample (100,000 shots, seed 2026) reproduces `pilot.calibrated_stim_with_idle` bit for bit (detection
fraction 0.11593791666666667, 59 all-zero rows). `tests/test_emerald_pilot.py` runs submit and recover
against a fake connection, on a ledger that already holds the Garnet row.

## The stop/go checks (`job_plan.json: pilot.band`, plus MINOR 3)

| check | hardware | band | | stand-in (sim) | model, no idle | model, idle 360 ns |
|---|---|---|---|---|---|---|
| `memory.npz` shape | (1000, 33) | (1000, 33) | **PASS** | (1000, 33) | | |
| all-zero rows | 0 | ≤ 5 | **PASS** | 1 | 119 per 100,000 | 59 per 100,000 |
| X-type round-1 bits (clbits 0, 2, 5, 7; QB14, QB30, QB28, QB44), pooled mean | 0.4835 | [0.4, 0.6] | **PASS** | 0.513 | 0.5002 | 0.5002 |
| Z-type round-1 bits (clbits 1, 3, 4, 6; QB21, QB39, QB19, QB37), pooled mean | 0.0853 | < 0.3 | **PASS** | 0.0563 | 0.0612 | 0.0612 |
| detection fraction (3,307 events / 1,000 shots × 24 detectors) | 0.1378 | [0.0402, 0.2319] | **PASS** | 0.0804 | 0.0884 | 0.1159 |
| **MINOR 3:** `meta.json` `physical_qubits` | eA's 17 | eA's 17 | **PASS** | | | |
| **MINOR 3:** the ledger row (task, device, calset, qubits, `1 x 1000`, job id) | eA's 17, all fields as planned | eA's 17 | **PASS** | | | |
| `meta.json` `qubit_mapping` (logical qubit v → `placement[stim_qubits[v]]`) | as planned | the placement | **PASS** | | | |

The round-1 bits show that the entangling gates ran and that the bit order is right: the X-type
ancillas are random, and the Z-type ones are 0 up to their gate and readout errors. The Z-type mean is
1.39 times the model's (0.0853 against 0.0612). eA's placement is the one thing the bands cannot see (a
valid automorphic placement gives the same Stim record), so MINOR 3's two checks read it from the
ledger row and from `meta.json`, both written from the run request that was sent.

**Detection fraction against the model.** Hardware 0.1378 against the with-idle prediction 0.1159:
**1.19×**. On Garnet it was **1.28×** (0.2213 against 0.1725). Emerald's absolute rate is also lower
than Garnet's (0.138 against 0.221), as the two models predict (0.116 against 0.172).

## The timeline and the per-shot cost

`meta.json`'s `timeline`, the server's own record (UTC, 2026-10-24):

| source | status | time | Δ from created |
|---|---|---|---|
| iqm-server | created | 22:45:20.758 | 0 |
| iqm-station-control | received | 22:45:20.891 | 0.133 s |
| iqm-station-control | validation_started / _ended | 22:45:20.987 / .990 | |
| iqm-station-control | compilation_started / _ended | 22:45:21.077 / .236 | 159 ms of compilation |
| iqm-station-control | execution_started | 22:45:21.522 | 0.764 s (0.29 s after compilation) |
| iqm-station-control | execution_ended | 22:45:22.880 | **1,357.7 ms of execution** |
| iqm-station-control | post_processing_started / _ended | 22:45:22.881 / .901 | |
| iqm-station-control | ready | 22:45:22.902 | |
| iqm-server | completed | 22:45:23.310 | 2.55 s in all |

- **Per shot:** execution_started → execution_ended is 1,357.7 ms over 1,000 shots: **1,357.7 µs per
  shot**. The plan assumed 401.8 µs (Emerald's 400 µs `reset_wait` plus the 1.848 µs circuit), so this
  is **3.38 times** the plan, and 957.7 µs per shot beyond the reset wait.
- **Emerald does not show what Garnet showed.** Garnet's span was 50.7 µs per shot, less than its 298 µs
  `reset_wait` alone (0.17 times its plan). Emerald's is longer than its `reset_wait` plus the circuit.
  So on Emerald the span is consistent with the server waiting `reset_wait` between shots, with about
  0.96 ms per shot more on top; on Garnet it cannot be.
- **What the job data can and can't tell.** The timeline is the only timing there is: iqm-client
  35.0.3's job data has no QPU-time, per-shot or billing field. One job of one circuit at one shot
  count cannot separate a per-shot cost from a fixed per-job cost inside the execution span (instrument
  set-up, waveform upload, the data return). 1,357.7 ms could be 1,000 × 1.36 ms, or about 400 ms of
  waits plus about 0.96 s fixed, or anything between. Nor can it say whether `reset_wait` itself is
  honoured. Two shot counts (the batch's jobs, for instance) would give the affine fit that separates
  them.
- **The plan, re-estimated.** If the whole span is per shot (1,355.9 µs per shot beyond each circuit's
  duration, the plan's 0.1 s per circuit kept), the 15-job Emerald plan (313,000 shots) comes to
  **442 s**, against the plan's 143 s. That is still inside its 619 s pessimistic column (Garnet's CLOPS
  bound). If most of the 0.96 s is fixed per job, it is closer to 143 s + 15 × 0.96 s ≈ 157 s. Plan
  against 442 s until the batch's first jobs settle it.
- **Wall clock:** 2.55 s from created to completed, with no queue wait to speak of (0.29 s between
  compilation and execution). Garnet's job waited 30.2 s there.

## Decoding (PyMatching, R = 3)

Priors: `emerald.calibrated_noise(<the before snapshot>, spec, idle_ns=360)`. 360 ns is
`emerald.readout_window_ns`: the slowest of eA's ancillas' `measure` (every measure is 360 ns). The
readout window's idle is counted once (data qubits p_xy + p_z, ancillas p_xy). PyMatching runs on the
noisy copy of the circuit that ran: its DEM builds with `decompose_errors=True`, has 294 error
mechanisms, and its graphlike distance is 3.

| | hardware (1,000 shots) | calibrated model (100,000 shots, seed 2026) |
|---|---|---|
| P_L, PyMatching | 129 / 1000 = **0.129 [0.110, 0.151]** | 0.0915 [0.0898, 0.0933] |
| P_L, "always 0" (raw observable flips) | 336 / 1000 = 0.336 [0.307, 0.366] | 0.2071 [0.2046, 0.2096] |

The intervals are Wilson 95%. The decoder cuts the raw failure rate by 2.6 times, and the two hardware
intervals are far apart. Unlike Garnet's, **the model's P_L (0.092) is below the hardware's interval**:
the hardware's P_L is 1.41 times the model's, and its raw flip rate 1.62 times. The priors
under-predict Emerald's noise, as the detection fraction (1.19×) and the per-round table say too.

**ε_L, a single point:** ε = ½[1 − (1 − 2P_L)^(1/3)] = **0.0473 [0.0396, 0.0566]**, from R = 3 alone.
That is one point, not a fit: the fit 2P_L = 1 − A(1 − 2ε_L)^R needs the batches' round counts.
Garnet's single point was 0.0825 [0.0713, 0.0955].

**Per-round detection fraction** (detector round t, Stim's t: t = 0 is the 4 Z-type checks' first
readout, t = 1 and 2 have 4 X-type and 4 Z-type detectors each, t = 3 is the 4 final data-qubit
detectors):

| t | 0 | 1 | 2 | 3 (final) |
|---|---|---|---|---|
| hardware | 0.0853 | 0.1674 | 0.1723 | 0.0623 |
| calibrated model (idle 360 ns) | 0.0612 | 0.1213 | 0.1335 | 0.1248 |
| hardware / model | 1.39 | 1.38 | 1.29 | 0.50 |

For comparison, Garnet's pilot gave 0.079, 0.267, 0.284 and 0.146 against its model's 0.0795, 0.176,
0.199 and 0.206: 1.0, 1.5, 1.4 and 0.7.

Round 0 and the bulk rounds fire about 1.3–1.4 times as often as the model on Emerald. The final data
detectors fire half as often. On Garnet only the bulk rounds were above the model. The final data
detectors are below the model on both devices. Lag correlations (`derived/`, `summary.json`) are above
the stand-in's: same-stabilizer lag 1 0.039 (stand-in 0.017), lag 2 0.019 (0.0075), lag 3 0.0057
(0.0025), cross-stabilizer lag 0 0.017 (0.0064) and lag 1 0.0089 (0.0031). Every one of them is below
Garnet's pilot (0.063, 0.025, 0.0084, 0.037 and 0.012). This is one pilot's worth of data:
the batches will say more, and this structure is what the fine-tuned decoder is for.

## The record

Written by the record's own writers (`write_dataset`, `write_derived`, `write_ler`), labelled
`"label": "HARDWARE"`, with `simulated = 0` on every shot:

- `InfoRecord/emerald/datasets/E1/ep0_E1_d3_Z_r03_eA/`: `device: emerald`, batch 0, job
  `01a17017-…`, split `finetune+validation` (800 finetune, 200 validation, by
  `suite.split("EP0", 0, 1000)`; EP0 is on the fine-tune side, as the plan says). The calibration
  paths are the before and after snapshots (`../../data/calibration/emerald/…T224404Z.json` and
  `…T224614Z.json`, relative to `InfoRecord/emerald/`), and `raw` points at the committed `memory.npz`.
  `raw_meas.npy` equals `memory.npz`'s `c0`. `check_run` finds no problem.
- `InfoRecord/emerald/derived/E1/ep0_E1_d3_Z_r03_eA/`: p_ij, rates by round and by shot bucket,
  streaks and syndrome weights. `summary.json` carries `label: HARDWARE` and `device: emerald`.
  `pilot_checks.json` holds every stop/go number, the full timeline and its spans, the decoder and the
  decode, the model's prediction, Garnet's reference numbers and the pre-submission checks.
- `InfoRecord/emerald/ler/E1/ep0_E1_d3_Z_r03_eA/`: `ler_points.csv` has the one point, with its Wilson
  interval and the single-R ε. `ler_fits.json` carries `label`, `device`, `decoder`, `noise` and
  `zero_failure_points` (none), and no fit (there is one R).

Calibration snapshots: Emerald `data/calibration/emerald/23e12f6b-…T224404Z.json` (before) and
`…T224614Z.json` (after), both calibration set 23e12f6b. Garnet's were taken at 22:43:58Z and 22:46:07Z
(a18a947f), because `resonance_check.py` snapshots both devices. The snapshot log's `FAIL
hw.list_devices()` is resonance-check's known finding 1: the fallback lists the devices.

## What is left

- **E-B1** needs its own allowance. Before it, settle the per-shot figure: 1,358 µs measured on
  Emerald against the plan's 402 µs, and Garnet's 50.7 µs against its 300 µs. The first batch job's
  span, at another shot count, separates the per-shot cost from a fixed one.
- The priors under-predict Emerald: P_L by 1.41×, the bulk detection rate by 1.3–1.4×, and the round-0
  Z-type bits by 1.39×. They over-predict the final data detectors (0.5×). That is a note for the
  decoder work, not a stop: every band passes.
- Not done, as the brief keeps them out: a resubmission of any kind, `:mock`, a timeslot, and Pulla.
