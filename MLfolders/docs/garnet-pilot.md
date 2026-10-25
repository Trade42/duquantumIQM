# garnet-pilot: the first hardware job (P0 on IQM Garnet)

garnet-pilot task, 2026-10-24. One job, one circuit, 1,000 shots, as Sal pre-approved it (19:50 UTC):
P0 = `E1_d3_Z_r03_pA`, the d = 3 rotated surface-code memory in the Z basis, R = 3 rounds, no reset, on
Garnet's single zero-SWAP qubit set, with the plan's default barriers (`docs/garnet-design.md`, the
pilot section).

**Result: PASS.** Every stop/go check is inside its band. PyMatching beats "always 0": P_L(R=3) = 0.209
[0.185, 0.235] against 0.379 [0.349, 0.410]. All numbers here are **HARDWARE**.

## The job

| | |
|---|---|
| job id | `01a17000-6394-755f-a58b-b4f9c01893b0` |
| device | `garnet` |
| calibration set | `a18a947f-0408-4628-81ea-1715637e434f`: the design's own, and the server default before, at and after the job |
| circuits x shots | 1 x 1,000 (the whole allowance) |
| physical qubits | QB1, QB2, QB4–QB16, QB19, QB20 (17), zero SWAPs |
| submitted | 2026-10-24T22:20:36Z (the ledger row); completed 22:21:06.88Z |
| ledger | `orchestration/qpu-ledger.md`: exactly one `garnet-pilot` row, `1 x 1000` |
| raw | `InfoRecord/raw/20261024-garnet-pilot/01a17000-6394-755f-a58b-b4f9c01893b0/` (`meta.json`, `memory.npz`), as `hw.save_raw` wrote it |
| code | `meta.json.commit` = `f33ac04`. The circuit, `suite.build` and `hw.submit` are main's (`9979b57`). The driver `scripts/garnet_pilot.py` was committed on this branch (`d0857ff`) before the job |

It was submitted once, through `hw.submit` only, armed for that one command with `DQ_QPU=run
DQ_QPU_TASK=garnet-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000`. Nothing failed: `save_raw` wrote the
result itself, and the recovery path was not used. Re-pulled read-only by id (`hw.retrieve`) into
scratch, the memory, the calibration set and the timeline are identical to the committed raw data.

## How to run it

```
uv sync --locked
uv run python scripts/garnet_pilot.py snapshot                 # live, read-only: resonance_check.py + patch check
uv run python scripts/garnet_pilot.py verify --snapshot <S> --live
DQ_QPU=run DQ_QPU_TASK=garnet-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \
    uv run python scripts/garnet_pilot.py submit --snapshot <S>   # done once; it now refuses (ledger row, raw dir)
uv run python scripts/garnet_pilot.py analyze \
    --before data/calibration/garnet/a18a947f-0408-4628-81ea-1715637e434f.20261024T221752Z.json \
    --after  data/calibration/garnet/a18a947f-0408-4628-81ea-1715637e434f.20261024T222152Z.json
uv run pytest tests/test_garnet_pilot.py
```

`analyze` is offline and needs no `DQ_QPU`. It rebuilds the suite (about 17 s), reads the committed raw
data and writes into fresh `datasets/`, `derived/` and `ler/` directories: to rerun it, use
`--root <scratch>`. Nothing here needs the sim lock.

## Before the submission

**Calibration.** The live snapshot (`snapshot`, 22:17:52Z) was calibration set `a18a947f`, the one the
suite was designed on, so the circuits needed no rebuild against a new set. The patch check passed on
it anyway: `prx` and `measure` (the implementation each locus actually runs) on all 17 qubits, and
`cz` on all 24 couplers. The job ran with `calibration_set_id=a18a947f` passed explicitly, the before
snapshot's. `submit` checked that this was still the server default, and `IQMClient.create_run_request`
validated the circuit against the live dynamic architecture, before anything was sent.

**The circuit** (`verify`, rebuilt with `suite.build(device.live_calibration(<the committed a18a947f
snapshot>))`):

| check | value |
|---|---|
| `native_sha256` vs `InfoRecord/circuits/E1/placement.json` | `9c6dba2b…26be73`, equal |
| `stim_sha256`, `qasm_sha256` | equal to `placement.json` |
| ops (serialized) | prx 106, cz 72, measure 33, barrier 17: only {prx, cz, measure, barrier} |
| cz loci | 72, on 24 distinct couplers, all couplers of Garnet, exactly the patch's |
| measurements | 33; measurement j in clbit j, on the device qubit its Stim qubit is placed on |
| SWAPs | 0 |

**Pinning the 17 qubits** (from iqm-client 35.0.3's code, before the submission):
- `hw.submit` calls `IQMBackend.create_run_request(circuits, shots, qubit_index_to_name=...)`. That
  only serializes: there is no transpile and no layout pass.
- P0's transpiled circuit has a layout whose registers are its own, so `serialize_instructions` maps
  each qubit through the layout to its physical index, then through `qubit_index_to_name` to a name.
  We passed the map the circuit was placed and checked with (`HardwareCircuit.qubit_index_to_name`),
  so the live backend's own index order never entered.
- `IQMClient.create_run_request` only validates those loci against the dynamic architecture. It builds
  a `RunRequest` with `qubit_mapping=None`: the loci are physical names. The request carries no layout
  or routing option. Its options were heralding `none`, DD `disabled`, `active_reset_cycles=None`
  (reset by waiting), `max_circuit_duration_over_t2=None` (the server's default check), and the MOVE
  flags (`strict`, `full`), which do nothing on Garnet: it has no MOVE gate.
- `verify --live` built that very request, exactly as `hw.submit` calls it, on the read-only
  `hw.backend("garnet", a18a947f)` (whose `run` and `submit_*` refuse). Its instructions hash to the
  planned `native_sha256`, and `hw._sent_qubits` is the 17 placement qubits.
- Afterwards, the ledger row's qubits, `meta.json`'s `physical_qubits`, and its `qubit_mapping`
  (logical qubit v → `placement[stim_qubits[v]]`) all equal the plan.

**Rehearsal.** Before the token was used, the whole path ran offline on the real P0 circuit. Real:
`hw.submit`, `hw._connect`, `IQMBackend.create_run_request` and IQM's own validation against the
snapshot's architecture. Faked, with sockets blocked: the server, which returned the dry run's sim-P0
memory, keyed per measurement and formatted by IQM's own `IQMJob.result()`. `save_raw`'s `memory.npz`
came back column for column identical to the input. A failure after the job id was also rehearsed,
recovered read-only by `recover`. `analyze` on sim-P0 reproduces `job_plan.json`'s `pilot.stand_in`
numbers exactly.

## The stop/go checks (`job_plan.json: pilot.band`)

| check | hardware | band | | stand-in (sim) | calibrated model, idle 556 ns |
|---|---|---|---|---|---|
| `memory.npz` shape | (1000, 33) | (1000, 33) | **PASS** | (1000, 33) | |
| all-zero rows | 1 | ≤ 5 | **PASS** | 0 | 19 per 100,000 |
| X-type round-1 bits (clbits 0, 2, 5, 7), pooled mean | 0.494 | [0.4, 0.6] | **PASS** | 0.4805 | 0.4997 |
| Z-type round-1 bits (clbits 1, 3, 4, 6), pooled mean | 0.079 | < 0.3 | **PASS** | 0.056 | 0.0795 |
| detection fraction (events / shots × 24 detectors) | 0.2213 | [0.05179, 0.34496] | **PASS** | 0.1036 | 0.1725 |
| `meta.json` `physical_qubits` and `qubit_mapping` | as planned | the placement | **PASS** | | |

The round-1 bits prove that the entangling gates ran and that the bit order is right. The X-type
ancillas are random, and the Z-type ones are 0 up to their gate and readout errors, matching the model's
0.0795 to the third decimal. The detection fraction is 1.28 times the calibrated model's prediction,
well inside the band.

## The timeline and the per-shot cost

`meta.json`'s `timeline`, the server's own record (UTC, 2026-10-24):

| source | status | time | Δ from created |
|---|---|---|---|
| iqm-server | created | 22:20:36.123 | 0 |
| iqm-station-control | received | 22:20:36.248 | 0.125 s |
| iqm-station-control | validation_started / _ended | 22:20:36.273 / .276 | |
| iqm-station-control | compilation_started / _ended | 22:20:36.299 / .383 | 84 ms of compilation |
| iqm-station-control | execution_started | 22:21:06.573 | 30.45 s (30.19 s waiting after compilation) |
| iqm-station-control | execution_ended | 22:21:06.623 | **50.66 ms of execution** |
| iqm-station-control | post_processing_started / _ended | 22:21:06.626 / .655 | |
| iqm-station-control | ready | 22:21:06.659 | |
| iqm-server | completed | 22:21:06.884 | 30.76 s in all |

- **Per shot:** execution_started → execution_ended is 50.66 ms over 1,000 shots: **50.7 µs per
  shot**. The plan assumed 300.34 µs (the 298 µs `reset_wait` plus the 2.34 µs circuit), so this is
  0.169 times the plan. Beyond the circuit it is 48.3 µs per shot, against the plan's 298 µs.
- **What that means:** 50.7 µs is less than the 298 µs passive-reset wait alone. So either the server
  does not wait `reset_wait` between shots, or this execution span does not bracket the whole shot
  loop. The timeline cannot tell the two apart. iqm-client 35.0.3's job data has no QPU-time or
  billing field (`JobData`: status, timeline, messages, errors, compilation), so the timeline is all
  there is.
- **The plan, re-estimated** with 48.3 µs per shot beyond each circuit's duration (the plan's 0.1 s per
  circuit kept, which one circuit cannot test): **88 s** for the 14 jobs and 1,116,000 shots, against
  the plan's 366 s. 26 s of the 88 s is that assumed per-circuit overhead. If the span does not cover
  the shot loop, 366 s stays the estimate to plan against.
- **Wall clock:** the job took 30.8 s from created to completed, 30.2 s of it waiting after
  compilation: the on-demand queue, or the instruments' preparation.

## Decoding (PyMatching, R = 3)

Priors: `calibrated_noise(<the before snapshot>, idle_ns=556)`, the fixed version on main. 556 ns is
the readout window: the slowest ancilla's measure (QB5 and QB7). PyMatching runs on the noisy copy of
the circuit that ran: its DEM builds with `decompose_errors=True`, has 294 error mechanisms, and its
graphlike distance is 3.

| | hardware (1,000 shots) | calibrated model (100,000 shots, seed 2026) |
|---|---|---|
| P_L, PyMatching | 209 / 1000 = **0.209 [0.185, 0.235]** | 0.1989 [0.1964, 0.2013] |
| P_L, "always 0" (raw observable flips) | 379 / 1000 = 0.379 [0.349, 0.410] | 0.3453 [0.3424, 0.3483] |

The intervals are Wilson 95%. The decoder roughly halves the raw failure rate, and the two intervals
are far apart. The model's prediction, 0.199, is inside the hardware's interval.

**ε_L, a single point:** ε = ½[1 − (1 − 2P_L)^(1/3)] = **0.0825 [0.0713, 0.0955]**, from R = 3 alone.
That is one point, not a fit: the fit 2P_L = 1 − A(1 − 2ε_L)^R needs the batches' round counts.

**Per-round detection fraction** (detector round t; t = 3 is the final data-qubit detectors):

| t | 0 | 1 | 2 | 3 (final) |
|---|---|---|---|---|
| hardware | 0.079 | 0.267 | 0.284 | 0.146 |
| calibrated model | 0.0795 | 0.176 | 0.199 | 0.206 |

Round 0 (the Z-type checks' first readout) matches the model. The bulk rounds fire 1.4–1.5 times more
than the model, and the final data detectors 0.7 times as often. Lag correlations (`derived/`,
`summary.json`) are well above the stand-in's: same-stabilizer lag 1 0.063 (stand-in 0.018), lag 2
0.025 (0.016), lag 3 0.0084 (0.0002), cross-stabilizer lag 0 0.037 (0.008). So the hardware's
round-to-round noise is larger and more correlated than the priors assume. This is one pilot's worth
of data, and the batches will say more. It is the structure the fine-tuned decoder is for.

## The record

Written by the record's own writers (`write_dataset`, `write_derived`, `write_ler`), labelled
`"label": "HARDWARE"`, with `simulated = 0` on every shot:

- `InfoRecord/datasets/E1/p0_E1_d3_Z_r03_pA/`: `device: garnet`, batch 0, split
  `finetune+validation` (800 finetune, 200 validation, by `suite.split("P0", 0, 1000)`; P0 is on the
  fine-tune side, as the plan says). The calibration paths are the before and after snapshots
  (`../data/calibration/garnet/…T221752Z.json` and `…T222152Z.json`, relative to `InfoRecord/`), and
  `raw` points at the committed `memory.npz`. `raw_meas.npy` equals `memory.npz`'s `c0`.
  `check_run` finds no problem.
- `InfoRecord/derived/E1/p0_E1_d3_Z_r03_pA/`: p_ij, rates by round and by shot bucket, streaks and
  syndrome weights. It also holds `pilot_checks.json`: every stop/go number, the full timeline and its
  spans, the decoder and the decode, and the pre-submission checks.
- `InfoRecord/ler/E1/p0_E1_d3_Z_r03_pA/ler_points.csv`: the one point with its Wilson interval and
  the single-R ε. There is no fit (one R).

Calibration snapshots, all calibration set `a18a947f` (Garnet) and `23e12f6b` (Emerald):
`data/calibration/garnet/a18a947f-…T221752Z.json` (before), `…T222152Z.json` (after), and Emerald's at
22:17:59Z and 22:21:58Z. The snapshot log's `FAIL hw.list_devices()` is resonance-check's known
finding 1: the fallback lists the devices.

## What is left

- **P1**, the barrier A/B (2 × 4,000 shots), needs its own allowance. Before B1, decide which per-shot
  figure the plan uses (above).
- The priors under-predict the bulk rounds and over-predict the final data detectors, while P_L agrees.
  That is a note for the decoder work, not a stop.
- Not done, as the brief keeps them out: a resubmission of any kind, `:mock`, a timeslot, and Pulla.
