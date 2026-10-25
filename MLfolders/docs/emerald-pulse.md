# emerald-pulse: EP0's circuit as a pulse-level job we compiled ourselves

<!-- result:begin -->
**What ran:** one pulse-level job on IQM Emerald, **`01a170f0-8c09-74b8-b7d7-ed83d24602bc`**. It is EP0's circuit, compiled by us into an IQM pulse playlist (sha256 `2f20cfbb…`), 1 segment × 1,000 shots, on calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`. Submitted 2026-10-25 02:42:55 UTC, status `completed`. **Resonance accepted a pulse-level job from our account.** It has one ledger row (`emerald-pulse-pilot`, `1 x 1000`), and its raw data is committed.

**The one-line result:** decoded with EP0's own decoder, the pulse job's **P_L(R = 3, Z) is 117/1000 = 0.117 [0.099, 0.138]**, against EP0's 0.129 [0.110, 0.151]. The difference is −0.012 [−0.041, +0.017] (Fisher p = 0.45). Every per-round detection fraction agrees with EP0's within ±0.012. The pulse-level path reproduces the circuit-level job within what 1,000 shots can resolve.
<!-- result:end -->

## What it is

The challenge's advanced functionality. It is the Emerald pilot circuit EP0 (`E1_d3_Z_r03_eA`: d = 3, Z basis, 3 rounds, 17 qubits, 24 couplers, zero SWAPs), compiled **locally** with IQM Pulla's standard compiler into a pulse playlist. That playlist was sent as one pulse-level job through our guarded, ledgered path (`dqiqm.hw.submit_playlist`), and the shots were decoded exactly as EP0's circuit-level job was.

## Run it

```
PYTHONHASHSEED=0 uv run python scripts/emerald_pulse.py compile    # live, read-only: snapshot, compile, record
PYTHONHASHSEED=0 DQ_QPU=run DQ_QPU_TASK=emerald-pulse-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \
    uv run python scripts/emerald_pulse.py submit                  # the one job (done; never again)
uv run python scripts/emerald_pulse.py recover --job-id J          # read-only; only if submit failed after the job id
uv run python scripts/emerald_pulse.py standin                     # offline: EP0's own shots through the playlist path
uv run python scripts/emerald_pulse.py analyze                     # offline: decode the job, compare with EP0
uv run pytest tests/test_emerald_pulse.py                          # offline, against a fake Pulla
```

None of these needs the sim lock. The compile takes about 45 s of CPU, `standin` 20 s, and `analyze` about 30 s.

## The record: `InfoRecord/emerald/raw/20261025-emerald-pulse-pilot/`

`hw.save_playlist_raw` names the folder after the armed task (`DQ_QPU_TASK=emerald-pulse-pilot`), so the folder is `…-emerald-pulse-pilot/`, not `…-emerald-pulse/`. Our own record sits beside the raw job, in the same folder.

| File | What |
|---|---|
| `compile.json` | The snapshot, the frozen-circuit checks, the compile settings, the playlist and run-definition hashes, the shape, IQM's `qpu_runtime` estimate, the segment duration per channel, the 33 readout labels and their classical bits, `integration_occurrences`, the compiled schedule's readout triggers, and every HTTP request (12 GETs, all 200). |
| `run_definition.pb.gz` | The serialized run definition (`serialize_run_definition`, protobuf, gzip), as compiled, before Pulla gave it a `run_id`. |
| `<job id>/meta.json`, `<job id>/sweep_results.npz` | `hw`'s raw save, never edited: one array per readout label, as the server returned it. |
| `run_definition.submitted.pb.gz` | The same run definition after `Pulla.submit_playlist` set its `run_id` and `sweep_id`: what was POSTed. |
| `submit.json` | The submission: job id, status, the ledger row, the server's own copy of the payload (`get_submit_run_payload`, a GET) checked against what we sent, or the refusal's class, HTTP status and scrubbed message. |
| `analysis.json` | The decode and the comparison with EP0. |

This makes the playlist record self-contained, which answers hw-hardening2 review's MINOR 2. A reviewer can rebuild the shots from `sweep_results.npz` and `compile.json`'s `readout_map` alone, and the run definition is kept.

## Step 1: the snapshot (read-only)

At 2026-10-25 02:39:35 UTC, the server's default calibration set for Emerald was **`23e12f6b-c23c-4736-8736-b0142989c0e6`**, the set EP0 ran on and the circuits were placed on. Pulla's stash of it holds 7,011 observations (sha256 of the sorted observations: `f6096180…`). The live dynamic architecture serves `prx` and `measure` on all 17 of EP0's qubits and `cz` on all 24 of its couplers (`patch_gates`: nothing missing).

## Step 2: the compile

**Under pulla_check's guard.** `compile` connects through `pulla_check.connect`. That installs the class-level guard: `Pulla.submit_playlist` and every writing server call refuse, and only GETs are sent. No `DQ_QPU*` is set.

**The circuit is the frozen EP0.**
- It is rebuilt from the committed snapshot (`emerald.build(emerald.live_calibration(DEFAULT_SNAPSHOT))`).
- Its `native_sha256` is `f70ccfd7…`, equal to `placement.json`'s.
- The compiled `iqm.pulse.Circuit` is made of exactly the instructions EP0's run request sent (`circuits.sent`): 106 `prx`, 72 `cz`, 33 `measure` and **17 barriers, kept**, with qiskit's measurement keys `c_33_0_<j>`. Those instructions hash to the same `native_sha256`.
- The committed QASM gives the same instructions (`pulla_check.load_emerald`: sha256, op counts and qubits against `placement.json`).

**The compiler.** `Pulla.get_standard_compiler(loading_rules=[LatestFromStash(<the 23e12f6b stash>)])`, with default stage options. Then `get_settings(circuits=[EP0])`, `set_shots(1000)` (so `averaging_bins` = 1,000, single-shot data) and `compile`. It is the same compiler emerald-compile used, here with the `prepend_reset` pass kept, because the job really runs.

| | |
|---|---|
| playlist sha256 | **`2f20cfbb1782bce3f5276d3a321dddc70a2ee12128d6433d2329ba13013d0dc3`** |
| run definition (canonical sha256) | `0a74dfd05f3d9f28ffa9a42aaff00351bfcac8c058e5d197bbb5503717de2b35` |
| segments × soft-sweep spots × `playlist_repeats` | 1 × 1 × 1,000 = **1,000 shots** (the ledger's `1 x 1000`) |
| segment duration | **403,956 ns** on all 95 channels: the 400 µs reset wait + 3,956 ns of circuit, as emerald-compile found |
| readout triggers | at 400,256, 401,548 and 402,880 ns, each 1,036 ns, on 5 probe lines (PL-1, 2, 3, 4, 8) |
| schedule | 214 `IQPulse` (106 PRX + 108 `prx_12` of the shelved terminal readout), 144 `FluxPulse`, 144 `VirtualRZ`, 18 `ReadoutTrigger` |
| `end_delay` | 5 µs |
| **IQM's `qpu_runtime` estimate** | **0.409 s**: (5 µs + 403.956 µs) × 1,000. The cap was 10 s. |
| return parameters | 33, `<qubit>__c_33_0_<j>`, one per classical bit; each label's qubit is the one EP0's placement measures into bit j |
| qubits in the request | **all 54**: `additional_run_properties["qubits"]`, so the ledger row names 54. The standard compiler's `probe_all` adds a terminal `measure_fidelity` (probe only) on every qubit of the chip, as emerald-compile found. The circuit's gates act on EP0's 17 only. |

**The estimate** is IQM's own formula (`station_control/interface/models/sweep.py`, `_qpu_runtime`): `end_delay` × segments × repeats, plus repeats × the segment's duration, times the soft-sweep spots. `options.playlist_repeats`, read through `find_by_name` (what the formula reads), equals `controllers.options.playlist_repeats` (what `hw.playlist_shape` reads): 1,000 both ways (hw-hardening2 review, MINOR 2's last suggestion).

**The context.** Pulla deep-copies `context` *after* the POST (`pulla.py:200-204`), so a context that can't be copied would raise once the job exists. We pass only what result post-processing reads, `readout_metrics` and `mapped_readout_keys` (empty: no key is 64 characters or longer). The compile checks that it deep-copies, and that IQM's own `construct_circuit_execution_results`, run on EP0's shots laid out as sweep results under this context, gives back EP0's 33 columns exactly.

**Determinism, and `PYTHONHASHSEED=0`.** Two compiles in one process give the same playlist. Two processes did not at first: the standard compiler builds the probe-all measurement from `tuple(missing)`, where `missing` is a set of qubit names, so the order depends on Python's string hashing. `compile` and `submit` therefore refuse unless `PYTHONHASHSEED=0`, and with it a second process reproduced the record exactly. The run definition's raw protobuf bytes also differ between processes for the same content, because `serialize_run_definition` packs the sweep definition into an `Any` whose bytes keep map fields in arbitrary order. The record's `run_definition_sha256` therefore hashes the run definition with the sweep definition serialized deterministically (`canonical_sha256`). `run_definition_gz_sha256` is the sha256 of the file itself.

## The submit

`submit` runs in a fresh process that never installs pulla_check's guard. If it did, `hw.submit_playlist` would hit the refusing class method inside its `try`, and write a fail-closed row for a job never sent; `preflight` refuses if the guard is installed. It:

1. checks the arming is exactly the brief's (`emerald-pulse-pilot`, 1 job, 1,000 shots, the real ledger) and `PYTHONHASHSEED=0`;
2. rebuilds the frozen EP0, reads the server's default calibration set (`23e12f6b`), and pins the same stash (same digest as the record);
3. recompiles with `hw.pulla("emerald")`, the instance-level read-only Pulla. It refuses unless the playlist hash, the canonical run-definition hash, the shape, the `qpu_runtime` estimate, the return parameters, the readout map and `integration_occurrences` all equal the committed record;
4. refuses if the estimate is over 10 s, if the ledger already has an `emerald-pulse-pilot` row, if the record folder already holds a job, or if the slim context does not deep-copy;
5. calls `hw.submit_playlist(run_definition, context=<slim>, device="emerald", calibration_set_id="23e12f6b-…", purpose=…)` once (`use_timeslot` forced off), with `emerald_pilot.LedgerWatch` printing the job id the moment the row lands.

A refusal by the server is a result: `submit.json` records `JobMayExistError`'s fail-closed row and the original exception's class, HTTP status and scrubbed message. Nothing is retried.

<!-- analysis:begin -->
## The result: job `01a170f0-8c09-74b8-b7d7-ed83d24602bc`

**The run.** These are the timeline's server timestamps, 2026-10-25 UTC:
- created 02:42:55.12;
- validation 02:42:56.51 to 02:42:58.86;
- execution 02:43:01.42 to 02:43:03.09, which is 1.67 s for IQM's 0.409 s estimate;
- completed 02:43:03.32.

The ledger row lists 54 qubits, `1 x 1000`, and the purpose `emerald-pulse EP0 E1_d3_Z_r03_eA as a locally compiled Pulla playlist [pulse,playlist=2f20cfbb]`. `sweep_results.npz` holds 33 arrays, one per readout label, each 1,000 `float64` values, all 0 or 1 (thresholded single shots).

**What the server ran is what we sent.** `payload-check` re-pulls the job's stored payload (`get_submit_run_payload`, a GET). Its canonical hash equals that of `run_definition.submitted.pb.gz`, its playlist hash is `2f20cfbb…`, and its `run_id` matches (`server_payload_check.json`). `submit.json`'s `server_payload_matches_submitted: false` compares raw protobuf bytes, and those differ only in the packed sweep definition's map order (see "Determinism" above).

**Sanity, before any decode.**
- 1 shot of 1,000 is all zeros, so the counts are not all zeros.
- The mean detection fraction is 0.134, against EP0's 0.138 and the compiled prior's model at 0.137.

**Decoded as EP0 was.**
- The shots are rebuilt from `sweep_results.npz` by `compile.json`'s `readout_map`: column j = classical bit j, each label's qubit checked against the placement.
- The decoders are the ones `pulla_check.py --compare` applies to EP0, both from EP0's before-snapshot of the same calibration set:
  - `pilot_record`: EP0's own decoder (`pre_fix_noise(emerald.calibrated_noise(idle_ns=360))`);
  - `compiled`: `device.hardware_noise` with the compiled 1,036 ns window.
- Both datasets go through the same code (`compare`), in the same run.
- The model columns are each prior's own noisy circuit: 100,000 Stim shots, seed 2026.
- P_L intervals are Wilson 95%. A difference's interval is Newcombe's hybrid score interval, built from the two Wilson intervals. Fisher's exact test is two-sided.

| prior | EP0 (circuit job `01a17017…`) P_L | pulse job P_L | pulse − EP0 (Newcombe 95%) | Fisher p | ε single point, pulse (R = 3) |
|---|---|---|---|---|---|
| `pilot_record` | 129/1000 = 0.129 [0.110, 0.151] | 117/1000 = **0.117** [0.099, 0.138] | −0.012 [−0.041, +0.017] | 0.45 | 0.0425 [0.0353, 0.0512] |
| `compiled` | 136/1000 = 0.136 [0.116, 0.159] | 129/1000 = **0.129** [0.110, 0.151] | −0.007 [−0.037, +0.023] | 0.69 | |

**Per-round detection fractions.** t = 0 is round 1, Z-type only; t = 3 is the final data detectors. Each fraction is the detections over detectors × 1,000 shots, with a Wilson 95% interval.

| detector round t | detectors | EP0 fraction | pulse fraction | pulse − EP0 (Newcombe 95%) | model, compiled prior |
|---|---|---|---|---|---|
| 0 | 4 | 0.0853 [0.0770, 0.0943] | **0.0845** [0.0763, 0.0935] | −0.0008 [−0.0130, +0.0115] | 0.0612 |
| 1 | 8 | 0.1674 [0.1594, 0.1757] | **0.1608** [0.1529, 0.1690] | −0.0066 [−0.0181, +0.0049] | 0.1735 |
| 2 | 8 | 0.1722 [0.1641, 0.1807] | **0.1643** [0.1563, 0.1725] | −0.0080 [−0.0196, +0.0036] | 0.1898 |
| 3 | 4 | 0.0622 [0.0552, 0.0702] | **0.0673** [0.0599, 0.0754] | +0.0050 [−0.0058, +0.0158] | 0.0936 |

The compiled prior's detector log-likelihood is −9,309 on the pulse job, against −9,500 on EP0. The pulse job's bulk rounds sit a little below the model (0.93× and 0.87× at t = 1, 2), where EP0's sat at 0.96× and 0.91×.

**What 1,000 shots can show:**
- **The pulse path works end to end.** Our own compile, the guarded submit, the raw save and the decode by readout label give syndrome data that decodes like the circuit job's. This is not the same shots re-labelled: the stand-in check below is that.
- **The two jobs agree.** The P_L difference is −0.012, with a 95% interval of [−0.041, +0.017]. A P_L difference larger than about ±0.04 is excluded. The per-round detection fractions agree within about ±0.012 (pooled over 4,000 to 8,000 detector-shots per round).
- **No sign of a compile mismatch.** If our local compile differed materially from the server's circuit-job compile (timing, readout, the shelved terminal readout), the bulk-round detection fractions would move. They don't, at the 0.01 level.

**What it can't show:**
- **Small differences.** At P_L ≈ 0.12–0.13 with 1,000 shots each, the smallest difference detectable at 80% power (two-sided α = 0.05) is about **0.041** absolute, a third of P_L itself. "No significant difference" is not "equal".
- **The round fractions' precision is optimistic.** Their intervals treat detector-shots as independent, but detection events within a shot are correlated (a single error flips two detectors), so the true uncertainty is somewhat wider. The minimum detectable round difference quoted in `analysis.json` (about 0.016) is optimistic for the same reason.
- **ε_L.** One round count (R = 3) and one basis give a single-point ε only. A fit of ε_L over rounds needs the batch's round series.
- **Why they would differ.** The two jobs ran 4 hours apart (22:45 and 02:43 UTC) on the same calibration set. Drift, and any difference between our default-stage compile and the server's circuit-job compile, are confounded at this size.
- **The decoder choice.** As for EP0, the two priors are not separated by 1,000 shots. The pulse job's P_L under them differs by 0.012.

## Caveats

- **The server's circuit-job compile is not visible to us.** We compiled with Pulla's standard compiler and its default stage options. The compiled schedule matches emerald-compile's (3,956 ns body + 400 µs reset wait, the same readout triggers and the shelved terminal readout), but whether the server compiled EP0's circuit job with exactly these options is not known (pulla-check's caveat). The agreement above says any difference doesn't show at this size.
- **54 qubits in the ledger row.** `probe_all` reads all 54 qubits in the terminal layer, so the job's request names all 54. Only EP0's 17 carry gates. The 37 others get a probe-only `measure_fidelity` (and the shelving `prx_12` pulses), with no data returned.
- **Pulse-level rights.** The challenge text says PulLA needs a paid organisation. Our account's submission was accepted, executed and returned data. How it is billed is for Sal to check on the Resonance dashboard; this task records only the job.
<!-- analysis:end -->

## The stand-in dry run (offline)

`standin` lays EP0's own committed shots (`memory.npz`) out as the playlist's sweep results, one array per readout label of the record. It sends them through `to_raw` and the same `compare` as `analyze`. It gives EP0 back exactly: the same 1,000 × 33 array, 129 failures under EP0's own decoder (`pilot_record`) and 136 under `hardware_noise`'s compiled prior, as in `docs/pulla-check.md`, and a difference of zero in every row. `tests/test_emerald_pulse.py` checks the same layout against the EP0 dataset's `det_events.npy`.

## Tests

`uv run pytest tests/test_emerald_pulse.py` (16 tests, offline; `conftest.py` blocks sockets):
- the committed record follows from its labels and the placement, and all its requests are GETs;
- the readout map refuses a label on the wrong qubit, a missing bit, and a bit with two labels;
- EP0's shots laid out as sweep results come back, and complex, short or non-binary arrays refuse;
- the arming must be exactly the brief's, and the real ledger;
- `preflight` refuses without connecting: guard installed, another calibration set, an existing row, an existing job folder, a recompile that differs from the record, a non-per-shot shape, an estimate over 10 s, a context that won't copy;
- one job, one row, the raw saved, the slim context and `use_timeslot=False` passed (test_hw_playlist's fake Pulla);
- a server refusal (HTTP 403) leaves exactly one fail-closed `UNKNOWN-…` row, records the status, and a retry is refused.
