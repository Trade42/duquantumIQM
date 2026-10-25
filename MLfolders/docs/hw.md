# `dqiqm.hw`: the one guarded path to IQM hardware

`src/dqiqm/hw.py` is the only module in the repo that may build a backend able to submit a job to IQM Resonance (`https://resonance.iqm.tech`). Everything else transpiles against the read-only `hw.backend(device)` and submits through `hw.submit(...)`. Native `iqm.pulse.Circuit` jobs (`prx_12`, `measure_012`) would go through `hw.submit_native(...)`, which is not to be used until reviewed and merged. Pulse-level jobs would go through `hw.submit_playlist(...)`, which must not be used yet (see the last section).

Importing `dqiqm.hw` makes no network call and imports nothing from `iqm`. Every test in `tests/test_hw.py`, `tests/test_hw_native.py` and `tests/test_hw_playlist.py` runs offline, against fakes.

## API

| Function | Arming | What |
|---|---|---|
| `submit(circuits, *, device, shots, purpose, ...)` | yes | Submit already-transpiled qiskit circuits as one job, write the ledger row, save the raw result. Returns a `Submission`. |
| `submit_native(circuits, *, device, shots, purpose, ...)` | yes | Submit native `iqm.pulse.Circuit`s as one job (C3's `prx_12`). **Not to be used until reviewed and merged; C3 runs only after a pilot.** Returns a `NativeSubmission`. |
| `submit_playlist(run_definition, *, context, device, calibration_set_id, purpose, ...)` | yes | Submit a Pulla run definition as one pulse-level job. **Not to be used yet.** Returns a `PlaylistSubmission`. |
| `backend(device, calibration_set_id=None)` | no | A read-only qiskit `IQMBackend`, for transpiling. Its `run`, its client's `submit_*`, and its server client's writing calls all refuse. It sends GETs only. |
| `pulla(device)` | no | A read-only Pulla, for compiling (`get_standard_compiler()`). Its `submit_playlist` refuses, and its server client sends GETs only. |
| `retrieve(job_id, device)` | no | Re-pull a circuit job's qiskit `Result` by id. All three `retrieve*` refuse an `UNKNOWN-…` placeholder (see "When the submit call raises"). |
| `retrieve_native(job_id, device)` | no | Re-pull a native job by id: a `NativeJob` with its `JobData`, results, and the server's payload (circuits rebuilt as `iqm.pulse.Circuit`, and `CircuitJobParameters`). `retrieve` can't read these jobs, because their measurement keys are not qiskit's. |
| `save_native_raw(results, *, circuits, job_id, task, device, calibration_set_id, shots, status, ...)` | no | A native job's `meta.json` and `measurements.npz`. |
| `retrieve_playlist(job_id, device)` | no | Re-pull a Pulla job by id: `(JobData, sweep results)`, the results `None` unless the job completed. |
| `save_raw(result, *, task, device, qubit_mapping, shots, ...)` | no | Write a circuit job's `meta.json` and `memory.npz`. It works on a fresh result or on a re-pulled one, and never overwrites. |
| `save_playlist_raw(sweep_results, *, job_id, task, device, calibration_set_id, shape, status, ...)` | no | The same for a Pulla job: `meta.json` and `sweep_results.npz`. |
| `list_devices()` | no | The quantum computers the token sees. |
| `architecture(device, calset)`, `calibration_set_id(device)`, `calibration_fidelities(device, calset)` | no | Calibration reads, as before. |
| `check_native_ops(circuits)` | no | The op allowlist alone, offline. Use it to check circuits before arming. |
| `compilation_options_record(options)`, `compilation_options_tag(record)` | no | What `meta.json` records for a `CircuitCompilationOptions`, and the ledger suffix it gets. |
| `playlist_shape(run_definition)` | no | What a run definition will execute: segments, soft-sweep spots, shots, qubits and playlist hash. |

`submit`'s keyword arguments:

| Argument | Default | Meaning |
|---|---|---|
| `calibration_set_id` | `None` (server default) | Pass the set of the read-only backend you transpiled against. |
| `qubit_index_to_name` | backend's map | The physical index → device qubit name map, as in `layout.HardwareCircuit`. |
| `circuit_compilation_options` | `None` (its defaults) | An `iqm.iqm_client.models.CircuitCompilationOptions`. It sets DD, heralding, active reset and the T2 limit. |
| `circuit_names` | `None` | One name per circuit, in batch order, e.g. `E1_d3_Z_r03_pA`. |
| `native_sha256` | `None` | One hash per circuit, in batch order: `experiments.circuits.native_sha256(hc)`. A hash that is not the hash of what the request sends refuses. |
| `data_root` | `data/hw` | Where the raw result goes. |
| `timeout` | `None` | Passed to `job.result()`. |

`submit` takes no other run option. The deprecated `heralding_mode=` keyword, `circuit_callback` and `use_timeslot` are refused, so every option goes through `circuit_compilation_options` and is recorded.

## Arming and the allowance

Both submit functions refuse, before any network call, unless the process is armed:

- `DQ_QPU=run` and `DQ_QPU_TASK=<task>`;
- `DQ_QPU_MAX_JOBS` and `DQ_QPU_MAX_SHOTS`: the task's allowance. Shots are total shots, circuits × shots summed over the task's jobs. A batch is one job. **Unset means zero.**

Usage is counted from the task's rows in the QPU ledger (`DQ_QPU_LEDGER`, default `/home/sal/duquantumIQM/orchestration/qpu-ledger.md`). A row that does not parse makes both functions refuse, for every task.

Every check that needs no network runs first, in this order:

1. the inputs;
2. the op allowlist (`submit`) or the run definition's shape (`submit_playlist`);
3. the arming;
4. the allowance.

Then, for `submit`:

1. The run request is built once.
2. It is checked again: its serialized ops, and its option fields against what `meta.json` will record.
3. Its physical qubits, measured qubits and circuit hashes are read from it.
4. Under an exclusive `flock` on the ledger (`ledger_lock`), the allowance is checked again, the request is submitted, and the ledger row is appended the moment the job id returns. So of two concurrent submissions for one task, only those that fit go through.
5. The lock is released before the wait for results.
6. `save_raw` writes the raw result.

If anything fails after the ledger row, the job is spent: re-pull it with `retrieve` and save it with `save_raw`. Never submit it again. A failed or cancelled job counts towards the allowance.

The ledger file is never edited by hand. A correction is a new row, and it keeps the `N x M` form in `circuits x shots`.

## When the submit call raises

On all three paths, the library's submit call (`submit_run_request`, or Pulla's `submit_playlist`) can raise after the server has accepted the job. Examples are a `requests` read timeout, a reply the client can't parse into its `JobData`, or Pulla's `deepcopy(context)` after the POST. Then no job id comes back, but the job may exist, and be billed.

So `hw` fails closed. Inside the ledger lock, after the last allowance check, it appends one row before re-raising:

- the same task, device, calibration set, qubits and `N x M` as a real row;
- the job id `UNKNOWN-<UTC stamp>`, for example `UNKNOWN-20261025T013000Z`, which can't be mistaken for a real one (a UUID);
- the purpose, tagged `[no job id: submit raised <exception type>]`. The tag holds the type only, never the message.

The row parses under main's `read_ledger` at `79092cf`, the code the running batch uses (`test_mains_parser_reads_the_fail_closed_rows`). It counts against the allowance like any row, so a retry with the same allowance is refused.

- **The error raised** is `hw.JobMayExistError`, chained from the original, with the placeholder in `ledger_job_id`. It is not a `QPUGuardError`, because nothing was refused. A `KeyboardInterrupt` stays a `KeyboardInterrupt`, with the same message as a note.
- **No row is written** for anything that fails before the submit call: the pure checks, the arming, the allowance, connecting, building the request (IQM's `validate_circuit`, which refuses `measure_012`), or the second check of the built request.
- **`retrieve`, `retrieve_native` and `retrieve_playlist`** refuse an `UNKNOWN-…` id with a clear error.

**What to do.** Never resubmit first. Stop and report to the orchestrator.

1. **Check Resonance's job list, read-only.** Use the Jobs page of the dashboard at `https://resonance.iqm.tech`: iqm-client 35.0.3 has no job-list call, so `hw` has none. Look for a job on that device near the row's time, with the row's circuits and shots.
2. **Append one correction row.** Never edit the `UNKNOWN-…` row. Write the correction with `hw`'s own row builder, so it parses:

   ```python
   from dqiqm import hw
   line = hw.ledger_line(task="<task>", device="<device>", calibration_set_id="<calset>", qubits=["QB1", "QB3"],
                         circuits=N, shots=M, job_id="<id>", purpose="correction: <...>")
   hw.append_ledger(hw.ledger_path(), line)
   hw.read_ledger(hw.ledger_path())  # must not raise
   ```

   - **The job exists:** `job_id` is its real id. The other columns, `N x M` included, are the `UNKNOWN-…` row's. The purpose is `correction: the job of UNKNOWN-<stamp>`. Then re-pull it by that id and save it as usual (`retrieve` and `save_raw`, or `retrieve_native` and `save_native_raw`).
   - **No such job:** `0 x 0`, `job_id` `none`, and the purpose `correction: UNKNOWN-<stamp> made no job (job list checked <UTC time>)`.

**The accounting is fail-closed.** `usage()` counts every row as one job with its `N x M`, so both rows count. A job that exists counts twice, and one that doesn't still counts once. The orchestrator reconciles the pair and raises `DQ_QPU_MAX_JOBS` or `DQ_QPU_MAX_SHOTS` in the next brief if the task needs them back.

## The op allowlist

A circuit may send only `prx` (qiskit's `r`), `cz`, `measure`, `barrier` and `delay` (`hw.NATIVE_OPS`). `submit` serializes each circuit with IQM's own serializer, using placeholder qubit names, before it connects. It refuses any other serialized op with a `QPUGuardError` that names the circuit and counts the ops. Examples are `reset`, `cc_prx` (a classically controlled gate, from `if_test`) and `move`. It also refuses a circuit the serializer cannot take, such as an untranspiled `h` or `cx`. The built run request is checked the same way just before it is sent.

## The ledger row's purpose

A job with default compilation options keeps its purpose unchanged. Each non-default option adds a compact suffix, in field order:

| Field | Tag | Example |
|---|---|---|
| `max_circuit_duration_over_t2` | `t2_limit` | `t2_limit=0.0` |
| `heralding_mode` | `herald` | `herald=ZEROS` |
| `move_gate_validation` | `move_validation` | |
| `move_gate_frame_tracking` | `move_tracking` | |
| `active_reset_cycles` | `reset` | `reset=2` |
| `dd_mode` | `dd` | `dd=ENABLED` |
| `dd_strategy` | `dd_strategy` | `dd_strategy=1a2b3c4d` (the first 8 hex digits of the sha256 of its JSON) |

For example, `E1 A/B, DD on [dd=ENABLED]`. A pulse-level job's purpose ends in `[pulse,playlist=<first 8 hex digits of the playlist's sha256>]`. The purpose in `meta.json` is the same tagged string.

## `meta.json` of a circuit job: `data/hw/<YYYYMMDD>-<task>/<job id>/`

Unchanged keys: `job_id`, `task`, `device`, `calibration_set_id`, `qubit_mapping`, `physical_qubits`, `shots`, `circuits`, `circuit_names`, `success`, `commit`, `purpose`, `saved_utc`, `timeline` and `memory`.

New keys, all written by `submit`:

| Key | What |
|---|---|
| `measured_qubits` | Per circuit: memory column (as a string) → the device qubit measured into it. It is read from the request's measurement keys and loci, laid out as `IQMJob` lays out a shot's bits. For an unrouted circuit it equals `qubit_mapping`. For a routed one it is `final_index_layout`, while `qubit_mapping` stays the initial placement. |
| `circuit_order` | Per circuit, in submission order: `{position, name, native_sha256}`. IQM returns no names (`circuit_names: [null]`), so `position` is what links `memory.npz`'s `c<i>` to a circuit. `native_sha256` is always the hash of the instructions sent. `name` is the caller's `circuit_names[i]`, else the qiskit circuit's name. |
| `compilation_options` | Every `CircuitCompilationOptions` field the request carried, defaults included: `max_circuit_duration_over_t2`, `heralding_mode`, `move_gate_validation`, `move_gate_frame_tracking`, `active_reset_cycles`, `dd_mode` and `dd_strategy`. Enums are recorded as their wire values (`"enabled"`, `"zeros"`). `dd_strategy` is the strategy's fields with `target_qubits` sorted, or `null`, which means the server's default strategy. |

`circuits` stays the number of circuits, an integer, because `decode.load_memory` reads it as one. The brief named the new list `circuits`; it is `circuit_order` instead.

A record saved by `save_raw` without the new arguments has the new keys as `null`. That covers the committed pilots, and `recover`-style re-pulls. Old records are not rewritten.

A default job's `compilation_options`:

```json
{"max_circuit_duration_over_t2": null, "heralding_mode": "none", "move_gate_validation": "strict",
 "move_gate_frame_tracking": "full", "active_reset_cycles": null, "dd_mode": "disabled", "dd_strategy": null}
```

## The DD on/off A/B

Compilation options belong to a job, not to a circuit, so an A/B is two jobs: the same circuits, one with `CircuitCompilationOptions(dd_mode=DDMode.ENABLED, ...)` and one with the defaults. The ledger tells them apart by the `[dd=ENABLED]` suffix, and `meta.json` by `compilation_options`.

- **Name the strategy.** Pass `dd_strategy=STANDARD_DD_STRATEGY` (from `iqm.iqm_client.models`), or your own `DDStrategy`, rather than `None`. Then `meta.json` holds the sequences that ran, not "server default".
- **Heralding.** `heralding_mode=ZEROS` post-selects on the server, so a job returns fewer rows than `shots`. `decode.load_memory` currently refuses a record whose row count differs from `meta["shots"]`. A heralded job needs that changed first. It is not in this task's paths.
- **Active reset.** Setting `active_reset_cycles` also turns off IQM's conversion of terminal measurements to `measure_fidelity` (pulla-check, point 5).

## `list_devices()`

`IQMClient(SERVER_URL)` with no device resolves one in its constructor. It raises `Quantum computer not selected` once the token sees several devices, which ours does (resonance-check, finding 1). `list_devices()` now builds iqm-client's server client with no device selected. It is read-only and calls only `get_quantum_computers()`. It returns `[{"id", "alias", "display_name"}, ...]`.

## Native circuits: `submit_native`

**Not to be used until reviewed and merged; C3 runs only after a pilot.**

garnet-calib's C3 leakage design needs `prx_12`, with `angle`/`phase` arguments. iqm-client 35.0.3's qiskit serializer can't send it: it would emit `p0`/`p1`, which IQM's validation rejects. `submit_native` takes `iqm.pulse.Circuit` objects: device qubit names, `CircuitOperation(name, locus, args)` instructions. They are what C3's `design.json` lists, one `Circuit(name=c["name"], instructions=tuple(CircuitOperation(**i) for i in c["instructions"]))` per entry.

- **The ops** are `hw.NATIVE_JOB_OPS`: `NATIVE_OPS` plus `prx_12` and `measure_012`. Anything else refuses before any connection, as do a repeated measurement key and a non-`Circuit` input.
- **`measure_012` passes the allowlist, but iqm-client 35.0.3's own validation refuses it** ("Unknown quantum operation"). It needs a newer client, and the device must have it calibrated (Garnet: only QB1, QB3, QB9 and QB14).
- **The guard is `submit`'s:**
  - the arming and the allowance, before any connection;
  - IQM's own `create_run_request` validation against the calibration set (an op on an uncalibrated locus refuses);
  - the built request checked again (ops, options);
  - an optional `native_sha256` per circuit, checked against what is sent;
  - the ledger lock, and the ledger row the moment the job id returns.

  `circuit_compilation_options` works as in `submit`, and so does the purpose tag.
- **The calibration set is pinned in the request.** When `calibration_set_id` is None, the server default's id at connection is pinned, so the set recorded is the set that runs. `submit` passes None through to the server instead.
- **The raw save** is `measurements.npz`: one array `a<j>` per circuit and measurement key, as IQM returned it, `(shots, locus)` of 0/1, or 0/1/2 from `measure_012`. There is no qiskit bitstring, so the ternary values survive. `meta.json` holds:
  - `kind: "native"`, `calibration_set_id`, `shots`, `circuits` (an integer);
  - `circuit_order` (position, the circuit's name, `native_sha256`);
  - `measurement_keys` (per circuit: key → qubits);
  - `physical_qubits`, `compilation_options`;
  - `status`, `success`, `commit`, `purpose`, `timeline`;
  - `arrays`: each array's key, circuit, measurement key, qubits, shape and dtype.
- **Re-pulls.** `retrieve_native(job_id, device)` returns the server's own payload. iqm-client 35.0.3's `CircuitJob.payload()` hands its circuits back as dicts, and `retrieve_native` rebuilds them as `iqm.pulse.Circuit` objects, with the same `native_sha256` and measurement keys. `save_native_raw(job.results, circuits=job.circuits, ..., compilation_options=hw.compilation_options_record(job.parameters))` into a scratch directory reproduces the fresh save, except `saved_utc`. `save_native_raw` also takes the payload's dicts as they come.

## Pulse level: `submit_playlist`

**`hw.submit_playlist` is not to be used until Sal and the mentors confirm pulse-level submission rights.** Resonance's read access to Pulla works (pulla-check). The challenge text says PulLA needs a paid organisation, and whether a pulse-level submission from our account would be accepted, or billed, is unknown. The function is tested only offline, against a fake Pulla. No pulse-level job has ever been sent.

What it would do:

- **The shape.** It reads what will run from the run definition itself, and fails closed with a `ValueError` if it can't (`playlist_shape`):
  - **shots** are `settings.controllers.options.playlist_repeats`, what `SettingNode.set_shots` writes;
  - **circuits** are the playlist's segments × the soft-sweep spots (the whole playlist runs once per spot);
  - **qubits** are `additional_run_properties["qubits"]`;
  - **the playlist hash** is the sha256 of `pack_playlist(playlist)`'s protobuf, serialized deterministically. That is how the run definition sends it.
- **The guard is `submit`'s:** the same arming, the same allowance (counted as circuits × shots above), the same ledger lock, and the ledger row the moment the job id returns. `calibration_set_id` is required, because a run definition does not carry one. Pass the set you compiled from. `use_timeslot` is always off, and any other option is refused.
- **The raw save.** `sweep_results.npz` holds one array `a<j>` per return parameter and soft-sweep spot, as the server returned them. `meta.json` holds:
  - `kind: "playlist"`, `playlist_sha256`, `physical_qubits`;
  - `segments`, `spots`, `circuits`, `shots`, `averaging_bins`;
  - `status`, `success`, `commit`, `purpose`, `timeline`;
  - `arrays`: each array's key, return parameter, spot, shape and dtype.
- **Re-pulls.** `retrieve_playlist(job_id, device)` and `save_playlist_raw(...)` re-pull a job and save it into a scratch directory for comparison.

## Tests

`uv run pytest tests/test_hw.py tests/test_hw_native.py tests/test_hw_playlist.py`, offline. The socket-blocking `tests/conftest.py` and the ledger isolation apply.

| Item | Tests |
|---|---|
| Compilation options | `test_a_default_job_records_the_defaults_and_keeps_its_purpose`, `test_compilation_options_are_recorded_and_tag_the_ledger` (through IQM's own `create_run_request`), `test_the_tag_names_only_what_differs_from_the_defaults`, `test_only_circuit_compilation_options_are_accepted`, `test_a_request_whose_options_differ_from_the_record_refuses` |
| Circuit order | `test_the_circuit_order_is_recorded_by_position` (against `experiments.circuits.native_sha256`), `test_without_names_the_order_takes_the_serialized_circuit_names`, `test_a_circuit_order_that_does_not_fit_refuses`, `test_a_wrong_native_sha256_refuses_before_submitting` |
| `list_devices` | `test_list_devices_works_when_the_token_sees_several` (the real server client, with only its HTTP layer stood in; it also shows the old route raising), `test_list_devices_works_with_one_device` |
| Op allowlist | `test_the_op_allowlist_refuses_before_any_connection` (`reset`, `cc_prx`, `move`, `h`), `test_native_ops_pass_the_allowlist`, `test_the_built_request_is_checked_again` |
| `submit_native` | all of `tests/test_hw_native.py`: `prx_12` through IQM's own `create_run_request`, per-key and ternary raw save, options and tag, refusals before any connection, IQM refusing `prx_12` on an uncalibrated qubit and `measure_012` in 35.0.3, a wrong hash, a failed job, a re-pull through iqm-client's own `CircuitJob.payload()` from the request's JSON |
| The submit call raising | per path, `test_a_submit_call_that_raises_leaves_one_fail_closed_row` (one row, the allowance consumed, a retry refused, the placeholder refused by `retrieve*`), `test_a_keyboard_interrupt_in_the_submit_call_also_leaves_the_row`, `test_no_row_for_a_failure_before_the_submit_call` (connect, build, request check, arming, allowance), `test_no_row_when_connecting_fails` (playlist), `test_mains_parser_reads_the_fail_closed_rows` (main's `read_ledger` at `79092cf`, from git) |
| `submit_playlist` | all of `tests/test_hw_playlist.py`: shape, refusals, ledger-first, raw save, a failed job, a re-pull, the read-only Pulla |
| Parked fixes | `test_every_submit_route_of_the_read_only_backend_refuses` (15 routes, bound, unbound and private, on the real `IQMClient` and `_IQMServerClient`), `test_a_routed_circuit_records_where_each_clbit_is_measured`, `test_measured_columns_follow_iqms_own_memory_layout`, `test_of_two_concurrent_submissions_only_one_fits_one_job`, `test_the_ledger_lock_refuses_a_missing_ledger` |
