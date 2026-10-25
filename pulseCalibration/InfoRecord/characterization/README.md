# InfoRecord/characterization: Garnet's coherent-error record

The coherent errors of the gates our d=3 surface code uses on Garnet, measured at circuit level (no pulse access): per qubit the PRX amplitude and axis errors, per coupler the CZ conditional phase and the residual local Z phases. Device: **garnet**, calibration set `a18a947f-0408-4628-81ea-1715637e434f`. The design, the model and the numbers are in `docs/garnet-calib.md`.

**Nothing here has run on hardware.** `circuits/` and `job_plan.json` are the design and the plan for Sal's approval. `sim/` is the SIMULATED closure test on the stand-in. `raw/`, `datasets/` and `derived/` at this level are for hardware data, once it exists, and are never ignored.

## Layout

| Path | What |
|---|---|
| `circuits/patch.json` | the pilot patch (17 qubits, 24 couplers, the 4 CZ layers, roles) and the calibration priors the plan used |
| `circuits/C1/`, `C2/`, `C4/` | one `.qasm` per circuit (the placed, transpiled circuit) and `placement.json` |
| `circuits/C3/design.json` | the leakage design as IQM native instructions; **blocked**: it needs `prx_12`, which `dqiqm.hw` cannot send |
| `job_plan.json` | the jobs in order, with shots, QPU-time estimates, the pilot and its stop/go bands |
| `sim/` | the closure test: `closure.json` (coverage and the pilot dry run), `closure.log`, `derived/<world>/r<NN>/` (`table.csv`, `fits.json`), `raw/` (`hw.save_raw` format) |

## Conventions

- **Columns.** Every circuit measures all 17 patch qubits at the end, clbit j into column j, and column j is `column_qubits[j]` (`placement.json`), the qubits sorted QB1, QB2, QB4, …, QB16, QB19, QB20. A raw record is `hw.save_raw`'s: `memory.npz` with `c<i>` of shape (shots, 17) per circuit, in the job's order.
- **Submission.** `hw.submit([circuits in job_plan order], device="garnet", shots=…, qubit_index_to_name=placement.json's, calibration_set_id=…)`. Each circuit's `native_sha256` is of the instructions IQM's serializer sends; a rebuild (`scripts/garnet_calib_build.py --check`) must reproduce it before anything is submitted.
- **No merging.** Every repeated gate has a barrier after it, and the circuits are placed at optimization level 0; `placement.json` records `ops` (the circuit's `count_ops()`) and `sent` (the serialized counts), which equal the design.
- **Angles** are in rad. The table's columns are in `src/dqiqm/characterization/table.py`.
- **SIMULATED** marks every file under `sim/`: the `label` in each JSON, `simulated=True` and `change_status` in each table row, and the purpose in each raw `meta.json`.

## Regenerate

```
uv run python scripts/garnet_calib_build.py                 # circuits/ and job_plan.json, ~20 s, offline
uv run python scripts/garnet_calib_build.py --check         # rebuild and compare every native_sha256 and qasm_sha256
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/garnet_calib_closure.py --clean
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/garnet_calib_compensate.py
uv run python scripts/garnet_calib_report.py                # the numbers, as markdown
```

A regeneration reproduces every number; only the time fields differ (`timestamp_utc`, `saved_utc`, `seconds_simulating`, the log). The closure is seeded (Aer seed `2026 + crc32(job label) % 10^6`, injected errors seed `1000 + replica`). Git keeps every `derived/` file and the raw records of replica 1 of each world and of the pilot; `.gitignore` here lists the rest, which the command above regenerates.
