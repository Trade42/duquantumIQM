# Live Garnet benchmark

Commands run from `ethan/`. Historical artifact paths below refer to the original
repository's local `results/` archive. The portable release evidence is described
in [benchmarks/README.md](../benchmarks/README.md).

The first live run completed on 24 October 2026 using IQM Resonance Garnet,
calibration `a18a947f-0408-4628-81ea-1715637e434f`. It collected 12,000 shots:
1,000 for each X/Z, 1/3/5-round, DD-off/XX condition. All 12 jobs completed.

Artifacts are in `results/garnet/20261024T191558Z/`: `report.md`,
`comparison.png`, `manifest.json`, raw counts and measurement/syndrome arrays,
Stim/DEM/QPY circuits, calibration observations and structured server job timelines.
These generated artifacts are ignored by Git.

| Memory | Rounds | DD off | XX DD | Relative reduction |
|---|---:|---:|---:|---:|
| X | 1 | 181/1000 (18.1%) | 103/1000 (10.3%) | 43.1% |
| X | 3 | 324/1000 (32.4%) | 191/1000 (19.1%) | 41.0% |
| X | 5 | 379/1000 (37.9%) | 287/1000 (28.7%) | 24.3% |
| Z | 1 | 47/1000 (4.7%) | 59/1000 (5.9%) | -25.5% |
| Z | 3 | 171/1000 (17.1%) | 181/1000 (18.1%) | -5.8% |
| Z | 5 | 279/1000 (27.9%) | 256/1000 (25.6%) | 8.2% |

At three rounds, X failure fell by 13.30 percentage points; the pointwise 95%
independent-shot Newcombe interval is 9.49–17.06 points. All Z differences have
intervals spanning zero. These intervals do not include device drift or adjustment
for the six comparisons. Pair order and DD order within pairs were randomized
with seed 1901; conditions used separate physical shots and interleaved jobs.

A multi-round binomial fit with a separate preparation/readout amplitude gives
X rates of 11.566%/round without DD (95% profile interval 9.368–13.909%) and
6.813%/round with XX (5.576–8.090%): a 41.1% relative reduction in the fitted
rate. These are conditional model estimates. The Z/off fit is strained
(binomial deviance 9.81); its fitted rate should not replace the raw observations.
This single-distance experiment does not establish below-threshold scaling.

Warmed correlated PyMatching throughput on the measured three-round records was
206k/220k shots/s for X off/XX and 219k/203k for Z off/XX. Build time, conversion
time and decode time are recorded separately. This is data-dependent batch CPU
throughput, not online streaming latency or a claimed decoder speed improvement.
Timing uses the original acquisition order, preserved in the saved memory and
measurement arrays; it does not group repeated records using aggregate counts.
Server compilation took a median 0.083 s; server execution intervals had median
1.139 s and range 0.121–8.955 s per 1,000-shot job. These intervals can include
device setup and contention; they are not pulse durations. Client completion
times additionally include queueing, polling and retrieval.

## Reproduce or retrieve

Install the pinned hardware extras in an isolated environment:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e '.[hardware,test]'
```

Save a token locally as raw text or JSON with a `token`, `IQM_TOKEN`, `api_key`,
`apikey` or `access_token` field. `apikey.jom` is ignored by Git. Its contents
are never written to benchmark artifacts. The token file is read only by the
explicit CLI entry point, never on import or in tests.

Prepare circuits and inspect calibration without submitting:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_garnet --token-file apikey.jom
```

This preparation retrieves live architecture/calibration but consumes no shots.
Actual execution requires the explicit `--submit` flag:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_garnet --token-file apikey.jom --submit --shots 1000
```

The runner saves each accepted job ID immediately and waits up to 30 minutes.
To retrieve existing submitted jobs after interruption or timeout:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_garnet --token-file apikey.jom --resume results/garnet/20261024T191558Z
```

Resume never submits additional jobs. A failure during submission is recorded
in `error.json`; inspect the manifest before starting another run. The runner
does not automatically retry a potentially accepted submission.

## Placement and model scope

Garnet uses its live named coupling graph. Exhaustive subgraph embedding found
four native placements for the 17-qubit distance-3 patch; gate/readout calibration
ranked those placements. The selected physical layout and decoder weights were
held fixed across all conditions. There were 24 native CZ gates per round and
zero routing SWAPs. Exact graph search in this runner is limited to 20 physical
qubits; larger devices use the existing geometric placement interface.

`prepare_qiskit` and `run_qiskit` now accept an explicit `qubit_map` mapping sparse
Stim indices to physical QB names. The map must cover exactly the circuit's
qubits, be injective, and satisfy the live couplings. Validation uses IQM's
actual serialized physical wire names. IQM 35 may retain virtual registers:
ordinary circuit bit-list indices do not always identify the physical wires.

Calibration parsing selects active native implementations, ignores invalid
observations, checks time units and retains original observations. Gate duration
observations with blank units use the seconds defined by IQM pulse parameter
schemas. The calibration snapshot retains an explicitly assumed initialization
error of 0.002; initialization fidelity was not measured in this run.

The decoder uses a gate/readout Pauli approximation. Native R decomposition,
exact pulse timing, leakage and non-QND measurement are model mismatches. T1/T2
are saved but are not applied as a timed idle channel in these decoder weights.
Readout leakage/seepage/QND observations are retained without converting them
into a per-gate leakage probability. DD conditions use identical decoder weights.

The IQM server accepted the requested XX DD strategy on the nine data qubits.
No compiled pulse schedule or inserted-pulse count is exposed in the retrieved
artifacts, so DD pulse insertion itself is not independently verified. Reported
gains compare execution conditions configured with DD off and XX.

Primary SDK references: [IQMProvider](https://docs.iqm.tech/iqm-client/api/iqm.qiskit_iqm.iqm_provider.IQMProvider.html),
[compilation options](https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html),
[DDStrategy](https://docs.iqm.tech/iqm-station-control-client/api/iqm.station_control.interface.models.circuit.DDStrategy.html).
