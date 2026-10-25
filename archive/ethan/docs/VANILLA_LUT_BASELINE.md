# Vanilla surface-code / lookup-table Garnet baseline

Commands run from `ethan/`. Historical artifact paths below refer to the original
repository's local `results/` archive; the baseline's replay files are bundled
under [benchmarks/](../benchmarks/README.md). For an existing repository-root
token, use `--token-file ../apikey.jom`.

This implements the requested encoded baseline for the IQM challenge in
[reference/README.md](../reference/README.md). The challenge describes an offline
memory pipeline and recommends PyMatching as a starting point; it permits other
decoders and does not require a lookup table. The LUT here is the user's requested
baseline. It is a surface-code baseline, not an unencoded-qubit reference.

## Baseline definition

- Distance-3 rotated memory: nine data qubits and eight reused ancillas.
- X and Z memories, with 1, 2 and 3 syndrome rounds.
- No ancilla reset between rounds; generated two-round detection parities and
  initialization/terminal boundaries are retained.
- Canonical Stim four-CX-layer schedule, converted literally to H-CZ-H layers.
  Custom H cancellation and ASAP scheduling are disabled. Qiskit uses optimization
  level 0; mandatory IQM native R synthesis remains.
- One fixed native layout, selected deterministically from valid embeddings
  without calibration ranking. Every interaction is native; zero routing SWAPs.
- DD explicitly disabled. No logical gate sequence, leakage mitigation, custom
  pulse compilation, heralded shot exclusion or calibration-weighted decoder.
- Fixed uniform decoder prior `p=0.003`: H/CZ depolarization, initialization and
  readout flips, and data depolarization before each round. These are model
  probabilities, not claims about measured Garnet errors.
- Complete full-history lookup table, frozen before any QPU submission. No
  training shots or evaluation labels construct the table. Every shot is scored.

`src/iqm_surface_code/benchmark_vanilla_lut.py` owns preparation, explicit submission,
retrieval, scoring, fits and reports. `src/iqm_surface_code/decoders/lookup_table.py` owns
the independent LUT backend. Existing shared runner/decoder registries are not
changed.

## Measured results

All six jobs completed on 24 October 2026 at approximately 15:46 EDT, collecting
18,000 real Garnet shots, 3,000 per point. Calibration was
`a18a947f-0408-4628-81ea-1715637e434f` and was held fixed. Its observations are
archived for context; they did not tune the layout or decoder.

Artifacts: `results/decoders/lookup_table/20261024T194536Z/`.

| Rounds | X failures / shots | X probability, 95% Wilson CI | Z failures / shots | Z probability, 95% Wilson CI |
|---:|---:|---|---:|---|
| 1 | 423/3000 | 14.10% (12.90–15.39%) | 115/3000 | 3.83% (3.20–4.58%) |
| 2 | 972/3000 | 32.40% (30.75–34.10%) | 409/3000 | 13.63% (12.45–14.91%) |
| 3 | 1220/3000 | 40.67% (38.92–42.44%) | 583/3000 | 19.43% (18.06–20.89%) |

The primary fit follows the challenge exactly:

`P(r) = [1 - (1 - 2 epsilon)^r]/2`.

It gives X **18.517%/round** (95% profile interval 17.748–19.314%) and Z
**6.722%/round** (6.326–7.135%). However, its binomial deviances are **61.03**
and **56.94** on two nominal degrees of freedom. Both fits are poor. Their
conditional intervals should not be interpreted as reliable uncertainty on a
single physical rate that describes all three round counts.

After observing that mismatch, an exploratory model separated the first-round
visibility from subsequent repeated-cycle decay:

`P(r) = [1 - V1 (1 - 2 epsilon)^(r-1)]/2`.

| Basis | Effective post-first-cycle LER | 95% profile interval | First-round visibility V1 | Deviance / nominal df |
|---|---:|---:|---:|---:|
| X | **24.996%/round** | 23.186–26.835% | 0.7171 | 0.322 / 1 |
| Z | **9.715%/round** | 8.836–10.604% | 0.9213 | 2.383 / 1 |

This model fits these three points substantially better. It is a post-hoc
diagnostic with only one residual degree of freedom, not the challenge's primary
formula. It allows a distinct first-round exposure; it does not identify its
physical cause. Its intervals assume stationary independent logical flips after
the first round and omit model error, drift and model-selection uncertainty.
Use the directly observed failure curve when comparing implementations.

These shots use a different schedule/layout, decoder and time window from the
earlier DD run. Comparing their headline rates cannot isolate a DD-only or
decoder-only improvement. No unencoded-storage advantage or distance-scaling
claim follows from this baseline.

## Lookup algorithm and resource cost

An independent Stim DEM fault is a Bernoulli XOR mask over all detector bits and
the logical bit. A Walsh-Hadamard convolution sums the probability of every fault
combination. For each complete detector history `s`, the table stores
`argmax_L P(s,L)`. Separator-marked components remain one correlated fault.
There is no runtime matching call, truth input, sampled-table miss or hidden
fallback. Ties select logical zero. Circuit-to-DEM extraction explicitly allows
the disjoint-channel approximation.

For `n` detector bits, preprocessing uses `2**(n+1)` probability entries and the
decision table has `2**n` bytes. At 1/2/3 rounds, `n=8/16/24`, giving
256/65,536/16,777,216 decision entries. This is why the complete-history baseline
uses three rounds at most. Four rounds would require 4 GiB for decisions alone
and much larger construction workspace; the implementation rejects it.

The largest tables were built in 5.4–5.6 seconds each. Each decision table is
16 MiB; transform workspace can reach roughly 512 MiB. Whole-process RSS sampled
every 20 ms peaked near 953 MiB, including the IQM stack. It is not an isolated
decoder allocator measurement. The workspace estimate was corrected in metadata
after the run; all frozen prediction-table hashes stayed unchanged.

At three rounds, warmed 3,000-shot batch throughput was approximately **23.0M
shots/s for X** and **23.7M for Z**, excluding build, table loading, result parsing
and syndrome conversion. Original acquisition order is preserved. These are
amortized batch lookup rates, not single-shot end-to-end or streaming latencies.
The manifest records 31 timing repetitions, host/package information and server
execution intervals separately.

The uniform independent-fault model is a deliberate baseline. It omits actual
spatial noise variation, native R decomposition errors, exact pulse timing,
leakage and non-QND readout. Maximum likelihood is with respect to this model,
within floating-point precision, not the QPU's unknown true distribution.

## Reproduce and retrieve

Use the existing isolated environment with the pinned hardware extras. Preparation
retrieves architecture/calibration, validates circuits in matched-noise simulation
and builds all six tables; it submits no jobs without `--submit`:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_vanilla_lut --token-file apikey.jom
```

To explicitly run a new 18,000-shot experiment:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_vanilla_lut --token-file apikey.jom --shots 3000 --submit
```

To retrieve already submitted jobs without submitting again:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_vanilla_lut --token-file apikey.jom --resume results/decoders/lookup_table/20261024T194536Z
```

Job IDs are persisted immediately after acceptance. Ambiguous submission errors
are not automatically retried. Tokens are read only on explicit CLI invocation,
kept out of artifacts, and redacted from printed exception text. Imports and tests
make no QPU calls and read no credentials.

Saved evidence includes original-order memory and counts, measurement/detector/
observable arrays, ideal/noisy Stim circuits, full DEMs, QPY native circuits,
frozen LUTs and hashes, calibration observations, job IDs, timelines and fits.

Validation: **274 tests passed**, including brute-force fault enumeration against
complete LUT decisions, correlated DEM separators, deterministic/half-probability
faults, reset/no-reset X/Z simulation, native IQM serialization and ideal Aer record
checks. All 18,000 hardware shots were retained; no jobs remain pending.

Primary DEM semantics: [Stim detector error model format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md).
