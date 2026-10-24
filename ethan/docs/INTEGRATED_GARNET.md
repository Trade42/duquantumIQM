# Integrated Garnet path: diagnosis and measured improvement

The implementation now lives under `ethan/src/iqm_surface_code/`. Historical
`results/...` paths refer to the original repository's local acquisition archive;
selected validation shots and frozen decoders are included in
[the portable replay bundle](../benchmarks/README.md).

The confirmed distance-3 memory policies reduce three-round decoded failure by
37.1% in X and 35.9% in Z relative to the previous optimized XX/MWPM stack.
These are hardware observations, with 1,500 fresh shots per point. They are
whole-experiment failure probabilities, not automatically per-round LERs.

## What was preventing improvement

1. **Terminal readout was compiled incorrectly for our intended use.** Every
   Stim `TICK` became a global barrier, including a trailing barrier. IQM 35's
   `subscribe_and_probe` discovers terminal measurements by traversing the
   instructions backwards. The barrier eclipses every qubit, so the returned
   final bits retain `measure.constant`. The compiler then adds
   `measure_fidelity` probes on all 20 qubits under `__ONLY_PROBE`; their better
   readout results are discarded. X readout also had local basis-uncompute
   rotations after measurement, creating the same obstruction.
2. **Fewer barriers were not sufficient.** Removing every barrier permits drive
   and readout scheduling that performed worse on this device. The rejected
   aggressive pilot/confirmation study is retained, including its regressions.
3. **The decoder described a different gate schedule.** Previous weights used
   logical CZ/H operations. Actual execution uses native R rotations, pulse
   scheduling and different intermediate/terminal readout implementations.
   Logical-frame and measurement-order agreement are necessary before changing
   those weights.
4. **Extra features were being treated as guaranteed improvements.** Longer DD
   sequences add imperfect pulses. Fast readout changes disturbance as well as
   duration. A leakage stress simulator does not remove physical leakage. A
   faster offline decoder cannot change errors already recorded by the QPU.
   Transversal algorithm simplification has no logical operations to remove in
   an identity memory experiment.

The readout diagnosis is supported by the installed compiler source and
compiled operation traces. The Z read-only recompilation reproduces the winning
submitted playlist byte for byte. The comparison recompilation has the same
operation counts and timings, but differs in serialization of unused probes;
it is not claimed to be byte identical to the older submitted control.

## Changes now used

- **Measurement fences:** retain boundaries around readout regions, remove
  unnecessary unitary-region barriers, and omit the trailing barrier.
- **X terminal fusion:** after the final CZ, move remaining local basis rotations
  before measurements on other wires. Discard local uncomputes after a wire's
  final measurement; they cannot change any returned probability. Reject
  repeated measurements in this terminal tail. Final ancilla and data readout
  then form one joint terminal group. Intermediate no-reset checks are retained.
- **Real pulse execution:** PulLA compiles the exact submitted playlist against
  the pinned calibration. Before/after pulse traces count inserted DD pulses,
  with DD targeted to the nine data qubits. This verifies actual insertion,
  rather than reporting only a requested strategy.
- **Native decoder:** rebuild the Clifford circuit from actual native R/CZ/M
  instructions. Remap every classical record and detector/observable reference.
  Use calibrated native gate/readout priors. When available, the compiled
  readout trace overrides assumptions about terminal/fast measurement.
- **Evidence-based selection:** separate pilot shots select the physical profile,
  decoder family and residual model. Freeze all choices before interleaved fresh
  confirmation jobs. Keep the previous stack if the candidate fails the release
  gate. No observable truth is used during inference; no shots are filtered.

Current release configuration:

| Memory basis | Physical profile | Decoder | Frozen residual parameters |
|---|---|---|---|
| X | `fused_xy4` | Native correlated MWPM | phase=0, backaction scale=0.5 |
| Z | `fenced_xy4` | Native full-history ML LUT | phase=0, backaction scale=0 |

The backaction channel is a Pauli proxy built from measured QND/leakage metrics;
it is not a calibrated qutrit leakage process. X selected it on pilot shots. Z
did not select it. Neither result demonstrates physical leakage removal.
The decoder remains an effective Pauli model; it does not simulate analog DD
waveforms or individually locate every inserted DD pulse fault. Actual hardware
comparisons include those pulse errors as well as unmodeled crosstalk.
The LUT retains joint faults in its independent DEM, with explicit
disjoint-channel approximation during circuit-to-DEM conversion.

## Fresh hardware evidence

Garnet calibration: `a18a947f-0408-4628-81ea-1715637e434f`. All experiments use the
same native 17-qubit patch, with zero routing SWAPs. X and Z use separate fresh
validation batches on that calibration; they are not simultaneous estimates of
one unknown-state logical channel.

| Basis | Rounds | Previous errors / 1500 | Released errors / 1500 | Previous failure | Released failure |
|---|---:|---:|---:|---:|---:|
| X | 1 | 169 | 35 | 11.27% | 2.33% |
| X | 2 | 263 | 121 | 17.53% | 8.07% |
| X | 3 | 315 | 198 | 21.00% | 13.20% |
| Z | 1 | 75 | 33 | 5.00% | 2.20% |
| Z | 2 | 154 | 127 | 10.27% | 8.47% |
| Z | 3 | 262 | 168 | 17.47% | 11.20% |

At the primary three-round endpoint, current-minus-new absolute reductions are
7.80 percentage points in X and 6.27 in Z. Their 97.5% Newcombe intervals are
4.73–10.86 and 3.41–9.13 points. This applies a Bonferroni correction over the
two final memory-basis endpoints. It does not cover every exploratory study,
unmeasured device drift or uncertainty in future calibrations. Both intervals
exclude zero, so both candidates pass the release gate.

The entire pipeline changed. These comparisons do not independently identify
the contribution of each compiler, DD and decoder change.

| Three-round execution metric | X previous → new | Z previous → new |
|---|---:|---:|
| Native R instructions | 141 → 112 | 114 → 114 |
| Native CZ instructions | 72 → 72 | 72 → 72 |
| Additional DD microwave pulses | 136 → 144 | 108 → 144 |
| Total microwave pulses | 277 → 256 | 222 → 258 |
| Compiled schedule, including passive initialization | 306.268 → 301.900 µs | 304.240 → 301.900 µs |

X removes 29 R instructions (20.6%); total microwave pulses drop only 7.6%
because DD adds pulses. Z improves despite an increased pulse count. Gate count
alone therefore does not predict logical failure. In the Z timing audit the
first data pulse is approximately 298 µs into the schedule; recorded terminal
data readout starts at 302.104 µs in the control and 300.864 µs in the new path.
The relevant pre-readout interval falls from 4.104 to 2.864 µs. The whole
schedule duration includes a large passive initialization wait and should not
be interpreted as quantum-memory exposure.

### Per-round LER is more conditional

Both conventions are saved. The challenge fit is
`P(r) = [1 - (1 - 2e)^r] / 2`. Its fitted rates are X 9.456% → 4.156% and Z
5.901% → 3.835% per round, but the new fits have poor deviance (19.10 and
15.77 with two nominal degrees of freedom). Those headline rate reductions
would overstate the evidence for a stationary repeated-cycle improvement.

The common first-round-offset model is
`P(r) = [1 - V1 * (1 - 2e)^(r-1)] / 2`. It separates first-round preparation
and terminal effects from an effective later-round rate:

| Basis | Previous effective rate, 95% profile CI | New effective rate, 95% profile CI | Deviance previous / new, df=1 |
|---|---|---|---|
| X | 6.975% [5.209, 8.708] | 6.052% [5.107, 7.017] | 0.76 / 0.00 |
| Z | 7.077% [5.897, 8.265] | 5.381% [4.440, 6.330] | 2.11 / 3.24 |

The X rate intervals overlap substantially. Much of the strong X memory-result
gain is terminal/preparation related. The raw fresh-shot endpoint supports the
release; it does not establish a universal 37% reduction in repeated-round LER.
There is no unencoded-qubit break-even result or distance-scaling claim.

### Decoder-only evidence and cost

On the exact same new Z three-round records, the old decoder fails 187/1500 and
the native LUT 168/1500. It rescues 55 shots and harms 36; exact McNemar
`p=0.0586`. This improvement alone is not conclusive at 95%. On the new X
records, native MWPM fails 198 versus the old decoder's 214; the physical-path
change is still essential to the overall result.

For a warmed 1,500-shot Z batch, 31 timing repetitions gave 285,307 shots/s for
old correlated MWPM and 22,796,350 shots/s for native LUT: **79.9× throughput**.
This excludes graph/table construction, loading and measurement conversion;
it is offline batch throughput, not streaming latency or a QPU error reduction.
The three-round LUT uses 16 MiB for decisions and about 5.5 s preprocessing,
with much larger temporary transform arrays. It supports one observable and
at most 24 detector bits; longer distance-3 histories fail explicitly. MWPM
remains available for longer histories via an explicit profile selection.

## Run and reproduce

Use the pinned isolated hardware environment (`.[hardware]`). Imports and tests
never access credentials, retrieve device data or submit jobs.

```python
from iqm_surface_code.benchmark_garnet import read_token
from iqm_surface_code.run_on_hardware import run_optimized_memory, decode_hardware_results

syndromes = run_optimized_memory(
    token=read_token("apikey.jom"), distance=3, rounds=3, basis="Z", shots=1500,
)
probability, standard_error = decode_hardware_results(syndromes)
```

`run_optimized_memory` defaults to Garnet and the released basis-specific policy.
It saves source/native/decoder circuits, the exact pulse playlist, readout
mapping, calibration/quality data, original-order measurements, dataset hashes
and pulse audit under a unique `results/integrated_api/` directory. Decoder
selection travels with the returned syndromes. `run_qiskit` still supports the
previous path for controls; `hardware_profile=` explicitly enables this pulse
path. A profile owns DD settings, so conflicting legacy DD options are rejected.
Logical-operation/frame options are forwarded, but those workloads have not
received this memory benchmark's hardware validation. The audited pulse path
currently refuses non-Garnet backends.

To reproduce the two bounded selection/validation designs with **new hardware
submissions**, use:

```powershell
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_integrated --token-file apikey.jom --submit --profile-set protective --pilot-shots 500 --confirmation-shots 1500
.venv/Scripts/python.exe -m iqm_surface_code.benchmark_integrated --token-file apikey.jom --submit --profile-set terminal --bases X --pilot-shots 500 --confirmation-shots 1500
```

Use `--resume <existing-run-directory>` to retrieve/continue an interrupted run;
saved job IDs prevent duplicate submission. Keep the token in the ignored local
file or `IQM_TOKEN`, outside source and result artifacts. A changed calibration
requires a new pilot/confirmation before assuming the same ranking.

## Evidence locations and limits

- Z release: `results/integrated/20261024T204754Z/` — 24 completed jobs,
  24,000 shots, frozen selection, all records and reports, decoder benchmark,
  read-only timing audit.
- X release: `results/integrated/20261024T212753Z/` — 9 completed jobs,
  10,500 shots, frozen selection, all records and reports, public-path reproduction
  hashes and combined release summary/figure.
- Rejected aggressive study: `results/integrated/20261024T203720Z/` — 34
  completed jobs, 44,000 shots. Its forced aggressive candidate worsened X
  three-round failure from 25.70% to 34.70% and Z from 21.25% to 28.50%.
  The release guard rejects both. This negative result is preserved.
- Initial end-to-end pulse smoke: `results/integrated/20261024T202959Z/`,
  2 shots. Total hardware acquisition for this investigation: **78,502 shots**.

The supplied paper's directional tile codes, native iSWAP exchanges and
data/check role swapping are different circuits. The saved Garnet architecture
exposes CZ, not native iSWAP, and this challenge pipeline remains a no-reset
surface memory. Its leakage-removal mechanism and reported 100× tile-code
advantage are not implemented or claimed here. Calibration-aware scheduling,
explicit noise mismatch, exact detector frames and decoder co-design are the
applicable lessons. Persistent leakage and coherent-noise simulators remain
useful sensitivity tests; they do not prove which mechanism dominates Garnet.

References: [supplied paper](https://arxiv.org/abs/2606.19482),
[IQM measurement implementations](https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.measure.html),
[IQM standard DD strategy](https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.STANDARD_DD_STRATEGY.html),
[PulLA configuration and usage](https://docs.iqm.tech/iqm-pulla/Configuration%20and%20Usage.html).
