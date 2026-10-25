# Improvements retained in the final Garnet pipeline

The final experiment stores one logical qubit in a distance-three rotated
surface code on IQM **Garnet**: nine data qubits and eight reused check
ancillas. It performs identity memory in X and Z, with one, two or three
syndrome rounds and no mid-circuit ancilla resets.

The improvements combine physical execution and classical inference. The
physical footprint remains **17 qubits**. Logical gate compression is not part
of this memory experiment. The measured comparison evaluates the complete
pipeline; it does not measure an independent improvement factor for each step.

## Final configuration and vanilla comparator

| Setting | Vanilla surface code | Final X memory | Final Z memory |
|---|---|---|---|
| Code and workload | Distance-three X/Z identity memory | Same | Same |
| Placement | Deterministic legal native embedding | Calibration-ranked native embedding | Same optimized embedding |
| Routing | Zero routing SWAPs | Zero routing SWAPs | Zero routing SWAPs |
| Compilation | Literal four-CX-layer schedule lowered through H-CZ-H; optimization level zero | Native rotation cancellation and terminal-readout fusion | Native rotation cancellation and measurement fences |
| Idle protection | DD off | Data-qubit XY4 | Data-qubit XY4 |
| Decoder prior | Uniform circuit Pauli probability `p=0.003` | Native calibration plus readout-disturbance proxy | Native calibration; guarded training-only correlation fitting |
| Decoder | Complete modeled logical-ML lookup | Correlated minimum-weight perfect matching | Exact modeled posterior updated with training counts, `alpha=10` |
| Hardware training | None | No labeled decoder training | 6,000 independent shots per round count |

The vanilla baseline already has a complete logical-class decoder. The final
Z policy improves the prior and adds measured training information to that
kind of inference. Initial preparation, hook-safe check ordering and correct
no-reset boundary equations are shared foundations.

## 1. Calibration-aware native placement

The mapper enumerates legal embeddings of the surface-code interaction graph
into Garnet's actual coupling graph. It ranks them using a negative-log
gate/readout-fidelity proxy and fixes the selected embedding for the optimized
experiment. Every required interaction is available directly on a native
coupler.

This assigns work to a favorable physical patch while avoiding additional
routing gates. Both comparison arms already have zero routing SWAPs: the
increment over vanilla is calibration ranking among legal placements.

The routing certificate independently checks all twelve optimized native
circuits, their couplers and measurement maps. One, two and three rounds have
24, 48 and 72 CZ gates respectively, with no added entanglers. This establishes
the minimum routing SWAP count for this fixed schedule, without claiming a
globally optimal logical error rate.

Implementation: [native placement](ethan/src/iqm_surface_code/benchmark_garnet.py),
[compiler validation](ethan/src/iqm_surface_code/run_on_hardware.py),
[routing certificate](docs/GARNET_ROUTING_PROOF.md).

## 2. Native CZ lowering, cancellation and scheduling

Syndrome extraction starts from four hook-safe interaction layers. CX operations
are expressed using CZ and basis rotations. The compiler cancels redundant local
rotations and schedules disjoint work as early as its dependencies permit.
The native optimizer further simplifies single-qubit rotations.

Removing unnecessary operations reduces control-error opportunities. The
transformation preserves the required entangling interactions, QEC rounds,
detectors and logical observable. The noise prior is constructed for the
resulting native Clifford circuit rather than attaching errors to gates that
have been removed.

Implementation: [CZ lowering](ethan/src/iqm_surface_code/surface_code.py),
[native optimization](ethan/src/iqm_surface_code/hardware_optimization.py).

## 3. Terminal-readout fusion for X memory

X preparation and readout require basis rotations around native Z measurement.
After the final entangler, eligible rotations commute with measurements on
other wires. The optimizer moves those rotations before the final measurement
group while preserving same-wire dependencies. A local uncompute after its
wire's last measurement is removed when it cannot affect any returned outcome.

This lets final ancilla and data measurements form a terminal readout group
that benefits from the hardware compiler's terminal-measurement treatment.
Intermediate ancilla measurements remain present.

Implementation: `optimize_native(..., mode="fused")` in
[hardware optimization](ethan/src/iqm_surface_code/hardware_optimization.py).

## 4. Measurement fences for Z memory

Barriers preserve boundaries between unitary regions and measurement regions.
The optimizer can cancel gates within unitary regions while retaining the
readout structure. Omitting a trailing barrier exposes terminal measurement
to the compiler's terminal fidelity-readout pass.

Readout grouping, implementation and drive timing affect physical performance
even when two circuits have the same ideal action. Fencing is the retained Z
policy; terminal fusion is the retained X policy.

Implementation: `optimize_native(..., mode="fenced")` in
[hardware optimization](ethan/src/iqm_surface_code/hardware_optimization.py).

## 5. Data-qubit XY4 dynamical decoupling

Data qubits wait while ancillas are measured. The pulse compiler inserts the
SDK sequence `YXYX` in eligible data-drive waits, using ASAP placement,
contiguous-wait merging, and skipped leading/trailing waits. The sequence
returns the ideal state to its original frame up to global phase while
refocusing selected slowly varying unwanted evolution.

The original wait duration is preserved. Only the nine data qubits are targets.
Before/after pulse audits record actual insertion:

| Syndrome rounds | Additional compiled IQPulse instructions |
|---:|---:|
| 1 | 0 |
| 2 | 72 |
| 3 | 144 |

These count compiled pulse instructions, not named logical gates. DD adds
control pulses and their errors. It does not reverse T1 population loss, clear
leakage, or correct assignment errors.

Implementation: `dd_strategy`, `PulseExecutor` and the pulse-audit helpers in
[hardware optimization](ethan/src/iqm_surface_code/hardware_optimization.py).

## 6. Exact source/native measurement-frame alignment

Optimization can change native measurement order. The pipeline records an
explicit permutation between source and native measurement columns and keeps
sparse Stim IDs, dense circuit indices and physical QB names distinct.
Both converters must produce identical detection events and reference-relative
logical flips on every returned record.

With reused ancillas, the bulk detector is `m[t] XOR m[t-2]`; initialization,
second-round and terminal checks use their generated boundary equations.
This alignment makes inference describe the actual optimized experiment.
It prevents bookkeeping errors from appearing as physical improvements.

Implementation: [native frame reconstruction](ethan/src/iqm_surface_code/hardware_optimization.py),
[detector conversion](ethan/src/iqm_surface_code/extract_syndromes.py),
[record verification](challenge/replay_rounds.py).

## 7. Calibration prior for the executed native schedule

The decoder prior uses gate and readout calibration for the compiled native
Clifford circuit. The declared depolarizing approximation converts average
gate infidelity `r` into probability `1.5r` for one-qubit gates and `1.25r`
for two-qubit gates. Native half-turns account for their rotation composition.
Initial preparation has an assumed error probability of `0.002`.

Readout assignment probability is `1 - fidelity` for the implementation
identified by the compiled measurement trace, including terminal fidelity
readout. Native gate/readout probabilities are bounded to `[0, 0.45]`.
Both retained policies use `phase_round=0`.

This is a qubit Pauli approximation. It does not explicitly model timed idle
decoherence, individual inserted DD-pulse errors, coherent interference,
non-unital dynamics or persistent population outside the qubit subspace.

Implementation: `native_noise` in
[hardware optimization](ethan/src/iqm_surface_code/hardware_optimization.py).

## 8. Intermediate-readout disturbance proxy for X

The X prior adds ancilla depolarization after nonterminal readout. For measured
QNDness `Q` and repeated-readout leakage metric `l`, its probability is

```text
clip(backaction_scale * (0.5*(1-Q) + 0.25*l), 0, 0.45)
```

The retained scale is `0.5` for X and `0` for Z. These are decoder-prior
settings frozen before evaluation. The term approximates state disturbance
that an assignment-error model misses. It does not remove physical leakage,
interpret repeated-readout leakage as per-CZ leakage, or expose hidden state
to the decoder.

Implementation: `native_noise` in
[hardware optimization](ethan/src/iqm_surface_code/hardware_optimization.py).

## 9. Correlated minimum-weight perfect matching for X

Matching connects detection events to each other or boundaries using fault
weights derived from the native prior. The logical parity of the inferred
fault explanation predicts the observable flip.

Correlation handling is enabled during both PyMatching graph construction
and decoding. It preserves useful relationships between graphlike pieces of
decomposed faults. The decoder consumes the complete measured history and
returns one logical-flip prediction per shot.

Implementation: [matching decoder](darawork/decoders/matching.py),
[final policy adapter](challenge/decoders/hybrid.py).

## 10. Guarded training-only detector-correlation fitting for Z

Detector firing rates and pairwise coincidences estimate fault-edge
probabilities. The fitter rescales existing detector-error-model mechanisms,
preserving their joint detector/observable targets. Fitted/model ratios are
clipped to `[0.1, 10]`; the final mechanism probability is capped at `0.4`.

Each round count has 6,000 independent training shots. Half fits weights;
the other half compares native and fitted matching predictions on common
records. Weights are refitted on all training records only when fitting wins
that internal comparison. The selected prior is fitted at R1/R2 and native
at R3. Evaluation shots never choose those weights.

Implementation: [correlation fitting and guard](darawork/decoders/dem_fit.py).

## 11. Exact modeled logical-class probabilities for Z

Multiple fault configurations can produce the same detector history while
implying different logical flips. Logical-class inference sums all modeled
explanations for each logical outcome before choosing between them.

Each independent DEM mechanism contributes an XOR mask over detector and
logical bits. A Walsh-Hadamard transform calculates their XOR convolution,
producing the complete joint distribution `P_model(s, L)`. Separator-linked
components of one mechanism remain one joint fault. Stim's permitted
disjoint-channel approximation is recorded explicitly.

The posterior is

```text
q_model(s) = P_model(s, 1) / (P_model(s, 0) + P_model(s, 1))
```

The summation is exact within the specified independent DEM. Its accuracy for
hardware remains limited by that model. The same inference principle already
exists in the vanilla lookup; the final Z policy combines it with the native
prior and training update below.

Implementation: [XOR/FWHT lookup](ethan/src/iqm_surface_code/decoders/lookup_table.py),
[`model_posterior`](challenge/decoders/hybrid.py).

## 12. Training-count update with Bayesian shrinkage for Z

For history `s`, let `n_s` be its training count and `k_s` its number of logical
flips. The final policy uses

```text
q_hat(s) = (k_s + 10*q_model(s)) / (n_s + 10)
prediction(s) = 1 if q_hat(s) > 0.5 else 0
```

Common histories can correct model bias using measured counts. Rare histories
remain regularized toward the model; unseen histories retain its decision.
Ties choose logical zero. The strength `alpha=10` and the policy are frozen
before held-out evaluation. Truth labels are training/scoring data and are
never inference inputs.

R1/R2/R3 have 8/16/24 detector bits. Their byte decision tables occupy
256 bytes, 64 KiB and 16 MiB. The largest transform has an estimated 512 MiB
temporary footprint. This is exponential preprocessing followed by direct
table lookup, with no scalable claim for arbitrarily long histories.

Implementation: [`build_decoder` and frozen inference](challenge/decoders/hybrid.py).

## Measurement, uncertainty and reproducibility

The comparison contains 18,000 vanilla and 36,000 optimized evaluation shots,
all scored, under calibration `a18a947f-0408-4628-81ea-1715637e434f`.
Training/selection precedes evaluation; optimized evaluation has two acquisition
blocks. Each implementation has its own fixed placement and acquisition window.

| Memory | Vanilla fitted LER, 95% profile interval | Final fitted LER, 95% profile interval | Relative reduction |
|---|---:|---:|---:|
| X | 18.517% (17.748â€“19.314%) | 5.808% (5.492â€“6.136%) | 68.6% |
| Z | 6.722% (6.326â€“7.135%) | 4.255% (4.066â€“4.449%) | 36.7% |

At three rounds, the directly measured decoded failure probability changes
from 40.667% to 18.200% for X and from 19.433% to 14.3375% for Z.

The challenge fit is `P_L(r) = [1 - (1 - 2*epsilon_L)^r] / 2`, estimated by
joint binomial maximum likelihood across R1â€“3. Individual points have Wilson
intervals. Comparison intervals use 20,000 bootstrap resamples stratified by
round and acquisition block, with Bonferroni adjustment over the two bases.
All four constant-rate fits have substantial lack of fit, so the fitted LER
is reported alongside the measured round-dependent probabilities. Calibration
drift and model mismatch add uncertainty beyond fixed-shot sampling variation.

Artifact hashes, measurement-column maps, raw-record conversion and frozen
decoder states make the result replayable offline. The original published
bundle, including its historical source-hash record, is preserved unchanged.
[`runtime_sources.json`](challenge/runtime_sources.json) ties that bundle to
checksums of the retained runtime after cleanup. Replay validates the saved
artifacts and the current retained code without importing removed research.

Exact bootstrap-interval replay uses the recorded NumPy 2.5.4 runtime.

```powershell
python -m examples.reproduce
python -m challenge.audit_garnet_routing
```

See [Methods](docs/METHODS.md), [Evaluation](docs/EVALUATION.md),
[Error analysis](docs/ERROR_ANALYSIS.md) and
[Reproduction](docs/REPRODUCIBILITY.md) for the detailed evidence and limits.
