# The retained Garnet pipeline

The retained result is a complete memory experiment, not a decoder in isolation.
The physical compiler comes from `iqm_surface_code`; shared adapters combine it
with correlated matching for X and a model/data posterior for Z. Both choices
were kept after a separate follow-up. Hardware acquisition has ended; the
supported entry point is [offline reproduction](REPRODUCIBILITY.md).

## Scope and exact configuration

The measured system is a distance-three rotated surface-code patch with one
logical qubit, 9 data qubits and 8 check ancillas on Garnet's native coupling
graph. It performs identity memory: prepare, repeat checks, measure. The selected
patch and calibration are fixed across the comparisons. There are no routing
SWAPs or mid-circuit ancilla resets. The initial preparation is still present.

| Setting | X memory | Z memory |
|---|---|---|
| Native optimization | `fused` | `fenced` |
| Data-qubit DD | XY4, SDK sequence `YXYX`, ASAP in eligible waits | Same |
| Decoder family | `mwpm` | `posterior_10` |
| Graph/model correlations | Correlated matching at construction and decoding | Joint DEM fault probabilities in the modeled posterior |
| Native prior residual phase | `phase_round=0` | `phase_round=0` |
| Native prior backaction scale | `backaction_scale=0.5` | `backaction_scale=0` |
| Empirical posterior strength | Not used | `alpha=10` |
| Qualified rounds | 1, 2, 3 | 1, 2, 3 |

The backaction settings are **decoder-prior parameters**. They do not change the
QPU pulse sequence and do not mean that only X memory experiences physical
backaction. They were frozen empirically. The physical channel is only
approximated by the native Pauli model.

The acquisition calibration ID is
`a18a947f-0408-4628-81ea-1715637e434f`. Qualification belongs to that recorded
configuration. It is not a promise about a new calibration or a larger patch.

## From logical memory to a scored shot

### 1. Build an annotated circuit

The generator supplies the stabilizer schedule, qubit coordinates, detectors
and one logical observable. Four interaction layers implement each round's
local checks. The schedule orders interactions to control hook-error
propagation; using fewer apparent gates is not helpful if it reduces the
effective code distance or changes a boundary parity.

No-reset ancillas retain their previous measured state. Therefore a measured
ancilla bit is not a fresh stabilizer value. In the bulk, detection compares
measurements two rounds apart, `m[t] XOR m[t-2]`. Initialization, the second
round and the final data-readout boundary have their own generated equations.
The pipeline uses those annotations and Stim's measurement-to-detector
converter rather than applying one bulk formula everywhere.

### 2. Use the actual native layout and circuit

The Garnet mapper searches legal embeddings in the native coupling graph and
ranks them with a gate/readout calibration proxy. The selected placement is
then fixed. Sparse Stim qubit IDs, dense circuit indices and physical `QB`
names are separate mappings; no `QB number minus one` assumption is made.

CZ and native single-qubit rotations replace the abstract gate schedule. The
optimizer preserves measurement dependencies. It records the actual native
Clifford circuit and the permutation between source and native measurement
columns. Both circuits must produce identical detectors **and observable
flips** on the same raw records.

### 3. Preserve readout structure while optimizing

X uses terminal fusion: after the last entangling gate, remaining rotations can
move past measurements on other wires. They cannot move past a measurement on
their own wire. Local uncomputes after a wire's last measurement can be removed
when they cannot affect any returned outcome. Final measurements can then form
a joint terminal readout group.

Z uses measurement fences: retain barriers around measurement regions, while
allowing redundant single-qubit gates inside unitary regions to cancel. This
protects the physical readout schedule and the compiler's choice of measurement
implementation. The choices are algebraically checked for the ideal circuit
and qualified on hardware; ideal circuit equivalence alone does not establish
equivalent performance under scheduling and readout noise.

### 4. Protect eligible data-qubit waits with XY4

The pulse compiler inserts `YXYX` on data-qubit drive channels during suitable
idle windows. The ideal net action is the identity up to global phase; toggling
the axes can average slowly varying unwanted evolution. Ancillas are not
blanket-targeted, and leading/trailing waits are skipped. The compiler audits
the actual before/after pulse schedules, not just a requested DD flag.

For the retained R1/R2/R3 schedules, the audit records 0/72/144 extra `IQPulse`
instructions. These count compiled pulse instructions, not named pi rotations:
one rotation can require multiple native pulses. More pulses are not assumed
better; the longer adaptive trial did not qualify in fresh confirmation.

### 5. Construct the prior in the native frame

The prior attaches calibrated one-/two-qubit and readout probabilities to the
actual native Clifford schedule. Readout probabilities follow the compiled
measurement implementation, including terminal fidelity readout. Average gate
infidelity is converted to a declared depolarizing approximation. The X
backaction term is a phenomenological ancilla Pauli proxy derived from repeated
readout metrics; it is not a measured per-gate leakage rate or a qutrit model.

This prior does not explicitly model inserted DD-pulse errors, timed idle
decoherence, persistent leakage or all spectator effects. A decoder can be
exact for this prior while still being mismatched to the physical device.
[Error analysis](ERROR_ANALYSIS.md) describes the consequences.

### 6. Decode the complete measured history

X builds a correlated PyMatching graph from the native detector error model
(DEM). It finds a likely fault explanation and returns the predicted logical
flip. The retained graph is calibration-based; it is not refitted from hardware
training labels. Pilot labels still participate in choosing which policy to
release.

Z first calculates `P_model(logical_flip=1 | detector_history)` by summing joint
DEM fault probabilities. For each history `s`, training supplies its occurrence
count `n_s` and logical-flip count `k_s`. The frozen decision is

$$
\widehat q(s)=\frac{k_s+10q_{\rm model}(s)}{n_s+10},
\qquad \widehat L(s)=\mathbf 1[\widehat q(s)>1/2].
$$

Frequent histories can correct model bias. Rare/unseen histories retain the
model's regularization instead of relying on a noisy empirical majority. The
complete table covers every input history; there is no hidden MWPM fallback
for unseen syndromes. The exponential table is capped at 24 detector bits,
which limits this qualified Z policy to R1–3.

### 7. Score every shot and fit the requested endpoint

For this one-observable experiment, a failed shot is a disagreement between
prediction and reference-normalized observable flip. Truth is supplied to the
scorer, not the decoder. Raw logical bits and observable flips are different
quantities; a virtual Pauli readout correction must not be applied twice.

Independent fixed-shot points at R1, R2 and R3 are fit to the challenge equation.
All shots are retained. The result includes raw failures, intervals, fit
deviance and paired/independent comparisons, with their multiplicity adjustment.
[Results](RESULTS.md) explains why the fitted value is descriptive here.

## The Z prior is not refitted at every round count

The family is always the same posterior-plus-count rule, but its native DEM
is selected using an internal split of the 6,000 training shots per round.
The first half estimates detector correlations; the second half checks whether
those fitted weights improve matching. The fit is retained only if it wins,
then refit on all training shots. Outer pilot validation and confirmation remain
separate from these training decisions.

| Z rounds | Internal training errors: native / fitted | Prior kept for the posterior |
|---|---:|---|
| 1 | 46 / 45 | Fitted DEM |
| 2 | 374 / 371 | Fitted DEM |
| 3 | 491 / 520 | Original native DEM |

Thus, “training-fitted Bayesian ML” is shorthand for a guarded fitting pipeline,
not a claim that fitting helped every round. At R3, the original prior was
better on the internal check and was kept. Training counts still regularize
the final posterior decision there. This is parameter selection within one
decoder family, not a family switch inside the reported LER fit.

## Why X and Z use different techniques

### They measure different kinds of logical failure

| Memory basis | Ideal stored state/read observable | A logical error that reverses that observable |
|---|---|---|
| X | Logical plus state / logical X | Logical Z or Y |
| Z | Logical zero state / logical Z | Logical X or Y |

The names do not mean “X decoder receives only physical X errors” or “Z decoder
receives only Z errors.” Each decoder sees the detector history produced by
the actual circuit, including correlated mechanisms and Y faults. Under a
perfect X/Z-dual schedule and symmetric noise, one would expect corresponding
decoding problems to be related. This device and compilation do not guarantee
those symmetries.

### The physical implementations break the ideal symmetry

Garnet executes native CZ/rotations and Z readout. X preparation/readout needs
basis rotations. The fixed placement assigns different physical qubits to
different checks and logical supports, with different gate/readout quality.
The boundary detectors and terminal rotations also differ. Measurements,
coherence loss and control errors can therefore produce different conditional
logical-flip distributions in the two memory experiments.

Terminal fusion and fencing are chosen for their actual compiled schedules,
not because an abstract X or Z code universally demands that compiler. Likewise,
the different backaction priors are empirical approximations, not proof of a
basis-specific physical mechanism.

### The deciding evidence is empirical

The Z posterior hybrid reduced the original confirmation estimate from 3.920%
for native model-only ML to 3.763%, with an adjusted positive gain interval.
Most of its additional corrected failures occurred at R2. In X, the fitted
model-only ML candidate's lower aggregate point estimate did not survive its
adjusted interval and increased R3 failures. X therefore kept correlated MWPM.

The follow-up did not overturn these choices. Its X refit-only gain was small
and uncertain, while adaptive DD plus fitting was worse at the point estimate.
The recurrent model did not beat the retained project policy. These outcomes
support the chosen policies in this measured scope; they do not prove that
matching is fundamentally superior for X or that an ML decoder cannot improve X.

Summing logical fault classes and shrinking empirical counts can help when the
prior and data are informative. The same operation can fail to help when model
mismatch or finite data dominates. That is the reason to validate X and Z
separately rather than impose one algorithm on both.

## Source map and use

| Responsibility | Source |
|---|---|
| Static circuit and no-reset construction | [`surface_code.py`](../ethan/src/iqm_surface_code/surface_code.py) |
| Native embedding/calibration | [`benchmark_garnet.py`](../ethan/src/iqm_surface_code/benchmark_garnet.py) |
| Fencing, fusion, native frame, prior and pulse audit | [`hardware_optimization.py`](../ethan/src/iqm_surface_code/hardware_optimization.py) |
| Correlated matching | [`ensemble_decoder.py`](../darawork/decoders/ensemble_decoder.py) |
| Guarded training-only DEM fitting | [`dem_fit.py`](../darawork/decoders/dem_fit.py) |
| Exact XOR/FWHT lookup | [`lookup_table.py`](../ethan/src/iqm_surface_code/decoders/lookup_table.py) |
| Posterior/count adapter and frozen states | [`hybrid.py`](../challenge/decoders/hybrid.py) |
| Current adopted-policy loader | [`load_adopted`](../challenge/benchmark_rounds.py) |
| Small offline entry point | [`replay_winners.py`](../examples/replay_winners.py) |

Use the root adopted-policy loader for the final Z hybrid. The isolated
package's older `validated_memory_profile` still describes its model-only Z
LUT baseline; it is not the complete final posterior policy. Similarly, the
generic simulator's default matching path is not a registry of all research
backends. [Reproducibility](REPRODUCIBILITY.md) gives the unambiguous current use.
