# Technique guide

This guide covers the retained pipeline and the alternatives present in the
repository. **Retained** means part of the qualified Garnet policy. **Evaluated**
means tested within a recorded finite search, not automatically promoted.
**Research/simulation** means available code or a modeling capability without
a claim that it improves the retained hardware result.

The full selected configuration is in [Winning pipeline](WINNING_PIPELINE.md).
Measured comparisons are in [Results](RESULTS.md). General mechanisms below do
not substitute for those hardware comparisons.

## 1. Encoding, syndrome history and logical frames

### Rotated surface code and hook-safe extraction — retained

The distance-three patch encodes one qubit in nine data qubits. Eight ancillas
measure local X/Z parity checks. Stabilizers commute with the logical
observables, so their outcomes reveal error information without directly
measuring the stored logical value. A change in a predicted parity is a
detection event. The final data measurement ends the memory experiment.

An ancilla fault can propagate through later entangling gates into several
data faults, called a hook. Ordering the four interaction layers appropriately
controls that propagation relative to the logical boundaries. The generator
starts from a hook-safe schedule and tests the resulting distance/parities.
Distance three supplies limited redundancy; it does not make multiple faults,
leakage spread or arbitrary correlated faults harmless. For the general code
construction, see [Fowler et al.](https://arxiv.org/abs/1208.0928).

### No-reset parity conversion — retained correctness requirement

Without resetting an ancilla, its next measured bit also depends on its prior
state. If the reset-based stabilizer value is `s_t`, the corresponding no-reset
records satisfy a generated relation of the form `s_t=m_t XOR m_(t-1)`.
Comparing successive stabilizer values then gives the bulk detector
`s_t XOR s_(t-1)=m_t XOR m_(t-2)`.

The first/second rounds and the terminal data boundary are different. The code
generates those parities and uses Stim's converter. This makes the decoder
interpret the experiment that was actually measured. It does not suppress a
physical fault by itself. Ancilla readout is not assumed to clear leakage.

### Reference-normalized observables and virtual frames — retained

The inference target is whether the logical observable flipped relative to the
ideal reference, not the raw final bit. Logical Pauli operations can change
the expected reference without changing error-correction quality. For logical
workload extensions, frame tracking records that change rather than treating
it as a failure. Applying the same readout XOR again to already normalized
observable-flip labels would create an artificial error or improvement.

### Source/native measurement alignment — retained

Qiskit's bit order, sparse Stim IDs and physical qubit indices differ. Native
optimization can also reorder commuting measurements. The saved column
permutation translates acquisition-order measurements to the native record.
Source and native converters must agree on every detector and observable on
the same raw shot. This catches an error a width-only array check cannot find.

Implementation: [`surface_code.py`](../ethan/src/iqm_surface_code/surface_code.py),
[`native_stim`](../ethan/src/iqm_surface_code/hardware_optimization.py),
[`extract_syndromes.py`](../ethan/src/iqm_surface_code/extract_syndromes.py).
Stim's [DEM specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)
defines the detector/observable fault representation.

## 2. Physical compilation and control

### Native, calibration-aware placement — retained

An edge that is not physically available requires routing or a different
embedding. Routing adds entangling operations and error opportunities. The
Garnet path searches legal graph embeddings, checks all required couplings and
uses a negative-log gate/readout-fidelity proxy to rank them. It then fixes one
patch across the comparisons. A fixed placement makes decoder and compiler
comparisons meaningful instead of changing the physical experiment between arms.

The proxy is not a measured LER, and its optimum is limited to the enumerated
family. The generic grid-placement tools also support coherence/leakage costs;
those broader options should not be confused with the exact scoring function
used for this Garnet study. No distance-five Garnet placement is claimed.

### CZ lowering and local single-qubit cancellation — retained

The abstract check circuit is lowered to the device's CZ and native rotations.
Adjacent inverse or redundant rotations can cancel, reducing physical work
while preserving ideal Clifford action. The noise model is constructed after
this transformation. A model for the original abstract gate schedule would
otherwise assign errors to operations that were changed or removed.

Fewer gates are a useful mechanism, not a sufficient LER proof: compiler
changes can alter readout grouping, implementation selection and idle exposure.
This is why the project retains actual native circuits and pulse audits.

### Measurement fences — retained for Z

Fences preserve boundaries between unitary work and readout regions while
permitting optimization inside those regions. They prevent an apparently
innocent cancellation from restructuring a measurement schedule that the pulse
compiler treats differently. They retain useful physical scheduling constraints;
they do not reduce an assignment error mathematically.

### Terminal-readout fusion — retained for X

After the final entangling gate, a rotation commutes with measurements on other
wires. Moving those rotations before a final measurement group can improve
the terminal physical schedule. A local uncompute after its own wire's last
measurement can be omitted when no later returned quantity depends on it.
Same-wire dependencies and repeated measurements are checked explicitly.

This transformation preserves ideal returned probabilities in the allowed
tail. Its effect on noisy hardware comes from the resulting physical schedule
and readout implementations. The retained X/Z distinction is empirical; neither
transformation is universally required by the corresponding abstract code.

### XY4 dynamical decoupling — retained for both bases

A pi pulse reverses the sign of some unwanted Hamiltonian terms in a toggling
frame. Alternating axes can average slowly varying error evolution while
returning the ideal state to its original frame. The selected SDK sequence is
`YXYX`, a four-pulse XY-family sequence, placed ASAP in eligible data-qubit waits.
The compiler preserves the wait's total duration and audits actual insertion.
See the [IQM DD example](https://docs.iqm.tech/iqm-pulla/Example%20-%20Simple%20Dynamical%20Decoupling.html)
for the scheduling mechanism.

XY4 does not restore population lost through T1 relaxation, remove leakage, or
correct measurement assignment errors. Real control pulses also have errors.
Timing and pulse interval matter, so a longer sequence is not automatically
better; [Ezzell et al.](https://doi.org/10.1103/PhysRevApplied.20.064027)
study that tradeoff on other superconducting hardware. That literature motivates
the experiment, not its Garnet headline numbers.

### Centered/adaptive DD and fast readout — evaluated, not retained

Centered XY4 changes where pulses sit inside the same wait. The adaptive trial
uses centered XY8, XY4 or XX according to eligible window length. It seeks
stronger refocusing, at the cost of more imperfect pulses. At R3 the audit
counts 300/304 extra IQ instructions for adaptive X/Z versus 144 for retained
XY4. The independent X confirmation did not qualify adaptive DD.

Fast ancilla readout shortens the window in which data must wait. Its pulse
implementation was calibrated, but fast SSRO fidelity was unpublished in the
saved snapshot; the decoder used an explicitly labeled standard-readout proxy.
It did not win pilot selection. Faster measurement is not necessarily less
disturbing measurement. QND readout exists as an SDK capability but was not
calibrated in the acquired Garnet architecture, so it was not attempted.
The [measurement API](https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.measure.html)
describes implementation classes; availability must come from device calibration.

### Alternative Hadamard synthesis and physical merging — evaluated

The independent paths used literal-H or sqrt-Y-based decompositions with
different barriers/default server DD. A merged path applied the sqrt-Y rewrite
before the native terminal/fenced optimizer and data XY4. These can be ideally
equivalent but physically different schedules. The merged path did not earn
promotion. Algebraic compatibility is not evidence that two individually
reasonable optimizations improve LER when composed.

Implementation: [`hardware_optimization.py`](../ethan/src/iqm_surface_code/hardware_optimization.py),
[`rounds_hardware.py`](../challenge/rounds_hardware.py),
[`garnet_dataset.py`](../darawork/garnet_dataset.py).

## 3. Calibration and detector priors

### Native Pauli error model — retained approximation

The prior attaches gate and readout probabilities to the actual native circuit.
For average gate infidelity `r`, the declared depolarizing conversion is
`p=1.5r` for a qubit and `p=1.25r` for a two-qubit gate. These conversions assume
the calibration metric really is average gate infidelity. They are not exact
descriptions of coherent, non-unital or qutrit dynamics.

Readout priors follow the compiled measurement implementation. The initial
reset/preparation error prior is assumed, not a measured reset fidelity. The
native winner prior omits explicit timed idle errors and errors from inserted
DD pulses. The simulator's duration-aware coherence model is a different model.

### Repeated-readout backaction proxy — retained in the X prior

After a nonterminal ancilla measurement, the prior can add ancilla depolarization
using a scaled combination of loss of QNDness and repeated-readout leakage
metrics. It approximates unmodeled state disturbance in a qubit DEM. The
retained scale is 0.5 for X and zero for Z; residual phase is zero for both.

Those settings are empirical decoder parameters. Repeated-readout leakage is
not a per-CZ leakage probability, and no per-shot hidden flag is supplied to the
decoder. The term does not physically remove leakage or prove its causal share
of the observed error rate.

### Detector-correlation fitting — retained conditionally in Z

Training measurements estimate detector firing probabilities and pairwise
coincidences. Their covariance can reveal graph edges whose true error rates
differ from the calibration-only prior. The implementation estimates edge
probabilities, clips unstable ratios and rescales the existing DEM mechanisms.
It retains separated components as parts of a joint fault rather than silently
discarding hyperedges. The underlying correlation-estimation approach is
described in the [2021 cyclic-QEC study](https://www.nature.com/articles/s41586-021-03588-y).

Fitting can chase finite-sample fluctuations. `choose_dem` fits on half the
training data, compares native/fitted matching on the other half, and refits
all training data only if fitting wins. The retained Z prior is fitted at R1/R2
and native at R3. Fresh confirmation is never used to choose those weights.

### Coherence-aware prior selection — evaluated alternative

Other paths use T1/T2 and an assumed or selected measurement duration to model
data-idle exposure. A validation search chooses a readout-time hypothesis.
This can improve a prior when idling is omitted, but the selected time is a
decoder-model parameter, not a measurement of the physical pulse duration.
It did not replace the native retained pipeline.

Implementation: [`native_noise`](../ethan/src/iqm_surface_code/hardware_optimization.py),
[`dem_fit.py`](../darawork/decoders/dem_fit.py),
[`finetune_hw.py`](../iyanu/finetune_hw.py).

## 4. Classical decoders

### Correlated minimum-weight perfect matching — retained for X

With graphlike independent faults of probability `p_e`, a natural edge cost is
`log((1-p_e)/p_e)`. Matching finds a low-cost explanation connecting detection
events to each other or a boundary, and maps that explanation to logical flips.
Correlation-aware construction/decoding also uses relationships between parts
of decomposed faults. Both switches are enabled in this project.

Matching is efficient and has an explicit calibrated structure. It does not
generally sum every fault configuration in a logical equivalence class. A
most-likely configuration and a most-likely logical class need not agree.
[PyMatching's documentation](https://pymatching.readthedocs.io/en/stable/)
and [Sparse Blossom](https://arxiv.org/abs/2303.15933) describe the algorithm;
their throughput benchmarks are not this project's hardware error rates.

### Exact modeled logical ML through XOR/FWHT — component of retained Z

Each DEM fault contributes an XOR mask of detector and logical bits. Independent
faults combine by XOR convolution. A Walsh-Hadamard transform diagonalizes that
convolution, allowing the implementation to calculate the full joint
distribution `P(s,L)` and sum every modeled explanation for each logical class.
The modeled decision compares `P(s,L=0)` and `P(s,L=1)`.

Separators inside one error instruction preserve one joint Bernoulli event;
repeated detector/observable targets cancel by XOR. Stim's permitted
disjoint-channel approximation remains explicit. “Exact” means exact for that
DEM, not for the unknown physical device distribution.

The decision table has `2^n` entries for `n` detector bits: R1/R2/R3 use
8/16/24 bits, giving 256 B / 64 KiB / 16 MiB of uncompressed byte decisions.
Building the largest joint distribution has an estimated 512 MiB transform
temporary peak. This is exponential preprocessing, not a scalable claim for
arbitrary histories. The complete decoder rejects more than 24 detector bits.

### Bayesian posterior plus empirical counts — retained for Z

The model provides a prior probability for each full history. Independent
training counts update it through `(k_s + alpha*q_model)/(n_s + alpha)`.
`alpha=10` acts as a pseudo-count strength, not ten extra physical shots per
syndrome. More common histories rely increasingly on data; rare/unseen
histories keep a model-supported prediction. It balances model mismatch against
the variance of sparse empirical counts.

This mechanism combines Ethan's modeled posterior calculation and Dara's
posterior/data rule. Alpha and the guarded DEM fit are chosen before held-out
confirmation. No observable labels are inputs at inference. The positive Z
comparison, not the mathematical elegance of the combination, justifies keeping it.

### Empirical majority lookup with matching fallback — evaluated alternative

A sparse dictionary memorizes the majority logical flip for observed histories.
Minimum-count thresholds avoid trusting very rare examples; unseen or
insufficiently sampled histories go to a named matching fallback. This can
learn hardware-specific patterns without a complete physical model, but the
history space grows exponentially. It is a different family from the complete
posterior table and was not the selected final policy.

### Perturbed matching ensemble — evaluated alternative

Several matchers use log-normal perturbations of the DEM probabilities, with
one unperturbed member; an odd-member vote gives the prediction. It can hedge
against uncertain edge weights or ambiguous paths. Agreement among correlated
members is not a calibrated confidence interval, and voting is not exact
logical ML. The tested ensemble did not justify replacing the winners.

### Neural decoders and predecoders — evaluated/research

An MLP maps an entire detector vector to an observable prediction. Spatial/time
convolutional models exploit local detector structure. A predecoder predicts
local corrections before a conventional decoder. Simulated pretraining teaches
patterns from the declared model; hardware fine-tuning can compensate for
model mismatch. Missing-detector masks distinguish unmeasured boundaries from
measured zeros. Finite data, simulation mismatch and checkpoint selection can
all limit generalization.

The follow-up residual model uses a bidirectional GRU over measured detector
history, fixed coordinate masks and a correlated-MWPM prediction. It predicts
a correction to that prediction. It is **offline full-history** inference, not
a causal real-time controller. Current-domain replay prevents the larger
archive from overwhelming the new physical profile. Its tested training budget
did not establish a better retained policy.

[AlphaQubit](https://www.nature.com/articles/s41586-024-08148-8) motivates learned
history-dependent inference on other experiments. This repository's GRU is
not an AlphaQubit reproduction and has no hidden leakage or soft-readout inputs.
The general ConvGRU research implementation in `darawork` is also distinct from
the bounded residual GRU used in the follow-up.

### Tensor-network logical ML — research, not a retained hardware winner

A tensor network sums products of fault probabilities subject to syndrome and
logical XOR constraints. An exact contraction can implement modeled logical
ML without enumerating the entire history table. Runtime depends on contraction
width; boundary sweeps, pruning or finite-bond approximations introduce their
own limits. “Exact ML” requires an exact representation and untruncated
contraction. This backend was not a promoted reference in the two headline studies.

### Union-find, belief propagation and OSD — research adapters

Union-find grows/merges defect or erasure clusters to obtain a correction.
Belief propagation exchanges local messages on a parity-check factor graph;
cycles and quantum-code degeneracy can make its answer unreliable. Ordered
statistics decoding (OSD) or union-find postprocessing can resolve residual
inconsistency. The adapters convert DEM faults into detector/observable parity
matrices and return logical predictions. Their presence does not establish a
Garnet improvement, and optional LDPC dependencies are separate from winner replay.

Sources: [`hybrid.py`](../challenge/decoders/hybrid.py),
[`history.py`](../challenge/decoders/history.py),
[`ensemble_decoder.py`](../darawork/decoders/ensemble_decoder.py),
[`lut_decoder.py`](../darawork/decoders/lut_decoder.py),
[`nn_decoder.py`](../darawork/decoders/nn_decoder.py),
[`tn_decoder.py`](../darawork/decoders/tn_decoder.py),
[`neural_gpu.py`](../iyanu/neural_gpu.py),
[`conv_decoders.py`](../iyanu/conv_decoders.py),
[`ldpc_decoders.py`](../iyanu/ldpc_decoders.py).

## 5. Simulation models and logical extensions

These tools explain possible failure mechanisms and test correctness. They do
not turn a simulated result into a measured Garnet headline.

| Model | What it changes and why |
|---|---|
| `uniform` | Gate, preparation, measurement and idle Pauli faults; a simple controlled baseline |
| `biased_z` | Favors Z relative to X/Y at fixed one-qubit/idle error budget; tests sensitivity to bias |
| `readout_heavy` | Increases assignment error to stress temporal/boundary decoding |
| `correlated_zz` | Adds pairwise ZZ mechanisms to test correlated faults |
| `coherent_twirl` | Replaces a Z overrotation by its Pauli-twirled probability; removes coherent phase interference |
| `leakage` | Uses persistent hidden leakage flags, partner corruption, spread and seepage assumptions |
| `leakage_stress` | Increases generation/persistence to expose model mismatch |
| `calibrated_pauli` | Uses per-qubit/per-edge measured calibration where supplied |
| `calibrated_coherence` | Adds duration-aware, Pauli-twirled T1/T2 idle effects |
| `calibrated_leakage` | Combines calibrated/coherence channels with explicit leakage assumptions |

The leakage simulator reduces qutrit-like behavior to a phenomenological
Clifford/Pauli process; flags are never decoder inputs. Seepage is per scheduled
moment, not per round. The separate Aer trajectory/Kraus path retains coherent
rotations and non-unital relaxation for a slower non-Pauli comparison; it is
not a qutrit leakage simulator. See [implementation/model definitions](../IMPLEMENTATION.md).

Transversal logical Pauli strings and H with explicit frame/boundary updates
are supported extensions. Logical simplification can remove cancelling
operations from such workloads. These operations are absent from the retained
identity-memory experiment, so their gate savings do not explain its headline.
Static-patch S/T are unsupported. The separate transversal CNOT helper is a
primitive, not a completed two-block syndrome/decoder pipeline.
See [logical gates](../TRANSVERSAL_GATES.md).

Qrisp remains an optional translation path; Qiskit/native pulse execution is the
measured winner path. Historical bias-tailoring/twirling/axis-scrambling work was
not retained in the current isolated solution. No walking-code leakage-removal
protocol is claimed by this static no-reset experiment.

## 6. Experimental and statistical techniques

Training/validation/confirmation separation prevents a decoder from appearing
better because it learned the test labels. Frozen state hashes make the
selected inference reproducible. Paired failure comparisons exploit agreement
on common shots; different physical arms require independent-sample treatment.
Round/block stratification preserves the acquisition design.

Fixed-shot Wilson intervals describe each failure probability. Multi-round
binomial maximum likelihood estimates the challenge LER, with a profile interval
and lack-of-fit test. Bonferroni adjustment accounts for the declared comparison
family. A positive point estimate alone does not qualify an improvement.
All-shot scoring prevents an apparent gain from discarding hard/leaky records.

Optional failure-count stopping/confidence-sequence machinery exists for
simulation sweeps. It was not the fixed-shot hardware headline protocol. A
Wilson interval cannot be made optional-stopping-safe merely by ending a run
after enough failures. [Results](RESULTS.md) states the actual acquisition and
fitting rules used here.
