# Methods: from surface-code memory to logical inference

Our response to the IQM challenge combines physical control and classical
inference. Native compilation and XY4 change the experiment's error exposure;
the decoder uses the measured record to infer its logical outcome. The
evaluation compares this complete pipeline with vanilla surface-code memory.

## 1. Encode and measure a logical memory

A distance-three rotated surface code uses nine data qubits and eight ancillas
to encode one logical qubit. Four X-type and four Z-type checks measure local
parities. Their operators commute with the logical observable, so syndrome
extraction reveals fault information while preserving the stored logical value.

X memory prepares the logical plus state and measures logical X. Logical Z or
Y reverses that observable. Z memory prepares the logical zero state and
measures logical Z; logical X or Y reverses it. Both experiments contain the
full circuit's physical X, Y and Z fault mechanisms.

Each round has four interaction layers. The ordering controls **hook errors**:
an ancilla fault can spread through subsequent entangling gates to several data
qubits. The rotated-code schedule orients this propagation relative to the
logical boundaries to preserve the intended distance. See
[Fowler et al.](https://arxiv.org/abs/1208.0928) for the code construction.

### Reused ancillas and detector equations

The same ancillas are measured repeatedly without mid-circuit reset. If `m_t`
is a measured ancilla bit, its parity includes its incoming state. A stabilizer
value has the form `s_t = m_t XOR m_(t-1)`. Comparing successive stabilizers
therefore produces the bulk detection event

```text
d_t = s_t XOR s_(t-1) = m_t XOR m_(t-2).
```

Initial, second-round and terminal-data checks have their own boundary equations.
We generate these equations in the annotated Stim circuit and use its
measurement-to-detector converter. The final logical label is an observable
flip relative to the ideal circuit reference. The decoder predicts that flip.

Implementation: [`surface_code.py`](../ethan/src/iqm_surface_code/surface_code.py),
[`extract_syndromes.py`](../ethan/src/iqm_surface_code/extract_syndromes.py).

## 2. Place the code on native connectivity

The mapper enumerates legal embeddings of the code's interaction graph into
Garnet's coupling graph. It ranks embeddings using a negative-log gate/readout
fidelity proxy: multiplying favorable operation fidelities becomes an additive
placement cost. The chosen seventeen-qubit embedding supplies every required
coupling directly, with zero routing SWAPs.

The [routing proof](GARNET_ROUTING_PROOF.md) checks the saved native circuits,
including physical edge multiplicities, ordered interactions on every wire and
measurement placement. All twelve optimized cases use exactly `24*rounds` CZ
gates and zero added entanglers. A legal static embedding achieves zero SWAPs;
nonnegative gate counts make this routing SWAP count optimal.

This reduces the need for additional entangling operations and assigns checks
to a calibration-aware physical patch. The mapping is fixed for the optimized
experiment. Stim qubit IDs, dense circuit indices and Garnet `QB` names each
have explicit maps in the measurement bundle.

Implementation: [`benchmark_garnet.py`](../ethan/src/iqm_surface_code/benchmark_garnet.py).

## 3. Compile checks and readout together

The abstract check schedule becomes native CZ gates and single-qubit rotations.
Local cancellation removes adjacent redundant rotations. We reconstruct the
actual native Clifford circuit after optimization and preserve its measurement
column permutation. Converting the source and native records yields the same
detectors and reference-relative logical flips on every recorded shot.

### Measurement fences for Z

Barriers mark boundaries between unitary regions and measurement regions.
Optimization cancels redundant gates within unitary regions while preserving
readout dependencies. Omitting the final trailing barrier makes terminal
readout visible to the IQM compiler's terminal-measurement pass. This allows
the compiled terminal fidelity-readout implementation to supply the returned bits.

Fencing is useful because drive/readout timing and measurement grouping affect
the physical schedule as well as its abstract gate count.

### Terminal-readout fusion for X

X preparation and readout require basis rotations around native Z measurement.
After the last CZ, a local rotation commutes with measurements on other wires.
We move those rotations before the terminal measurement group while preserving
same-wire dependencies. A local uncompute after a wire's final measurement
can be removed when it cannot influence any returned outcome.

The resulting final ancilla/data readout forms one terminal group. Repeated
intermediate ancilla measurements remain in the syndrome circuit. Fusion aligns
the X basis change with the hardware's terminal-readout treatment.

Implementation: [`hardware_optimization.py`](../ethan/src/iqm_surface_code/hardware_optimization.py).

## 4. Protect data-qubit waits with XY4

While ancillas are measured, data qubits wait. Slowly varying unwanted
Hamiltonian terms can accumulate phase during those waits. A pi pulse reverses
the sign of some terms in a toggling frame; alternating axes refocuses their
accumulation while returning the ideal logical action to identity up to phase.

The pulse compiler inserts SDK sequence `YXYX` in eligible data-drive waits,
using ASAP placement. It merges eligible contiguous waits and skips leading
and trailing waits. The nine data qubits are the DD targets. Wait duration is
preserved, and a before/after pulse audit records actual inserted instructions.

| Syndrome rounds | Additional compiled IQPulse instructions |
|---:|---:|
| 1 | 0 |
| 2 | 72 |
| 3 | 144 |

A native rotation can contain several pulse instructions, so these counts
describe the compiled waveform schedule. Refocusing competes with imperfect
pulse control; the selected sequence balances both effects. DD acts on unwanted
idle evolution, while relaxation and assignment error require other parts of
the pipeline. The [IQM DD example](https://docs.iqm.tech/iqm-pulla/Example%20-%20Simple%20Dynamical%20Decoupling.html)
describes insertion, and [Ezzell et al.](https://doi.org/10.1103/PhysRevApplied.20.064027)
analyze sequence and timing tradeoffs.

## 5. Build a detector prior in the native frame

The noisy decoder circuit follows the executed native Clifford schedule.
Gate and readout probabilities come from the recorded calibration and compiled
measurement implementation. For average gate infidelity `r`, the declared
depolarizing approximation uses `p=1.5r` for one-qubit gates and `p=1.25r` for
two-qubit gates. Initial preparation uses an assumed error probability of 0.002.

The X prior also applies a phenomenological ancilla disturbance term after
nonterminal readout. It combines repeated-readout QND/leakage metrics with
`backaction_scale=0.5`. Z uses scale zero. Both use `phase_round=0`.
These values parameterize inference in a qubit Pauli model; pulse execution
is determined separately by the compiled schedule.

For repeated-readout QNDness `Q` and leakage metric `l`, the disturbance
probability is `clip(backaction_scale * (0.5*(1-Q) + 0.25*l), 0, 0.45)`.
It is applied as ancilla depolarization after nonterminal measurement. Readout
assignment probability is `1 - fidelity` for the implementation named by the
compiled measurement trace. Native gate/readout probabilities are bounded
between zero and 0.45 in this prior.

Stim converts the noisy circuit into a detector error model (DEM). An error
instruction specifies a fault probability and the detectors/logical observables
that flip together. The extraction settings permit the disjoint-channel
approximation. The posterior retains separator-marked pieces of an instruction
as one joint fault. [Stim's format specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)
defines these semantics.

### Training-only detector-correlation fitting for Z

Detector firing rates and pairwise coincidences estimate covariance. Correlated
firings reveal fault edges whose probabilities differ from the calibration-only
prior. The fitter clips unstable estimates and rescales existing DEM mechanisms.
This follows the detector-correlation approach described in the
[cyclic-QEC study](https://www.nature.com/articles/s41586-021-03588-y).

For detector means `m_i`, `m_j` and coincidence `c_ij`, define covariance
`C_ij=c_ij-m_i*m_j` and parity expectation `B_ij=1-2*m_i-2*m_j+4*c_ij`.
The graph-edge estimate is

$$
p_{ij}=\frac{1-\sqrt{\operatorname{clip}(1-4C_{ij}/B_{ij},0,1)}}{2}.
$$

A nonpositive parity denominator uses the implementation's zero-edge estimate.
Boundary-edge probabilities account for the remaining single-detector rate
after the fitted adjacent edges. Fitted/model probability ratios are clipped
to `[0.1, 10]`. Each joint DEM mechanism is rescaled by the geometric mean of
its component ratios, with a final probability cap of 0.4. Its correlated
detector/observable targets remain together.

Each round count has 6,000 independent training shots. One half fits weights;
the other compares native and fitted matching predictions. When fitting improves
this internal check, weights are refit on all training shots. This procedure
uses fitted priors for R1/R2 and the native prior for R3. The empirical posterior
update below uses training counts at every round count. Evaluation shots enter
only the final score.

Implementation: [`native_noise`](../ethan/src/iqm_surface_code/hardware_optimization.py),
[`dem_fit.py`](../darawork/decoders/dem_fit.py).

## 6. Decode X with correlated MWPM

An independent graphlike fault with probability `p` receives edge weight
`log((1-p)/p)`. Matching connects detection events to each other or boundaries
with a low-weight fault explanation. The logical parity of that explanation
is the predicted observable flip.

The decoder uses the native calibration prior. Correlation handling is enabled
both when constructing the PyMatching graph and when decoding. This accounts
for relationships between graphlike pieces of decomposed faults. The graph
provides an efficient structured inference rule for the X detector distribution.
See [PyMatching](https://pymatching.readthedocs.io/en/stable/) and
[Sparse Blossom](https://arxiv.org/abs/2303.15933).

## 7. Decode Z with logical-class probabilities and training counts

Several physical fault explanations can have the same detectors but different
logical outcomes. Logical-class maximum likelihood sums all modeled explanations
for each outcome, then compares their total probabilities.

### Exact XOR convolution with a Walsh–Hadamard transform

Each independent DEM fault contributes an XOR mask over detector bits and one
logical bit. Combining faults is XOR convolution. In the Walsh–Hadamard domain,
convolution becomes multiplication: each fault contributes a factor
`(1-p) + p*(-1)^(mask dot frequency)`. The inverse transform gives the complete
joint distribution `P_model(s,L)` for history `s` and logical flip `L`.

The modeled posterior is

$$
q_{\rm model}(s)=\frac{P_{\rm model}(s,1)}{P_{\rm model}(s,0)+P_{\rm model}(s,1)}.
$$

This summation is exact within the specified DEM. The model's approximation to
hardware is described in [Error analysis](ERROR_ANALYSIS.md).

### Bayesian shrinkage toward the modeled posterior

Let `n_s` be the number of training occurrences of history `s`, and `k_s` the
number accompanied by a logical flip. We use

$$
\widehat q(s)=\frac{k_s+10q_{\rm model}(s)}{n_s+10},\qquad
\widehat L(s)=\mathbf{1}[\widehat q(s)>1/2].
$$

Frequent histories can correct model bias through measured counts. Rare
histories retain the prior's regularization; an unseen history receives the
modeled decision. `alpha=10` is a pseudo-count strength. The final decision
table contains every history, and ties choose logical zero. At inference the
only input is the detector vector.

R1/R2/R3 have 8/16/24 detector bits. Their byte decision tables occupy
256 B / 64 KiB / 16 MiB. The largest transform has an estimated 512 MiB
temporary footprint. Preprocessing builds the table; inference packs each
detector history into its table index and returns one binary logical prediction.

Implementation: [`lookup_table.py`](../ethan/src/iqm_surface_code/decoders/lookup_table.py),
[`hybrid.py`](../challenge/decoders/hybrid.py).

## Why X and Z use different policies

The stored observables respond to different logical faults: X memory probes
logical phase preservation, while Z memory probes logical bit preservation.
Native Z readout, X basis rotations, terminal detector boundaries and the
physical assignment of check qubits make their compiled noise distributions
different.

The X schedule benefits from fusing its terminal basis rotations and readout.
Its calibrated correlated graph supplies the inference rule. The Z schedule
uses measurement fences to preserve its readout structure, and its inference
combines logical-class probability sums with training-based correction of the
native prior. Each policy is selected independently using training/validation
shots before scoring the measured evaluation records.

The mechanism is complementary: compilation and XY4 control physical exposure;
calibration and fitted correlations describe likely detector patterns; matching
or the posterior maps those patterns to logical flips. The complete pipeline
is measured against the vanilla implementation in [Evaluation](EVALUATION.md).
