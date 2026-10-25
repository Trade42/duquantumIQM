# Residual error pathways

After decoding, three-round failure probabilities are 18.200% for X memory
and 14.3375% for Z memory. The measurement record reflects a combination of
gate, idle, preparation and readout faults. Several mechanisms can produce
the same detector history, so the remaining error budget is described through
its physical and inference pathways.

## Physical channels

| Pathway | Effect on memory | Technique used | Residual contribution |
|---|---|---|---|
| Entangling and rotation faults | Propagation through checks changes data and measured parity | Native placement, cancellation and hook-safe ordering | Multiple faults, coherent control error and gate-context dependence |
| Ancilla assignment error | A reported bit changes temporal detection events | Complete no-reset history and implementation-specific priors | State-asymmetric and history-dependent assignment probabilities |
| Terminal data readout | Data-bit errors change final checks and logical parity | Terminal-readout treatment and joint boundary inference | Ambiguity at the final destructive measurement |
| Measurement disturbance | Ancilla state or neighboring data changes during readout | Measurement-aware scheduling and X backaction proxy | Non-QND dynamics and spectator coupling beyond a Pauli proxy |
| Idle phase accumulation | Data dephases while ancillas are measured | Data-qubit XY4 | Imperfect pulses, timing-dependent filtering and fast noise |
| T1 relaxation | Excited-state population decays during the circuit | Reduced native work and calibrated effective errors | Irreversible population loss |
| Leakage | Population outside the qubit subspace persists or corrupts partners | Inference from measured syndrome correlations | Unobserved leakage state and correlated subsequent faults |
| Crosstalk | Concurrent drives/readout create coupled faults | Native scheduling, joint DEM faults and correlation fitting | Context-dependent interactions and slow temporal correlations |
| Preparation | Initial parity/reference is imperfect | Initial boundary detectors and explicit preparation prior | Uncertainty in the assumed preparation probability |
| Distance-three redundancy | Ambiguous multi-fault histories share the same syndrome | Logical decoding across complete history | Logical classes that binary measurements cannot fully distinguish |

An assignment error changes the reported classical bit. Measurement disturbance
changes the physical post-measurement state and can affect subsequent checks.
This distinction matters when ancillas are reused. The backaction term represents
effective qubit disturbance using repeated-readout metrics; persistent leakage
requires a richer temporal model.

XY4 averages selected unwanted idle evolution. Real pulses introduce control
error, and relaxation transfers population irreversibly. The compiled audit
therefore records both native operations and inserted pulses. At R3, the
optimized schedule contains 144 additional DD pulse instructions and a
301.900 microsecond total schedule, including 298 microseconds of passive
initialization. The active memory exposure is a subinterval of that schedule.

## Prior and training error

The native prior compresses calibration into Pauli probabilities. It contains
gate/readout channels and the selected residual disturbance settings. Timed
idle decoherence, individual DD-pulse errors, coherent interference, non-unital
dynamics and persistent qutrit state are additional physical effects beyond
this prior.

Training-based correlation fitting corrects some differences between calibration
and measured syndromes. Its estimates have finite-sample variance, controlled
by an internal training check. The Z posterior further regularizes history
counts toward the model. At R3 there are 16,777,216 possible histories and
6,000 training records; sparse coverage makes that regularization consequential.

Matching chooses a likely fault explanation. Logical-class inference sums
modeled explanations. Both encounter histories compatible with opposite logical
flips. Their remaining mistakes combine this ambiguity with differences between
the physical process and the prior.

## Statistical interpretation

The R1–3 curves have substantial binomial deviance under a constant per-round
rate. The [evaluation](EVALUATION.md) reports both the challenge fit and the
individual failure probabilities. Profile intervals quantify uncertainty within
the fitted equation. Comparison bootstrap intervals quantify fixed-shot sampling
variation across the measured acquisition blocks. Hardware variation between
windows adds a separate contribution.

The data establishes the complete pipeline's improvement over vanilla memory.
Individual channel fractions require targeted characterization: repeated-readout
experiments for disturbance, leakage-sensitive measurements for population
outside the computational subspace, and timing/context sweeps for idle and
spectator effects.
