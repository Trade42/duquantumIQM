# References and provenance

General literature explains why the techniques are plausible. All claims about
**this project's Garnet performance** come from its own fixed-shot records and
declared comparisons. Performance on another processor, or an SDK class being
available, is not evidence of a Garnet improvement or calibration.

## Primary scientific references

| Source | Used here for |
|---|---|
| Fowler, Mariantoni, Martinis and Cleland, [Surface codes: Towards practical large-scale quantum computation](https://arxiv.org/abs/1208.0928), Phys. Rev. A 86, 032324 (2012) | Stabilizers, logical observables, distance and circuit-level surface-code context |
| Gidney, [Stim: a fast stabilizer circuit simulator](https://arxiv.org/abs/2103.02202), Quantum 5, 497 (2021) | Efficient Clifford circuit representation/sampling used by construction and validation |
| Higgott and Gidney, [Sparse Blossom: correcting a million errors per core second with minimum-weight matching](https://arxiv.org/abs/2303.15933), Quantum 9, 1600 (2025) | Matching-decoder algorithmic background; its benchmark rates are not imported into our result |
| Google Quantum AI, [Exponential suppression of bit or phase errors with cyclic error correction](https://www.nature.com/articles/s41586-021-03588-y), Nature 595, 383–387 (2021) | Source identified by the challenge for its LER fitting convention; detector-correlation analysis background |
| Ezzell, Pokharel, Tewala, Quiroz and Lidar, [Dynamical decoupling for superconducting qubits: A performance survey](https://doi.org/10.1103/PhysRevApplied.20.064027), Phys. Rev. Applied 20, 064027 (2023) | Pulse-sequence/timing tradeoffs motivating DD trials; no transfer of another device's gains |
| Bausch et al., [Learning high-accuracy error decoding for quantum processors](https://www.nature.com/articles/s41586-024-08148-8), Nature 635, 834–840 (2024) | Motivation for learned history-dependent decoding; our residual GRU is not an AlphaQubit reproduction |

## Software and protocol references

- [Stim detector-error-model format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md): detector/observable targets and separator semantics.
- [PyMatching documentation](https://pymatching.readthedocs.io/en/stable/) and [correlation-enabled interface implementation](https://github.com/oscarhiggott/PyMatching/blob/master/src/pymatching/matching.py): graph construction/decoding behavior.
- [IQM PulLA dynamical-decoupling example](https://docs.iqm.tech/iqm-pulla/Example%20-%20Simple%20Dynamical%20Decoupling.html): insertion inside eligible pulse waits.
- [IQM measurement implementations](https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.measure.html): API-level measurement variants. The saved architecture/calibration, not the online class list, establishes availability in this experiment.
- [Original IQM challenge brief](../challenge/README.md): pipeline, memory task and requested statistical endpoint.

Online SDK documentation can advance beyond the acquired environment. The
artifact manifests record the actual software versions, including IQM client
35.0.3 and PulLA 14.0.3 for the studied pipeline. Reproduction uses the frozen
code/states and saved calibration rather than assuming the current online SDK
defaults match the historical device.

## Implementation provenance

The retained physical path and exact XOR/FWHT modeled posterior come from the
isolated [`iqm_surface_code`](../ethan/) implementation. The guarded DEM fitting,
matching adapters and posterior/data rule draw on [`darawork`](../darawork/).
[`iyanu`](../iyanu/) supplies independent neural, coherence-prior and LDPC
research paths. Shared adapters and explicit runners perform the controlled
selection and frozen confirmation. Original contributor directories remain in
place so that integration does not erase those sources.

The current Z posterior is a specific integration: native physical compilation
plus exact modeled posterior, guarded training-only prior fitting, and
`alpha=10` empirical shrinkage. The current X path retains native correlated
MWPM. Algorithm code is linked from [Winning pipeline](WINNING_PIPELINE.md)
and [Techniques](TECHNIQUES.md); raw results and limitations are in
[Results](RESULTS.md) and [Error analysis](ERROR_ANALYSIS.md).

Historical general-implementation and logical-gate notes contain additional
paper comparisons. They describe separate capabilities and are not used to
claim that the retained static memory implements walking-code leakage removal,
all logical gates, or the performance of another code family.
