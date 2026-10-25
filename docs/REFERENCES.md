# References

| Reference | Technique |
|---|---|
| Fowler et al., [Surface codes: Towards practical large-scale quantum computation](https://arxiv.org/abs/1208.0928), Phys. Rev. A 86, 032324 (2012) | Stabilizers, logical operators, distance and syndrome extraction |
| Gidney, [Stim: a fast stabilizer circuit simulator](https://arxiv.org/abs/2103.02202), Quantum 5, 497 (2021) | Annotated Clifford circuits, detector conversion and simulation |
| Higgott and Gidney, [Sparse Blossom](https://arxiv.org/abs/2303.15933), Quantum 9, 1600 (2025) | Efficient minimum-weight matching |
| Google Quantum AI, [Exponential suppression of bit or phase errors with cyclic error correction](https://www.nature.com/articles/s41586-021-03588-y), Nature 595, 383–387 (2021) | Challenge LER convention and detector-correlation analysis |
| Ezzell et al., [Dynamical decoupling for superconducting qubits: A performance survey](https://doi.org/10.1103/PhysRevApplied.20.064027), Phys. Rev. Applied 20, 064027 (2023) | DD sequence and timing mechanisms |

Software documentation:

- [Stim DEM format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md): fault masks, observables and correlated separators.
- [PyMatching](https://pymatching.readthedocs.io/en/stable/): graph construction and correlation-aware decoding.
- [IQM dynamical decoupling](https://docs.iqm.tech/iqm-pulla/Example%20-%20Simple%20Dynamical%20Decoupling.html): insertion in eligible pulse waits.
- [IQM measurement implementations](https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.gates.measure.html): native readout compilation.
- [IQM challenge brief](../challenge/README.md): memory pipeline and evaluation task.

The [methods](METHODS.md) link each technique to its implementation. The
[measurement manifest](../benchmarks/garnet/manifest.json) records the software
versions used for acquisition.
