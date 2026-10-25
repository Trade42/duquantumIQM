# Implementation

The challenge pipeline implements distance-three surface-code memory on native
Garnet connectivity. It combines annotated no-reset syndrome extraction,
measurement-aware native compilation, data-qubit XY4 and basis-specific logical
inference.

| Component | Source |
|---|---|
| Submission inference and evaluation | [`challenge/submission.py`](challenge/submission.py) |
| Surface-code generator | [`surface_code.py`](ethan/src/iqm_surface_code/surface_code.py) |
| Native embedding | [`benchmark_garnet.py`](ethan/src/iqm_surface_code/benchmark_garnet.py) |
| Native optimization, noise and pulse audit | [`hardware_optimization.py`](ethan/src/iqm_surface_code/hardware_optimization.py) |
| Correlated matching | [`ensemble_decoder.py`](darawork/decoders/ensemble_decoder.py) |
| Detector-correlation fitting | [`dem_fit.py`](darawork/decoders/dem_fit.py) |
| Exact full-history lookup | [`lookup_table.py`](ethan/src/iqm_surface_code/decoders/lookup_table.py) |
| Posterior/count adapter | [`hybrid.py`](challenge/decoders/hybrid.py) |

[Methods](docs/METHODS.md) explains the complete implementation and X/Z choices.
[Evaluation](docs/EVALUATION.md) compares it with vanilla surface code, and
[Reproduction](docs/REPRODUCIBILITY.md) describes the data and execution interface.
