# Native surface-code compilation

The `iqm_surface_code` package supplies the Garnet challenge pipeline's circuit
generator, native mapping, pulse-aware optimization and exact logical lookup.

| Component | Role in the submission |
|---|---|
| `surface_code.py` | Rotated-code schedule, no-reset detectors and logical observables |
| `benchmark_garnet.py` | Native graph embedding and calibration-aware placement |
| `hardware_optimization.py` | Readout fences/fusion, native Clifford frame, calibration prior and XY4 pulse audit |
| `extract_syndromes.py` | Measurement order and detector conversion |
| `decoders/lookup_table.py` | Exact joint-DEM XOR convolution and complete logical decision lookup |

Install from the repository root:

```powershell
python -m pip install -e ".[test]" -e ./ethan
python -m examples.reproduce
```

The submission's Z decoder combines this lookup construction with training
counts. Use [`challenge.submission.load_policy`](../challenge/submission.py)
for the complete X/Z inference interface.

Read the [methods](../docs/METHODS.md), [Garnet evaluation](../docs/EVALUATION.md)
and [reproduction guide](../docs/REPRODUCIBILITY.md). Package regressions run with
`python -m pytest -q -c ethan/pyproject.toml ethan/tests` from the repository root.

[Every retained Garnet improvement](../improvements.md) is documented at the repository root.
