# Native-package validation

From the repository root:

```powershell
python -m pytest -q -c ethan/pyproject.toml ethan/tests
python -m examples.reproduce
```

The suite covers circuits, no-reset boundaries, logical frames, native compiler
equivalence and lookup inference. Reproduction checks the Garnet measurement
comparison. See [validation](../VALIDATION.md) and
[reproduction](../docs/REPRODUCIBILITY.md).
