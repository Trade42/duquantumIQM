# Validation

Reproduce the complete vanilla/optimized Garnet comparison:

```powershell
python -m examples.reproduce
python -m pytest -q tests/test_submission.py tests/test_rounds_replay.py
```

The replay checks measurement and decoder hashes, source/native detector and
observable frames, measurement order, all 54,000 shot predictions, challenge
fits and bootstrap intervals. The tests also exercise corrupted-state rejection
and equal-width observable-frame errors.

Run all retained core regressions:

```powershell
python -m pytest -q
```

For the native compiler package:

```powershell
python -m pytest -q -c ethan/pyproject.toml ethan/tests
```

See [Reproduction](docs/REPRODUCIBILITY.md) for installation and
[Evaluation](docs/EVALUATION.md) for the measured comparison.
