# Garnet surface-code evaluation data

[`garnet/manifest.json`](garnet/manifest.json) describes 54,000 measured shots:
18,000 vanilla surface-code shots and 36,000 optimized-pipeline shots. It lists
the raw/source/native records, frozen decoder state, native measurement maps,
calibration, versions, pulse audits and file hashes.

[`garnet/evaluation.json`](garnet/evaluation.json) gives the failure counts,
pointwise Wilson intervals, challenge LER fits and independent-arm comparison.

```powershell
python -m examples.reproduce
```

See [Methods](../docs/METHODS.md), [Evaluation](../docs/EVALUATION.md) and
[Reproduction](../docs/REPRODUCIBILITY.md).
