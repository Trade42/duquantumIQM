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

## Emerald

[`emerald/`](emerald/README.md) contains the **18,000-shot vanilla surface-code
baseline**: X/Z memory at rounds 1–3, fixed native layout, DD off, uniform-prior
lookup. [Methods and measurements](../docs/EMERALD_BASELINE.md).

```powershell
python -m examples.replay_emerald
```
