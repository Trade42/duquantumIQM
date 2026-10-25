# Emerald vanilla memory baseline

The [vanilla bundle](vanilla/20261025T022556Z/) contains six distance-three
X/Z surface-code cases at rounds 1–3, with **18,000 measured shots** on Emerald.
DD is off, the native embedding is fixed, and a uniform-prior full-history lookup
decodes every shot.

| Memory | Challenge LER | 95% profile interval |
|---|---:|---:|
| X | 12.352% | 11.773–12.953% |
| Z | 4.749% | 4.424–5.089% |

[Baseline method and fit diagnostics](../../docs/EMERALD_BASELINE.md).

```powershell
python -m examples.replay_emerald
```

The [manifest](vanilla/20261025T022556Z/manifest.json) records raw measurements,
source/native circuits, frozen requests, decoder state, calibration, seed,
versions and timing. Reproduction uses saved records without QPU access.
