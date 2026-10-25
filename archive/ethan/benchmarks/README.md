# Replayable Garnet evidence

This bundle retains **18 physical cases, 36,000 shots**, with original-order
measurements/detectors/observable labels, frozen decoder inputs, LUTs and
predictions. `manifest.json` lists every file hash, expected failure count,
decoder family, original run ID, calibration ID and fit convention.

| Study | Included cases | Physical shots |
|---|---:|---:|
| Vanilla surface code + full-history LUT, X/Z rounds 1–3 | 6 | 18,000 |
| Z prior-stack versus validated XY4/LUT, rounds 1–3 | 6 | 9,000 |
| X prior-stack versus validated fused XY4/MWPM, rounds 1–3 | 6 | 9,000 |

```powershell
python tools/replay_benchmarks.py
```

Run from `ethan/` after installing the package. Replay checks file hashes, both
source/native detector frames, all shot counts and exact archived decoder
scores. It needs core dependencies only, and never accesses IQM or a token.
Rebuilding the exponential LUTs is unnecessary: their frozen decision tables
are included. Archived records are evaluation data, not a fresh test set for
training a decoder.

At three rounds, the validated X result is 315→198 failures /1500 and Z is
262→168 /1500. The separate vanilla baseline is 1220/3000 X and 583/3000 Z.
These are logical failure probabilities, not per-round LERs. X/Z validated
comparisons came from separate fresh batches on one calibration, and do not
isolate each compiler/DD/decoder contribution.

The challenge's unshifted fitted rates for vanilla→validated are X
18.517%→4.156% and Z 6.722%→3.835% per round; those fits have substantial
mismatch. First-round-offset fits give X 24.996%→6.052% and Z
9.715%→5.381%, conditional on their stationary binomial model. Earlier and
later acquisition/model changes limit comparisons beyond these datasets.

See [the hardware report](../docs/INTEGRATED_GARNET.md) and
[vanilla baseline](../docs/VANILLA_LUT_BASELINE.md) for uncertainty, negative
studies, decoder scaling and fit limitations. The complete local raw archives
also retain calibration and pulse audits; private service responses and token
files are excluded from this versioned replay bundle.
