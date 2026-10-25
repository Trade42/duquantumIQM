# Immutable Garnet headline evidence

60 study jobs, 300,000 shots, calibration
`a18a947f-0408-4628-81ea-1715637e434f`. Original-order raw measurements and
detector/observable arrays are retained for all 24 pilot and 36 confirmation
jobs. Every acquired shot is retained. The separate `-policy-X` and `-policy-Z`
directories each contain a two-shot public-entry-point smoke test; those four
shots do not enter headline fitting.

`frozen_selection.json` contains final choices and the protocol frozen before
confirmation. `classical_selection.json`, `neural_selection.json` and
`posterior_selection.json` preserve intermediate pilot choices. The final
candidate is X fitted ML and Z fitted ML posterior plus training counts
(`alpha=10`), both on Ethan's unchanged physical path. The adjusted promotion
test retains native MWPM for X and adopts the hybrid only for Z.

`frozen_artifacts.json` hashes every selected decoder state and model. These
exact inference files are included. Unselected trial model binaries remain in
the full local acquisition archive; their validation counts, budgets and
metadata are retained in `manifest.json` and the archived selections.

Source, actual native Clifford and prior circuits are included for every
study job. Published raw NPZ files preserve Stim measurement order and actual
acquisition order. Source/native measurement permutations are in the manifest.
All raw-record converters, predictions, failure counts and state hashes were
checked again after acquisition. Credentials, private job objects and the
isolated optional-dependency installation are excluded. Full local archives
also retain QPY circuits, serialized pulse playlists and original counts.

Offline reproduction from the repository root, with both root and `ethan/`
packages installed:

```powershell
python -m challenge.benchmark_headline --run results/decoders/hybrid/20261024T234900Z --analyze --phase confirmation
```

Replay uses saved inference state only: no training, credentials, network or
new hardware jobs. The final verification additionally blocked network sockets
and obtained identical counts, fits, comparisons and adopted policy.

The primary estimator is the IQM challenge equation with amplitude fixed to
one over rounds 1, 2, 3. **The stationary model is rejected**; its estimates are
descriptive for this window. Raw failure probabilities and lack-of-fit
diagnostics remain visible. No R3 suppression, distance scaling or universal
constant per-cycle error rate is inferred from the fitted number.

Raw data and frozen states are immutable. Replaying a published test set does
not make it fresh held-out evidence for a newly tuned decoder. New selection
requires new confirmation shots.
