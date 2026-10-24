# Merge and integration notes

Prepared on `codex/ethan-surface-code-bundle`, based on upstream `d064196`
(`iyanu: trial GPU neural decoder vs PyMatching, with accuracy and latency benchmark`).

This change adds only **`ethan/`**. Shared tracked files, including `challenge/`,
the root configuration/tests and `iyanu/`, match upstream. Earlier local
surface-code improvements have been copied into this independent package;
their original working files were archived privately before restoring the
shared files to upstream. Full raw results remain locally available.

## Import and decoder boundaries

Install with `python -m pip install -e "./ethan[test,qiskit,hardware]"` from the
repository root, or the equivalent `.[...]` command inside `ethan/`.

```python
from iqm_surface_code import make_stim_circuit
from iqm_surface_code.decoding import make_matcher, decode_syndromes
from iqm_surface_code.decoders.lookup_table import build_lookup_table
```

The package does not register a replacement for `challenge`. Teammates can keep
their current imports while integrating selected helpers deliberately.
Decoder interchange uses binary detection events `(shots, num_detectors)` and
logical predictions `(shots, num_observables)` in the same shot order. Adapter
calls accept `enable_correlations=True`. Observable truth is for scoring only.
Circuit parity/frame identity must match, not just dimensions.

The `benchmarks/` bundle provides common original-order detector arrays for
decoder comparisons without contacting IQM. Keep train/pilot and evaluation
data separate; these archived validation labels have already been inspected.

## Runtime and validation

Core simulation needs Stim, PyMatching, NumPy, SciPy and Matplotlib. Qiskit/Aer
and IQM/PulLA extras are optional; hardware versions are pinned to the tested
SDK interface. The short-history LUT is exponential and explicitly capped.
Imports never submit QPU work or read credentials.

Run the package suite and immutable replay before pushing. The source move
changes the import namespace and file locations; the validated X/Z policy,
physical circuits and noise/decoder algorithms are retained. Synthetic results
remain labeled synthetic, and the bundled hardware fits retain their assumptions.

Packaging checks: 327 package tests pass; all 18 replay cases match their saved
failure counts across 36,000 hardware shots. The wheel builds and generates a
distance-3 circuit from outside the repository with editable loading disabled.
A six-case synthetic smoke run completes 6,000 shots. The installed environment
passes dependency compatibility checks. No hardware was contacted for these
packaging checks.

The unchanged shared suite passes 215 tests; Iyanu's optional PyTorch module is
skipped in this environment. Separate imports of `challenge` and
`iqm_surface_code` resolve to their own folders.

The reverted bias-tailoring/long-round experiments are excluded from executable
code and the advertised release. Their full historical artifacts remain local.

## Push

After reviewing the prepared commit:

```powershell
git push -u origin codex/ethan-surface-code-bundle
```

No force push is needed. If `main` advances again, the folder-only diff keeps
integration independent of edits to teammates' implementations.
