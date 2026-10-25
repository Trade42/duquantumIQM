# Reproduce the Garnet surface-code comparison

Use a Git checkout and Python 3.11+. The evaluation reads saved measurement
records and frozen inference state; it runs offline with the base packages.

## Install

```powershell
python -m venv .venv
```

Activate in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in Bash/zsh:

```bash
source .venv/bin/activate
```

Install both local distributions from the repository root:

```powershell
python -m pip install -e ".[test]" -e ./ethan
python -m examples.reproduce
```

The packages supply Stim, NumPy, SciPy, PyMatching, Matplotlib and
`iqm_surface_code`. Add `--json` to obtain structured measurements. Expected
console output includes:

```text
X: vanilla 18.517% -> optimized 5.808% LER (68.6% reduction)
Z: vanilla 6.722% -> optimized 4.255% LER (36.7% reduction)
Verified 54,000 recorded shots, circuit frames, decoder states and comparison intervals.
```

## Evaluation bundle

The separate [routing certificate](GARNET_ROUTING_PROOF.md) verifies the
seventeen-qubit placement and zero routing overhead from the saved native
circuits without decoding measurements:

```powershell
python -m challenge.audit_garnet_routing
```

[`benchmarks/garnet/`](../benchmarks/garnet/) contains eighteen physical cases:
six vanilla cases and twelve optimized cases across two acquisition blocks.
Their total is 54,000 shots. The [manifest](../benchmarks/garnet/manifest.json)
specifies calibration, placement, versions, pulse audits and SHA-256 hashes.

| File | Contents |
|---|---|
| `*.source.stim` / vanilla `*.stim` | Annotated source circuit with detector/observable definitions |
| `*.native.stim` | Executed native Clifford frame |
| `*.decoder.stim` / `model.dem` | Declared noisy prior and detector error model |
| `*.shots.npz` | Original-order measurements, detector events and observable flips |
| `table.npz` / `*.lut.npz` | Complete binary lookup decisions |
| `metadata.json` | Decoder family, training configuration and model information |
| `evaluation.json` | Reproducible counts, intervals, fits and comparisons |

Reproduction verifies artifact and retained-runtime source hashes, reconstructs the
source/native detector frames, checks native measurement-column order, reloads
the inference state and scores every shot. It then refits the challenge equation
and recomputes the independent bootstrap comparison. The example disables
network connections for the calculation.

## Inference interface

```python
from challenge.submission import load_policy

decoder = load_policy("X", rounds=3)
predictions = decoder.decode_batch(detection_events)
```

X/Z memory at R1/R2/R3 has 8/16/24 detector bits. Input is a binary
`(shots, num_detectors)` array and output is binary `(shots, 1)` in the same
order. The detector definitions, source/native mapping and logical reference
must match the bundled circuit. Truth labels enter the scorer after inference.
Use `arm="vanilla"` to load the reference lookup.

The canonical API is [`challenge/submission.py`](../challenge/submission.py).
The [methods](METHODS.md) give the physical and inference configuration.

## Verify changes

```powershell
python -m pytest -q tests/test_submission.py tests/test_rounds_replay.py
git diff --check
```

[CI](../.github/workflows/offline.yml) reproduces the comparison and runs these
checks on Windows and Ubuntu with Python 3.12. All retained regression
suites can be run with:

```powershell
python -m pytest -q
python -m pytest -q -c ethan/pyproject.toml ethan/tests
```

Recorded acquisition versions include Stim 1.16.0, PyMatching 2.4.0, Qiskit
2.1.2, IQM client 35.0.3 and PulLA 14.0.3. Reproduction of the measurement
comparison uses the saved circuits and inference state with the base environment.
Repository attributes preserve the byte representation used by the hashes.

## Retained runtime source integrity

The original published Garnet bundle remains byte-for-byte unchanged. Its
`source_dependencies` field records acquisition-time provenance, including
source paths retired during cleanup. Current source validation uses
[`challenge/runtime_sources.json`](../challenge/runtime_sources.json), which
pins the original measurement manifest and every retained implementation
dependency. No historical source checksum is presented as a checksum of the
cleaned runtime. The unchanged saved records, decoder states and measured
scores are verified by offline replay.

Exact bootstrap-interval reproduction uses the recorded NumPy 2.5.4 runtime.
Other NumPy versions can produce different binomial draws with the same seed,
while preserving all failure counts and point estimates.
