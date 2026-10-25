# Offline reproduction

Hardware acquisition is complete. These commands use saved records and frozen
states; they do not submit jobs, read an API key or require an IQM account.
Run them from the **repository root of a Git checkout**. Package installation
may download dependencies; the replay itself runs offline.

## 1. Install the small winner environment

Use Python 3.11 or newer; Python 3.12 was used for the recorded validation.

```powershell
python -m venv .venv
```

Activate that environment using the command for your shell:

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Bash / zsh
source .venv/bin/activate
```

Then install the two local packages:

```powershell
python -m pip install -e ".[test]" -e ./ethan
```

The root distribution provides `challenge`/`darawork`; the isolated distribution
provides `iqm_surface_code`. Both are required by the final policy. Base
dependencies include Stim, NumPy, SciPy and PyMatching. This winner-only path
does not need Qiskit, the IQM SDK, PyTorch, a GPU or credentials.

## 2. Reproduce the retained policies

```powershell
python -m examples.replay_winners
```

Expected essentials:

```text
X: mwpm; failures R1/R2/R3=[159, 404, 728]; shots=[4000, 4000, 4000]
   Challenge R1-3 LER: 5.808% (95% profile CI 5.492-6.136%)
Z: posterior_10; failures R1/R2/R3=[145, 639, 1147]; shots=[8000, 8000, 8000]
   Challenge R1-3 LER: 4.255% (95% profile CI 4.066-4.449%)
Verified 36,000 saved shots. Stationary fits rejected; LERs are descriptive.
```

Add `--json` for structured output. The command verifies the inference seal,
loads the adopted states, decodes both later confirmation blocks and checks
the result against the published report. It recomputes failures from predictions;
it does not merely print a table of stored scores. It writes no artifacts and
blocks network connections for the duration of the replay.

Those 36,000 shots are the retained-policy subset of the later 96,000-shot
study, not its pilots or competing physical arm. Reusing saved shots here
consumes no new hardware budget.

## 3. Use the decoder interface

```python
from challenge.benchmark_rounds import load_adopted

policy, decoder = load_adopted(
    "results/decoders/history/20261025T004600Z", "X", 3
)
logical_flip_predictions = decoder.decode_batch(detection_events)
```

Input is binary `(shots, num_detectors)`; output is binary `(shots, 1)` in the
same shot order. R1/R2/R3 have 8/16/24 detectors. Correct width is necessary but
not sufficient: use the saved circuit's detector/observable definitions and
source/native measurement map. Predictions are flips relative to the ideal
reference. They are not raw logical bits or per-qubit physical corrections.

There is no truth-label argument at inference. If evaluating accuracy, compare
the returned predictions with independent `obs_flips` only after decoding.
Do not apply the virtual Pauli readout correction again to normalized labels.
The loader rejects unqualified bases/rounds rather than extending the Z table
with an undisclosed fallback.

Use this root loader for the final Z posterior hybrid. The isolated package's
older default Z model-only LUT and the generic simulator's matching default
are distinct baselines, not alternative names for the final winner.

## 4. Audit all later-study evidence

The complete audit includes recurrent candidates, so it additionally requires
a compatible PyTorch installation. The recorded versions are in the
[manifest](../results/decoders/history/20261025T004600Z/manifest.json); PyTorch was
CPU-only. Other research libraries such as a tensor-network optimizer are not
needed to load these frozen candidates.

```powershell
python -m challenge.replay_rounds --run results/decoders/history/20261025T004600Z
```

An optional isolated dependency directory can be supplied without changing the
environment:

```powershell
python -m challenge.replay_rounds --run results/decoders/history/20261025T004600Z --runtime PATH_TO_OPTIONAL_PACKAGES
```

The default audit is read-only; omit `--write` when verifying published data.
It checks:

- 82 unique completed jobs and 96,000 requested/returned shots, with no overlap
  between new jobs and archived training jobs;
- final freeze before confirmation, and completion of the first acquisition
  window before the second was submitted;
- every raw shot in source and native frames, including measurement order;
- 138 saved prediction groups and 20 challenge fits;
- six adjusted bootstrap comparisons and the frozen release rule.

The CLI blocks network sockets and does not access credentials. The original
verification is [`verification.json`](../results/decoders/history/20261025T004600Z/verification.json).
The input artifact list is [`frozen_artifacts.json`](../results/decoders/history/20261025T004600Z/frozen_artifacts.json).

An older, distinct evidence bundle is also available:

```powershell
python ethan/tools/replay_benchmarks.py
```

It verifies **18 cases and 36,000 earlier hardware shots** in `ethan/benchmarks`.
Those are not the 36,000 later winner-validation shots used by the new example.

## 5. Run tests

The small offline CI surface requires only the base environment:

```powershell
python -m pytest -q tests/test_winner_example.py tests/test_rounds_replay.py
```

The workflow in [offline.yml](../.github/workflows/offline.yml) runs this check
and winner replay on Windows and Ubuntu. It supplies no QPU secret and does not
install hardware SDKs. A local pass does not itself assert that a remote CI run
has completed.

For broader regression coverage in an environment with the optional research
dependencies:

```powershell
python -m pytest -q
python -m pytest -q -c ethan/pyproject.toml ethan/tests
```

Optional Qiskit/IQM fixtures and neural/tensor libraries affect which research
tests can run. Use the recorded hardware-stack versions if reproducing those
fixtures; no active account is needed for fake-backend tests. Do not install
new global packages or change dependency pins just to exercise an individual
backend. A separate environment or `--runtime` directory keeps the core path small.

## Provenance and read-only artifacts

The selected states originate in
[`hybrid/20261024T234900Z`](../results/decoders/hybrid/20261024T234900Z/).
The later retained-policy decision and independent raw validation are in
[`history/20261025T004600Z`](../results/decoders/history/20261025T004600Z/).
The two studies use one recorded calibration but different acquired shots.

| Artifact | Meaning |
|---|---|
| `*.source.stim` / `*.native.stim` | Ideal annotated circuits and their actual native Clifford frame |
| `*.decoder.stim` / `model.dem` | Declared noisy prior and its detector error model |
| `*.shots.npz` | Original-order raw/source-native measurements, detectors and observable flips |
| `metadata.json`, `table.npz`, `gru_*.npz` | Frozen decoder configuration/state; NPZ loading disables pickle |
| `manifest.json` | Case IDs, shot ledger, calibration, maps, versions, timing and pulse audits |
| `frozen_selection.json` / `frozen_artifacts.json` | Selected rule and inference-file hashes |
| `report.json` / `results.json` | Counts, estimates, intervals, comparisons and diagnostics |

Repository attributes preserve the recorded checkout bytes. Two recorded Dara
dependencies use CRLF; the isolated package uses LF. Prefer a Git checkout:
downloaded source archives do not necessarily apply checkout filters. A hash
failure after formatting or newline conversion is a reproducibility failure to
resolve, not a reason to disable verification. Keep the published commit and
artifacts together; a future semantic source change needs a new qualification.

Do not regenerate reports into published run directories. The recommended
replays do not retrain, rewrite weights, overwrite raw measurements or retrieve
jobs. Historical `--submit`/`--analyze` commands remain in completed protocol
documents for provenance, not as the current getting-started path.
