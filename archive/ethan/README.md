# Ethan's IQM surface-code solution

Self-contained no-reset surface-code pipeline, isolated in `ethan/` for team
integration. The Python package is **`iqm_surface_code`**; it installs alongside
the repository's existing `challenge` package and Iyanu's decoder.

Includes the generator, ten noise models, calibration-aware native placement,
persistent-leakage and coherent-noise simulation, transversal Pauli/H frames,
DD, correlated MWPM, a small-history ML lookup decoder, and audited Garnet
pulse execution. The later twirling/XZZX/scrambling experiments were reverted;
this bundle contains the preceding validated implementation.

## Install and verify

Run from this folder, using a virtual environment and Python 3.11+:

```powershell
cd ethan
python -m pip install -e ".[test,qiskit]"
python -m pytest -q
python tools/replay_benchmarks.py
```

For IQM hardware, install `python -m pip install -e ".[test,qiskit,hardware]"`.
`requirements-validated.txt` records the observed Python 3.12 package versions;
it is a reproducibility reference rather than a complete dependency lockfile.
The wheel exposes only `iqm_surface_code`, with its own distribution name
`ethan-iqm-surface-code`.

Offline smoke experiment:

```powershell
python -m iqm_surface_code.experiment --synthetic-calibration --models uniform --rounds 1 3 5 --shots 1000 --output results/smoke
```

Module imports, tests and benchmark replay are offline. Run commands from
`ethan/` to keep newly generated output in its ignored `results/` directory.

## Validated hardware entry point

Set credentials externally in `IQM_TOKEN`, then explicitly submit:

```python
from iqm_surface_code.run_on_hardware import run_optimized_memory, decode_hardware_results

syndromes = run_optimized_memory(basis="Z", distance=3, rounds=3, shots=1500)
probability, standard_error = decode_hardware_results(syndromes)
```

The policy uses terminal-readout fusion + XY4 + native MWPM for X, and
measurement fences + XY4 + native full-history LUT for Z. Pulse execution is
currently Garnet-only. The LUT supports at most 24 detector bits (three d=3
rounds), one observable; longer histories fail explicitly. Explicit profiles
remain available for controlled experiments.

Credentials stay in local environment variables or ignored files. An existing
token at the repository root can be passed to benchmark CLIs with
`--token-file ../apikey.jom` when running from this folder.

## Measured results and replay

Three-round distance-3 validation, 1,500 fresh shots per arm:

| Basis | Previous XX stack | Validated policy | Relative failure reduction |
|---|---:|---:|---:|
| X | 315/1500 = 21.00% | 198/1500 = 13.20% | 37.1% |
| Z | 262/1500 = 17.47% | 168/1500 = 11.20% | 35.9% |

These X/Z comparisons use separate acquisition batches on one calibration.
Failure probability at one round count is distinct from per-round fitted LER.
The unshifted challenge fit is poor on several datasets; conditional
first-round-offset fits and their limits are documented in
[the Garnet report](docs/INTEGRATED_GARNET.md).

[Bundled evidence](benchmarks/README.md) contains 18 immutable replay cases:
vanilla LUT plus the X/Z validation controls and selected policies, covering
36,000 physical shots. Source/native/decoder circuits, original-order shot
arrays, frozen LUTs, predictions and hashes are included. Replay needs no
credentials or QPU access. Full local acquisition archives remain separate.

## Guide to the folder

- [Merge and integration notes](MERGE_NOTES.md)
- [Pipeline and noise models](IMPLEMENTATION.md)
- [Transversal gates and logical frames](TRANSVERSAL_GATES.md)
- [DD controls](docs/DYNAMICAL_DECOUPLING.md)
- [Vanilla baseline](docs/VANILLA_LUT_BASELINE.md)
- [Validated Garnet pulse/readout fixes](docs/INTEGRATED_GARNET.md)
- [Agent instructions](AGENTS.md) and [decoder coordination](docs/DECODER_AGENT_GUIDE.md)
- [Original challenge brief](reference/README.md)
