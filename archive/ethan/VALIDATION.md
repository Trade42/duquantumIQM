# Validation — 24 October 2026

## Isolated package and preserved hardware evidence

The relocated `iqm_surface_code` suite passes **327 tests**. All 26 original
implementation modules and the copied tests retain their algorithms; changes
are limited to the package namespace and the challenge-reference location.
Two subpackage initializers make decoder/calibration modules explicit wheel
contents. Shared source and Iyanu's decoder are preserved independently.

Offline replay verifies **18 cases / 36,000 previously acquired Garnet shots**,
including file hashes, source/native measurement-to-detector conversion,
frozen decoders, predictions and exact failure counts. Stim evidence is stored
without Git line-ending conversion so the original hashes survive cloning.
This is reproducibility validation, not a new hardware experiment.

The same replay passes from a snapshot of the exact staged Git blobs, using
only the staged source package and installed dependencies. The staged-content
scan, including decompressed NPZ members, finds no local IQM credential or
private host path. Challenge reference copies normalize whitespace only.

The independent wheel builds and imports from outside the repository with
editable/source-tree loading disabled. It contains 28 Python modules under
`iqm_surface_code` plus distribution metadata. A synthetic smoke run completes
six X/Z uniform-noise cases at rounds 1/3/5, totaling **6,000 simulated shots**.
No QPU jobs were submitted while packaging.

After restoring shared files, the root suite passes **215 tests**. Iyanu's neural
decoder module is skipped because its optional PyTorch dependency is absent.
Both `challenge` and `iqm_surface_code` import concurrently from their respective
folders. The 92-package validation environment passes dependency compatibility
checks.

Current hardware results and limitations are in
[the integrated Garnet report](docs/INTEGRATED_GARNET.md); commands and dependency
versions are in [README.md](README.md) and `requirements-validated.txt`.

## Earlier development checks

The entries below record earlier milestones. Their test counts and statements
about which hardware jobs had been submitted describe those milestones only.
Historical `results/...` paths refer to the original local repository archive;
the portable subset is now under `benchmarks/`.

DD extension: **234 tests passed**. Focused checks cover XX/XY4/XY8 identity,
idle-window fitting, target selection, unchanged parity/distance, decoder-weight
schedule reuse, IQM options/submission forwarding, static-phase refocusing and
unreduced Markovian decay. **1.2 million DD benchmark shots** compare 24 physical
cases using synthetic calibration. See [DD results and assumptions](docs/DYNAMICAL_DECOUPLING.md).
No IQM DD job was submitted; server integration is mocked locally.

Later transversal-gate extension: **215 tests passed**, including Pauli/H
logical semantics, code-frame changes, scheduled corrections, circuit distance,
Qiskit execution and a regression test for SPAM fits near the amplitude boundary.
The benchmark completed **5,000,000 independently simulated shots** across
18 before/after comparisons (identical controls reuse their samples) and 20
additional fixed-shot cycle-rate points. See [TRANSVERSAL_GATES.md](TRANSVERSAL_GATES.md)
for operation counts, rate estimates, intervals, literature and limitations.

`python -m pytest -q`: **97 passed**.

Validated deterministic syndrome/observable parities, X/Z memories, reset and
no-reset operation, CX and CZ schedules, and graphlike circuit distance at
d=3, 5 and 7. Tests include end-to-end Qiskit/Aer measurements, calibration
unit/fidelity conversion, exact coherent rotations, non-unital T1 relaxation,
leakage persistence/reset behavior, Pauli-limit agreement with Stim, optional
stopping and zero-failure intervals, SPAM/LER fitting and offline hardware
preparation against an explicit 64-qubit square-grid backend.

Offline d=3, three-round preparation used **17 physical qubits, 72 native CZ
gates, zero routing SWAPs**. The geometric placement search found 128 legal
placements on the supplied 8×8 example grid. These are local compiler checks,
not measurements on Emerald.

Demo command:

```powershell
python -m iqm_surface_code.experiment --rounds 1 3 5 --shots 2000 --batch-size 1000 --synthetic-calibration --models all --output results/iqm_demo
```

Completed **60 experiments / 120,000 shots**: ten models, X/Z bases, three round
counts, no reset. Outputs are in `results/iqm_demo/` (ignored generated data).
`logical_errors.png` shows point confidence intervals and per-round fits;
`results.json` contains raw failures, fits, diagnostic rates and provenance.

Example fitted per-round LERs from this small **synthetic** run:

| Model | X memory | Z memory |
|---|---:|---:|
| Uniform p=0.001 | 0.00134 | 0.00117 |
| Readout stress | 0.00212 | 0.00286 |
| Correlated ZZ | 0.00257 | 0.00224 |
| Persistent leakage | 0.00784 | 0.00883 |
| Leakage stress | 0.03927 | 0.04289 |
| Synthetic calibrated Pauli | 0.00697 | 0.00639 |
| Synthetic calibration + idle coherence | 0.02045 | 0.01364 |
| Synthetic calibration + coherence + leakage | 0.03017 | 0.02346 |

These are smoke-test results under specified assumptions, not hardware LERs,
paper reproductions or evidence of experimentally measured dominant mechanisms.
For this fixture they motivate testing measurement-idle decoherence and leakage
first; replacing assumed rates with measured rates can change that ranking.
The full result file includes uncertainty; 2,000 shots per point is insufficient
for precision claims at the lowest error probabilities.

Local package versions: Stim 1.16.0, PyMatching 2.4.0, Qiskit 2.5.2,
NumPy 2.2.6, SciPy 1.18.0; Python 3.13.14. IQM hardware dependencies use the
organisers' separate Qiskit 2.1.2 / iqm-client 35.0.3 pins and were not installed
or tested against a live account. No actual calibration or QPU job was fetched.
