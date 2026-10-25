# Garnet calibration (copied into the SIMULATED record: the stand-in's noise is this calibration): the pre-training package

Calibration set `a18a947f-0408-4628-81ea-1715637e434f`, fetched 2026-10-24T19:13:13.190422Z UTC, read from `InfoRecord/calibration/garnet/20261024T191313Z-a18a947f-0408-4628-81ea-1715637e434f.json`
(`dqiqm.experiments.device.calibration_from_snapshot`). This is what the laptop needs to build the
team's `CalibrationSnapshot` and calibrated noise models for pre-training, before any hardware
shot exists:

| File | What |
|---|---|
| `qubits.csv` | per qubit: T1, T2 Ramsey, T2 echo, readout P(1\|0) and P(0\|1) of the default `measure` (not the heralded figure), the 1Q error, readout leakage, durations; times in ns |
| `couplers.csv` | per coupler: CZ error (interleaved RB), CZ error (Clifford RB), CZ duration |
| `durations.json` | prx, CZ and measure durations per qubit or coupler, and the passive-reset wait |
| `placement.json` | per experiment: Stim qubit -> device qubit, its role (data, X_ancilla, Z_ancilla) and check, and the Stim coordinates; E1's two orientations pA and pB, E2's line, E5's block |

Regenerate offline from the committed snapshot: `uv run python scripts/garnet_calibration_package.py`.
Every later snapshot (E6, before and after each batch) gets its own package next to it.
