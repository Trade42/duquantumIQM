# Garnet calibration: the pre-training package

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

`durations.json` has `units` ("ns"), the calibration set and its fetch time, then `prx_ns` and
`measure_ns` per qubit (`QB<n>`), `cz_ns` per coupler (`QB<a>-QB<b>`), and `reset_wait_ns`: the
passive-reset wait between shots (298,000 ns = 298 µs, `gates.reset_wait` in the snapshot), as
Garnet has no active reset.

**Pre-training.** Build synthetic shots from this package **with idle terms**: the T1/T2 decay over
each readout window (the slowest ancilla's `measure_ns` in that layer), plus any `delay` (E4).
`dqiqm.experiments.device.calibrated_noise(cal, ..., idle_ns=<readout window + delay>)` is that
model in Stim. Don't pre-train on `InfoRecord/sim/`: the stand-in has no idle noise and about twice
the benchmarked gate noise. E1 has two placements here, pA and pB: choose by each shot's `patch`.

**Device.** This package is Garnet's: `InfoRecord/` is Garnet's record. Emerald's record will be
`InfoRecord/emerald/`, with the identical layout inside it.

Regenerate offline from the committed snapshot: `uv run python scripts/garnet_calibration_package.py`.
Its template (`dqiqm.inforecord.write_calibration_package`) doesn't hold the three paragraphs above
yet, so regenerating drops them until it does.
Every later snapshot (E6, before and after each batch) gets its own package next to it.
