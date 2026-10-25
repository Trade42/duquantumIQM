# Emerald calibration: the pre-training package

Device **emerald**. Calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`, fetched 2026-10-24T19:13:21.046969Z UTC, read from
`InfoRecord/emerald/calibration/emerald/20261024T191321Z-23e12f6b-c23c-4736-8736-b0142989c0e6.json` (`dqiqm.experiments.device.calibration_from_snapshot`). The snapshot itself,
`20261024T191321Z-23e12f6b-c23c-4736-8736-b0142989c0e6.json`, is a byte copy of the resonance-check task's
`data/calibration/emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261024T191321Z.json`.

| File | What |
|---|---|
| `qubits.csv` | per qubit (54): T1, T2 Ramsey, T2 echo, readout P(1&#124;0) and P(0&#124;1), the 1Q error, readout leakage, durations; times in ns |
| `couplers.csv` | per calibrated coupler (85 of the 90 static ones): CZ error (interleaved RB), CZ error (Clifford RB), CZ duration |
| `durations.json` | prx, CZ and measure durations per qubit or coupler (ns), and `reset_wait_ns`, the passive-reset wait between shots |
| `placement.json` | E1's two patches eA and eB: Stim qubit -> device qubit, its role (data, X_ancilla, Z_ancilla), check and Stim coordinates; the full ranking |

## Where each figure comes from

| Column | Snapshot field |
|---|---|
| `t1_ns` | `characterization.model.<q>.t1_time` |
| `t2_ramsey_ns` | `characterization.model.<q>.t2_time` (QB3 has none: `device.t2_ns` falls back to the echo T2 there) |
| `t2_echo_ns` | `characterization.model.<q>.t2_echo_time` |
| `p1_given_0`, `p0_given_1` | `metrics.ssro.measure.constant.<q>.error_0_to_1` / `error_1_to_0`: the default `measure` circuits use, **not** the heralded `measure_fidelity` figure of `hw.calibration_fidelities` (optimistic) |
| `readout_error` | the mean of the two |
| `error_1q` | 1 − `metrics.rb.prx.drag_crf_sx.<q>.fidelity` |
| `readout_leakage` | `metrics.rrc.measure.constant.<q>.leakage` |
| `prx_ns`, `measure_ns` | `gates.prx.drag_crf_sx.<q>.duration`, `gates.measure.constant.<q>.duration` (seconds in the snapshot) |
| `cz_error_irb` | 1 − `metrics.irb.cz.<impl>.<a>__<b>.fidelity` (either order; Clifford RB if missing) |
| `cz_error_clifford_rb` | 1 − `metrics.rb.clifford.uz_cz.<a>__<b>.fidelity` |
| `cz_ns` | `gates.cz.<impl>.<a>__<b>.duration` |
| `reset_wait_ns` | the largest `gates.reset_wait.reset_wait.<q>.duration`: 400 µs |

**The 5 CZ-less couplers** (QB6–QB7, QB15–QB23, QB24–QB32, QB49–QB50, QB50–QB54) are not in
`couplers.csv`; no planned circuit uses them.

## Pre-training: read this first

- **Pre-train on noise built from this package, with idle terms**: the T1/T2 decay of every qubit
  over the ancilla readout window (the slowest ancilla's `measure_ns`, about 360 ns here) plus any
  delay, on top of the gate and readout errors. The data qubits idle through every ancilla
  readout; without reset, a decayed ancilla also fires detectors two rounds apart.
- **Do not pre-train on `sim/`.** The stand-in there has no idle noise at all and roughly doubles
  the gate errors (Aer adds relaxation on top of RB figures that already include it), so its
  shots are not Emerald's.
- **Choose the placement by each shot's `patch`** (`eA` or `eB`, in `shots.csv` and the manifest's
  `spec.patch`): the two patches are different qubit sets, so a noise model built for one does not
  fit the other.
- The ML split's `validation` is the fine-tuning side's held-out 20%: load it as finetune or
  validation, **never as test**. Only batch 3 is test.

Regenerate offline from the committed snapshot: `uv run python scripts/emerald_plan.py` (it also
rewrites the job plan). Every later snapshot (before and after each batch) gets its own package.
