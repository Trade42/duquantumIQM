# Real devices: the calibration build, its extensions, and the InfoRecord loader

`mlDecoder/hardware.py` turns the mini PC's InfoRecord into decoder inputs. It only reads `challenge/`.

## Calibration packages

`python -m mlDecoder.hardware vendor --device garnet` copies `InfoRecord/calibration/garnet/` (`qubits.csv`,
`couplers.csv`, `durations.json`, `placement.json`) into `mlDecoder/calibration/garnet/<calset8>/`, byte for byte. It also
extracts from the snapshot JSON the per-qubit figures the noise build reads into `metrics.json`: the
`measure_fidelity` readout errors, init errors, and repeated-readout leakage, seepage and QND-ness. `SOURCE.json` holds
the sha256 of every source file and every vendored file. A spec therefore rebuilds from the repo alone, and a changed
file is refused.

A device calibration is named in `DatasetSpec.calibration` by a canonical string that carries every choice, e.g.

```
inforecord:garnet:a18a947f:pA:t2=ramsey:window=1096:terminal=measure_fidelity:leak=none:readout=asym:qnd=excess
```

`hardware.snapshot` builds the team's `CalibrationSnapshot` for that patch:

| Team field | From the package |
|---|---|
| `single_qubit_infidelity` | `error_1q` (RB) |
| edge infidelity | `cz_error_irb` (interleaved RB); the team's model converts average infidelity with (D+1)/D |
| `readout_error` | mean of P(1\|0) and P(0\|1) of `measure` (the team's readout is symmetric) |
| `reset_error` | heralded init error (passive reset between shots); missing for QB8, QB19 → the measure_fidelity init error, else the device median (listed in `metadata["imputed"]`) |
| `t2_ns` | `t2=ramsey` (t2_time, the default: no dynamical decoupling on the hardware jobs), `echo`, or `mid` (geometric mean), capped at 2 T1 |
| `leakage_per_gate` | 0, or with `leak=readout` each ancilla's per-readout leakage spread over its CZs per round, 1 − (1 − L)^(1/w); data qubits 0 |
| durations | 1q = longest prx, 2q = longest CZ, **measure = the readout window** (1,096 ns on Garnet's pA/pB), reset = 1 ns |

Two durations need a note:
- **measure.** The team's model idles every unmeasured qubit for a measurement moment's duration. So the readout window, not the probe pulse (`measure_ns`), is the right value.
- **reset.** On hardware the passive reset happens between shots and takes no circuit time. But the generator splits an X memory's initial RX (data) and R (ancillas) into two moments, so a long reset duration would idle half the qubits.

Qubits are mapped to QB names **by coordinates** (`hardware.qubit_map`): every circuit qubit must match the patch's
placement, injectively.

## The readout-window extension (`readout_window`), after the team's `add_noise`

The hardware compiles each round's readout as a window of about 1.1 µs: acquisition delay, integration, and 76 ns
dead time, 1,096 ns on Garnet's pilot patch. The terminal layer (the last round's ancillas and the data qubits) is
one simultaneous `measure_fidelity` readout. The team's model differs in two ways: it idles only the qubits not
measured in a moment, and it runs the last ancilla readout and the data readout as two moments. The extension:

- **mid-circuit readouts:** the measured ancillas also idle through the window. That is the team's T1/T2 Pauli twirl,
  `model.coherence_channel`, placed after the measurement, so it flips the post-measurement state that the next round
  inherits. Unmeasured qubits already idle for the window, since measure equals the window;
- **terminal layer:** no idle for anyone, and every readout takes the `measure_fidelity` error (the mean of its two
  directions);
- `calibrated_pauli` (no coherence) gets only the terminal readout errors. The ideal circuit is unchanged (checked).

Two options, both from the calibration's own readout figures:

- **`readout=asym`.** Each readout flips with P(1\|0) or P(0\|1) according to the value read. Stim has no
  state-dependent channel, so the sampler draws the record without readout flips (`strip_readout_flips`) and flips it
  afterwards (`apply_readout_flips`, seeded); detection events then come from the ideal circuit's m2d converter. The
  decoder prior keeps the symmetric mean.
- **`qnd=excess`.** After each mid-circuit readout, the ancilla flips (X) with probability 1 − qndness (rrc `measure`),
  less the X/Y flip the window idle already gives. Without reset, a state flip appears as a lag-1 pair of detection
  events.

### Against the real pilot (E1 Z R=3 pA, 1,000 shots, from dq-orch-2)

These pilot statistics were used **only to choose the pre-training noise mixture**, never for training or evaluation.

| | t0 | t1 | t2 | final | all | lag-1 p_ij | corr MWPM P_L |
|---|---:|---:|---:|---:|---:|---:|---:|
| HARDWARE pilot | .079 | .267 | .284 | .146 | .221 | .063 | .209 |
| calibrated_coherence, Ramsey | .079 | .201 | .222 | .145 | .178 | .033 | .095 |
| + asym readout + QND flips | .067 | .199 | .232 | .136 | .178 | .040 | .094 |
| calibrated_leakage p=0.003, leak=readout, asym, QND | .084 | .230 | .280 | .194 | .216 | .061 | .168 |

The simulated rows are SIMULATED, 40,000 shots each. Lag-1 is the mean p_ij over same-stabilizer pairs one round
apart, the same definition as InfoRecord's `summary.json`; it reproduces the stand-in's 0.0179. The device is noisier
in the bulk rounds than any calibrated build, with stronger lag-1 correlation and a quieter final layer. So the
pre-training mixture leans on the leakage-heavy variants, and fine-tuning on real shots does the rest.

## The InfoRecord loader

`hardware.load_run(run_dir)` checks:
- the format;
- every array's sha256, shape and bit-packing;
- `shots.csv` against the manifest (shot order, patch, basis, rounds, barriers, split values);
- `detectors.csv` against the circuit's detector coordinates;
- the run's `.stim` against `stim_sha256`;
- that `det_events`/`obs_flips` are the m2d conversion of `raw_meas`.

`data.inforecord_dataset(spec, run, split)` also checks the run's circuit against the team's for the spec
(measurement order, detector and observable definitions as measured-qubit records), and the device, patch and
placement. It returns one split, in shot order, with the raw record. All 45 E1 circuits in InfoRecord match the
team's generator exactly.

Splits: `finetune` trains; `validation` (20% of the fine-tuning batches) only early-stops; `test` (batch 3, whole)
only scores. Runs are filtered on `barriers` to the batches' setting.

## Limits

- The window is Garnet's measured pilot value. Emerald's is not measured yet.
- Readout-induced leakage enters through the team's gate-based leakage sampler, spread over an ancilla's CZs. Seepage
  stays the team's global per-moment value.
- `qnd=excess` reads IQM's QND-ness figure as a state-flip probability per readout. E3's repeated readouts in batch 1
  will measure the state-flip rate directly.
