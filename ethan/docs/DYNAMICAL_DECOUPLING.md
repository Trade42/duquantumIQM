# Dynamical decoupling for idle windows

DD is opt-in. Available sequences are `xx`, `xy4` and `xy8`, plus `off`.
The default target is data qubits, particularly while ancillas are measured.
Hardware DD uses IQM's pulse compiler. Local simulations use an explicit timed
window plan and noisy pulse model; those are different implementations.

## IQM hardware

```python
from iqm_surface_code.run_on_hardware import run_qiskit, prepare_qiskit
from iqm_surface_code.dynamical_decoupling import DDConfig

# Read-only preparation with an existing backend:
prepared = prepare_qiskit(backend, distance=3, rounds=3, calibration=cal, dd='xx')
print(prepared['report']['dd'])

# This explicit call submits a hardware job:
syndromes = run_qiskit(
    distance=3, rounds=3, shots=10000, backend=backend, calibration=cal,
    dd=DDConfig(sequence='xx', target='data'),
)
```

`run_qiskit(dd=...)` creates IQM `CircuitCompilationOptions` with enabled
`DDMode` and a centered `DDStrategy`. It maps selected sparse Stim data indices
through the chosen placement into physical QB names. Existing unrelated
compilation options, including heralding/reset configuration, are preserved.
`dd=None` preserves caller-supplied options; `dd=False` or `dd='off'` explicitly
disables DD and clears an existing strategy. Existing SDK options can still be
passed through `circuit_compilation_options`.

The server determines actual idle durations, pulse timing and feasibility.
No untimed `XX` identity is added to the gate circuit and then called physical
DD. Actual pulse counts are unknown until a server pulse schedule is available.
Unsupported server options/errors propagate rather than silently disabling DD.
No hardware job was submitted during local validation.

The hardware decoder currently retains its original static CZ/H noise weights.
It preserves detector/observable parity structure but does not infer the actual
server-added DD pulse errors or filtering. The hardware report explicitly marks
this decoder-model mismatch. A physically calibrated post-DD DEM requires the
server schedule and measured residual noise; a claimed improvement factor is
not substituted for those measurements.

## Local simulation

```python
from iqm_surface_code.calibration import synthetic_snapshot
from iqm_surface_code.build_emerald_qubit_rotated import rank_placements
from iqm_surface_code.surface_code import make_stim_circuit
from iqm_surface_code.noise_models import NoiseModel
from iqm_surface_code.experiment import simulate_memory
from iqm_surface_code.dynamical_decoupling import DDConfig

cal = synthetic_snapshot()  # Explicit example, not IQM measurements.
ideal = make_stim_circuit(3, 3, gate_basis='cz')
mapping = rank_placements(ideal, calibration=cal)[0].stim_to_physical
model = NoiseModel(calibration=cal, coherence=True, idle=0)
result, physical_circuit, decoder_circuit = simulate_memory(
    3, 3, model=model, qubit_map=mapping, shots=10000,
    dd=DDConfig(sequence='xx', target='data'),
)
print(result['dd'])
```

`make_stim_circuit(..., noise=model, dd='xx')` also accepts DD directly. With
`noise=None`, use a precomputed `DDPlan` for a noiseless DD circuit; a bare
sequence name cannot supply missing physical timings. `plan_dd` and `insert_dd`
are public helpers in `src/iqm_surface_code/dynamical_decoupling.py`.

CLI examples:

```powershell
python -m iqm_surface_code.experiment --synthetic-calibration --models calibrated_coherence --rounds 1 3 5 --shots 10000 --dd xx
python -m iqm_surface_code.experiment --synthetic-calibration --models calibrated_coherence --rounds 1 3 5 --shots 10000 --dd xx --idle-detuning-std 0.0008
python -m iqm_surface_code.benchmark_dd --shots 50000 --output results/dd_demo
```

The experiment CLI also accepts `--dd-target data|all`, `--dd-pulse-ns` and
`--dd-pulse-error`. The Python config adds `min_idle_ratio`, `max_pulse_error`,
leading/trailing-wait controls and explicit `durations_ns` for a non-calibrated
timing fixture. The default pi-pulse duration is the snapshot's `1q` duration;
override it when measured pi pulses have a different duration. Per-pulse error
defaults to the model's one-qubit nonidentity Pauli probability; an explicit
override is a modeling assumption and is recorded.

The local planner:

- Uses the longest operation in each existing TICK moment as its duration.
- Targets qubits idle throughout that moment; never pulses a qubit while it is
  gated, measured or reset. Leading/trailing waits are skipped by default.
- Fits the entire DD sequence inside the existing idle duration, with symmetric
  pulse centers and reserved pulse durations. It adds no TICKs or modeled time.
- Uses minimum window/pulse-duration ratios 5 for XX, 9 for XY4 and 17 for XY8.
  These conservative defaults can differ from IQM's stock DD strategy.
- Can exclude qubits whose modeled pulse error exceeds `max_pulse_error`.
- Records pulse count, protected qubit-time, timing assumptions and circuit hash.
  A physical DD plan is reused when testing different decoder weights; changing
  decoder noise does not silently change the executed pulse schedule.

Tagged Stim DD pulses preserve ideal detector/observable parity, measurement
order, QEC-round count and two-qubit count. The pulse strings are identities up
to global phase. Stim tags do not encode a pulse schedule: the Qiskit converter
rejects a locally tagged DD circuit instead of exporting an untimed identity
that could be canceled by a transpiler. Use the IQM server path for hardware.

## What DD can and cannot suppress in these models

The existing T1/T2 model is Markovian. **Its full-window background decay is
retained with DD.** DD never reduces T1, multiplies T2 by an arbitrary gain,
clears leakage flags or removes calibrated two-qubit/readout faults. Extra
DD pulse errors can make that model perform worse.

An optional extra channel,
`NoiseModel.idle_detuning_std_rad_per_ns`, represents slow Z detuning constant
within an individual idle window, independently sampled between windows.
If its Gaussian standard deviation is sigma, the averaged phase-error channel
is `pZ = [1-exp(-(sigma*t_signed)^2/2)]/2`. The signed time integrates the
toggling sign between pi pulses. Symmetric even-pulse sequences cancel a
constant detuning ideally; short unprotected windows keep their noise.

This is a window-correlated slow-phase stress model, analytically averaged into
a Pauli channel. It is not a model of arbitrarily long temporal correlations,
finite-pulse filter distortions or coherent errors across the whole algorithm.
Pulse errors are an additive independent channel; imperfect-pulse/filter
correlations are not explicitly simulated. Full-window background decay is
retained even during reserved pulse intervals, making pulse-error accounting
conservative when a calibrated pulse infidelity already includes that decay.

The extra slow component is **not inferred from T2**. Appending it to a measured
T2 can double-count noise unless separate Ramsey/echo/noise-spectrum information
supports the split. The default benchmark deliberately adds it as an independent
stress assumption, not a prediction for Emerald. Replacing calibration's
Markovian model with a measured spectral model is separate characterization work.

`aer_simulation.simulate_nonpauli(..., dd='xx', idle_detuning_rad_per_ns=...)`
also supports an exact fixed-detuning/coherent trajectory check. It evolves
thermal channels and Z rotations between pulse centers while applying actual
X/Y operations. It retains instantaneous-pulse/midpoint timing assumptions;
the Gaussian stress field is rejected on this path rather than silently
reinterpreted. Local tests show static-detuning refocusing and unchanged
Markovian Ramsey decay with error-free pulses.

## Saved benchmark

`results/dd_demo/results.json`, `report.md` and `comparison.png` contain a
fixed-patch d=3, three-round comparison. All calibration values are synthetic.
Each of 24 physical cases uses 50,000 shots: **1.2 million shots total**.
Shared DD-off baselines yield 18 on/off comparisons with per-comparison
uncertainty intervals. These are logical-failure probabilities for the full
three-round memory, not fitted per-round LERs.

Representative XX results:

| Model | Basis | DD off | DD on | Interpretation |
|---|---|---:|---:|---|
| Markov-only calibration | X | 7.162% | 7.238% | No resolved gain; extra pulse cost remains |
| Added slow-phase stress | X | 39.554% | 8.164% | 79.4% relative reduction; 95% interval 78.4–80.3% |
| Slow phase plus persistent leakage | X | 40.462% | 10.830% | 73.2% relative reduction; leakage is still present |
| Added slow-phase stress | Z | 5.384% | 5.418% | No resolved gain for this observable |

XX adds 72 pulses for the X experiment and 54 for Z under this schedule. XY4
and XY8 add more pulses and perform worse than XX in these particular models:
their ideal static-Z cancellation is the same while their pulse-error cost is
higher. That is model-specific, not a general claim that XX is the best sequence
under real coherent control errors or other noise spectra. Null and negative
comparisons remain in the report.

## References

- [IQM compilation options](https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html)
  and [DD strategy](https://docs.iqm.tech/iqm-station-control-client/api/iqm.station_control.interface.models.circuit.DDStrategy.html):
  server DD enablement, target qubits, patterns, window ratios and alignment.
- [IQM model source](https://github.com/iqm-finland/iqm-client/blob/main/src/iqm/iqm_client/models.py):
  compilation-options dataclass and DD option definitions. No installed SDK or
  live server was used for the mocked integration tests.
- [Bylander et al., superconducting-qubit DD and noise spectroscopy](https://arxiv.org/abs/1101.4707):
  low-frequency dephasing can be refocused while irreversible relaxation remains.
