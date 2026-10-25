# InfoRecord: everything we collect from IQM Garnet

Format `dqiqm.inforecord/1`, written by `scripts/garnet_dryrun.py` and `dqiqm.inforecord` (the
garnet-design task, 2026-10-24). Sal's request: collect all the data Garnet gives, from circuits
designed to carry as much information as possible, especially temporal correlations, with every
column useful to an ML decoder. The experiments and the hardware job plan are in
`docs/garnet-design.md`.

## Devices

This root, `InfoRecord/`, is IQM Garnet's record. Emerald's will be `InfoRecord/emerald/`, with the
identical layout inside it (another task builds it). Every run's `manifest.json` names its
`device` (in `sim/`, the stand-in's). The circuits' `placement.json` files and the `ler/` records
don't carry a `device` key yet: their device is the root they sit in. Adding the key is on the
backlog, as it regenerates `circuits/` and `sim/`.

## What is real and what is simulated

| Path | What | Status |
|---|---|---|
| `architecture/garnet.json` | the live dynamic and static architecture, grid coordinates | **hardware** (calibration set `a18a947f-0408-4628-81ea-1715637e434f`) |
| `calibration/garnet/` | the live snapshot the circuits and noise were built from (E6 adds one before and after each batch), and its pre-training package: `qubits.csv`, `couplers.csv`, `durations.json` (prx, measure and CZ durations per qubit or coupler, in ns, and `reset_wait_ns`: the passive-reset wait between shots, 298 µs, as there is no active reset), `placement.json` (`scripts/garnet_calibration_package.py`, offline) | **hardware** |
| `circuits/` | the planned hardware circuits (`target="hardware"` only, zero SWAPs), `placement.json`, `job_plan.json` | planned, not yet run |
| `raw/`, `datasets/`, `derived/`, `ler/` | hardware results | **empty until the hardware jobs run** |
| `sim/` | the dry run on the stand-in, in exactly this layout | **SIMULATED** (`"label": "SIMULATED"` in every manifest, `simulated = 1` in every shot) |

**What git holds of `sim/`.** The full dry run is 4,479 files, 37.7 MB. Git holds its
README, `FILES.json`, `checks/`, `ler/`, `calibration/`, `architecture/`, `raw/` and a few representative runs
(`InfoRecord/.gitignore` lists them); the rest stays on disk. Regenerate the whole tree, seeded, in about
9 min under the lock:

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/garnet_dryrun.py --clean
```

It keeps `circuits/job_plan.json`'s `pilot` block (the pilot's sanity bands) while the calibration
set is the same. Otherwise run `scripts/garnet_plan.py` after it, which writes the block.

`sim/FILES.json` (sha256 `4e87b26fc174f70acc3fc3cbf05fce343b13b01fc1241624988445d63d3e69ea`) lists every file the run writes with its size, sha256 and a canonical
sha256 that ignores wall-clock fields; `dqiqm.inforecord.compare_trees` checks a regenerated tree against it.
Hardware data is different: its `raw/` is committed in full, always (it cannot be regenerated).

The stand-in is `IQMFakeGarnet().copy_with_error_profile(...)` with the live calibration's noise
(`dqiqm.experiments.device`): per-qubit T1, Ramsey T2, 1Q and CZ errors, and the asymmetric
readout. It puts no noise on `delay` or on idle qubits, so E4's idles are checked for structure
only in `sim/`.

## Layout

```
architecture/garnet.json
calibration/garnet/<UTC>-<calset>.json
circuits/<experiment>/<name>.stim       the circuit's meaning: detectors, observables, QUBIT_COORDS (noise-free)
circuits/<experiment>/<name>.qasm       exactly what runs (OpenQASM 3 of the transpiled circuit)
circuits/<experiment>/placement.json    per circuit: Stim qubit -> device qubit, roles, checks, gate counts, hashes
circuits/job_plan.json                  the jobs, their circuit order, shots and estimated QPU time
raw/<YYYYMMDD>-<task>/<job id>/         dqiqm.hw.save_raw: meta.json + memory.npz (c<i>: uint8 shots x clbits); never edited
datasets/<experiment>/<run id>/         one circuit in one batch, ML-ready (run id: b<batch>_<circuit>, p0_ for the pilot)
    raw_meas.npy det_events.npy obs_flips.npy shots.csv detectors.csv measurements.csv qubits.csv manifest.json
derived/<experiment>/<run id>/          pij.npy pij_lags.csv rate_by_round.csv rate_by_bucket.csv streaks.csv
                                        syndrome_weight.csv summary.json; E3 also assignment.csv readout_corr.npy readout_flips.csv;
                                        a pilot run also pilot_checks.json (its stop/go checks, timeline and decode)
ler/<experiment>/<run id>/              ler_points.csv ler_fits.json ler.png, one per (basis or kind, patch, batch)
sim/                                    all of the above, SIMULATED; sim/checks/ holds the dry run's own checks
```

## Conventions

- `round` is Stim's detector time coordinate t: 0-based, with the final data-qubit detectors at
  t = R. A measurement's `round` is the same t: round k's readout at t = k − 1, the final data
  readout at t = R.
- `x`, `y` are Stim's detector coordinates, exactly as `circuit.get_detector_coordinates()` gives
  them. The rows of `detectors.csv` and the columns of `det_events.npy` are in the circuit's
  detector order, which for E1 is that of the team's `make_stim_circuit`.
- Column j of `raw_meas.npy` is Stim measurement j = clbit j = column j of the raw `memory.npz`.
- Arrays are `uint8` 0/1. One with more than 16,777,216 entries is bit-packed along its
  columns (`np.packbits(a, axis=1, bitorder="little")`): the manifest's `array_info` says
  `"packed": true` and gives the unpacked shape. `dqiqm.inforecord.load_array(run_dir, "dets")`
  reads either. The dry run's arrays are all unpacked.
- Detection events and observable flips are against the circuit's noiseless reference sample
  (Stim's m2d converter): a noiseless shot is all zeros.
- `shots.csv` is CSV: `pyarrow` is not in `uv.lock` (and the brief forbids adding it), so no
  Parquet. `sim/checks/sizes.json` has what a million shots cost per file.
- The decoder column `pm_pred` is PyMatching on a noisy copy of the circuit that ran
  (`dqiqm.decode`), with the noise named in the manifest's `decoder`.
- Hashes, in `placement.json` and each manifest: `qasm_sha256` is the sha256 of the `.qasm` file;
  `stim_sha256` is the sha256 of the Stim text **without the file's trailing newline**
  (`str(circuit)`), so hashing the `.stim` file as stored doesn't match; `native_sha256` is
  `dqiqm.experiments.circuits.native_sha256` of the hardware circuit.
- Submitting: the `.qasm` files document what runs, but they can't be loaded here
  (`qiskit.qasm3.loads` needs `qiskit_qasm3_import`, which is not in `uv.lock`). The hardware
  worker rebuilds the circuits with `dqiqm.experiments.suite.build(device.live_calibration(<the
  snapshot>))` and, before it submits, checks each circuit's `circuits.native_sha256` against its
  `placement.json` entry. A mismatch stops the job.

## Pre-training the ML decoder

Pre-train on synthetic shots built from the calibration package (`calibration/garnet/`) **with
idle terms**: the T1/T2 decay over each mid-circuit readout window, plus any `delay` (E4), and none
after the last round. In Stim, `dqiqm.experiments.device.hardware_noise(cal, spec)` builds the
hardware prior: the compiled readout window (1,096 ns for E1's ancillas, from
`data/calibration/garnet/pulse/compile.<calset>.json`; the `measure` probe alone, 556 ns, is about
half of it), a symmetric idle on the data (the biased twirl, in each data qubit's hardware frame,
is the `idle_channel="pauli"` switch), and `measure_fidelity`'s unheralded errors on the final
readout layer, which IQM compiles terminal measurements to. `calibrated_noise(cal, ..., idle_ns=...)`
is the same with its switches (`idle_channel`, `t2`, `final_readout`) left at the old defaults; its
`idle_channel="pauli"` without `frames` is in Stim's frame, which is right for a Stim simulation but
not for hardware. On the pilot, the biased twirl in the hardware frame fits the detection counts best
and decodes worst, so choose the hardware prior per basis by decoded P_L on the finetune split's
validation shots, never on test, and not by detector log-likelihood alone. Like
the stand-in, both add relaxation over each gate on top of the RB and IRB figures, so their gate
noise is on the high side (on the backlog). On the pilot, the model still has fewer detection events
than the hardware in the bulk rounds (`docs/garnet-design.md`, "Decoder priors for hardware"). **Don't pre-train on `sim/`:** the stand-in has no idle
noise, and its gate noise is about twice the benchmarked figures (median ×2.05 for 1Q gates, ×1.82
for CZ), so its shots are not Garnet's. `calibration/garnet/placement.json` holds E1's two
placements, pA and pB: choose the placement by each shot's `patch` (`shots.csv`).

## The ML split (fixed before any data)

Batch 3, whole, is the **test** set and is never fine-tuned on. In every other job (the pilot phase
included) a seeded draw of 20% of the shot indices, the same for every circuit of the job, is
**validation**, the rest **finetune** (`dqiqm.experiments.suite.split`). Each shot's split is in
`shots.csv`'s `split`; the rule, the run's role and the counts in the manifest's `split`.

`validation` is the fine-tuning side's held-out 20%. A loader with only finetune and test loads it
as finetune (or as its own validation set), **never as test**: test is batch 3 only.

There is no P1 on hardware. The barrier A/B was dropped: IQM's compiler multiplexes each round's
readout only with barriers, and the pilot compiles to 7,748 ns without them against 3,900 ns with
them (`docs/pulla-check.md`), so barriers stay on. The barrier-off circuit
(`circuits/E1/E1_d3_Z_r03_pA_nobar.*`) stays as built, and no job runs it; its `.stim` is
byte-identical to the barrier-on circuit's. A loader should still **filter on `barriers`**
(`shots.csv`, or the manifest's `spec.barriers`) and keep only the runs with barriers on.

## Reading a run

```python
import json, numpy as np, csv
from dqiqm.inforecord import load_array
run = "InfoRecord/sim/datasets/E1/b1_E1_d3_Z_r10_pA"
m = json.load(open(f"{run}/manifest.json"))
dets, obs, raw = (load_array(run, k) for k in ("dets", "obs", "raw_meas"))
detectors = list(csv.DictReader(open(f"{run}/detectors.csv")))
```

## The plan in one line

14 jobs, 1,116,000 shots, about 6 min of QPU time at the assumed overheads
(`circuits/job_plan.json`, `docs/garnet-design.md`).

## Column dictionary

Printed from `dqiqm.inforecord.COLUMNS`, which also writes and checks every file.

### `datasets/shots.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `simulated` | bool |  | 1 for a simulated shot (InfoRecord/sim), 0 for hardware |
| `job_id` | str |  | the job that ran the shot (a sim job id starts with 'sim-') |
| `batch` | int |  | the time-separated batch (1, 2, 3; 0 for the pilot phase) |
| `position_in_job` | int |  | the circuit's place in its job's run order (0-based) |
| `shot` | int |  | the shot's index within the circuit's shots, in the order returned |
| `split` | str |  | finetune, validation or test (the fixed ML split; rule in the manifest) |
| `time_bucket` | int |  | shot * 10 // shots: blocks of consecutive shots, for drift |
| `circuit` | str |  | the circuit's name (circuits/&lt;experiment&gt;/&lt;name&gt;.stim) |
| `experiment` | str |  | E1 ... E5 |
| `family` | str |  | surface_memory, repetition, readout or stability |
| `rounds` | int |  | stabilizer rounds R (readouts in a row for E3) |
| `basis` | str |  | the basis the data are prepared and read in (Z or X) |
| `patch` | str |  | pA / pB (E1 orientations), line19, all20, block3x3 |
| `prep` | str |  | the prepared state: 0, 1, + or, for E3, one bit per qubit |
| `delay_ns` | float | ns | the idle after each round's readout but the last (E4) |
| `barriers` | bool |  | 1 if each layer is fenced by barriers (the default) |
| `raw_weight` | int |  | the number of 1s in the shot's measurement record |
| `all_zero` | bool |  | 1 if the whole record is 0 |
| `total_events` | int |  | the number of detection events |
| `longest_streak` | int | rounds | the longest run of consecutive rounds one stabilizer fired |
| `sw_t<k>` | int |  | the syndrome weight (detection events) at detector round t = k |
| `obs_flip` (optional) | bool |  | the observable's flip (the raw logical outcome against the noiseless one) |
| `pm_pred` (optional) | bool |  | PyMatching's predicted flip (decoder in the manifest) |
| `pm_fail` (optional) | bool |  | 1 if pm_pred != obs_flip: a logical failure of the decoder |

### `datasets/detectors.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `index` | int |  | the detector's index: its column in det_events.npy |
| `round` | int |  | Stim's time coordinate t (0-based; final data detectors at t = R) |
| `x` | float |  | Stim's detector x coordinate |
| `y` | float |  | Stim's detector y coordinate |
| `t` | float |  | Stim's detector t coordinate, as given (equal to round) |
| `stabilizer` | str |  | the check's label, e.g. X(2,0), ZZ(3), M(QB5) |
| `type` | str |  | the check's Pauli type, X or Z |
| `ancilla_qb` | str |  | the device qubit that measures the check |
| `ancilla_stim` | int |  | its Stim qubit index |
| `data_qbs` | str |  | the device qubits in the check's support, ';'-separated |
| `record` | str |  | the measurement indices it XORs, most recent first, ';'-separated (m_k;m_(k-2);...) |
| `boundary` | str |  | first (t = 0), second (t = 1), bulk, or final (it reads the data readout) |
| `grid_row` | int |  | the ancilla's row on the device grid (architecture/garnet.json) |
| `grid_col` | int |  | the ancilla's column on the device grid |

### `datasets/measurements.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `index` | int |  | the measurement's index j: column j of raw_meas.npy, clbit j |
| `round` | int |  | t: round k's readout at k - 1, the final data readout at R |
| `stim_qubit` | int |  | the measured Stim qubit |
| `qb` | str |  | the measured device qubit |
| `role` | str |  | data, ancilla or readout |
| `stabilizer` (optional) | str |  | the ancilla's check label (empty for data) |
| `type` (optional) | str |  | the check's Pauli type (empty for data) |
| `moment` | int |  | the barrier-delimited layer (moment) of the circuit it is in (0-based) |
| `t_ns` | float | ns | estimated start of the readout from the start of the shot (calibrated durations) |
| `grid_row` | int |  | the qubit's row on the device grid |
| `grid_col` | int |  | the qubit's column on the device grid |

### `datasets/qubits.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `qb` | str |  | device qubit (all of the device's, used or not) |
| `stim_qubit` (optional) | int |  | its Stim qubit in this circuit (empty if unused) |
| `role` | str |  | data, ancilla, readout, or idle (not used) |
| `stabilizer` (optional) | str |  | the check it measures, if an ancilla |
| `grid_row` | int |  | row on the device grid |
| `grid_col` | int |  | column on the device grid |
| `t1_ns` | float | ns | T1 at collection time |
| `t2_echo_ns` (optional) | float | ns | T2 (echo) |
| `t2_ramsey_ns` (optional) | float | ns | T2 (Ramsey) |
| `p1_given_0` | float |  | readout P(1\|0) of the default measure |
| `p0_given_1` | float |  | readout P(0\|1) of the default measure |
| `error_1q` | float |  | 1Q gate error (RB infidelity) |
| `readout_leakage` (optional) | float |  | readout-induced leakage |
| `measure_ns` (optional) | float | ns | readout duration |
| `cz_errors` | str |  | CZ error (IRB infidelity) per coupler, 'QBx=err' ';'-separated |
| `cz_error_mean` | float |  | the mean of cz_errors |

### `derived/rate_by_round.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `detector` | int |  | detector index |
| `stabilizer` | str |  | its check |
| `round` | int |  | its t |
| `rate` | float |  | firing rate over the run's shots |
| `shots` | int |  | shots |

### `derived/rate_by_bucket.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `bucket` | int |  | time bucket (blocks of consecutive shots) |
| `shot_lo` | int |  | first shot of the bucket |
| `shot_hi` | int |  | last shot of the bucket |
| `stabilizer` | str |  | check |
| `rate` | float |  | mean firing rate of the check's detectors in the bucket |
| `shots` | int |  | shots in the bucket |

### `derived/pij_lags.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `stabilizer_i` | str |  | the earlier detector's check |
| `stabilizer_j` | str |  | the later detector's check |
| `lag` | int | rounds | t_j - t_i: 0, 1 (k -> k+1) or 2 (k -> k+2) |
| `pij_mean` | float |  | mean p_ij over the pairs (Google's pairwise error probability) |
| `pij_max` | float |  | largest p_ij among the pairs |
| `pairs` | int |  | number of detector pairs averaged |

### `derived/streaks.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `stabilizer` | str |  | check |
| `length` | int | rounds | length of a run of consecutive rounds in which the check fired |
| `count` | int |  | number of such runs over all shots |

### `derived/syndrome_weight.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `round` | int |  | detector round t |
| `weight` | int |  | detection events in that round |
| `count` | int |  | shots with that weight |

### `derived/assignment.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `qb` | str |  | device qubit |
| `prep` | int |  | prepared bit |
| `layer` | int |  | readout layer (0-based) |
| `p_wrong` | float |  | P(read != prep) |
| `p_one` | float |  | P(read 1) |
| `shots` | int |  | shots |

### `derived/readout_flips.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `qb` | str |  | device qubit |
| `prep` | int |  | prepared bit |
| `repeats` | int |  | readouts in a row |
| `p_det_bulk` | float |  | mean P(D_k), 2 <= k <= K-1 |
| `record_flip` | float |  | record flips per readout: mean p_ij(D_k, D_(k+1)) |
| `state_flip` | float |  | state flips per readout gap: p_det_bulk - 2 record_flip |
| `p_wrong_first` | float |  | misread fraction at the first readout |
| `p_wrong_last` | float |  | misread fraction at the last readout |
| `shots` | int |  | shots |

### `calibration/qubits.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `qb` | str |  | device qubit |
| `calibration_set_id` | str |  | the calibration set the figures describe |
| `fetched_utc` | str |  | when the snapshot was fetched |
| `grid_row` | int |  | row on the device grid |
| `grid_col` | int |  | column on the device grid |
| `t1_ns` | float | ns | T1 (characterization.model.&lt;q&gt;.t1_time) |
| `t2_ramsey_ns` | float | ns | T2 Ramsey (.t2_time) |
| `t2_echo_ns` | float | ns | T2 echo (.t2_echo_time) |
| `p1_given_0` | float |  | readout P(1\|0): metrics.ssro.measure.constant.&lt;q&gt;.error_0_to_1 (not the heralded figure) |
| `p0_given_1` | float |  | readout P(0\|1): ...error_1_to_0 |
| `readout_error` | float |  | the mean of the two |
| `error_1q` | float |  | 1Q gate error: 1 - RB fidelity of the default prx |
| `readout_leakage` (optional) | float |  | readout-induced leakage (metrics.rrc...leakage) |
| `prx_ns` | float | ns | prx duration |
| `measure_ns` | float | ns | measure duration |

### `calibration/couplers.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `qb_a` | str |  | first qubit of the coupler |
| `qb_b` | str |  | second qubit |
| `calibration_set_id` | str |  | the calibration set |
| `fetched_utc` | str |  | when the snapshot was fetched |
| `cz_error_irb` | float |  | CZ error, 1 - interleaved-RB fidelity |
| `cz_error_clifford_rb` (optional) | float |  | CZ error from Clifford RB (for reference) |
| `cz_ns` | float | ns | CZ duration |

### `ler/ler_points.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `experiment` | str |  | E1, E2, E4 or E5 |
| `run_id` | str |  | the LER run: basis (or kind/prep), patch, batch |
| `circuit` | str |  | circuit name |
| `dataset` | str |  | the dataset directory the point comes from |
| `rounds` | int |  | R |
| `shots` | int |  | shots n |
| `failures` | int |  | decoder failures k (pm_fail) |
| `p_l` | float |  | P_L = k / n |
| `wilson_lo` | float |  | Wilson 95% lower end |
| `wilson_hi` | float |  | Wilson 95% upper end |
| `eps_single` | float |  | eps_L from this R alone: (1 - (1 - 2 P_L)^(1/R)) / 2 |
| `eps_single_lo` | float |  | its 95% lower end (Wilson, mapped) |
| `eps_single_hi` | float |  | its 95% upper end |
| `raw_failures` | int |  | failures of the 'always 0' decoder (obs_flip) |
| `p_raw` | float |  | raw_failures / n |

### Arrays

| file | contents |
|---|---|
| `datasets/raw_meas.npy` | uint8 (shots, measurements): column j = Stim measurement j = clbit j, the raw record |
| `datasets/det_events.npy` | uint8 (shots, detectors): detection events, against the noiseless reference (Stim m2d) |
| `datasets/obs_flips.npy` | uint8 (shots, observables): observable flips, against the noiseless reference |
| `derived/pij.npy` | float32 (detectors, detectors): p_ij for every pair; the diagonal is each detector's rate |
| `derived/readout_corr.npy` | float64 (qubits, qubits): correlation of misassignments between qubits (E3) |

### `datasets/manifest.json`

| key | meaning |
|---|---|
| `format` | 'dqiqm.inforecord/1' |
| `label` | SIMULATED or HARDWARE |
| `simulated` | true for the dry run |
| `experiment` | E1 ... E5 |
| `circuit` | the circuit's name |
| `run_id` | this directory's name: &lt;batch&gt;_&lt;circuit&gt; |
| `batch` | the time-separated batch (0: pilot phase) |
| `job_id` | the job that ran it |
| `position_in_job` | the circuit's place in the job's run order |
| `split` | the ML split: its rule (test batch, validation fraction, seed), this run's role, and the counts |
| `spec` | what the circuit is: family, d, rounds, basis, no_reset (true), gate_basis (the Stim source's: R, H, CX, M), kind, prep, patch, delay_ns, barriers, orientation |
| `device` | the device (or the stand-in, for the dry run) |
| `calibration_set_id` | the calibration set the job ran on (the noise's source, for the dry run) |
| `calibration` | paths of the snapshots before and after the batch |
| `shots` | shots |
| `num_measurements` | columns of raw_meas |
| `num_detectors` | columns of det_events |
| `num_observables` | columns of obs_flips |
| `arrays` | {'dets': 'det_events.npy', 'obs': 'obs_flips.npy', 'raw_meas': 'raw_meas.npy'} |
| `array_info` | per file: shape (unpacked), dtype, packed, bitorder, sha256 of the file |
| `placement` | Stim qubit -> device qubit |
| `circuit_files` | the circuit's .stim and .qasm under circuits/ |
| `stim_sha256` | sha256 of the Stim source's text |
| `qasm_sha256` | sha256 of the .qasm |
| `native_sha256` | sha256 of the serialized native instructions (what the server receives) |
| `raw` | the raw result it was read from: path of memory.npz and its key c&lt;i&gt; |
| `decoder` | the decoder of pm_pred: PyMatching on the noisy copy of the circuit that ran, and its noise |
| `duration_ns` | the circuit's estimated duration (calibrated durations, barriers as sync points) |
| `n_buckets` | time buckets in shots.csv |
| `created_utc` | when this directory was written |
| `job_timeline` | the server's status timestamps (hardware), from the raw meta.json |
| `commit` | git HEAD of the code that wrote it |
| `versions` | package versions |
| `seeds` | the seeds (simulation) |
| `simulator` | how the dry run simulated it (simulation only) |

