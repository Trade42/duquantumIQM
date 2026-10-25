## Part B: the contract

In stage 0, copy this part verbatim into `repo/mlDecoder/CONTRACT.md`. In later stages, where this file and `CONTRACT.md` disagree, `CONTRACT.md` wins.

### The layout

```
repo/mlDecoder/
  CONTRACT.md  README.md  RESULTS.md  NOTES.md  requirements-ml.txt  __init__.py
  data.py        # specs, circuits, datasets, frozen test sets, detector layout
  evaluate.py    # comparison with MWPM, intervals, the fits, speed benchmarks
  train.py       # a generic training loop (backends may use it or their own)
  run.py         # CLI: python -m mlDecoder.run --backend <name> --config mlDecoder/configs/<name>_<size>.json
  report.py      # regenerates RESULTS.md from every result JSON (idempotent)
  models/__init__.py   (docstring only: there is no registry. run.py imports mlDecoder.models.<name>)
  models/<name>.py     # one file per backend
  configs/<name>_smoke.json, configs/<name>_full.json
  docs/<name>.md       # the design, the papers it follows, its limits
  tests/test_*.py      # offline, fast: python -m pytest -q mlDecoder/tests
  figures/             # small committed PNGs (stage 4)
  results/             # NOT committed (the repo ignores results/): datasets/, <backend>/<run-id>/
```

### The backend API (`models/<name>.py`)

```python
def build(circuit: stim.Circuit, config: dict) -> "Decoder": ...    # sized from the circuit (detector count, coordinates)

class Decoder:
    name: str
    def fit(self, train_spec_or_data, val: "Dataset", *, device: str, seed: int, log=print) -> dict: ...
        # trains, then freezes; returns epochs, steps, shots seen, wall time, device, final val loss and val P_L
    def decode_batch(self, det_events: np.ndarray, *, enable_correlations=None) -> np.ndarray: ...
        # (N, num_obs) bool, in shot order. Accepts the team matcher's keyword (and records that it's ignored).
        # Raises on a wrong width.
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path, circuit: stim.Circuit, device: str) -> "Decoder": ...
    def metadata(self) -> dict: ...
        # architecture, parameter count, config, torch version, training cost
```

A decoder may use the circuit (detector coordinates, its DEM) and the detection events. Never the observables at decode time, and never the hidden leakage flags.

### Data (`data.py`)

```python
@dataclass(frozen=True)
class DatasetSpec:
    distance: int = 3; rounds: int = 3; basis: str = "Z"
    noise: str = "uniform"            # a noise_suite key
    p: float = 0.001
    gate_basis: str = "cz"; no_reset: bool = True
    calibration: str | None = None    # "synthetic:<seed>" for the calibrated models

@dataclass
class Dataset:
    spec: DatasetSpec; split: str     # "train" | "val" | "test"
    dets: np.ndarray                  # (N, num_detectors) uint8
    obs: np.ndarray                   # (N, num_observables) uint8
    detector_coords: np.ndarray       # (num_detectors, 3): x, y, t from circuit.get_detector_coordinates()
    circuit: stim.Circuit             # the noisy circuit
    manifest: dict                    # spec, split, shots, seed, versions, sha256 of the arrays and of str(circuit)

def make_circuits(spec) -> tuple[stim.Circuit, stim.Circuit]: ...      # (ideal, noisy), as simulate_memory builds them
def make_dataset(spec, split, shots, *, seed=None) -> Dataset: ...
def sample_batch(spec, n, rng) -> tuple[np.ndarray, np.ndarray]: ...   # fresh training shots (streaming)
def frozen_test_set(spec, shots) -> Dataset: ...                       # cached under results/datasets/<sha12>/
def detector_layout(circuit) -> dict: ...                              # per detector: round, stabilizer id, X/Z type, (x, y)
```

- **Seeds:** split-specific streams (`numpy.random.SeedSequence` keyed by split and spec), so train, val and test never overlap.
- **Test sets** are generated once and frozen: `.npz` plus `manifest.json`. A later load refuses a hash mismatch.
- **Test size per point:** take a pilot P_L from correlated MWPM, then choose N so that MWPM expects **≥ 100 failures**, capped at 10⁷ shots. Record the N, and if the cap binds, say so.

### Evaluation (`evaluate.py`)

On the same frozen test shots, every decoder against **correlated MWPM** and **uncorrelated MWPM**:

- **Failures:** counts, P_L with a Wilson 95% interval, and the **paired** comparison: the discordant counts (ML wrong and MWPM right, and the reverse) and a paired-bootstrap 95% interval for ΔP_L.
- **P_L against rounds:** `fit_ler` → ε_L per round, with and without the SPAM amplitude. Report both, and label which form is the challenge's own (no SPAM).
- **Speed:**
  - warmed decode throughput (shots/s) and latency per shot, at batch sizes {1, 64, 4096, 65536} (capped at the test size);
  - on the CPU (thread count stated) and on CUDA, for every decoder **and** for PyMatching, on this laptop;
  - training time, graph building, device transfer and inference reported separately. Speeds are only comparable on the same machine.
- **Output:** one JSON per run in `results/<backend>/<run-id>/` (spec, dataset hashes, config, metrics, timings, versions, device, the git commit). Then `python -m mlDecoder.report` rebuilds `RESULTS.md`.

### The LER record

*Added 2026-10-24 at the orchestrator's (`dq-orch-1`) instruction, an amendment from Sal; the updated ML-DECODERS.md carrying it follows by scp. Until then this section is the spec.*

**Every simulation ends with a detailed LER record** (the challenge's steps 03, decode, and 04, extract the LER), for each (decoder, d, basis, noise, p):

- **Step 03:** measurements → detection events (the sampler's, or the m2d of the circuit that ran) → decode. A shot fails if ANY observable is wrong.
- **Step 04, the points:** P_L at every R in the sweep (at least 5 distinct R), with shots, failures and a Wilson 95% interval.
- **Step 04, the fits** of 2·P_L = 1 − A(1−2ε)^R: `challenge.decoding.fit_ler(rounds, errors, shots, fit_spam=False)` (the challenge's formula, A=1) and `fit_spam=True`. For each: ε_L with a 95% interval, A, and a goodness-of-fit.
- **The same for correlated MWPM on the same shots,** plus the ratio ε_L(decoder)/ε_L(MWPM) with a paired-bootstrap interval, and Λ = ε_L(d)/ε_L(d+2) for d = 3 → 5 → 7.
- **Files** in `results/<backend>/<run-id>/ler/`:
  - `ler_points.csv`, columns `decoder,d,basis,noise,p,rounds,shots,failures,P_L,wilson_lo,wilson_hi`;
  - `ler_fits.json`;
  - `ler_<d>_<basis>_<noise>_p<p>.png`: P_L against R with error bars, both fitted curves, MWPM for reference, labelled axes.

  `report.py` puts each run's LER table and plots into RESULTS.md.
- **Rules:** ≥ 100 MWPM failures per point, or the shortfall stated. Fixed shots. Label everything simulated, with its noise model.
- **When you report to Sal,** give the ε_L numbers with their intervals.

Implementation choices, approved by the orchestrator:
- p is in the PNG names, since the baseline sweeps two values of p.
- The goodness of fit is Pearson χ² and the deviance, with p-values on #R − #parameters degrees of freedom.
- The ratio's interval is a paired bootstrap: shots resampled jointly for both decoders at each R, 2,000 resamples, in the A = 1 form, refit by a vectorised MLE checked against `fit_ler` to 1e-6.
- The SPAM-form ratio gets a reduced bootstrap: 200 of the same resamples, refit with `fit_ler(fit_spam=True)`, and labelled as reduced.
- Λ comes from an independent binomial bootstrap of each point, since different d means independent shots. A = 1 form.
- Points with fewer than 100 MWPM failures are listed with their R.

### Hybrid training: Sal's choice, 2026-10-24 at about 19:47 UTC

*Added at the start of stage 1 from ML-DECODERS.md: Part B's "Hybrid training" section, then points 1–6 of its STAGE-1 UPDATE (`dq-orch-2`, 2026-10-24, about 22:10 UTC, from Sal's decisions between 20:56 and 21:42 UTC). Points 1–6 override the rest of this file wherever they disagree.*

**Pre-train on synthetic data, then fine-tune and test on real IQM shots.** AlphaQubit was trained this way.

- **The synthetic pre-training data includes the device's calibrated noise.** The mini PC's `InfoRecord/` delivers, by Sal's copy and before any hardware data, `calibration/<device>/`: `qubits.csv` (T1, T2, readout P(1|0) and P(0|1), the 1q error, leakage), `couplers.csv` (the CZ errors), `durations.json` and `placement.json` (Stim qubit → QB name, per patch). Build the team's `CalibrationSnapshot` and `qubit_map` from these (read `challenge/calibration.py` and `challenge.experiment.main()`). Then take `calibrated_pauli` / `calibrated_coherence` / `calibrated_leakage` from `noise_suite(p, calibration=...)`, next to `uniform` and `leakage`.
- **Fine-tune and test on the hardware shots,** loaded with `data.external_dataset`. The test split is a whole held-out, time-separated batch, never fine-tuned on; InfoRecord's manifests define the split. Report synthetic-only against fine-tuned against correlated MWPM (built from the calibrated noisy circuit), and MWPM with uniform weights. The LER record applies to hardware data too. Label every number hardware or simulated.

#### 1. Real data for training and for every evaluation

Sal: "the decoders should be trained on Garnet data and evaluated only on Garnet data". Since 21:05 UTC the same goes for **IQM Emerald**: "train and gain data from emerald as well as you are doing for garnet".

Per device, the training is **hybrid**:
- **(a) Pre-train** on synthetic shots built from that device's calibration (point 4).
- **(b) Fine-tune** on that device's real shots.
- **(c) Evaluate only on that device's held-out real TEST batch.**

**Every number reported as a result comes from real held-out data.** Synthetic numbers are development only, and labelled so.

#### 2. Distances: d = 3 only

- Garnet fits d=3 with zero SWAPs on exactly one qubit set (patches pA and pB are its two orientations).
- Emerald fits d=3 on 15 sets. We use the best two, eA and eB.
- **d=5 doesn't fit Emerald** with zero SWAPs today: its only 49-qubit set needs couplers that have no working CZ.
- **d=7 needs 97 qubits,** and Emerald has 54.
- **So there is no d=5 or d=7 ML training.** Sal, 21:42 UTC: "No d=7 ML at all"; d=5 has no hardware.
- Stage 0's simulated MWPM numbers at d=5 and d=7 stay only as the write-up's scaling reference (Λ 7.7–8.9 on `uniform`).
- Everywhere else in this contract that says d ∈ {3, 5, 7} for training or evaluation, read d = 3.

#### 3. What the mini PC sends

Sal copies it into the laptop folder as `InfoRecord/`.
- **Garnet's record is the root, `InfoRecord/`. Emerald's is `InfoRecord/emerald/`,** with the identical layout inside. Every manifest names its `device`.
- **`InfoRecord/README.md` comes first.** It has the column dictionary and the conventions.
- **Garnet, now:** `calibration/garnet/` (`qubits.csv`: T1, the T2 echo and Ramsey times, readout P(1|0) and P(0|1) from `measure.constant`, and the 1q error; `couplers.csv`: the CZ errors; `durations.json`: prx, CZ and measure in ns, and `reset_wait_ns`; `placement.json`: qubit → QB name for both patches, pA and pB; its README) and `circuits/E1/*.stim`, the exact noise-free circuits that run on hardware. Their detectors are identical, in order and coordinates, to the team's `make_stim_circuit`.
- **Emerald, soon:** the same, under `InfoRecord/emerald/` (patches eA and eB).
- **The real shots, later:** `InfoRecord/datasets/E1/<run id>/` and `InfoRecord/emerald/datasets/…`: `det_events.npy`, `obs_flips.npy` and `raw_meas.npy` (bit-packed if large; the manifest's `arrays` map and `array_info` say which); `shots.csv`, one row per shot (`split`, `patch`, `barriers`, batch, rounds, basis, …); `detectors.csv` (`round` = Stim t, and x, y = Stim coordinates, in detector order); `measurements.csv`, `qubits.csv` and `manifest.json`.
- **They arrive in this order:** first the pilot, 1,000 shots of E1 Z R=3, a loader smoke test and not training data; then batches B1 and B2 (fine-tune) and B3 (TEST), at least 2 hours apart.

#### 4. The pre-training data: from the calibration package, with idle noise

- **Never pre-train on `InfoRecord/sim/`.** It is a stand-in simulator with no idle noise and about twice the benchmarked gate noise.
- **Use the team's calibrated models:** build the team's `CalibrationSnapshot` and `qubit_map` from the package, then take `calibrated_pauli`, `calibrated_coherence` (T1/T2 idling) and `calibrated_leakage` from `noise_suite(p, calibration=...)`.
- **Map the team's qubits to QB names by coordinates, not by index:** match `circuit.get_final_qubit_coordinates()` against the placement, and check that all 17 match.
- **One qubit map per patch:** pA and pB on Garnet, eA and eB on Emerald, chosen by each shot's `patch`.
- **The idle during each readout window is in the model.** Every qubit idles through it, data and ancillas. It is about 1.1 µs per round on Garnet, not the `measure_ns` of `durations.json` (only the probe pulse): each window is acquisition delay (≈465 ns) + integration + 76 ns of dead time, and the pilot patch's window is **1,096 ns** (set by ancillas QB5 and QB7). Use 1,096 ns for Garnet's pA/pB idle per round until InfoRecord ships per-round windows. **Terminal readouts are compiled as `measure_fidelity`**: its unheralded errors are as good or better on 16 of 17 qubits, but QB20's P(0|1) is 0.090, against 0.017 for `measure`. Where the team's models leave the idle out, it is a documented extension in `mlDecoder/`; `challenge/` is never edited.
- **The plain `leakage` model is mixed in as well.** Real devices leak, and stage 0 showed that leakage is where ML wins.
- The mini PC may later send calibrated noisy circuits (`*.noisy.stim`) that match the hardware decoder's priors exactly. If they arrive, sample them too.

#### 5. Loading the real shots (`data.external_dataset`)

- **The splits:** `finetune` → train; **`validation` → the validation set**, the fine-tuning side's held-out 20%, **never used as test**; `test` → test, a whole time-separated batch, never trained or early-stopped on.
- **Filter on `barriers`.** Garnet's pilot phase has a barrier-off arm (P1). Fine-tune and validate only on runs whose `barriers` matches the batches' setting, given in each manifest's `spec`.
- Check `device`, `patch`, `rounds` and `basis` from the manifest. Refuse a detector count that doesn't match the circuit.
- **The MWPM baselines on real data:** correlated MWPM from the calibrated noisy circuit (the team's `make_matcher` on the calibrated model), **and** MWPM with uniform weights.

#### 6. The LER record on real data

It follows "The LER record" above, plus:
- **the free-A fit is reported next to the challenge's A=1 form.** Stage 0 found that A=1 reads 5–25% low on no-reset circuits;
- **every number is labelled** HARDWARE (device, calibration set, batch) or SIMULATED (noise model).
