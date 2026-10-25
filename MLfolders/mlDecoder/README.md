# mlDecoder: a learned decoder for the team's surface code on IQM Garnet and Emerald

Sal's part of the DuQuantum 2026 IQM-track project. It decodes the team's no-reset, distance-3 rotated surface-code
memory (`challenge/`, CZ gates, both bases) on real IQM hardware, and is measured against the team's correlated
PyMatching on the same shots.

**The final decoder** is a two-layer cascade per device, for the best patch of each, both bases:

1. a recurrent **LSTM** (after Varbanov et al. 2025 and Baireuther et al. 2018). It is pre-trained on simulated shots of
   the device's calibrated noise, then fine-tuned on the device's real shots. It decodes every shot and gives P(flip).
2. **Relay-BP** (IBM, arXiv:2506.01779) takes the shots the LSTM is unsure of: confidence |2p − 1| below a threshold
   τ, chosen on validation by the one-standard-error rule. Where deferring does not beat paired noise, τ = 0 and the
   LSTM decides alone. Relay-BP runs on the decoder prior's detector error model: calibrated, or uniform weights
   where those won on validation (Emerald X).
3. **Correlated MWPM** takes any shot Relay-BP does not converge on, so every shot gets a decision.

On the held-out test batch, τ = 0 for three of the four models, so the cascade is the LSTM. On Emerald X it defers 23%
of shots and did not beat the LSTM alone (see Results).

| Device | Patch | Bases | Weights (float16) |
|---|---|---|---|
| IQM Garnet | pA (QB1, QB2, QB4–QB16, QB19, QB20) | Z, X | `weights/garnet/rnn_lstm_pA_{Z,X}.pt`, fine-tuned, 488 KiB each |
| IQM Emerald | eA (QB13, QB14, QB19–QB22, QB27–QB31, QB36–QB39, QB44, QB45) | Z, X | `weights/emerald/rnn_lstm_eA_{Z,X}.pt`, fine-tuned, 488 KiB each |

Each model has 243,714 parameters and decodes any number of rounds R. `weights/manifest.json` records each file's
sha256, provenance (run, commit, calibration set), decoder prior, and the cascade's τ.

## Install

The package lives at `MLfolders/mlDecoder/` in the team's repo and carries its own `pyproject.toml` and `uv.lock`. The
team's package (`challenge/`, the repo root) is an editable path dependency. The lock pins the rest: torch
2.14.1+cu132 (PyTorch's CUDA 13.2 index), relay-bp 0.2.2, stim 1.16.0, PyMatching 2.4.0, numpy, scipy and matplotlib.
On Windows (Git Bash):

```bash
cd MLfolders/mlDecoder
uv sync --extra test                                                       # .venv here, Python 3.13
cd ../..
MLfolders/mlDecoder/.venv/Scripts/python.exe -m pytest -q MLfolders/mlDecoder/tests    # offline, about 30 s
```

On Linux the interpreter is `MLfolders/mlDecoder/.venv/bin/python`. torch is the CUDA build, but the CPU works too, and
nothing imports torch until a model is used. This was tested at the push base, GitHub main `8e5d373`.

Windows or Linux only; on Linux uv downloads several GB of CUDA wheels.

`requirements-ml.txt` records the environment the models were trained in, a venv at the repo root of the laptop's
clone. The lock has the same torch, relay-bp, stim and PyMatching versions.

## Use

```python
from mlDecoder import final

decoder = final.load("garnet", "pA", "Z")                  # the LSTM alone, on the CPU by default
flips = decoder.decode_batch(det_events)                   # (shots, 1) bool, in shot order; R from the width

cascade = final.load("emerald", "eA", "X", cascade=True)   # LSTM -> Relay-BP below tau -> correlated MWPM
out = cascade.decode_detailed(det_events)                  # predictions, layer per shot (0, 1, 2), p, confidence
```

`det_events` are the circuit's detection events, (shots, 8·R) binary: InfoRecord's `det_events.npy`, or the
measurement-to-detector conversion of the team's `make_stim_circuit` record. From the command line:

```bash
python -m mlDecoder.final list
python -m mlDecoder.final decode --device garnet --patch pA --basis Z --dets <run>/det_events.npy --obs <run>/obs_flips.npy
```

**From another parent folder.** On the team's main the package lives at `MLfolders/mlDecoder/`. It uses package-relative
imports and resolves its weights, configs and vendored calibrations from its own location, so either of these works:

```bash
python -m MLfolders.mlDecoder.final list                 # from the repo root, as a sub-package of MLfolders/
PYTHONPATH=MLfolders python -m mlDecoder.final list      # or with MLfolders/ on the path
```

The team's `challenge` package must be importable; `uv sync` installs it editable from the repo root. InfoRecord, needed
only for training and evaluation, is found from `MLDECODER_INFORECORD` or the nearest ancestor folder holding
`InfoRecord/`.

## Results

### HARDWARE: the held-out test batch (B3)

Batch 3 of each device is the whole time-separated batch, never trained on, early-stopped on or tuned on. Garnet pA
has 11 runs of 5,000 shots per basis, Emerald eA 11 runs of 2,000, at R = 1–20. The models, the cascade's prior and
its τ were fixed on validation and committed (`f877b1a`) before B3 was opened. The mini PC's batch review passed
(APPROVE at batches `47eb36a`; all 417 circuits re-pulled from IQM match the raw data).

**Per-round logical error ε_L**, from the challenge's fit 2P_L = 1 − (1 − 2ε)^R over R = 1–20, with its 95% profile
interval. Under it is the ratio to correlated MWPM on the same shots (paired bootstrap, 95%):

| Device, basis | τ (prior; deferred) | LSTM, fine-tuned | cascade | Relay-BP | correlated MWPM | uniform-weight MWPM | LSTM, pre-trained |
|---|---|---|---|---|---|---|---|
| Garnet pA, Z | 0.00 (calibrated; 0.0% deferred) | 0.110 [0.108, 0.113]<br>×0.868 [0.849, 0.887] | 0.110 [0.108, 0.113]<br>×0.868 [0.850, 0.887] | 0.122 [0.119, 0.125]<br>×0.959 [0.943, 0.977] | 0.127 [0.124, 0.130] | 0.136 [0.133, 0.140]<br>×1.072 [1.050, 1.095] | 0.116 [0.113, 0.119]<br>×0.914 [0.897, 0.932] |
| Garnet pA, X | 0.00 (calibrated; 0.0% deferred) | 0.106 [0.104, 0.109]<br>×0.888 [0.866, 0.910] | 0.106 [0.104, 0.109]<br>×0.888 [0.866, 0.911] | 0.119 [0.116, 0.122]<br>×0.997 [0.979, 1.015] | 0.120 [0.117, 0.123] | 0.125 [0.122, 0.129]<br>×1.047 [1.024, 1.073] | 0.127 [0.124, 0.130]<br>×1.058 [1.032, 1.082] |
| Emerald eA, Z | 0.00 (calibrated; 0.0% deferred) | 0.044 [0.042, 0.045]<br>×0.964 [0.931, 0.996] | 0.044 [0.042, 0.045]<br>×0.964 [0.930, 0.998] | 0.045 [0.043, 0.047]<br>×0.993 [0.966, 1.021] | 0.045 [0.044, 0.047] | 0.047 [0.045, 0.048]<br>×1.031 [0.996, 1.067] | 0.045 [0.044, 0.047]<br>×0.998 [0.967, 1.028] |
| Emerald eA, X | 0.19 (uniform; 23.0% deferred) | 0.034 [0.033, 0.035]<br>×0.911 [0.881, 0.944] | 0.034 [0.033, 0.036]<br>×0.915 [0.883, 0.949] | 0.037 [0.036, 0.039]<br>×0.997 [0.960, 1.032] | 0.037 [0.036, 0.039] | 0.036 [0.034, 0.037]<br>×0.952 [0.919, 0.985] | 0.035 [0.033, 0.036]<br>×0.925 [0.892, 0.959] |

**P_L pooled over R** (the same shots at every R), with paired ΔP_L from the summed discordant shots (bootstrap 95%).
Uniform-weight MWPM is the fair baseline on Emerald X, where it won on validation:

| Device, basis | Shots | LSTM P_L | ΔP_L vs corr MWPM | ΔP_L vs uniform MWPM | Cascade P_L | Cascade ΔP_L vs corr MWPM | corr MWPM P_L | uniform MWPM P_L |
|---|---|---|---|---|---|---|---|---|
| Garnet pA, Z | 55,000 | 0.3677 | -0.0180 [-0.0223, -0.0137] | -0.0300 [-0.0347, -0.0253] | 0.3677 | -0.0180 [-0.0223, -0.0137] | 0.3857 | 0.3977 |
| Garnet pA, X | 55,000 | 0.3617 | -0.0174 [-0.0221, -0.0127] | -0.0235 [-0.0282, -0.0187] | 0.3617 | -0.0174 [-0.0221, -0.0127] | 0.3790 | 0.3851 |
| Emerald eA, Z | 22,000 | 0.2270 | -0.0059 [-0.0115, -0.0006] | -0.0100 [-0.0158, -0.0042] | 0.2270 | -0.0059 [-0.0115, -0.0006] | 0.2329 | 0.2370 |
| Emerald eA, X | 22,000 | 0.1926 | -0.0129 [-0.0180, -0.0078] | -0.0058 [-0.0108, -0.0009] | 0.1934 | -0.0121 [-0.0173, -0.0069] | 0.2055 | 0.1984 |

- **The fine-tuned LSTM beats correlated MWPM on all four,** with ε_L ratios of 0.87–0.96. Emerald Z wins only just:
  its interval reaches 0.996. The LSTM beats uniform-weight MWPM too. On Garnet the gain comes from R ≤ 8, since every
  decoder is near P_L = 0.5 by R ≈ 10–12. Emerald's P_L stays below 0.45 up to R = 20, and its gain is spread over R.
- **Fine-tuning matters most where the pre-trained model was weakest:** the pre-trained LSTM loses to MWPM on Garnet X
  (×1.058) and ties on Emerald Z (×0.998). Fine-tuning on 35,200–88,000 real shots turns both into wins.
- **The cascade adds nothing here.** τ = 0 on three of the four, so the cascade is the LSTM. On Garnet X it differs
  from the LSTM on 1 shot of 55,000, a tie at p ≈ 0.5. On Emerald X, τ = 0.19 defers 23% of shots to Relay-BP. On B3
  that is 17 failures worse than the LSTM alone (4,254 against 4,237 of 22,000), and about 18× slower per shot at
  R = 20. For speed, use `final.load(..., cascade=False)`.
- **Relay-BP** ties correlated MWPM, except on Garnet Z, where it is slightly ahead.
- **Emerald X's prior.** Uniform weights beat the calibrated prior on validation and again on B3 (ΔP_L −0.007
  [−0.012, −0.002]). The calibrated model's dephasing over the readout window, or its idle channel, is the suspect; a
  dynamical-decoupling A/B on the mini PC tests it.
- **Fit quality.** χ² = 83–680 on 10 degrees of freedom (χ²/dof 8–68), so the single-ε form does not follow P_L(R),
  which saturates near 0.5 on Garnet. Read ε_L as a summary; the pooled paired ΔP_L is the comparison with fewer
  assumptions. The team's free-A fit (`fit_ler(fit_spam=True)`) bounds A to [0, 1] and pins every fit here at A = 1.
  mlDecoder's own free-A refit lets A reach 2: it finds A > 1 (1.07–1.64), which gives ε_L 16–53% higher. The
  LSTM/MWPM ratios hold under it: 0.837, 0.843, 0.962 and 0.918 for Garnet Z, Garnet X, Emerald Z and Emerald X.
  Correlated MWPM has fewer than 100 failures at R = 1 on Emerald Z and R = 1–2 on Emerald X (listed in `RESULTS.md`).
- **Validation leans optimistic.** It served early stopping, τ and the prior choice. B3 is the honest number.

**Speed** on this laptop (Ryzen 9 8945HS, RTX 4060 Laptop GPU), warmed, for the cascade as shipped (with τ = 0 it is
the LSTM):

| Device, basis | R | τ | Cascade, 1 CPU thread, batch 1 | Cascade, CUDA, shots/s at batch | corr MWPM, 1 thread, batch 1 |
|---|---|---|---|---|---|
| Garnet pA, Z | 3 | 0.00 | 0.58 ms (p99 0.84) | 78,766 at 4,096 | 11 µs |
| Garnet pA, Z | 20 | 0.00 | 0.79 ms (p99 1.33) | 171,118 at 4,096 | 89 µs |
| Garnet pA, X | 3 | 0.00 | 0.62 ms (p99 0.96) | 92,937 at 4,096 | 11 µs |
| Garnet pA, X | 20 | 0.00 | 0.72 ms (p99 0.92) | 169,636 at 4,096 | 84 µs |
| Emerald eA, Z | 3 | 0.00 | 0.73 ms (p99 1.48) | 15,714 at 64 | 9 µs |
| Emerald eA, Z | 20 | 0.00 | 0.78 ms (p99 1.18) | 27,465 at 64 | 48 µs |
| Emerald eA, X | 3 | 0.19 | 0.61 ms (p99 1.75) | 10,537 at 64 | 8 µs |
| Emerald eA, X | 20 | 0.19 | 14.20 ms (p99 47.75) | 75 at 64 | 39 µs |

Emerald's runs have 2,000 shots per R, so its CUDA throughput was measured at batch 64 only, against 4,096 on Garnet.
At batch 1 on one thread, correlated MWPM is 50–85× faster per shot than the LSTM at R = 3, and 9–17× at R = 20.

**Device notes from the batch review.** Emerald's compiled readout window is 1,036 ns on every eA and eB qubit. The
batch's own MWPM records used the 360 ns `measure_ns` fallback; mlDecoder's prior uses the compiled value. Emerald eB's
QB10 (X-type) reads about 0.413 on its random first readout in every eB Z circuit, a readout asymmetry; eB is not one
of these patches.

### HARDWARE: pilot smoke tests (loader and decoders; not a result)

1,000 shots of E1 Z R=3 per device, the pilot run, all on the fine-tune side; not trained or evaluated on. P_L with
Wilson 95%:

| Device, patch | LSTM, synthetic only | correlated MWPM | uniform-weight MWPM | always 0 |
|---|---:|---:|---:|---:|
| Garnet pA | 0.224 [0.199, 0.251] | 0.217 [0.193, 0.244] | 0.222 | 0.379 |
| Emerald eA | 0.128 [0.109, 0.150] | 0.120 [0.101, 0.142] | 0.120 | 0.336 |

Both are ties (paired 95% intervals contain 0).

### SIMULATED development (pre-training; not the evaluation)

Each figure is the LSTM's per-round LER ε_L (the challenge's A = 1 fit over R = 1–20) divided by correlated MWPM's on
the same frozen shots, with paired-bootstrap 95% intervals. MWPM uses each set's own calibrated prior.

| Device, patch | basis | calibrated coherence | calibrated leakage | plain leakage p=0.003 |
|---|---|---:|---:|---:|
| Garnet pA | Z | 0.981 [0.976, 0.985] | 0.938 [0.934, 0.943] | 0.850 [0.846, 0.855] |
| Garnet pA | X | 0.986 [0.981, 0.991] | 0.963 [0.958, 0.968] | 0.882 [0.877, 0.887] |
| Emerald eA | Z | 0.962 [0.958, 0.967] | 0.912 [0.908, 0.916] | 0.865 [0.861, 0.870] |
| Emerald eA | X | 0.978 [0.973, 0.982] | 0.943 [0.938, 0.948] | 0.917 [0.912, 0.922] |

Garnet's calibrated leakage is at p = 0.003, Emerald's at p = 0.001.

**Speed** (this laptop: Ryzen 9 8945HS, RTX 4060 Laptop GPU; warmed, end to end):

| Decoder | Batch | R = 3 | R = 20 |
|---|---|---:|---:|
| LSTM, CUDA | 65,536 | about 0.4M shots/s | 0.3–1.4M shots/s |
| LSTM, one CPU thread | 1 | 0.5–0.6 ms/shot | 0.7 ms/shot |
| correlated MWPM, one CPU thread | 65,536 | 0.36M shots/s | 0.034M shots/s |
| correlated MWPM, one CPU thread | 1 | 8–10 µs/shot | about 39 µs/shot |
| Relay-BP, one CPU thread | per shot | 0.13–0.22 ms | 2–4 ms at R = 10 |

So the cascade's mean cost per shot is the LSTM's plus (deferred fraction × Relay-BP's). See `docs/cascade.md`.

## How it is built

- **Hybrid training** (`CONTRACT.md`, Hybrid training):
  - pre-train on simulated shots built from the device's calibration (`hardware.py`, `docs/hardware.md`);
  - fine-tune on the device's real shots (`finetune.py`);
  - evaluate only on the device's held-out, time-separated test batch.
- **The noise build.** The team's calibrated models get a documented extension:
  - every qubit idles through the ~1.1 µs readout window;
  - the terminal layer is one `measure_fidelity` readout;
  - readout is asymmetric;
  - measurement-induced ancilla flips are added.

  The pre-training mixture adds T2 variants and leakage. It was chosen against the pilots' detection statistics, which
  were not trained on.
- **Real shots** come from the mini PC's InfoRecord. Every file's hash, the m2d conversion, the circuit (against the
  team's generator), placement and splits are checked on load (`hardware.load_run`).

## Reproduce

| What | Command |
|---|---|
| Garnet pre-training (pA Z, X) and its SIMULATED dev scores | `python -m mlDecoder.run --backend rnn_lstm --config mlDecoder/configs/rnn_lstm_full.json --groups garnet_pA_Z garnet_pA_X` |
| Emerald pre-training (eA Z, X) | `python -m mlDecoder.run --backend rnn_lstm --config mlDecoder/configs/rnn_lstm_emerald.json` |
| Pilot smoke test (HARDWARE) | `python -m mlDecoder.finetune --backend rnn_lstm --from <pre-training run> --config mlDecoder/configs/finetune.json --pilot` |
| Fine-tune on B1/B2, tune τ on validation | `python -m mlDecoder.finetune --backend rnn_lstm --from <pre-training run> --config mlDecoder/configs/finetune.json --groups garnet_pA_Z garnet_pA_X` |
| Re-tune τ under the current rule (validation only, in place) | `... --retune-tau <fine-tuning run>` |
| Score on the B3 test batch | `... --use-finetuned <fine-tuning run>` |
| Promote models to `weights/` | `python -m mlDecoder.final promote --backend rnn_lstm --run <run>` |
| Rebuild `RESULTS.md` | `python -m mlDecoder.report` |

The hardware numbers above come from the fine-tuning runs `20261024-210821-finetune-20261024-191223` (Garnet) and
`20261024-211525-finetune-20261024-195414` (Emerald), and the test-batch runs `20261024-215152-finetune-20261024-191223`
and `20261024-220320-finetune-20261024-195414`. Run ids are the laptop's local time, US Eastern (EDT, UTC−4):
`20261024-215152` started at 01:51:54 UTC on 2026-10-25.

These four runs' `result.json` and `ler/` files are committed under `results_record/rnn_lstm/<run id>/`. The
fine-tuning runs' `cascade_history` entries show that τ and the prior were fixed on validation, last at 01:46–01:49
UTC, before B3 was opened at 01:51 UTC. `results_record/readme_b3_tables.py` rebuilds the three B3 tables above from
them (its docstring has the command).

Each run writes `results/<backend>/<run-id>/` (git-ignored) and rebuilds `RESULTS.md`. Garnet pre-training takes about
25 min per model on this laptop (9 of them training), fine-tuning a few minutes. Stage 0's MLP and MWPM baselines
(d = 3, 5, 7, simulated) are still reproducible with `configs/mlp_*.json` and `configs/mwpm_full.json`.

## Files

- `CONTRACT.md`: the rules (layout, API, data, evaluation, hybrid training).
- `RESULTS.md`: every number, generated.
- `NOTES.md`: notes for the team's `challenge/`.
- `docs/`: `hardware.md`, `rnn_lstm.md`, `rnn_tf.md`, `cascade.md`, `mlp.md`.
- Code:
  - `final.py`: the entry point;
  - `cascade.py`, `relay.py`: the cascade and Relay-BP;
  - `sequence.py`, `models/`: the decoders;
  - `hardware.py`: calibration packages, noise extension, InfoRecord loader;
  - `data.py`, `train.py`, `evaluate.py`, `run.py`, `finetune.py`, `report.py`;
  - `calibration/`: vendored InfoRecord packages;
  - `weights/`, `configs/`, `tests/`.

`challenge/` is the team's, read-only. AlphaQubit-lite (`models/rnn_tf.py`) and the other patches (pB, eB) are
implemented, but were left out of the final scope (Sal, 2026-10-25: proof of concept, LSTM, best patch).
