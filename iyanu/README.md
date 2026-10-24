# iyanu: GPU decoders, guarded IQM hardware runs, Garnet/Emerald results

d=3 rotated surface-code memory without reset (the repo's `challenge.surface_code` circuits), run on
IQM Garnet and Emerald through IQM Resonance, decoded offline by classical and GPU-trained decoders.

## Modules

| File | What |
|---|---|
| `neural_gpu.py` | MLP decoder (detection events → logical flip), CUDA / Apple MPS / CPU; timing and paired-comparison helpers |
| `conv_decoders.py` | 3D space-time convolutional decoder, and a convolutional pre-decoder that applies confident local DEM edges before PyMatching |
| `ldpc_decoders.py` | Union-Find, BP+OSD, BP+Union-Find (ldpc 2.x) on the Stim detector error model |
| `compare_decoders.py` | simulation benchmark of all decoders on the same shots at d = 3, 5, 7 |
| `garnet_sim.py` | offline Garnet rehearsal: IQMFakeGarnet layout/noise, Stim and Aer shots, network blocked |
| `qpu.py` | the guarded path to Resonance: read-only backend; `submit` refuses unless `DQ_QPU=run` and within `DQ_QPU_MAX_SHOTS`, checks every CZ in the serialized request against the couplers and the placement, logs the job id before waiting, saves raw data |
| `hw_run.py` | `prepare` (read-only), `run` (submit; `--dd`, `--heralding`, `--active-reset`), `decode` |
| `finetune_hw.py` | decoders pre-trained on simulation, fine-tuned on hardware training shots, scored on held-out hardware test shots |

Needs PyTorch (and `ldpc` for `ldpc_decoders.py`) on top of the package: `pip install -e '.[test,qiskit,hardware]' torch ldpc`.

```
python -m pytest -q iyanu                                                    # 16 tests
python -m iyanu.hw_run prepare --device garnet --rounds 1 2 3 4 --bases Z X   # read-only
DQ_QPU=run DQ_QPU_MAX_SHOTS=1000 python -m iyanu.hw_run run --device garnet --rounds 2 --shots 1000 --task pilot
python -m iyanu.hw_run decode iyanu/hw_data/<date>-<task>/<job id>
python -m iyanu.finetune_hw --train iyanu/hw_data/*-garnet-train2/*/ --test iyanu/hw_data/*-garnet-test/*/ --device cuda
```

The IQM token is read from `IQM_TOKEN` or the repo's `.env` (git-ignored) and is never printed.

## Hardware data (`hw_data/`)

One directory per job, `<YYYYMMDD>-<task>/<job id>/`: `meta.json` (device, calibration set, options, qubit
map per circuit, qubits sent, shots requested and returned), `calibration.json` (the live calibration
used), one `.stim` circuit per round count, and `measurements.npz` (per-circuit shot arrays in Stim
measurement order, rebuilt from the returned counts, so shot order is not preserved). `counts.json` is
kept locally only (git-ignored; same data). `ledger.jsonl` lists every submitted job; `NOTES.txt`
records two jobs IQM declined before execution (over the 20,000-shot limit).

All Garnet jobs on 2026-10-24 used calibration set `a18a947f-0408-4628-81ea-1715637e434f`; the placement
is the only 17-qubit footprint on Garnet (QB3, QB17, QB18 unused), 0 SWAPs, every CZ checked in the request.

## Results so far (Garnet, d=3, no reset, 20,000 shots per point, calibrated PyMatching unless stated)

eps_L is the per-round logical error rate fitted over rounds 1-4 with 2 P_L = 1 - (1 - 2 eps_L)^r, 95% profile interval.

| Hardware setting | Z eps_L | X eps_L | Shots kept |
|---|---|---|---|
| baseline (bracketing runs) | 0.068 to 0.078 | 0.159 to 0.251 (drifts) | 100% |
| dynamical decoupling | worse (+30%) | **0.0891 [0.0876, 0.0906]**, about -44% vs bracketing baselines | 100% |
| zeros heralding | **0.0629 [0.0616, 0.0642]**, about -7.5% | no gain with DD | about 84% (post-selected) |
| active reset (2 cycles) | much worse | much worse | 100% |

Decoders on the same held-out baseline test shots (job `01a16ffd-9995...`): coherence-aware calibrated
PyMatching (T1/T2 idles, readout duration chosen on training shots) improves eps_L by 3-5% over the Pauli
calibration model; GPU decoders (MLP and 3D conv) pre-trained on the coherence-aware model and fine-tuned
on separate Garnet training shots lead at most round counts. Full tables come from `finetune_hw.py`.

Simulation (uniform noise, rounds = d): PyMatching scales with distance (P_L 0.030, 0.024, 0.016 at
d = 3, 5, 7, p = 0.003); the MLP beats PyMatching at d = 3 but the learned decoders fall behind it at
d >= 5 with this training budget. The pre-decoder leaves 95.5% of d = 3 shots with no matching to do.
