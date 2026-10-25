# iyanu results: GPU decoders on Emerald and Garnet, compiled to nanosecond decoding

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

## Hardware data (`../archive/iyanu/hw_data/`)

One directory per job, `<YYYYMMDD>-<task>/<job id>/`: `meta.json` (device, calibration set, options, qubit
map per circuit, qubits sent, shots requested and returned), `calibration.json` (the live calibration
used), one `.stim` circuit per round count, and `measurements.npz` (per-circuit shot arrays in Stim
measurement order, rebuilt from the returned counts, so shot order is not preserved). `counts.json` is
kept locally only (git-ignored; same data). `ledger.jsonl` lists every submitted job; `NOTES.txt`
records two jobs IQM declined before execution (over the 20,000-shot limit).

All Garnet jobs on 2026-10-24 used calibration set `a18a947f-0408-4628-81ea-1715637e434f`; the placement
is the only 17-qubit footprint on Garnet (QB3, QB17, QB18 unused), 0 SWAPs, every CZ checked in the request.

## Final results (headline)

d=3 rotated surface code, no reset, on real IQM hardware; GPU decoders (MLP + 3D conv, ensembled) pre-trained on a
coherence-aware calibrated simulation and fine-tuned on separate hardware training jobs (9-10 jobs, 180-200k shots per
round count), scored once on a held-out test job (20,000 shots per round count, no post-selection). eps_L is the
per-round logical error rate fitted over rounds 1-4 with 2 P_L = 1 - (1 - 2 eps_L)^r, 95% profile interval.
Configuration: **Z memory on the vanilla circuit** (canonical Stim schedule, literal H-CZ-H, IQM scheduling;
`circuits.py`), **X memory on our circuit with dynamical decoupling**.

| Device | Basis | **GPU ensemble eps_L** | Calibrated PyMatching | Lookup-table baseline* | P_L at 1 / 2 / 3 / 4 rounds (GPU) |
|---|---|---|---|---|---|
| **Emerald** | **Z** | **0.0400 [0.0391, 0.0410]** | 0.0455 | 0.0672 | 1.73% / 7.27% / 11.96% / 16.05% |
| **Emerald** | **X** | **0.0620 [0.0608, 0.0632]** | 0.0717 | 0.1852 | 6.64% / 11.63% / 15.62% / 20.87% |
| Garnet | Z | 0.0545 [0.0533, 0.0556] | 0.0617 | 0.0672 | 2.31% / 8.27% / 15.44% / 22.98% |
| Garnet | X | 0.0867 [0.0852, 0.0882] | 0.0916 | 0.1852 | 9.95% / 14.89% / 21.96% / 26.05% |

*Team baseline (`ethan/docs/VANILLA_LUT_BASELINE.md`): vanilla circuit, DD off, uniform full-history lookup table,
Garnet, 3,000 shots per point, rounds 1-3. On the same test shots the GPU ensemble beats calibrated PyMatching at
every round count in all four rows except Emerald Z at 1 round (22 vs 35 one-sided failures, p = 0.1);
exact McNemar p from 1e-4 to 1e-37 elsewhere. Sources: `../archive/iyanu/hw_data/decoder_eval_final_emerald-20261025T013312.json`,
`../archive/iyanu/hw_data/decoder_eval_final_garnet-20261025T010252.json` and the matching `finetune_final_*.log`.

What moved the numbers (each measured against bracketing baselines on the same calibration set):
- **Circuit construction (Z):** the vanilla H-CZ-H construction cuts 1-round Z failures from about 6% to 1.8-2.6%
  versus the repo's H-cancelled/ASAP circuit; our shallower circuit stays better for X.
- **Dynamical decoupling (X):** -31% (Emerald) to -44% (Garnet) X eps_L; it hurts Z, so it is X-only.
- **Device:** Emerald (readout 0.3%, T2 about 15 us) beats Garnet (readout about 1%, T2 about 9 us) in both bases.
- **GPU decoders:** -5% to -17% eps_L versus calibrated PyMatching; fine-tuning on hardware shots is what makes the
  difference (simulation-only networks tie or lose). Heralding helped Garnet Z (-7.5%, 84% kept) but not Emerald;
  active reset hurt everywhere.

## Decoding latency: the GPU decoder compiled to nanoseconds

The fine-tuned GPU ensemble is compiled (`compiled.py`, run inside `finetune_hw.py --export`) into a lookup table
of its own answers, built on the GPU: complete (2^n bits) for up to 24 detectors (rounds 1-3, 2 MiB at 3 rounds),
and a hash table for 32 detectors (4 rounds) holding every training syndrome plus 20M simulated ones, with
the fine-tuned MLP as fallback for syndromes not in the table. Test shots never build a table. The C decoder
(`latency/decode_bench.c`: one bit read, or one hash probe plus an Accelerate/BLAS MLP with flush-to-zero)
decodes the held-out hardware test shots one at a time on an Apple M3; per-shot latency is measured as a
256-step dependency chain (each key depends on the previous answer), so it is latency, not throughput.

| Circuits | Median | p99 | **p99.9** | max | Accuracy vs GPU ensemble |
|---|---|---|---|---|---|
| Rounds 1-3, all 12 (Emerald/Garnet, X/Z) | **~1 ns** | ~1 ns | **1.1-1.3 ns** | < 35 ns | identical or within 1 shot in 20,000 |
| Emerald Z, 4 rounds (hash, 76% hit) | **4.4 ns** | 4.6 us | **6.7 us** | 28 us | +0.4% errors (fallback) |
| Emerald X, 4 rounds (63% hit) | **5.7 ns** | 4.6 us | **7.5 us** | 47 us | +0.4% |
| Garnet X, 4 rounds (62% hit) | **8.1 ns** | 4.7 us | **10.4 us** | 32 us | +0.4% |
| Garnet Z, 4 rounds (46% hit) | 3.6 us | 5.6 us | 13.8 us | 36 us | +0.4% |

LER of the compiled decoder (all rounds): Emerald Z 0.0404, X 0.0625; Garnet Z 0.0550, X 0.0873 (GPU ensemble:
0.0400 / 0.0620 / 0.0545 / 0.0867). Raw benchmark output: `latency/latency_m3.jsonl`; evaluations with compiled
errors and hit rates: `../archive/iyanu/hw_data/decoder_eval_lat_*.json`. For reference, the same network run directly takes about
25 us per shot in PyTorch on CPU and 390 us on the M3 GPU; PyMatching about 2.7 us. The 4-round fallback (a miss
rate set by how far hardware syndromes are from the simulated ones) is the remaining gap; a complete 2^32-bit
table (512 MiB) would put 4 rounds at ns at every percentile.

Reproduce: `python -m iyanu.finetune_hw ... --export DIR --device-tag emerald` (GPU), then
`python -m iyanu.latency.prepare DIR/emerald_Z_r3_compiled.npz <test job dir> Z 3 OUT` and
`cc -O3 -mcpu=native -o decode_bench iyanu/latency/decode_bench.c -framework Accelerate && ./decode_bench OUT/table.bin OUT/shots.bin`.
Compiled tables and weights (about 50 MB) are not committed; they are regenerated by the export run.

## Earlier results (Garnet, d=3, no reset, 20,000 shots per point, calibrated PyMatching unless stated)

eps_L is the per-round logical error rate fitted over rounds 1-4 with 2 P_L = 1 - (1 - 2 eps_L)^r, 95% profile interval.

| Hardware setting | Z eps_L | X eps_L | Shots kept |
|---|---|---|---|
| baseline (bracketing runs) | 0.068 to 0.078 | 0.159 to 0.251 (drifts) | 100% |
| dynamical decoupling | worse (+30%) | **0.0891 [0.0876, 0.0906]**, about -44% vs bracketing baselines | 100% |
| zeros heralding | **0.0629 [0.0616, 0.0642]**, about -7.5% | no gain with DD | about 84% (post-selected) |
| active reset (2 cycles) | much worse | much worse | 100% |

### GPU decoders on Garnet hardware shots

Held-out test job `01a16ffd-9995-728f-9388-fe89b1376a5b` (baseline settings, 20,000 shots per round count);
training: 10 separate jobs (task `garnet-train2`, 200,000 shots per round count, 90% fit / 10% validation).
`_sim` = trained only on the coherence-aware calibrated simulation; `_ft` = then fine-tuned on Garnet training
shots (checkpoint chosen on validation). Source: `../archive/iyanu/hw_data/decoder_eval_garnet-20261024T233359.json`,
log `../archive/iyanu/hw_data/finetune_garnet.log` (Colab NVIDIA L4).

| Decoder | Z eps_L (95%) | X eps_L (95%) |
|---|---|---|
| **3D conv, fine-tuned (GPU)** | **0.0653 [0.0641, 0.0666]** | **0.1970 [0.1940, 0.2001]** |
| **MLP, fine-tuned (GPU)** | 0.0659 [0.0646, 0.0672] | 0.1978 [0.1947, 0.2009] |
| MLP, simulation only (GPU) | 0.0673 [0.0660, 0.0686] | 0.2101 [0.2069, 0.2133] |
| PyMatching, coherence-aware calibration | 0.0686 [0.0673, 0.0699] | 0.2040 [0.2009, 0.2072] |
| conv pre-decoder + PyMatching | 0.0699 [0.0686, 0.0712] | 0.2034 [0.2003, 0.2066] |
| PyMatching, calibrated (Pauli) | 0.0707 [0.0694, 0.0720] | 0.2137 [0.2104, 0.2170] |
| lookup table from Garnet training shots | 0.0719 [0.0706, 0.0733] | 0.2065 [0.2034, 0.2098] |
| BP+OSD | 0.0722 [0.0709, 0.0736] | 0.2044 [0.2013, 0.2076] |
| PyMatching, uniform | 0.0745 [0.0732, 0.0759] | 0.2130 [0.2097, 0.2163] |
| lookup table from uniform simulation | 0.0752 [0.0739, 0.0766] | 0.2142 [0.2109, 0.2175] |
| BP+Union-Find | 0.0797 [0.0782, 0.0811] | 0.2041 [0.2010, 0.2073] |
| Union-Find | 0.1027 [0.1010, 0.1044] | 0.2544 [0.2504, 0.2584] |

The fine-tuned conv decoder lowers eps_L by 7.6% (Z) and 7.8% (X) relative to calibrated PyMatching; on the
same shots it fails less at 2, 3 and 4 rounds in both bases (exact McNemar p from 1e-4 to 1e-14), and ties
at 1 round. These are baseline-setting shots; the DD (X) and heralded (Z) runs are separate datasets.

Simulation (uniform noise, rounds = d): PyMatching scales with distance (P_L 0.030, 0.024, 0.016 at
d = 3, 5, 7, p = 0.003); the MLP beats PyMatching at d = 3 but the learned decoders fall behind it at
d >= 5 with this training budget. The pre-decoder leaves 95.5% of d = 3 shots with no matching to do.
