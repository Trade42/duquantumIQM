# ML telescoping: Emerald results and decoding time per round

The ML telescope uses a recurrent model for real-time surface-code decoding.
It replaces a lookup over every detector history with shared learned weights,
then sends uncertain predictions through additional inference stages.

## How the telescope works

A two-layer LSTM reads each round's detector events, existence masks and
no-reset round kind. Its learned state summarizes the history; the final head
combines that state with data-boundary events to predict the logical flip.

Confidence is `abs(2*p - 1)`. Predictions below the validation-selected threshold
go to Relay-BP. Correlated MWPM handles Relay-BP nonconvergence. On Emerald,
Z selects `tau=0`, while X selects `tau=0.19` and defers 23% of B3 test shots.

Each model has **243,714 parameters** and about **488 KiB of stored weights**.
The same weights handle the tested round counts from one to twenty. This removes
the exponential growth of lookup storage with history length: a table requires
`2^D` decisions for `D` detector bits, while the recurrent model reuses its
weights as successive rounds are processed.

## Comparison with vanilla Emerald

The reference is the [vanilla Emerald memory](EMERALD_BASELINE.md): a
distance-three patch with 17 qubits, literal native compilation, DD off and a
uniform-prior logical maximum-likelihood lookup at `p=0.003`.

Both columns use the challenge's one-to-three-round fit,
`P_L(R) = (1 - (1 - 2*epsilon)^R) / 2`. Vanilla contributes 9,000 shots per basis;
the ML telescope contributes 6,000 B3 test shots per basis.

| Memory | Vanilla LER, 95% profile interval | ML telescope LER, 95% profile interval | Relative reduction |
|---|---:|---:|---:|
| X | 12.352% (11.773–12.953%) | **2.179% (1.921–2.459%)** | **82.4%** |
| Z | 4.749% (4.424–5.089%) | **3.318% (2.994–3.665%)** | **30.1%** |

The measured failure counts behind those fits are:

| Rounds | Vanilla X | ML telescope X | Vanilla Z | ML telescope Z |
|---:|---:|---:|---:|---:|
| 1 | 341 / 3,000 | 19 / 2,000 | 46 / 3,000 | 17 / 2,000 |
| 2 | 681 / 3,000 | 74 / 2,000 | 301 / 3,000 | 162 / 2,000 |
| 3 | 859 / 3,000 | 161 / 2,000 | 456 / 3,000 | 202 / 2,000 |

At three rounds, X failure falls from **28.633% to 8.050%** and Z failure from
**15.200% to 10.100%**. The
[comparison record](../MLfolders/mlDecoder/results_record/emerald_vanilla_r1_r3.json)
stores the input hashes, counts, fits and calculated reductions.

## Decoding time per round

The main latency measure is **mean decoding cost per syndrome round**, calculated
as mean complete-history decoding time divided by the number of rounds. Recorded
CPU timings use a warmed Ryzen 9 8945HS with one thread and one shot at a time.

| Emerald memory | Rounds | LSTM cost per round | Full telescope cost per round |
|---|---:|---:|---:|
| X | 3 | 164.1 us | 204.8 us |
| Z | 3 | 245.0 us | 242.3 us |
| X | 20 | 38.2 us | 710.2 us |
| Z | 20 | 39.9 us | 38.9 us |

At twenty rounds, the first stage takes roughly **38–40 us per round**.
Confidence-based deferral adds work on difficult histories, particularly
Emerald X. The control-loop design therefore has two tuning targets: logical
error rate and the available time per syndrome round.

The comparison record retains the full-history mean, median and p99 timings
alongside the per-round calculation.

## Training and included evidence

Training begins with calibrated synthetic mixtures and fine-tunes on hardware.
Emerald uses 35,200 fine-tuning shots and 8,800 validation shots per basis.
Validation selects the model and telescope threshold; the time-separated B3
batch supplies 22,000 test shots per basis over eleven round counts:
1, 2, 3, 4, 5, 6, 8, 10, 12, 15 and 20.

The included work from `MLDecodeTest` commit `0383456` covers four frozen models
for Garnet and Emerald, with **154,000 B3 shots in 44 cases**. It includes raw
measurements, source circuits, calibration packages, training and validation
records, and the original evaluation.

- [Model package and usage](../MLfolders/mlDecoder/README.md)
- [Source provenance](../MLfolders/mlDecoder/UPSTREAM.md)
- [Emerald B3 evaluation](../MLfolders/mlDecoder/results_record/rnn_lstm/20261024-220320-finetune-20261024-195414/result.json)
- [Vanilla Emerald manifest](../benchmarks/emerald/vanilla/20261025T022556Z/manifest.json)
- [Frozen-model CPU replay](../MLfolders/mlDecoder/results_record/promoted_cpu_replay.json)

## Reproduce

Recompute the vanilla comparison and per-round timings:

```powershell
python -m MLfolders.mlDecoder.compare_emerald
python -m MLfolders.mlDecoder.compare_emerald --json
```

Audit the imported evidence and replay the frozen LSTM with the
[ML environment](../MLfolders/mlDecoder/README.md):

```powershell
python -m MLfolders.mlDecoder.evidence
python -m MLfolders.mlDecoder.evidence --decode
```
