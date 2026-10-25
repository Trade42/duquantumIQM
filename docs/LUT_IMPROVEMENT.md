# Hardware-calibrated logical lookup

The calibrated LUT learns corrections to the logical probabilities that the
native noise model gets wrong. It compiles those corrections into a complete
decision table, so inference remains detector-bit packing and one table read.

## Results

Re-decoding the same 36,000 optimized Garnet benchmark shots gives:

| Memory | Original modeled LUT LER | Calibrated LUT LER | Relative reduction |
|---|---:|---:|---:|
| X | 6.039% | **5.620%** | **6.9%** |
| Z | 4.283% | **4.162%** | **2.8%** |

Paired 95% intervals for the absolute LER reduction are **0.219–0.624 percentage
points in X** and **0.025–0.217 points in Z**, stratified by round and acquisition
block with 10,000 bootstrap draws, seed 20261025.

| Memory | Rounds | Shots | Original LUT failures | Calibrated LUT failures | MWPM failures |
|---|---:|---:|---:|---:|---:|
| X | 1 | 4,000 | 150 | **121** | 159 |
| X | 2 | 4,000 | 400 | **380** | 404 |
| X | 3 | 4,000 | 789 | **752** | 728 |
| Z | 1 | 8,000 | 148 | 148 | 148 |
| Z | 2 | 8,000 | 643 | **634** | 1,037 |
| Z | 3 | 8,000 | 1,152 | **1,109** | 1,451 |

The new fitted X rate is also below MWPM's 5.808%; MWPM retains the lower error
count at the three-round X point. The published Z posterior/count policy has a
4.255% fitted rate. Paired intervals for the new LUT's improvement over those
published X/Z policies are −0.022–0.399 and −0.003–0.190 percentage points.

The [frozen policy bundle](../benchmarks/calibrated_lookup/) includes all six
tables, the [selection records](../benchmarks/calibrated_lookup/frozen_selection.json),
[comparison report](../benchmarks/calibrated_lookup/report.json) and
[timing record](../benchmarks/calibrated_lookup/latency.json), plus
[source hashes and versions](../benchmarks/calibrated_lookup/reproducibility.json). It occupies 3.6 MB
compressed. The three-round tables each take about 200 seconds to construct on
the recorded CPU; construction includes compiling matching predictions for every
history. Warmed batch-one median cost per round was 2.7 us for X/R3 and 7.8 us
for Z/R3, measured as whole-history decoding time divided by three.

## What changes

The original LUT compares the two modeled logical probabilities for a detector
history. The new builder starts with that posterior and learns a regularized
logistic correction from hardware records. Its features are the prior log odds,
a correlated matching prediction, individual detector events and optional
detector-pair interactions. A history-count update can further correct common
syndromes.

Validation selects the correction order, regularization and count strength for
each basis and round count. The builder then evaluates the selected rule for
every history. Matching runs during construction; the saved decoder contains
the resulting byte table and uses measured detection events at inference.

## Training and validation

Each X/Z and R1–3 policy uses **9,000 training shots** and **5,800 validation
shots**, recovered from the original Garnet development records:

- Training: the first 6,000 original pilot shots and 3,000 earlier block-zero
  confirmation shots.
- Validation: the remaining 2,000 pilot shots, 3,000 earlier block-one
  confirmation shots and 800 later pilot shots.

The loader checks acquisition IDs, calibration, source/native circuits, raw
measurement conversion and record hashes. The published confirmation jobs are
reserved for scoring. All six policies are saved and their hashes sealed before
that scoring begins.

## Use the decoder

Load the included corrected policy directly:

```python
from challenge.decoders.calibrated_lookup import load_policy

decoder = load_policy("X", rounds=3)
predictions = decoder.decode_batch(det_events)
```

Build and evaluate a new set of policies from a full Git checkout:

```powershell
python -m challenge.improve_lookup build --run results/decoders/calibrated_lookup/NEW
```

Decode detector-event rows with a saved policy:

```powershell
python -m challenge.improve_lookup decode --policy results/decoders/calibrated_lookup/NEW/X_r3.npz --dets det_events.npy --output logical_flips.npy
```

```python
from challenge.decoders.calibrated_lookup import load

decoder = load("results/decoders/calibrated_lookup/NEW/X_r3.npz")
predictions = decoder.decode_batch(det_events)
```

Inputs have shape `(shots, 8*rounds)` and outputs `(shots, 1)`, in the same shot
order. The circuit's detector definitions and logical reference match the
selected policy. R1/R2/R3 decision tables occupy **256 B / 64 KiB / 16 MiB**.
Construction cost is paid once; the fitted model and matching graph are compiled
into the table.

The run directory records the fixed protocol, all validation candidates, selected
coefficients, training provenance, table hashes, paired LER comparisons and warmed
decode timings. The [earlier audit](LUT_MWPM_AUDIT.md) establishes the correctness
of the underlying probability calculation.
