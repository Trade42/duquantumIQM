# Why MWPM beat the LUT on three-round X memory

**No LUT implementation error was found.** An independent calculation reproduced
every decision in the three-round X table. MWPM's hardware advantage comes from
the noise model assigning the wrong conditional probabilities to some histories.

The vanilla LUT usually beats MWPM when both receive the same records and prior.
The large vanilla-to-improved gain combines circuit changes, calibration and
decoding.

The [hardware-calibrated LUT update](LUT_IMPROVEMENT.md) implements the correction
and includes frozen tables with lower fitted X/Z error rates on these records.

## Same-record comparisons

Re-decoding the vanilla records with correlated MWPM and the LUT's uniform
`p=0.003` circuit prior gives:

| Hardware records | Shots | LUT failures | MWPM failures |
|---|---:|---:|---:|
| Garnet vanilla, X/Z and R1–3 | 18,000 | **3,722** | 3,882 |
| Emerald vanilla, X/Z and R1–3 | 18,000 | **2,684** | 2,727 |

The LUT wins 10 of 12 individual cases. The two MWPM wins are Garnet X/R2 by
one shot and Emerald Z/R2 by eight shots. Garnet's pooled LUT advantage is
0.889 percentage points, with a paired 95% interval of 0.544–1.233 points.
Emerald's pooled advantage is 0.239 points, with an interval of −0.056–0.528.

On the improved Garnet circuits, using each frozen policy's same calibrated
DEM for both decoders:

| Memory | Rounds | Shots | Modeled ML LUT failures | Correlated MWPM failures |
|---|---:|---:|---:|---:|
| X | 1 | 4,000 | **150** | 159 |
| X | 2 | 4,000 | **400** | 404 |
| X | 3 | 4,000 | 789 | **728** |
| Z | 1 | 8,000 | 148 | 148 |
| Z | 2 | 8,000 | **643** | 1,037 |
| Z | 3 | 8,000 | **1,152** | 1,451 |

These Z LUT values use the modeled posterior. The final Z policy also learns
from independent training counts and scores 145, 639 and 1,147 failures.

## The three-round X result

MWPM wins by **61 failures out of 4,000 shots**, or **1.525 percentage points**.
The paired 95% interval is 0.575–2.450 points; exact McNemar p = 0.00137.
The decoders disagree on 353 shots: the LUT alone fails on 207, and MWPM alone
fails on 146.

Sampling the same calibrated detector error model reverses the result:

| Garnet X/R3 input | Shots | LUT failures | MWPM failures |
|---|---:|---:|---:|
| Recorded hardware | 4,000 | 789 | **728** |
| Independent-DEM simulation, seed 20261025 | 100,000 | **3,485** | 4,299 |

The LUT minimizes logical error under its supplied probability distribution.
It sums all modeled fault explanations for each logical class. MWPM uses a
different decision rule. When the hardware's conditional probabilities differ
from the model, MWPM can make the better choice for those histories.

On the 353 disputed hardware shots, the model assigns the LUT a mean error
probability of **34.55%**. Its measured error is **58.64%**. The probability model
therefore favors the logical class that loses on these records. Improving those
conditional probabilities is the useful next change.

## Implementation checks

- Rebuilt all twelve frozen vanilla tables: every table byte and every scored
  prediction matched.
- Compared the FWHT lookup with an independent probability-space convolution
  for all eight distinct R1/R2 models: zero decisions disagreed outside numerical
  ties, and each LUT achieved the independently calculated minimum error.
- Tested every fault combination in small random models, including probabilities
  0, 0.5, above 0.5 and 1; joint separator faults; repeated-block offsets;
  logical-class degeneracy; detector bit order; and save/load behavior.
- Independently convolved all 315 fault terms for Garnet X/R3 in probability
  space. Across all **33,554,432 joint detector/logical outcomes**, normalization
  was 0.9999999999999982 and the optimal modeled error was 3.4528255%. All
  **16,777,216 LUT decisions matched**, with zero changed hardware predictions.
- Simulated 100,000 identical shots per decoder for each of twelve distinct
  models, preserving joint DEM faults and using the recorded decomposition
  approximation.

The focused correctness suite contains **20 passing tests**. The audit changes
the diagnostic tools and tests; the published LUT implementation and frozen
policies retain their original bytes.

The observed advantage therefore gives a direction for future model work:
validate calibrated logical posteriors on independent training/validation records,
especially three-round X histories, before changing the inference rule.

## Reproduce

Run from the repository root with the core packages installed. Use a new output
directory for each run:

```powershell
python -m pytest -q tests/decoders/test_lookup_audit.py
python -m challenge.audit_lookup --output results/decoders/lookup_audit/NEW/report.json
python -m challenge.audit_lookup --direct-x-r3 --output results/decoders/lookup_audit/NEW/x_r3_direct.json
```

The reports contain model and record hashes, failure counts, paired intervals,
simulation seeds, versions, setup time, warmed batch decode time and LUT memory.
The full X/R3 direct check evaluates all **33,554,432 joint outcomes** using
bounded scratch memory and an independent XOR probability convolution.
