# Results and interpretation

The retained policy is **native/XY4 + correlated MWPM for X**, and
**native/XY4 + the frozen posterior/count hybrid for Z**. Hardware acquisition
is complete. The later study did not qualify a replacement. All figures below
come from saved hardware data, not synthetic calibration or simulator samples.

## What the reported LER means

For a shot with detector history `s`, the decoder predicts the logical
observable flip relative to the ideal reference. A shot fails when that
prediction disagrees with the measured reference-normalized truth. Every shot
is included. At one circuit length, `failures/shots` estimates **P_L(r)**.

The challenge endpoint is a multi-round binomial maximum-likelihood fit:

$$
P_L(r)=\frac{1-(1-2\epsilon_L)^r}{2}.
$$

If each round independently flips a logical bit with the same probability
`epsilon_L`, the final bit is wrong after an odd number of flips. Its expected
parity is `(1-2 epsilon_L)^r`, which gives this equation. The primary fit fixes
amplitude to one and uses **R=1,2,3** throughout. A single R3 failure probability
is not substituted for LER, and an offset/SPAM fit is not selected afterward.

The primary window was declared before acquisition because the complete
selected lookup supports at most 24 detectors. The longer R1–7 follow-up is a
separate diagnostic window. No decoder-family switch is hidden inside a fit.
The challenge's endpoint and its literature source are recorded in the
[original brief](../challenge/README.md) and
[2021 cyclic-QEC paper](https://www.nature.com/articles/s41586-021-03588-y).

## Original selection study

Run: [`20261024T234900Z`](../results/decoders/hybrid/20261024T234900Z/).
It acquired 192,000 pilot shots plus 108,000 fresh confirmation shots, for
300,000 study shots in 60 completed jobs. Each pilot case had 6,000 training
and 2,000 outer-validation shots. The final candidate and standalone references
were frozen before confirmation. Shared physical arms reused identical shots
for paired decoder comparisons.

| Policy | R1 / R2 / R3 failures | Shots per point | Fitted LER | Conditional 95% profile interval |
|---|---|---:|---:|---:|
| Retained X correlated MWPM | 176 / 585 / 993 | 6,000 | **5.220%** | 4.976–5.471% |
| Retained Z posterior/count hybrid | 80 / 456 / 753 | 6,000 | **3.763%** | 3.559–3.974% |
| Z native model-only ML reference | 86 / 498 / 756 | 6,000 | 3.920% | 3.712–4.136% |

The Z gain is **0.157 percentage points**, approximately a 4.0% relative point
reduction against native model-only ML. Its paired, six-comparison-adjusted
interval is **+0.037 to +0.274 percentage points**. It beats the three tested
standalone references in Z. The change is concentrated at R2: 42 fewer failures
there, compared with 6 at R1 and 3 at R3. It is not a large R3 improvement.

The original X fitted-model ML candidate gave 5.065% versus retained MWPM's
5.220%, but the adjusted gain interval was −0.078 to +0.381 points. Its R3
failures increased from 993 to 1,006. A lower fitted point estimate alone did
not qualify it, so the X policy remained unchanged.

| Reconstructed standalone reference | X LER | Z LER |
|---|---:|---:|
| Native compiler / original decoder | 5.220% | 3.920% |
| Dara physical path / selected classical decoder | 7.387% | 6.759% |
| Iyanu physical path / selected coherence prior | 9.894% | 7.813% |

These are the tested configurations on the common fixed placement, calibration
and finite training budget. They do not establish optimality over every
historical configuration, every neural training budget or every possible
placement. The protocol/report is [Garnet headline selection](GARNET_HEADLINE.md);
the machine-readable result is [`results.json`](../results/decoders/hybrid/20261024T234900Z/results.json).

Two separate two-shot policy executions, four shots total, verified the adopted
live entry points afterward. They are recorded outside the 300,000-shot study
and are not an additional precision estimate or LER fit.

## Bounded follow-up and later validation

Run: [`20261025T004600Z`](../results/decoders/history/20261025T004600Z/).
It acquired **96,000 new physical shots**, all returned in 82 completed jobs:
32,000 pilot shots and two successive 32,000-shot confirmation windows. There
were no retries or additional smoke jobs. Archived evidence could now train
new candidates; it was never counted as fresh confirmation.

The follow-up tested retained, centered and adaptive DD, plus fast ancilla
readout, guarded DEM fitting, posterior updates and a measured-history residual
GRU. The final pilot tie break preferred fitted MWPM over an exactly tied GRU.
Before any confirmation attempt, six component contrasts were frozen so a
combination had to beat both components as well as the released policy.
The original pilot rule/source revisions are retained alongside the final ones.

| X policy | R1 / R2 / R3 failures | Shots per point | Fitted LER |
|---|---|---:|---:|
| Retained native XY4 + correlated MWPM | 159 / 404 / 728 | 4,000 | **5.808%** |
| Adaptive DD + fitted MWPM | 141 / 446 / 757 | 4,000 | 6.066% |
| Adaptive DD alone | 142 / 480 / 757 | 4,000 | 6.239% |
| Fitted MWPM alone | 141 / 404 / 728 | 4,000 | 5.719% |

The combined candidate's reduction versus release is −0.2578 percentage points,
adjusted interval −0.8364 to +0.3534. Fitting alone reduces the point estimate
by 0.0886 points, adjusted interval −0.0148 to +0.1976. Neither meets the frozen
rule. The fitting-only benefit consists of fewer R1 failures; R2/R3 are unchanged.

| Retained later policy | R1 / R2 / R3 failures | Shots per point | Fitted LER | Conditional 95% profile interval |
|---|---|---:|---:|---:|
| X | 159 / 404 / 728 | 4,000 | **5.808%** | 5.492–6.136% |
| Z | 145 / 639 / 1,147 | 8,000 | **4.255%** | 4.066–4.449% |

Z's selected candidate was the unchanged released policy, so identical physical
arms shared the allocation. The descriptive recurrent rates on the retained
physical circuit were X 5.792% and Z 4.705%; they did not establish a better
project policy. Fast readout and centered XY4 did not win the pilot, so no
independent-confirmation superiority or inferiority is claimed for those paths.

The later values are reported as a new acquisition, not pooled with the earlier
study or substituted only when lower. Both used calibration
`a18a947f-0408-4628-81ea-1715637e434f`; the follow-up's two windows were roughly
00:58–01:03 and 01:03–01:08 UTC on 2026-10-25. They are separate acquisition
windows under one calibration, not a multi-day or multi-calibration stability test.

See [bounded-study protocol](GARNET_ROUNDS.md),
[`report.json`](../results/decoders/history/20261025T004600Z/report.json) and
[`EVIDENCE.md`](../results/decoders/history/20261025T004600Z/EVIDENCE.md).

## Uncertainty, multiplicity and lack of fit

Fixed-shot point failure probabilities have Wilson intervals. Fitted LERs have
asymptotic profile-likelihood intervals. Their 95% level is different from the
multiplicity-adjusted **comparison** intervals used for promotion.

Each study's six predeclared comparisons use Bonferroni adjustment, giving
99.1667% individual comparison intervals for 95% family coverage. Common-shot
decoder comparisons resample joint failure pairs; different physical arms use
independent samples. Resampling is stratified by round and acquisition block,
and the same challenge equation is refit on each draw.

The stationary model is rejected for the retained policies in both studies:

| Study | X deviance / df | Z deviance / df |
|---|---:|---:|
| Original selection | 103.80 / 2 | 156.79 / 2 |
| Later validation | 52.10 / 2 | 198.86 / 2 |

Consequently, fitted LERs are **descriptive challenge-equation estimates**, not
established intrinsic stationary error rates. A narrow profile interval does
not repair this lack of fit. The bootstrap compares the declared descriptive
endpoint; it does not explain the underlying physical nonstationarity or cover
all future drift. [Error analysis](ERROR_ANALYSIS.md) separates observations
from causal hypotheses.

## Reproducibility and claim boundaries

The frozen evidence retains source/native/decoder circuits, original-order
raw records, measurement maps, calibrated/assumed priors, versions, seeds,
trained states, failures, intervals and pulse audits. The follow-up full replay
checks 82 jobs, 96,000 shots, 138 prediction groups, 20 fits and all six adjusted
comparisons, with networking disabled. The small winner example replays the
36,000 later confirmation shots belonging to the retained policies alone.

The release claim covers this distance-three static memory and R1–3 policy.
It does not establish below-threshold scaling, unencoded break-even, physical
leakage removal, a real-time decoder or universal logical computation. R1–7
matching/recurrent diagnostics are not a silent extension of the R1–3 Z lookup.
Simulation and research backends are labeled separately in [Techniques](TECHNIQUES.md).
