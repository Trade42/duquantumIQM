# Garnet challenge LER comparison

All shots retained. Frozen pilot choices; independent interleaved confirmation blocks.
Primary equation: `P_L(r) = [1-(1-2 epsilon_L)^r]/2`, rounds 1, 2, 3; amplitude fixed at 1.

| Basis | Arm | LER / round (95% profile CI) | Errors at R1/R2/R3 | Fit deviance / df |
|---|---|---|---|---|
| X | candidate | 5.065% (4.825–5.312%) | [143, 557, 1006] / [6000, 6000, 6000] | 156.27 / 2 |
| X | dara | 7.387% (7.088–7.694%) | [542, 752, 1111] / [6000, 6000, 6000] | 30.29 / 2 |
| X | ethan | 5.220% (4.976–5.471%) | [176, 585, 993] / [6000, 6000, 6000] | 103.80 / 2 |
| X | iyanu | 9.894% (9.538–10.261%) | [646, 1040, 1427] / [6000, 6000, 6000] | 6.57 / 2 |
| Z | candidate | 3.763% (3.559–3.974%) | [80, 456, 753] / [6000, 6000, 6000] | 156.79 / 2 |
| Z | dara | 6.759% (6.476–7.050%) | [338, 771, 1114] / [6000, 6000, 6000] | 16.42 / 2 |
| Z | ethan | 3.920% (3.712–4.136%) | [86, 498, 756] / [6000, 6000, 6000] | 152.01 / 2 |
| Z | iyanu | 7.813% (7.505–8.130%) | [401, 900, 1232] / [6000, 6000, 6000] | 14.03 / 2 |

Improvement over every standalone in both bases established: **False**.

Positive differences favor candidate; intervals account for the six predeclared comparisons.
- X vs ethan: 0.155 percentage points; CI -0.078–0.381; paired=True.
- X vs dara: 2.322 percentage points; CI 1.803–2.837; paired=False.
- X vs iyanu: 4.829 percentage points; CI 4.254–5.406; paired=False.
- Z vs ethan: 0.157 percentage points; CI 0.037–0.274; paired=True.
- Z vs dara: 2.996 percentage points; CI 2.524–3.476; paired=False.
- Z vs iyanu: 4.051 percentage points; CI 3.548–4.553; paired=False.

Adopted policy (predeclared promotion/fallback rule):
- X: ethan / mwpm; promoted=False.
- Z: ethan / posterior_10; promoted=True.

- Distance 3, one Garnet calibration; no distance-scaling or bare-qubit claim
- R1-3 fixed before acquisition because released exact LUT supports <=24 detectors
- Rejected stationary fits remain descriptive challenge-equation estimates
- Dara uses its median/assumed logical prior; native profiles use measured native-gate priors
- No postselection, hidden leakage flags, or confirmation-label training
