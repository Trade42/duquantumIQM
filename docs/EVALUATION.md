# Garnet evaluation against vanilla surface code

The experiment measures distance-three X/Z memory at one, two and three
syndrome rounds on IQM Garnet. The comparison contains 18,000 vanilla shots
and 36,000 optimized-pipeline shots under calibration
`a18a947f-0408-4628-81ea-1715637e434f`.

## Define the reference experiment

Both implementations encode one logical qubit in nine data qubits and eight
reused ancillas. Both use generated no-reset boundary checks, native couplings,
zero routing SWAPs and the same R1–3 fitting equation.

| Element | Vanilla surface code | Our pipeline |
|---|---|---|
| Placement | Deterministic legal embedding, without calibration ranking | Gate/readout calibration-ranked native embedding |
| Compilation | Literal canonical four-CX-layer schedule lowered through H-CZ-H; optimization level 0 | Native rotations/CZ with local cancellation and measurement-aware optimization |
| Readout schedule | Literal circuit boundaries | X terminal fusion; Z measurement fences |
| Idle control | DD off | Data-qubit XY4 |
| Noise prior | Uniform `p=0.003` | Native gate/readout calibration; X disturbance proxy; Z guarded correlation fitting |
| Decoder | Complete modeled logical-ML lookup | X correlated MWPM; Z posterior/count lookup with `alpha=10` |
| Training labels | Model-only construction | X calibrated graph; Z 6,000 training shots per round |
| Evaluation shots per round | 3,000 per basis | X 4,000; Z 8,000 |

The vanilla prior applies uniform H/CZ depolarization, initialization/readout
flips and pre-round data depolarization. Its full-history lookup sums the
modeled logical fault classes. This provides an encoded surface-code baseline
with a complete decoder. The comparison measures the combined effect of physical
compilation, idle control and inference.

Each arm has a fixed native embedding and a separate acquisition window under
the recorded calibration. Optimized measurements are stratified into two
acquisition blocks. Every returned shot contributes to the score.

## Measure decoded logical failures

A shot fails when the predicted reference-relative logical flip differs from
the observed flip. Failure probability at round count `r` is `P_L(r)=k_r/n_r`.

| Basis | Rounds | Vanilla failures / shots | Vanilla P_L | Optimized failures / shots | Optimized P_L |
|---|---:|---:|---:|---:|---:|
| X | 1 | 423 / 3,000 | 14.100% | 159 / 4,000 | 3.975% |
| X | 2 | 972 / 3,000 | 32.400% | 404 / 4,000 | 10.100% |
| X | 3 | 1,220 / 3,000 | 40.667% | 728 / 4,000 | 18.200% |
| Z | 1 | 115 / 3,000 | 3.833% | 145 / 8,000 | 1.8125% |
| Z | 2 | 409 / 3,000 | 13.633% | 639 / 8,000 | 7.9875% |
| Z | 3 | 583 / 3,000 | 19.433% | 1,147 / 8,000 | 14.3375% |

![Decoded Garnet failure probabilities and 95% Wilson intervals](assets/garnet_comparison.png)

At three rounds, failure probability falls by 55.2% in X and 26.2% in Z.
The figure's error bars are 95% Wilson intervals, calculated from each point's
failure count and shot count.

## Fit the challenge's per-round LER

The IQM challenge specifies

$$
P_L(r)=\frac{1-(1-2\epsilon_L)^r}{2}.
$$

For independent logical flips of probability `epsilon_L`, an odd number of
flips reverses the final observable. The expected parity after `r` rounds is
`(1-2 epsilon_L)^r`, which gives the equation. We maximize the joint binomial
likelihood for the R1/R2/R3 counts with amplitude fixed to one. Intervals use
the profile likelihood of this one-parameter fit.

| Basis | Vanilla LER, 95% profile interval | Optimized LER, 95% profile interval | Absolute reduction | Relative reduction |
|---|---:|---:|---:|---:|
| X | 18.517% (17.748–19.314%) | **5.808% (5.492–6.136%)** | 12.709 percentage points | **68.6%** |
| Z | 6.722% (6.326–7.135%) | **4.255% (4.066–4.449%)** | 2.467 percentage points | **36.7%** |

An independent binomial bootstrap refits the equation on 20,000 resamples,
stratified by round and acquisition block, using seed `20261025`. Bonferroni
adjustment over the two bases gives 97.5% individual comparison intervals:

| Basis | Absolute LER reduction | Adjusted interval |
|---|---:|---:|
| X | 12.709 percentage points | 11.791–13.653 points |
| Z | 2.467 percentage points | 1.963–2.974 points |

These intervals describe sampling variation in the recorded whole-pipeline
comparison. Acquisition-window drift and model mismatch are additional sources
of variation.

### Fit diagnostics

| Basis | Vanilla deviance / df | Optimized deviance / df |
|---|---:|---:|
| X | 61.03 / 2 | 52.10 / 2 |
| Z | 56.94 / 2 | 198.86 / 2 |

All four deviances exceed the 5% chi-square critical value of 5.99 for two
degrees of freedom. The fitted LER is the challenge's descriptive summary of
R1–3; the measured failure probabilities
retain the round dependence. Initialization, terminal readout and repeated-check
exposure contribute different circuit boundaries. [Error analysis](ERROR_ANALYSIS.md)
describes the remaining pathways.

## Reproduce the calculation

[`manifest.json`](../benchmarks/garnet/manifest.json) lists the raw records,
circuits, measurement maps, decoder states, calibration, versions and hashes.
[`evaluation.json`](../benchmarks/garnet/evaluation.json) contains all counts,
pointwise intervals, fit diagnostics and comparison intervals.

```powershell
python -m examples.reproduce
```

The evaluator reconstructs detector frames and predictions from all 54,000
measurements before calculating the table. [Reproduction](REPRODUCIBILITY.md)
describes the environment and API.
