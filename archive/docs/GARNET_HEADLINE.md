# Garnet headline selection

> Completed acquisition study. Live commands below are historical protocol
> documentation. For the retained winners and credential-free use, see
> [winning pipeline](WINNING_PIPELINE.md), [results](RESULTS.md) and
> [offline reproduction](REPRODUCIBILITY.md).

This study optimizes the IQM challenge's **fitted per-round logical error rate**:

`P_L(r) = [1 - (1 - 2 epsilon_L)^r] / 2`.

The amplitude is fixed at one. The primary round counts are **1, 2, 3**, chosen
before acquisition because the released complete lookup decoder supports at
most 24 detector bits. Every primary comparison uses these same counts. A
single R3 failure probability, an offset/SPAM fit, postselected fidelity and a
simulation result are not substituted for this endpoint. The report retains
failure counts, Wilson intervals, profile intervals and binomial fit deviance.
A rejected stationary fit remains a descriptive challenge-equation estimate.

## What is combined, and why

The experiment separates physical compilation from classical decoding:

| Physical path | Implementation |
|---|---|
| Ethan | Existing native optimization, X terminal fusion / Z measurement fences, data-qubit XY4, PulLA pulse audit |
| Dara | Existing `sqrt_y_form` and measurement barriers, default server DD |
| Iyanu | Calibrated native literal-H circuit, default server DD |
| Physical combination | Dara sqrtY followed by Ethan terminal/fenced optimization and data-qubit XY4 |

All use distance-three no-reset X/Z memory, the same fixed native placement
and calibration, no routing SWAPs, no logical workload, and every returned shot.
Source and actual native Clifford circuits are saved. The native measurement
permutation must reproduce both source detector events and observable flips on
the *same actual hardware records*, including the ideal reference frame.

The decoder search reuses existing implementations: correlated PyMatching,
Dara training-only DEM fitting, the 21-member matching ensemble, Ethan's exact
modeled ML lookup, and empirical tables with explicit matching fallback.
Minimum empirical counts are 1, 3, 10 or 30. Joint DEM mechanisms and the
explicit disjoint-channel approximation are preserved. There is no truth-label
input to inference, hidden leakage input, or unresolved-shot exclusion.

A bounded second pilot stage evaluates Iyanu's MLP and convolutional decoder
on Iyanu's physical path and on the leading physical path in each basis.
This is a finite training-budget comparison, not a guarantee over every neural
architecture or training schedule. The original classical selection is retained
separately. Simulated pretraining uses independent Stim shots; hardware
fine-tuning/checkpoint selection uses only the pilot training portion. The
remaining pilot validation portion chooses the final candidate.

The MLP uses three 256-unit hidden layers and 1,000 pretraining batches of
1,024 simulated shots. The bounded convolutional trial uses two 8-channel
layers, a 64-unit head, and 1,000 batches of 256 simulated shots. Both then
fine-tune for 600 batches on 4,800 hardware training shots, choosing checkpoints
on the other 1,200 training shots. The separate 2,000 validation shots are used
only for the outer pipeline selection. Both run on CPU; no GPU speed claim is
made. These finite budgets do not establish optimality over all neural models.

The targeted posterior combination implements Dara's existing Bayesian rule
`(training_flips + alpha * P_model(flip | syndrome)) / (training_count + alpha)`.
It uses Ethan's exact short-history joint-DEM XOR/FWHT calculation, with prior
strengths 1, 10 or 100. A separated hyperedge remains one joint fault; repeated
detector/observable targets cancel by XOR. Unseen records retain model ML.
Additional Iyanu coherence-aware matching chooses readout duration from
300, 800, 1,500 or 3,000 ns using training observations only. These stages are
archived independently, and finish before any confirmation submission.

## Selection and confirmation

The first acquisition contains 8,000 shots per physical path, basis and round:
6,000 training plus 2,000 validation, **192,000 real Garnet shots** in 24 jobs.
One physical path and one decoder family are chosen per basis using minimum
validation challenge LER across all three round counts. There is no decoder
switch inside a fit.

The literal released Ethan decoder is retained as a reference: X native MWPM,
Z native exact ML. Dara chooses its best classical validation decoder; Iyanu
chooses among its calibrated/fitted matching, empirical, and evaluated neural
paths. Hardware priors are identified explicitly: Dara uses median calibration
and assumed durations/readout; Ethan's native path uses measured per-component
gate/readout calibration and frozen residual-noise parameters.

The adopted policy is also fixed by a rule before confirmation: promote a new
candidate per basis only if its adjusted LER reduction is positive against all
three references; otherwise retain the standalone chosen best on pilot data.
Confirmation labels cannot choose a new fallback or reopen pilot tuning.

After the final selection is frozen and hashed, confirmation obtains 6,000
fresh shots per distinct selected physical path, basis and round, in two
interleaved acquisition blocks. Shared physical paths reuse the same shots for
paired decoder comparisons. Different physical paths use independent shots.
No confirmation label trains or selects any decoder.

Candidate-minus-reference evidence is computed by resampling failure vectors
within each round/block and refitting the **same challenge equation** on every
bootstrap draw. Same-shot comparisons resample joint failure outcomes. The six
predeclared comparisons use Bonferroni-adjusted intervals with 95% family
confidence. The report explicitly states whether improvement over all three
standalones is established in both bases. Null or negative results do not
promote a combination.

## Run and replay

Install the root project and the isolated `ethan/` package in an environment
matching the existing hardware stack. Neural evaluation additionally needs
PyTorch; optional libraries may be installed in a separate directory passed
with `--runtime`. No dependency pins or shared interpreter packages are changed.

```powershell
python -m challenge.benchmark_headline --run results/decoders/hybrid/NEW_RUN --submit --phase pilot --token-file apikey.jom
python -m challenge.benchmark_headline --run results/decoders/hybrid/NEW_RUN --extend-neural
python -m challenge.benchmark_headline --run results/decoders/hybrid/NEW_RUN --extend-posterior
python -m challenge.benchmark_headline --run results/decoders/hybrid/NEW_RUN --extend-coherence
python -m challenge.benchmark_headline --run results/decoders/hybrid/NEW_RUN --submit --phase confirmation --token-file apikey.jom
```

`--resume` retrieves ledgered job IDs and never submits. It returns after a
bounded polling window if jobs remain; repeat with the same run directory and
phase. `--analyze --phase confirmation` replays frozen decoders and saved shots
offline. Imports and tests never read credentials or contact a QPU. An existing
phase cannot be resubmitted. All credentials remain outside saved artifacts.

Artifacts live in `results/decoders/hybrid/20261024T234900Z/`. The manifest
contains original-order raw records, circuit/shot hashes, native record maps,
calibration, versions, job IDs, pulse audits, source snapshots, decoder states
and the exact pilot/confirmation distinction. `report.md`, `results.json` and
`ler.png` are generated from those records. Setup and decode times are recorded
separately; no speed advantage is the primary claim. Timing during concurrent
validation/setup is exploratory, and GPU throughput is not inferred from CPU
neural training. No distance scaling, physical leakage removal or unencoded
break-even claim follows from this one-calibration distance-three study.

## Confirmed result and adopted policy

All 60 study jobs completed on Garnet calibration
`a18a947f-0408-4628-81ea-1715637e434f`: **300,000 physical shots**,
including 108,000 independent confirmation shots. No shots were postselected.

| Basis / policy | Challenge-fit LER per round | Conditional 95% profile interval | Failures at R1 / R2 / R3 (6,000 shots each) |
|---|---:|---:|---|
| X, retained native correlated MWPM | **5.220%** | 4.976–5.471% | 176 / 585 / 993 |
| Z, adopted posterior + data hybrid | **3.763%** | 3.559–3.974% | 80 / 456 / 753 |
| Z, previous standalone native ML | 3.920% | 3.712–4.136% | 86 / 498 / 756 |

The Z combination is Ethan's existing fenced native compiler and data-qubit
XY4, plus Dara's training-only fitted DEM and posterior-plus-count rule with
`alpha=10`. It reduces the challenge-fit Z estimate by **0.157 percentage
points**, a **4.0% relative point estimate**. The paired, multiplicity-adjusted
gain interval is **+0.037 to +0.274 percentage points**. It beats all three
tested standalone references in Z. The observed improvement is chiefly at R2;
R3 changes only from 756 to 753 failures. No large R3 gain is claimed.

The X fitted-ML candidate gives 5.065% versus 5.220% for MWPM, but its adjusted
gain interval is **−0.078 to +0.381 percentage points**. Its R3 failures rise
from 993 to 1,006. Therefore **X retains the existing MWPM** under the frozen
rule. Physical sqrtY/native merging, ensembles and the bounded neural trials
were not adopted. Improvement in both bases is **not established**.

The standalone comparison estimates on the same fresh primary window are:

| Reference | X LER | Z LER |
|---|---:|---:|
| Ethan | 5.220% | 3.920% |
| Dara | 7.387% | 6.759% |
| Iyanu, training-selected coherence prior | 9.894% | 7.813% |

These reconstruct the documented physical paths on this study's common fixed
placement and calibration, with the explicitly recorded finite decoder
budgets. They do not assert optimality over every historical placement,
training budget or untested configuration.

**The stationary challenge model is rejected.** Adopted X has deviance
103.80/2 df; adopted Z has 156.79/2 df. Both first-round probabilities are much
lower than their subsequent growth. These headline numbers are descriptive
fits to the exact requested equation over R1–3, **not established stationary
physical per-cycle rates**. Confidence intervals condition on the fit model;
the paired difference bootstrap resamples the observed failures. No shifted
fit replaces the requested primary estimator. Both confirmation blocks show
the same direction of the Z gain.

The full confirmation acquisition/compilation/retrieval/decode/report process reached an
observed Windows peak working set of **791.1 MiB**. R3 decisions occupy 16 MiB;
the FWHT builder's estimated temporary peak is 512 MiB, separate from process
RSS. Inference throughput excludes build, load and conversion. Exact raw
counts, per-block fits, paired comparisons and timing records are retained in
the run's `results.json`, `manifest.json` and `resource_usage.json`.

## Execute the adopted policy

The runner now executes the adopted frozen decoder and its matching physical
compiler directly. It verifies inference-file hashes, source/native/prior
identity and the active calibration before submission. A changed calibration
requires a new qualification. This executes one round count and reports a
failure probability; it does not fit a new LER from one point.

```powershell
python -m challenge.benchmark_headline --run results/decoders/hybrid/20261024T234900Z --submit --policy-basis Z --policy-rounds 3 --policy-shots 1000 --policy-output results/decoders/hybrid/NEW_POLICY_RUN --token-file apikey.jom
```

Use the same arguments with `--resume` instead of `--submit` to retrieve an
existing execution without creating another job. X selects retained MWPM;
Z selects the confirmed posterior hybrid. Both have separate raw-data ledgers.
Both public policy paths were smoke-tested on live Garnet with two shots each;
these four additional shots are excluded from the 300,000-shot study.

Final verification: 265 root tests, 327 isolated-package tests, the earlier
36,000-shot evidence replay, and the new 300,000-shot replay all pass. New
evidence replay was also run with network sockets blocked. Every source/native
frame, shot count, selected state hash and saved prediction was rechecked.
