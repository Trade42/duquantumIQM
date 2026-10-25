# Garnet bounded readout and history study

No new change qualified. The retained policy is the previously released
native/XY4 compiler with correlated MWPM for X and Bayesian ML posterior decoding
for Z. The experimental implementations remain available for reproduction;
they are not substituted into the released policy.

## Allocation and acquisition

Exactly 96,000 new physical shots were requested and returned in 82 completed
Garnet jobs: 40 pilot jobs × 800 shots, then two successive 32,000-shot
confirmation windows of 21 jobs each. No retries, smokes or replacement jobs
were submitted. The guard rejects totals of 100,000 or more; failed/ambiguous
attempts would remain charged. The unused 3,999-shot allowance was not spent.

The confirmation windows ran 00:58:41–01:03:09 and 01:03:44–01:08:10 UTC on
2026-10-25, through separate connections under calibration
`a18a947f-0408-4628-81ea-1715637e434f`. They are not independent days or
calibrations. Every requested shot is scored; heralding and postselection are off.
All experiments use the distance-three, static, no-reset memory patch.

Archived evidence, including the earlier confirmation, is now development data
only. The new job IDs are unique and disjoint from archived training jobs.
Another 8,200 archived R4/R5/R7 control shots were copied with their original
dataset hashes and verified identical native detector/observable frames. R6
recurrent training is explicitly synthetic. Recurrent optimization balances
round records and replays current-domain training without using the 200-shot
pilot validation split. No decoder sees hidden leakage flags or inference labels.

## Frozen decisions and independent results

The primary window was R1,2,3 throughout. The estimator is the IQM challenge
binomial MLE, with `P_L(r)=(1-(1-2 epsilon_L)^r)/2` and amplitude fixed to one.
R1 through R7 is a separately declared diagnostic window, not a replacement
headline selected after evaluation.

Pilot selection favored adaptive DD for X. A lexical tie initially selected a
GRU despite exactly the same validation failures as fitted MWPM. Before any
confirmation, the final tie rule preferred the simpler matching decoder and
added component comparisons. The original protocol, selection, artifact map
and training source revision are retained as `*.pilot.json` and
`benchmark_rounds.py.training.source.txt`. No weights were retrained during
finalization. The final freeze was 00:58:17 UTC, before every confirmation attempt.

The four frozen X arms were the released policy, adaptive DD plus fitted MWPM,
adaptive DD plus unchanged MWPM, and fitted MWPM on the unchanged physical
circuit. All six pairwise contrasts use 99.1667% intervals (Bonferroni over six,
12,000 bootstrap draws, seed 7321). Decoder-only comparisons share identical
shots; different physical arms use independent shots, stratified by round/block.
A combination must beat both components and the released policy. A simpler
component can replace the release only with an adjusted positive gain over it.
Z selected the unchanged released policy and shared its physical allocation.

| Policy | R1/R2/R3 failures | Shots at each round | Challenge LER |
|---|---|---:|---:|
| Retained X | 159 / 404 / 728 | 4,000 | 5.8078% |
| X adaptive DD + fitted MWPM | 141 / 446 / 757 | 4,000 | 6.0656% |
| X adaptive DD alone | 142 / 480 / 757 | 4,000 | 6.2387% |
| X fitted MWPM alone | 141 / 404 / 728 | 4,000 | 5.7192% |
| Retained Z | 145 / 639 / 1,147 | 8,000 | 4.2547% |

The X combined candidate's reduction versus release is −0.2578 percentage
points, adjusted interval −0.8364 to +0.3534. Fitting alone reduces X by 0.0886
points, adjusted interval −0.0148 to +0.1976. Neither qualifies. The retained
X profile interval is 5.4920–6.1360%; Z is 4.0659–4.4494%.

The recurrent model's descriptive R1–3 rates were X 5.7919% and Z 4.7049% on the
unchanged physical circuit. It did not beat the project's retained policy.
Its improvement over the weaker native Z MWPM reference is not used to claim a
new headline. The exact posterior policy is qualified only for R1–3; it is not
silently replaced by another decoder at longer rounds.

The stationary model is rejected: X deviance 52.095/2 df, p=4.87e−12; Z
198.862/2 df, p=6.57e−44. These are descriptive challenge-equation estimates,
not established stationary cycle rates. Fresh rates are higher than the earlier
study; this experiment does not establish the physical cause.

## Physical and model limitations

The source circuit, layout, no-reset t−2 parities, logical frame and native
terminal optimizer are unchanged. DD targets data qubits. R3 baseline and
centered XY4 insert 144 pulses; adaptive inserts 300 for X/304 for Z. The fast
ancilla trial inserts 152 and reduces total R3 schedule length from 301,900 to
300,948 ns. These totals include the same 298,000 ns initialization; they are
not coherent-memory lifetimes. Faster readout did not win pilot selection.

The active architecture calibrates constant/fast_constant measurement pulses,
but no QND implementation. Fast SSRO fidelity is unpublished here; its decoder
prior explicitly proxies standard-readout fidelity, without changing saved
calibration observations. The native Pauli prior omits explicit inserted-DD
pulse errors and idle decoherence. Fitting and learning are empirical
compensation, not a calibrated persistent-leakage model or an oracle.

## Compute and verification

All inference ran on CPU, with two Torch threads and one Python process.
Warmed retained-policy throughput on these batches was approximately 368,000
shots/s for X (2,000-shot batches) and 8.96 million shots/s for Z (4,000-shot
batches), excluding decoder loading and a 32-shot warmup. Per-case times are
in `report.json`. Template data/model preparation, fitting and saved decoder
construction took 52.37 s across 56 templates. The eight GRU optimization loops took 50.34 s total
(6.0–6.7 s each), excluding encoding/imports. The observed analysis-process
peak working set at scoring completion was 398.4 MiB, including imports and
decoder setup but excluding earlier training and subsequent figure rendering.
Full versions and seeds are in the manifest and model metadata.

Validation: 281 shared tests, 327 isolated Ethan tests. The earlier frozen
18-case/36,000-shot bundle replays exactly with sockets blocked. The new
network-blocked replay reconstructs all 96,000 raw shots in source/native frames,
matches 138 prediction groups exactly, refits 20 estimates, reproduces all six
adjusted intervals and verifies the frozen release decision. It accesses no
credential file. Run `python -m challenge.replay_rounds --run
results/decoders/history/20261025T004600Z` with the optional decoder dependencies;
`--runtime PATH` supports an isolated dependency installation.

All files required by the inference seal are published, including template QPY
circuits. Unsealed transport QPY files, redundant count JSON and private server
job metadata remain local. Raw shot arrays, source/native/decoder circuits,
calibration, manifests, model states, predictions and reports are retained.

![R1–7 diagnostic failure probabilities; these are not per-cycle LERs](ler.png)
