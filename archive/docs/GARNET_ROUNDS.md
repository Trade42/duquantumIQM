# Bounded Garnet improvement study

> Completed follow-up. Hardware access has ended; use the
> [offline reproduction guide](REPRODUCIBILITY.md). Submission commands below
> describe the completed protocol, not the current quickstart.

**Outcome: no new method qualified for release.** The current native/XY4 policy
remains: correlated MWPM for X and the frozen Bayesian ML posterior for Z.
All 96,000 newly requested physical shots returned, across 82 completed jobs.
There were no retries or smoke submissions outside this allocation.

| Fresh confirmation policy | Challenge R1–3 LER | 95% profile interval |
|---|---:|---:|
| Retained X: native XY4 + correlated MWPM | 5.808% | 5.492–6.136% |
| X adaptive DD + fitted MWPM candidate | 6.066% | 5.742–6.402% |
| X fitting alone on the retained physical circuit | 5.719% | 5.406–6.045% |
| Retained Z: native XY4 + frozen Bayesian ML | 4.255% | 4.066–4.449% |

The fitting-only X gain was 0.0886 percentage points, with a six-comparison
adjusted interval of −0.0148 to +0.1976 points. The combined candidate was worse
at its point estimate. Neither met the frozen release rule. The recurrent
decoder did not beat the retained policy. Fast readout and centered XY4 did not
win pilot selection. QND readout was unavailable in the active calibration.

The stationary challenge equation is rejected for both retained policies
(X deviance 52.10/2 df, Z 198.86/2 df). These are descriptive challenge fits,
not established stationary cycle rates. Higher fresh rates than the earlier
study are reported without attributing them to an unmeasured physical cause.
Both confirmation windows used one calibration; they are not separate-day tests.

See the [counts, fits and adjusted comparisons](../results/decoders/history/20261025T004600Z/report.md),
[evidence and limitations](../results/decoders/history/20261025T004600Z/EVIDENCE.md),
and [offline verification](../results/decoders/history/20261025T004600Z/verification.json).

Run `python -m challenge.benchmark_rounds --help` for the explicit entry point.
Imports, tests and `develop`/`analyze` remain offline. Hardware submission requires
`--action pilot|confirm0|confirm1 --submit`; `collect` only retrieves ledgered jobs.
Credentials are read only for an explicit hardware action. Never publish them.

The predeclared budget is 96,000 new physical shots: 32,000 pilot and 64,000
confirmation. The hard guard is 99,999. Each attempt is charged before calling
the server; failed or ambiguous attempts remain charged and are never retried
automatically. Shared physical arms reuse identical shots and receive the
combined allocation. Every returned shot is retained.

The primary IQM challenge estimator is the fixed-amplitude binomial fit
`P_L(r) = (1 - (1 - 2 epsilon_L)^r)/2` on rounds 1, 2, 3, matching the previously
released headline. Rounds 1 through 7 are a predeclared diagnostic. A rejected
stationary fit is reported as descriptive; no alternate window or SPAM fit is
substituted to improve the headline.

Physical trials preserve the released logical circuit, layout, no-reset
parities, native Clifford frame and terminal optimization. They change only
audited data-qubit DD placement/length or calibrated ancilla measurement:
baseline XY4 ASAP, centered XY4, centered adaptive XY8/XY4/XX, and fast ancilla
readout. The installed SDK exposes QND measurement, but the active Garnet
architecture does not calibrate that implementation, so it is not attempted.
The fast pulse is calibrated but lacks published fast SSRO fidelity here. Its
decoder prior explicitly proxies standard-readout fidelity; pilot measurements
fit the decoder independently. This proxy is not a fast-readout calibration.

The new decoder applies a bidirectional GRU to measured detection history with
fixed coordinate/missing-detector masks and a correlated MWPM prediction. It
predicts a correction to that prediction. This is offline full-history decoding;
it receives no hidden leakage state, observable truth, or evaluation-derived
rates. The fitted DEM retains round/detector-specific probabilities and joint
decomposed faults, with disjoint-channel approximation recorded explicitly.
The same recurrent architecture and matching families work throughout R1..7.
The exact posterior family and previously released Z policy are restricted to
their qualified R1..3 window; they are not silently extended with a fallback.

Archived shots, including previous confirmation, are now development evidence.
They train/tune models but never count as new confirmation. The pilot uses
600 training and 200 validation shots per physical case. Another 8,200 archived
R4/R5/R7 control shots have verified identical native frames and train only.
R6 recurrent training is explicitly synthetic. The minimum pilot R1..3 validation LER selects
one physical profile and decoder family per basis; all weights and inference
artifacts are sealed before confirmation. Both complete acquisition blocks
must finish before scoring. These are successive acquisition windows under one
calibration, not independent days/calibrations.

Release requires a strictly positive bootstrap interval for LER reduction over
the released policy, with Bonferroni adjustment across every predeclared pairwise
contrast. A combined candidate must also beat each component alone. Otherwise,
a simpler component may release only if it beats the previous policy; if neither
does, the previous policy stays. Decoder comparisons on identical physical shots are
paired; physical-arm comparisons are independent, stratified by round/block.
Raw failures, profile intervals, lack-of-fit, per-block fits and warmed decoding
times are recorded. No improvement means retaining the released policy.

The final component rule and complexity tie break were frozen before any fresh
confirmation. The original pilot rule, selection and source revision are kept
alongside the final versions. Identical pilot failure counts prefer matching
over a recurrent model; no model weights were retrained during finalization.

Recurrent training balances round counts and separately replays current-domain
training records so the larger archive cannot drown out a changed physical
profile. Internal fitting and tuning partitions remain disjoint; the fresh
200-shot pilot validation split is used only for final selection.

The native Pauli prior omits explicit inserted-DD pulse errors and idle
decoherence. Detector fitting and history learning compensate empirically;
neither guarantees a correct physical channel or models persistent leakage.

After confirmation, load the adopted R1..3 decoder without networking:

```python
from challenge.benchmark_rounds import load_adopted
policy, decoder = load_adopted("results/decoders/history/20261025T004600Z", "Z", 3)
predictions = decoder.decode_batch(measured_detection_events)
```

`policy` records the physical trial, calibration, placement and qualification.
Physical execution uses `TrialExecutor` with that explicit profile. R4..7
diagnostics do not silently extend the released policy's qualification.

Replay all published new evidence without credentials or network access:

```powershell
python -m challenge.replay_rounds --run results/decoders/history/20261025T004600Z
```

This uses the optional decoder dependencies. An isolated installation can be
supplied with `--runtime PATH`. Replay checks 96,000 raw shots, 138 frozen
prediction groups, 20 fits, six adjusted comparisons and the release decision.
Shared tests pass 281 cases; the isolated Ethan suite passes 327. The earlier
18-case, 36,000-shot frozen evidence bundle also replays exactly offline.
