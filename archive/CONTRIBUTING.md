# Contributing

Start with the [winning pipeline](docs/WINNING_PIPELINE.md),
[technique catalog](docs/TECHNIQUES.md) and
[reproduction guide](docs/REPRODUCIBILITY.md). The current target is a reproducible
offline presentation of completed Garnet experiments. Hardware acquisition has
ended; ordinary changes must not retrieve credentials or contact a QPU.

## Preserve the scientific contract

Detection-event input is binary `(shots, num_detectors)`. Prediction output is
binary `(shots, num_observables)` in the original shot order. Observable truth
belongs only in explicitly separated training/scoring. Do not give inference
hidden simulator flags, held-out rates or confirmation labels.

Keep the source circuit, actual native Clifford circuit, measurement permutation,
no-reset boundary parities and logical reference frame consistent. A matching
array width does not establish a matching experiment. Compare decoder changes
on identical held-out shots; compare physical changes as different arms.

The retained Z decoder is the frozen `posterior_10` hybrid, not the isolated
package's older model-only LUT. The retained X decoder is native correlated
MWPM. A research implementation becoming available is not a promotion decision.

## Respect frozen evidence

Published run directories are immutable. Add a new run ID for new analysis or
experiments. Preserve raw records, failures, seeds, versions, priors and model
approximations. Files listed in the inference seals, including their source
dependencies, are byte-sensitive. Repository attributes preserve the recorded
checkout bytes; do not run broad formatters over these files.

If an algorithm must change, create a new backend/version and an explicit
qualification rather than silently changing a sealed winner. Report null and
negative results. Do not swap fit windows, use postselection or add fallback
decoders inside a fit to improve the headline.

## Work in a narrow scope

Follow [AGENTS.md](AGENTS.md) and, for decoder work,
[the decoder guide](docs/DECODER_AGENT_GUIDE.md). Inspect existing changes and
active claims. Acquire the exact file claims before editing; release only your
own token after recording validation and handoff notes. Do not reset or reformat
someone else's unfinished work. Existing contributor directories retain their
separate ownership and provenance.

## Validate the change

For documentation/example changes, run the winner example and its focused tests:

```powershell
python -m examples.replay_winners
python -m pytest -q tests/test_winner_example.py tests/test_rounds_replay.py
```

For shared integration, run the relevant regression suites and full root tests;
isolated-package edits also require its suite and frozen replay. Exact commands
and optional dependencies are in [Reproducibility](docs/REPRODUCIBILITY.md).
Check local documentation links and `git diff --check`. Never commit credentials,
private server-job metadata, virtual environments or dependency caches.
