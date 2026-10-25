# Contributing

Read the [retained improvements](improvements.md), [challenge methods](docs/METHODS.md),
[evaluation](docs/EVALUATION.md) and [reproduction guide](docs/REPRODUCIBILITY.md).

## Scientific interfaces

Detection events are binary `(shots, num_detectors)` arrays. Decoders return
binary `(shots, num_observables)` predictions in original shot order.
Observable truth belongs to training or scoring; inference accepts the measured
detector record. Keep no-reset parities, source/native measurement maps and
logical reference frames consistent with the executed circuit.

Measurements and inference states in `benchmarks/garnet` are checksummed.
Scientific changes use a new evaluation bundle with explicit circuit, state,
training, seed and model metadata. Compare decoders on common shots and physical
changes on separately identified arms. Record counts, confidence intervals,
setup/decode cost and model assumptions.

## Development

Follow [AGENTS.md](AGENTS.md) and the
[decoder coordination guide](docs/DECODER_AGENT_GUIDE.md). Inspect active claims
and acquire exact file ownership before editing. Preserve other contributors'
unfinished work. Keep optional dependencies isolated and credentials ignored.

```powershell
python -m examples.reproduce
python -m pytest -q tests/test_submission.py tests/test_rounds_replay.py
git diff --check
```

Shared integration requires the full root regression suite. Native compiler
changes also require the `ethan/tests` suite. The reproduction guide lists both
commands. Imports, ordinary tests and measurement replay run offline.
