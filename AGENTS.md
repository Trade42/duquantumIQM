# Agent guide

Applies to this repository. For decoder work, read
[docs/DECODER_AGENT_GUIDE.md](docs/DECODER_AGENT_GUIDE.md) before editing.

## Start and coordinate

1. Inspect `git status --short` and the files relevant to your assignment.
   Existing modified/untracked files can contain other agents' unfinished work;
   do not reset, clean, overwrite or reformat them as part of your task.
2. Read active claims under the shared Git common directory's
   `agent-coordination/claims/`. Claim every file you will edit with an atomic
   exclusive lock before editing; the decoder guide defines the protocol.
3. Keep one decoder implementation in its own backend module, tests and output
   directory. Do not have multiple agents edit a central registry or runner.
4. If a file is already owned, continue in unclaimed files and leave an
   integration handoff. Silence or an old timestamp does not transfer ownership.
5. Before finishing, record tests, interface assumptions and outstanding work in
   your claim, then release only locks carrying your own token.

Use an isolated worktree when it helps; coordination still lives in the Git
common directory so agents in different worktrees can see the same claims.
Keep branches, commits and handoffs specific to the assigned task.

## File ownership boundaries

| Work | Normal ownership |
|---|---|
| New decoder | One agent per new `challenge/decoders/<backend>.py` and `tests/decoders/test_<backend>.py`; these directories are extension locations, not existing infrastructure |
| Decoder adapter/registration | One integration owner for `challenge/decoding.py`, package exports and any new `challenge/decoders/__init__.py` |
| Runners and result schema | Same integration owner, or a separately claimed handoff, for `challenge/experiment.py`, `challenge/benchmark_logical.py`, `challenge/run_on_hardware.py` |
| Circuit, noise, frame or layout changes | Separate explicit scope and file claims; do not change these to make a decoder pass |
| Dependencies, shared tests and docs | Claim exact files; backend agents propose changes in their own handoff rather than editing shared files concurrently |
| Generated results | `results/decoders/<backend>/<run-id>/`; never overwrite another run |

No pluggable decoder registry exists at the time this guide was written.
`make_matcher` builds PyMatching, and the simulation runner calls it directly.
Adding a backend module alone does not integrate it into the runner or CLI.
Inspect the live code before assuming that later integration has happened.

## Decoder invariants

- Input detection events: binary `(shots, num_detectors)`. Output: binary
  logical-flip predictions `(shots, num_observables)`, preserving shot order.
  Observable truth is for scoring only, never decoder input.
- Match the actual circuit's detectors, observables, measurement order,
  no-reset convention and logical frame. Matching array widths is insufficient.
- Preserve no-reset boundary handling. Bulk detection uses measurements two
  rounds apart; initialization and terminal checks have different equations.
  Use generated parities and Stim's measurement-to-detector converter.
- Observable flips are relative to the ideal reference. They are not raw
  logical bits. Apply a virtual Pauli readout correction only where required
  for actual logical-output interpretation, never twice during decoding.
- Correlated PyMatching is enabled at both graph construction and decoding.
  DEM decomposition currently explicitly allows disjoint-channel approximation.
  Record model approximations and fail explicitly on unsupported cases.
- Persistent leakage flags are hidden simulator state. A decoder may use
  measured syndrome history; it may not receive hidden flags as oracle data.
- The final pipeline is X/Z identity memory at distance three and rounds 1–3.
  Keep inference and compiled source/native measurement frames aligned.

## Validation and claims

Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for the pipeline and noise models,
and [improvements.md](improvements.md) for the retained Garnet improvements.
Run focused backend tests first; the integration owner runs the relevant shared
regressions and full `python -m pytest -q` before completing integration.
Do not change global packages or dependency pins for an individual backend.

Compare decoders on identical physical circuits and held-out shots. Report
failure counts, confidence intervals, setup/decode time, throughput, memory,
seeds, versions and model mismatch. Distinguish memory LER, workload failure
probability and effective computational-cycle LER. Synthetic calibration remains
labeled synthetic. Zero observed failures does not mean zero error rate.

Default work is local/offline. Imports and tests must not submit QPU jobs,
retrieve credentials or modify existing benchmark artifacts.
