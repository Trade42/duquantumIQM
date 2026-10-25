# Decoder agent coordination and handoff

This guide keeps parallel decoder implementations independent while preserving
the surface-code experiment's meaning. It documents current interfaces and a
coordination convention; it does not introduce a new decoder API or registry.
Read the live files before changing an interface that another agent may use.

## 1. Claim files before editing

Find the shared coordination location from the repository root:

```powershell
git rev-parse --path-format=absolute --git-common-dir
```

Use `<git-common-dir>/agent-coordination/claims/<agent-id>.json` for one
agent-owned status record and `agent-coordination/locks/<resource-hash>.json`
for exclusive file locks. This state lives outside tracked source files and is
shared by linked worktrees. Do not create a shared Markdown table that every
agent updates concurrently.

Each record contains `owner`, a fresh random `token`, `task`, repository-relative
`files`, `worktree`, `status` and `updated_utc`. Add `branch`/`base_head` when
available. Statuses: `active`, `waiting`, `handoff`, `done`.

For each resource, use the repository-relative path with `/` separators, no
leading `./`, and no `..`; case-fold it for this Windows repository. The lock
name is the SHA-256 of that normalized path, plus `.json`. Claim exact files,
including files you will create; wildcards and directory claims do not lock
their contents. Paths that alias the same file must use the same canonical
repository-relative name.

Create locks atomically, using exclusive creation rather than check-then-write:

```python
# resource is the canonical repository-relative filename.
# coordination_root is the absolute shared directory discovered above.
import hashlib
import json
from pathlib import Path

key = hashlib.sha256(resource.replace('\\', '/').casefold().encode()).hexdigest()
lock = Path(coordination_root) / 'locks' / f'{key}.json'
lock.parent.mkdir(parents=True, exist_ok=True)
with lock.open('x', encoding='utf-8') as stream:
    json.dump({**claim_record, 'resource': resource}, stream, indent=2)
```

`FileExistsError` means someone already holds the resource. Do not overwrite it.
When claiming multiple files, acquire them in normalized path order. If an
acquisition fails, release only locks acquired by this attempt, after checking
their token; do not hold a partial set while waiting for another owner's files.
Create your uniquely named claim record exclusively as well. Update only your
own status record, preferably with a same-directory temporary file and atomic
replacement so readers do not see partial JSON.

Read both claims and locks: a process can exit between creating a lock and
writing its status record. An old timestamp alone is not grounds for deletion.
Check the listed worktree and available task status; use an authorized team
channel for handoffs. If ownership cannot be resolved, work on other files.
Never infer permission to interrupt another agent or discard its changes.

At handoff, write validation and integration notes first, then delete only locks
whose owner **and token** still match yours. Keep the status record for context.
Locks are advisory coordination, not an operating-system write barrier; isolated
worktrees and scoped diffs remain useful protection against accidental edits.

## 2. Split work without shared edits

- Backend agent: claim its new implementation file, backend-specific tests and
  backend-specific documentation. Example: `challenge/decoders/bp_osd.py`,
  `tests/decoders/test_bp_osd.py`, `docs/decoders/bp_osd.md`.
- Another backend agent: use different names, such as `union_find.py`; do not
  create or edit the package's shared `__init__.py` as incidental scaffolding.
- Integration owner: claim shared adapters, registry/exports, runner selection,
  result-schema changes, common tests and dependency files before wiring backends.
- Benchmark agent: use an immutable input-shot dataset and its own run directory.
  Coordinate changes to shared benchmark code with the integration owner.

Ownership is assigned by the active claims, not by the example backend names.
These extension directories may not exist yet. Backend agents can create their
own files without claiming ownership of all future decoder work.
Do not launch parallel edits to `challenge/decoding.py`: it currently contains
both decoding **and statistics**, so even unrelated-looking changes can collide.

## 3. Current integration contract

| Location | Current responsibility |
|---|---|
| `challenge/decoding.py::make_matcher` | Noisy Stim circuit to correlated PyMatching graph |
| `challenge/decoding.py::decode_syndromes` | Validate shapes/circuit identity, predict logical flips, aggregate failures |
| `challenge/decoding.py::DecodeResult` | Shots, failures, probability, SE, Wilson interval and raw logical-flip diagnostic |
| `challenge/decoding.py::fit_ler` | Binomial maximum-likelihood rate fit, optional SPAM amplitude, profile interval |
| `challenge/submission.py::evaluate` | Frozen-policy replay, all-shot scoring and challenge LER comparison |
| `challenge/run_on_hardware.py::decode_hardware_results` | Hardware result wrapper; noisy decoder circuit is required |
| `challenge/replay_rounds.py::verify_case` | Source/native measurement order and detector/observable verification |

The core backend behavior is `decode_batch(det_events) -> logical_predictions`.
Accept a binary two-dimensional input, keep shot order and return a binary
two-dimensional array even for one observable or one shot. Do not squeeze axes,
return physical-qubit corrections in place of logical predictions, or silently
drop shots that fail to converge. Specify an explicit failure policy and count
or separately report unresolved shots; never improve accuracy by excluding them.

Keep backend-specific configuration on the backend builder/object. Supply a
small builder in your own module that accepts the noisy circuit or its DEM and
returns the decoding object plus reproducible configuration metadata. Keep
optional imports lazy and raise an actionable missing-dependency error.

For direct use through `decode_syndromes(..., matcher=adapter)`, the current
caller invokes `decode_batch(dets, enable_correlations=correlated)`. Implement an
appropriate adapter or coordinate a shared API change; do not silently ignore an
unsupported option. Record whether correlation handling is native, approximated
or unavailable. A new module is not automatically selectable from the CLI.

`obs_flips` is ground truth for scoring, never an inference feature. Train learned
weights on independent training shots; freeze them before held-out evaluation.
Per-detector rates estimated from evaluation shots are also evaluation-derived
information; do not use them for tuning without explicitly reporting adaptation.

## 4. Preserve physics, parities and frames

- `raw_meas`: `(shots, num_measurements)` in Stim record order. Qiskit's bit
  reversal and register spaces are handled by `counts_to_measurement_array`.
- `det_events`: `(shots, num_detectors)`; `obs_flips`:
  `(shots, num_observables)`. Both are relative to the ideal reference.
- A decoded shot fails if **any** predicted logical observable differs from
  truth: `np.any(predictions != obs_flips, axis=1)`. Do not average individual
  observable errors and call that the probability of a failed shot.
- The decoder and executed circuit must share the same noiseless instructions,
  detectors and observables. Different decoder weights/noise are allowed as an
  explicit model-mismatch experiment. Record physical and decoder circuit hashes.
- No-reset bulk checks compare raw measurements two rounds apart, not adjacent
  rounds. First-round, second-round and final-data boundaries require generated
  definitions; avoid hand-written bulk-only formulas.
- Sparse Stim qubit indices, dense hardware indices, measurement indices and
  detector indices are separate spaces. Do not interchange them.
- Preserve the generated X/Z memory frame and terminal readout. Observable
  flips are already reference-normalized; do not apply a second correction.
- Use the actual noisy schedule. Do not substitute a reset circuit, move logical
  gates, delete QEC rounds or alter the noise model to improve decoder scores.
- Static DEMs approximate persistent leakage. Hidden simulator flags and future
  unmeasured events are unavailable to a causal decoder. An offline full-history
  decoder may use the complete *measured* record, but must say it is offline.
- Actual leakage heralds, erasure flags or extra flag measurements require a
  specified measurement interface and matched circuit. They are different inputs
  from hidden simulator state and must be benchmarked as a different setup.
- Do not suppress non-deterministic-detector errors or silently discard
  hyperedges. Preserve `approximate_disjoint_errors` and decomposition settings
  in metadata. A fallback decoder must be named and timed, not hidden.
- The retained pipeline is single-logical identity memory. Do not claim
  multi-block logical-gate behavior from memory decoding.

## 5. Compare decoders fairly

Use the **same held-out detection events and observable labels** for each
decoder on the same physical setup. Publish immutable `.stim`/`.dem` definitions
and, when caching shots, a binary array dataset with a manifest identifying its
shapes, hashes, seeds, basis, distance, rounds, reset mode, gate basis, logical
schedule/frame, calibration source and noise parameters. A seed alone does not
guarantee identical shots across package versions or sampling/batching changes.

Write each run under `results/decoders/<backend>/<run-id>/`; record the dataset
hash and decoder configuration. Preserve raw failure counts. Do not overwrite
`results/iqm_demo`, `results/transversal` or another backend's outputs. Shared
datasets become read-only after publication; publish a new dataset ID to change
the physical setup.

Report graph/build or training cost separately from warmed decoding cost.
Measure decode latency, batch size, shots/second, memory and worker/thread
configuration; state which quantities were actually measured. Separate imports,
sampling, device transfers and decoder execution. Do not run competing CPU/GPU
timing jobs simultaneously; reserve the exact shared resource
`@resources/performance-timing` with the same hashing/exclusive-lock protocol.
This reserved resource is a coordination name rather than a repository file.
Ordinary correctness tests can run independently.

Use paired failure differences or a paired bootstrap when comparing decoders
on common shots. Physical before/after comparisons use independent samples; do not copy that
independence assumption into a common-shot decoder comparison. Report negative and null gains.

One round count gives logical failure probability. Per-round LER requires a
multi-round fit and its assumptions. Effective algorithm-cycle rates also depend
on the operation schedule and preparation offset. Zero-failure points retain a
positive uncertainty bound. Fixed-shot Wilson intervals do not become
optional-stopping-safe by stopping after a target number of errors; use a justified anytime-valid method.

## 6. Validate and hand off

Backend tests should exercise shape/order, noiseless decoding, one observable,
multiple observables, deterministic seeds when applicable, invalid inputs,
unsupported DEM features and explicit non-convergence/fallback behavior.
Exercise X/Z, reset/no-reset, matched/mismatched weights and H/frame boundaries
where supported. Test actual predictions, not just graph construction.
Do not claim support for an untested regime; state its limitation explicitly.

Run your focused tests, then relevant existing regressions, for example:

```powershell
python -m pytest -q tests/test_surface_code.py tests/decoders/test_hybrid.py
```

Before completing shared integration, the integration owner runs:

```powershell
python -m pytest -q
git diff --check
```

No expensive full benchmark rerun is required for a documentation-only change.
For a decoder change, use a small reproducible smoke evaluation first, then a
fixed-shot comparison sufficient for the claimed accuracy/throughput conclusion.
Install optional dependencies in an isolated environment; do not change the
shared interpreter or IQM's pinned hardware stack while another agent uses it.

Leave this handoff in your own claim record or backend-specific document:

```text
Owner / backend / branch / worktree:
Files changed and ownership released:
Builder and prediction interface:
Supported circuits / unsupported cases:
Dependencies and lazy-import behavior:
Tests run and results:
Dataset IDs / hashes / shot counts / seeds:
Accuracy, uncertainty, timing and memory evidence:
Fallbacks, model approximations and leakage/frame limitations:
Shared-file patch needed from the integration owner:
Next concrete step:
```

Source material: [pipeline and models](../IMPLEMENTATION.md),
[retained improvements](../improvements.md),
[validation history](../VALIDATION.md). Code is authoritative when a document's
test count or integration description has become stale.
