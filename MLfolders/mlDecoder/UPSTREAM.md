# Source and integration provenance

Imported from `Trade42/duquantumIQM`, branch `MLDecodeTest`, commit
`0383456080e783d6b2965385452112db81e028c2`.

- [Original ML work](https://github.com/Trade42/duquantumIQM/tree/0383456080e783d6b2965385452112db81e028c2/MLfolders/mlDecoder)
- [Original InfoRecord](https://github.com/Trade42/duquantumIQM/tree/0383456080e783d6b2965385452112db81e028c2/MLfolders/InfoRecord)
- [`upstream_manifest.json`](upstream_manifest.json): source-file SHA-256 hashes
  and exact branch/commit provenance.

The import selects the final LSTM backend, cascade/Relay-BP, training and
evaluation support, four promoted checkpoints, calibration packages, four
training/test result records, and the pA/eA B3 holdout. It does not merge the
branch's unrelated pulse-characterization tree or reinstate the removed
directional-tile work. MLP/transformer alternatives are outside this import.

## Preserved evidence

Original imported result JSON, LER tables, weights, calibration files and B3
measurements are preserved byte-for-byte. `evidence.py` verifies their hashes,
checks measurement-to-detector conversion, confirms B3 test-only membership,
and recomputes reported rate fits from saved failure counts. Reported paired
intervals and latency measurements retain their upstream provenance. The
original branch's complete device/job audit is described by its result records;
this integration does not repeat QPU retrieval.

The holdout has 44 cases: both bases at R=1,2,3,4,5,6,8,10,12,15,20 on Garnet
pA and Emerald eA. Per point there are 5,000 and 2,000 shots respectively.
Source Stim/QASM circuits, arrays, shot tables, detector tables and manifests
are included. Original manifests may refer to upstream raw-job archives and
before/after calibration records that are not needed by the offline array/frame
audit and are not included here. B1/B2 training arrays remain upstream; their
training/validation summaries and model-selection histories are included.

## Adaptations to the cleaned repository

1. Empty logical-operation circuit construction uses the current memory
   generator directly. A logical-gate implementation is not required for this
   identity-memory study.
2. Acquisition-era noise and leakage samplers needed by ML pre-training and
   priors are isolated in `_reference/`. Calibration and measurement utilities
   use the current core interfaces. They are separate from the submission's
   native prior.
3. `final.load` constructs the LSTM loader directly, avoiding an unnecessary
   dummy noisy circuit. Model architecture, parameter names and inference
   decisions for the same weights/inputs are preserved.
4. Checkpoint loading uses PyTorch's restricted weights-only loader, with
   explicitly scoped NumPy scalar/dtype support for the training metadata.
   Checkpoint hashes are checked before loading; unrestricted pickle loading
   is not enabled.
5. Input validation rejects nonbinary values before integer conversion. Final
   LSTM tests replace references to the omitted transformer.
6. `evidence.py` adds offline verification and optional full B3 LSTM replay.
   Source hashes identify upstream code; adapted code is not represented as
   byte-identical to its upstream source.

## Precision and runtime distinction

The upstream held-out scores precede promotion of the checkpoints to float16.
On the integration CPU runtime (Torch 2.13.0+cu130 / NumPy 2.2.6), the promoted
weights are loaded into float32 networks and yield a few different decisions.
The [local replay record](results_record/promoted_cpu_replay.json) keeps those
counts separate. Precision, device and runtime changes are not independently
isolated causes in this comparison.

Native Relay-BP could not be installed on this integration machine because its
source build could not initialize the required Rust toolchain. Its upstream
results remain recorded evidence; native-extension runtime tests are skipped
locally. Routing/fallback unit tests can use deterministic stand-ins, while
the LSTM models and complete raw holdout replay run without that extension.
