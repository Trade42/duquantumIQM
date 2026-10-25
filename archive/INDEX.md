# Experiment archive

This folder preserves superseded study reports, acquisition bundles and
documentation at their original byte hashes. Relative artifact paths are
preserved beneath this folder. Scientific source snapshots used by the seals
are present at the matching relative paths.

Run the full bounded-study audit from the repository root:

```powershell
python archive/replay.py --run results/decoders/history/20261025T004600Z
```

For an isolated optional runtime at the repository's `results/headline/runtime`:

```powershell
python archive/replay.py --run results/decoders/history/20261025T004600Z --runtime ../results/headline/runtime
```

Run the isolated-package evidence replay:

```powershell
python ethan/tools/replay_benchmarks.py
```

The archived reports preserve their study definitions, alternative fits and
experiment chronology. The replay commands read saved records and states.
