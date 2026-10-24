# Agent guide for Ethan's isolated solution

The repository's parent `AGENTS.md` coordination and validation rules apply.
Read `docs/DECODER_AGENT_GUIDE.md` before decoder work.

- Own implementation: `ethan/src/iqm_surface_code/`; tests: `ethan/tests/`;
  docs: `ethan/docs/`. Keep changes within this folder for independent merging.
- Claim exact **repository-relative** paths, including the `ethan/` prefix,
  in the shared Git common directory. Changing working directories does not
  change the canonical lock resource. Never use `src/...` as a lock alias.
- Keep a decoder in its own module and tests. Claim shared integration files
  before editing them. Preserve shot order, no-reset parities, source/native
  frames, correlation handling and explicit unsupported-case failures.
- Use `iqm_surface_code` imports. The shared `challenge` package and `iyanu/`
  have independent owners; changes there require their own explicit scope.
- From `ethan/`, run `python -m pytest -q` and
  `python tools/replay_benchmarks.py`; from the repository root use
  `python -m pytest -q -c ethan/pyproject.toml ethan/tests`.
- Imports, tests and evidence replay must stay offline. Hardware jobs require
  explicit user authorization and an explicit submission entry point.
- Keep credentials, environments, generated runs and dependency caches ignored.
  Treat `benchmarks/garnet/` as immutable published measurement evidence.
- Record validation and limitations in your own claim before releasing only
  locks with your matching owner and token. Preserve other agents' edits.
