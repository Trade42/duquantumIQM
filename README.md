# duquantumIQM

DuQuantum 2026, IQM challenge: a rotated surface-code memory experiment without resets on IQM Emerald, decoded offline, with the logical error rate fitted from P_L against rounds. See [the challenge](https://github.com/svijaymurugan/DuQuantum-IQM-Challenge-2026).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is pinned in `.python-version`.

```
uv sync
cp .env.example .env    # only for hardware access: put your IQM Resonance token in it
```

## Test

```
uv run pytest
```

The tests run offline and need no token. Tests that use hardware are marked `hw` and deselected by default.

## Layout

| Path | What |
|---|---|
| `src/dqiqm/` | the package; `hw.py` is the only path to IQM Resonance |
| `tests/` | offline tests |
| `data/` | calibration snapshots, raw hardware results, simulation tables (`data/README.md`) |
| `CLAUDE.md` | project rules, hardware rules and the review checklist |
