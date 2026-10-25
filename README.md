# duquantumIQM

DuQuantum 2026, IQM track: a no-reset rotated surface-code memory experiment on IQM Emerald.

Implemented generator, calibration-aware native placement, ten noise models,
persistent leakage simulation, correlated decoding, statistical fitting and
benchmark CLI. See [implementation and usage](IMPLEMENTATION.md).

```powershell
python -m pip install -e '.[test,qiskit]'
python -m pytest -q
python -m challenge.experiment --synthetic-calibration --models all --rounds 1 3 5 --shots 2000
```

The synthetic snapshot is an example, not measured Emerald calibration.
Original challenge brief remains in [challenge/README.md](challenge/README.md).

The controlled [Garnet headline study](docs/GARNET_HEADLINE.md) compares the
repository's physical compilation and decoder paths using the challenge's
unshifted multi-round LER equation. It selects on pilot shots, freezes all
decoder state, and tests on fresh interleaved hardware shots. Physical or
decoder combinations are retained only when the measured evidence supports
them. The runner is `python -m challenge.benchmark_headline`; hardware
submission requires an explicit `--submit` and credential file.

**Garnet result, 300,000 shots:** retain Ethan's native/DD compiler; X keeps
correlated MWPM, Z adds training-fitted Bayesian ML decoding. Challenge R1–3
fit: **X 5.220%, Z 3.763% per round**. Z improves over standalone native ML
(3.920%; adjusted gain interval +0.037–0.274 percentage points). X additions
and physical merging did not qualify. The stationary fit is poor, so these
are descriptive challenge-equation estimates; see the report for counts,
uncertainty and the runnable adopted policy.

Transversal Pauli/H gate profiles, scheduled logical operations, frame tracking
and measured gate-count/logical-failure comparisons:
[transversal implementation and results](TRANSVERSAL_GATES.md).

Agents implementing decoders: start with [AGENTS.md](AGENTS.md) and the
[decoder coordination guide](docs/DECODER_AGENT_GUIDE.md).
