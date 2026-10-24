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

Transversal Pauli/H gate profiles, scheduled logical operations, frame tracking
and measured gate-count/logical-failure comparisons:
[transversal implementation and results](TRANSVERSAL_GATES.md).

Agents implementing decoders: start with [AGENTS.md](AGENTS.md) and the
[decoder coordination guide](docs/DECODER_AGENT_GUIDE.md).
