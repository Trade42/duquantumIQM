"""The Garnet experiment suite, in priority order (``docs/garnet-design.md`` has the why of each).

* **E1** (:mod:`.memory`): the d=3 no-reset surface-code memory, Z and X, R in {1, ..., 20}, on
  Garnet's one zero-SWAP placement in two X/Z orientations; repeated in time-separated batches.
* **E2** (:mod:`.repetition`): bit-flip and phase-flip repetition codes on the longest line (19
  qubits, distance 10), up to 50 rounds.
* **E3** (:mod:`.readout`): readout characterisation on all 20 qubits: assignment (all-0, all-1,
  random states) and 15 readouts in a row without reset.
* **E4** (:mod:`.memory`): E1 with idles of 1, 2 and 5 µs between rounds.
* **E5** (:mod:`.stability`): the 9-qubit stability experiment, both duals.
* **E6**: a calibration snapshot before and after every batch (``scripts/resonance_check.py``'s
  format), no circuits.

:mod:`.device` reads the calibration and builds the noisy stand-in (IQMFakeGarnet with today's
noise); :mod:`.circuits` holds :class:`~.circuits.CircuitSpec` and the placement checks;
:mod:`.suite` builds the whole suite and the hardware job plan.
"""

from dqiqm.experiments.circuits import CircuitSpec, Stabilizer

__all__ = ["CircuitSpec", "Stabilizer"]
