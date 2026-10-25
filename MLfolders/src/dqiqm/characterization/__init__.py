"""Garnet's coherent-error characterization: the gates our surface code uses, measured at circuit level.

The goal is one table for the pilot patch (the 17 qubits and 24 couplers of
``E1_d3_Z_r03_pA``): per qubit the signed PRX amplitude error and its axis (Z) error, per coupler
the CZ conditional-phase error and the residual local Z phases, each with a 95% interval
(``docs/garnet-calib.md`` has the why of every experiment).

* :mod:`.patch`: the patch, its 24 couplers and the surface-code round's 4 parallel CZ layers, read
  from the pilot circuit itself; the terminal-readout priors (``measure_fidelity``).
* :mod:`.model`: the error model of one ``prx(pi/2, phi)`` and one CZ, and the exact forward model
  of the C1 sequences (Bloch vector with depolarizing, T1/T2 relaxation and readout).
* :mod:`.circuits`: the experiments C1 (PRX amplification), C2 (conditional Ramsey with repeated
  CZ) and C4 (isolated twins), as layers with a barrier after every layer; placement on Garnet and
  the checks (native ops, couplers, the per-qubit sequence, a barrier between every two gates).
  C3 (CZ leakage) is a native-instruction design only: it needs ``prx_12``, which our circuit
  path cannot send.
* :mod:`.fit`: maximum-likelihood fits with 95% intervals, C1 per qubit and C2 per coupler;
  :mod:`.analysis` runs them on saved records (``hw.save_raw``'s ``memory.npz``) and makes the rows.
* :mod:`.inject`: the closure test's twin of a placed circuit, with known coherent errors added
  as exact unitaries, for Aer.
* :mod:`.table`: the table as CSV (the runbook's Section 8 schema, adapted).
* :mod:`.plan`: the hardware job plan, the pilot and its stop/go bands.
* :mod:`.leakage`: C3's native-instruction design and its readout algebra (blocked: ``prx_12``).
* :mod:`.pulse`: Track B, the pulse-level sweeps centred on the saved per-locus values (designed only).
* :mod:`.compensate`: the local CZ phases cancelled by phase tracking in an E1 circuit.

Nothing here submits or connects: the hardware path is ``dqiqm.hw.submit``, after Sal's yes.
"""
