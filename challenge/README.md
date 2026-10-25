# IQM surface-code memory challenge

The retained implementation uses Stim, Qiskit and IQM's native pulse compiler
for distance-three X/Z memory on Garnet. Nine data qubits and eight reused check
ancillas store one logical qubit. Repeated checks use no mid-circuit reset.

The chain is:

```text
Annotated Stim memory -> native Qiskit/CZ placement -> audited XY4 pulses
-> measured records -> source/native detection events -> frozen X/Z decoder
-> logical failure probability versus rounds -> challenge LER fit
```

The challenge asks for a correct encoded memory, hardware-aware compilation,
low routing overhead and defensible logical-error statistics. The final
configuration uses terminal fusion/correlated matching for X and measurement
fences/posterior-count lookup for Z. Decoding occurs offline after acquisition.

- [Every retained improvement](../improvements.md)
- [Methods](../docs/METHODS.md)
- [Measured evaluation](../docs/EVALUATION.md)
- [Offline reproduction](../docs/REPRODUCIBILITY.md)
- [Scientific and software references](../docs/REFERENCES.md)

```powershell
python -m examples.reproduce
python -m challenge.audit_garnet_routing
```

Imports, tests and replay do not retrieve credentials or submit hardware jobs.
The separate Emerald bundle remains the vanilla surface-code reference.
