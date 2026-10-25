# Logical frames and gate interfaces

The challenge experiment executes identity memory with generated detector and
observable definitions. The logical reference determines whether a measured
observable has flipped.

The circuit library also provides logical Pauli strings and logical H with
explicit frame/boundary updates. Pauli readout correction applies to raw logical
bits; detector conversion already produces reference-relative observable flips.
Logical H exchanges X/Z roles and updates future check orientation and terminal
measurement.

Static-patch S/T and a joint two-block CNOT syndrome/decoder pipeline are outside
these interfaces. The transversal CNOT helper supplies a circuit primitive.

Sources: [`logical_gates.py`](challenge/logical_gates.py),
[`surface_code.py`](challenge/surface_code.py).
The memory method is described in [Methods](docs/METHODS.md).
