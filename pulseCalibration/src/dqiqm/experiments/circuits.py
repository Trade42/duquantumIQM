"""One experiment circuit, from its Stim source to the placed hardware circuit, and its checks.

A :class:`CircuitSpec` is everything the record needs about one circuit: the noise-free Stim
source with its detectors and observables (the circuit that ran, in Stim), the placed hardware
circuit (``target="hardware"`` only), the role and stabilizer of every qubit, the round of every
measurement, and a builder for the noisy copy a decoder uses.

How each family reaches hardware
--------------------------------
* **Surface code (E1), stability (E5), repetition code (E2) in |0> or |+>:** the Stim source goes
  through :func:`dqiqm.layout.native_circuit` and onto the placement.
* **States that need an X (E2 in |1...1>, E3):** the contract has no X, and the converter's
  hardware target refuses it. The X-free source is converted, then ``r(pi, 0)`` (an X) is
  prepended on the chosen qubits (:func:`prepend_x`). The Stim source *with* the X gates is the
  reference for every check.
* **Idles (E4):** ``delay`` is not in the converter either. The native circuit of the delay-free
  source gets a barrier, a ``delay`` on every qubit and a barrier after each round's readout but
  the last (:func:`insert_round_delays`).

Every placed circuit is checked by :func:`place` before it is returned:

* :func:`dqiqm.layout.check_hardware` (no SWAP, one ``cz`` per CX on a coupler, measurement j in
  clbit j on its placed qubit, the per-qubit sequence equal to the logical circuit's), against the
  source with its X gates, for every circuit without delays;
* for a circuit with delays, :func:`check_delays`: with its delays removed, every qubit's sent
  sequence equals that of the same circuit without them, which ``check_hardware`` proves; and each
  qubit has exactly the delays asked for, of the right duration;
* IQM's own ``validate_circuit`` against the backend (the native gate set and loci of the device);
* :func:`dqiqm.layout.check_equivalent` on the logical circuit (no delays): read back into Stim
  with the source's detectors, every detector and observable is deterministic and 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import stim
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate

from dqiqm import layout as L

SEED = L.SEED


@dataclass(frozen=True)
class Stabilizer:
    """An ancilla's check: ``label`` (unique in the circuit), ``pauli`` ("X" or "Z") and its data support."""

    label: str
    pauli: str
    support: tuple[int, ...]  # Stim qubits


@dataclass(frozen=True)
class CircuitSpec:
    """One circuit of the suite. ``hardware.circuit`` is what runs; ``stim`` is what it means."""

    experiment: str  # "E1" ... "E5"
    name: str  # unique and file-safe, e.g. "E1_d3_Z_r03_pA"
    stim: stim.Circuit  # noise-free, with detectors, observables and QUBIT_COORDS
    hardware: L.HardwareCircuit  # target="hardware", placed and checked
    params: Mapping[str, Any]  # family, rounds, basis, patch, prep, delay_ns, barriers, ...
    roles: Mapping[int, str]  # Stim qubit -> "data", "ancilla" or "readout"
    stabilizers: Mapping[int, Stabilizer]  # ancilla Stim qubit -> its check
    meas_round: tuple[int, ...]  # per measurement, its round (1-based; a final data readout: rounds + 1)
    noisy: Callable[[Any], stim.Circuit] | None = None  # NoiseModel -> the noisy copy, same detectors
    x_ancillas: tuple[int, ...] = ()  # ancillas that carry their own H (X-type checks)
    cx_pairs: tuple[tuple[int, int], ...] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)

    @property
    def placement(self) -> L.Placement:
        return self.hardware.placement

    @property
    def physical(self) -> dict[int, str]:
        return self.hardware.placement.physical

    @property
    def rounds(self) -> int:
        return int(self.params.get("rounds", 1))

    def stim_sha256(self) -> str:
        return hashlib.sha256(str(self.stim).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Native-circuit edits.
# ---------------------------------------------------------------------------


def prepend_x(native: QuantumCircuit, qubits: Iterable[int]) -> QuantumCircuit:
    """``native`` with an X, ``r(pi, 0)``, on each of ``qubits`` (logical indices) first, then a barrier."""
    qubits = sorted(set(qubits))
    if not qubits:
        return native
    out = QuantumCircuit(*native.qregs, *native.cregs, name=native.name)
    for q in qubits:
        out.append(RGate(3.141592653589793, 0.0), [out.qubits[q]])
    out.barrier()
    for ci in native.data:
        out.append(ci.operation, [out.qubits[native.find_bit(q).index] for q in ci.qubits],
                   [out.clbits[native.find_bit(c).index] for c in ci.clbits])
    return out


def insert_round_delays(native: QuantumCircuit, delay_ns: float, after_readouts: Sequence[int]) -> QuantumCircuit:
    """``native`` with barrier, ``delay(delay_ns)`` on every qubit, barrier, after the given readout layers.

    A readout layer is a maximal run of ``measure`` instructions (the converter, with barriers,
    writes each round's measurements as one). ``after_readouts`` are 1-based layer numbers.
    """
    if delay_ns <= 0:
        return native
    wanted = set(after_readouts)
    out = QuantumCircuit(*native.qregs, *native.cregs, name=native.name)
    layer = 0
    in_readout = False

    def close_readout():
        if layer in wanted:
            out.barrier()
            for q in out.qubits:
                out.delay(delay_ns, q, unit="ns")
            out.barrier()

    for ci in native.data:
        is_meas = ci.operation.name == "measure"
        if is_meas and not in_readout:
            layer += 1
            in_readout = True
        elif not is_meas and in_readout:
            in_readout = False
            close_readout()
        out.append(ci.operation, [out.qubits[native.find_bit(q).index] for q in ci.qubits],
                   [out.clbits[native.find_bit(c).index] for c in ci.clbits])
    if in_readout:
        close_readout()
    if max(wanted, default=0) > layer:
        raise ValueError(f"the circuit has {layer} readout layers; asked for a delay after {sorted(wanted)}")
    return out


# ---------------------------------------------------------------------------
# Placing and checking.
# ---------------------------------------------------------------------------


def place(
    source: stim.Circuit,
    native: QuantumCircuit,
    stim_qubits: Sequence[int],
    backend: Any,
    placement: L.Placement,
    *,
    plain: L.HardwareCircuit | None = None,
    delays: tuple[float, int] | None = None,
) -> L.HardwareCircuit:
    """Transpile ``native`` onto ``placement`` (hardware target, level 0) and check it.

    ``source`` is the Stim circuit the native one implements, X gates included. For a circuit
    with delays, pass ``plain`` (the same circuit placed without them) and ``delays`` =
    (duration in ns, delays per qubit); see the module docstring for what is checked.
    """
    physical = placement.physical
    initial = [backend.qubit_name_to_index(physical[q]) for q in stim_qubits]
    placed = transpile(native, backend=backend, initial_layout=initial, optimization_level=0, seed_transpiler=SEED)
    hc = L.HardwareCircuit(
        circuit=placed,
        qubit_index_to_name={p: backend.index_to_qubit_name(p) for p in initial},
        placement=placement,
        stim_qubits=tuple(stim_qubits),
        logical=native,
        initial_layout=tuple(initial),
        target="hardware",
    )
    if delays is None:
        L.check_hardware(hc, source, L.Architecture.from_backend(backend))
        L.check_equivalent(native, source, shots=16)
    else:
        if plain is None:
            raise ValueError("a circuit with delays needs its delay-free placement to be checked against")
        check_delays(hc, plain, *delays)
    validate(hc, backend)
    return hc


def validate(hc: L.HardwareCircuit, backend: Any) -> None:
    """IQM's own check: every instruction is native on the device's loci (``validate_circuit``)."""
    from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit

    if hc.target != "hardware":
        raise ValueError(f"{hc.circuit.name}: only target='hardware' circuits may be recorded or planned")
    validate_circuit(hc.circuit, backend, qubit_index_to_name=dict(hc.qubit_index_to_name))


def sent(hc: L.HardwareCircuit) -> list[Any]:
    """The instructions IQM's serializer makes of the circuit: what the server receives."""
    return L.sent_instructions(hc.circuit, hc.qubit_index_to_name)


def check_delays(hc: L.HardwareCircuit, plain: L.HardwareCircuit, delay_ns: float, per_qubit: int) -> None:
    """Raise ``AssertionError`` unless ``hc`` is ``plain`` plus ``per_qubit`` delays of ``delay_ns`` on every qubit."""
    got = L._per_qubit(sent(hc))  # noqa: SLF001 -- the comparison check_hardware itself uses
    want = L._per_qubit(sent(plain))  # noqa: SLF001
    if set(got) != set(want):
        raise AssertionError("the delayed circuit acts on other qubits than the plain one")
    for q in want:
        delays = [args for op, _, args in got[q] if op == "delay"]
        if len(delays) != per_qubit:
            raise AssertionError(f"{q}: {len(delays)} delays, expected {per_qubit}")
        for args in delays:
            (key, value), = args
            if key != "duration" or abs(value - delay_ns * 1e-9) > 1e-15:
                raise AssertionError(f"{q}: a delay of {value} s, expected {delay_ns} ns")
        if [x for x in got[q] if x[0] not in ("delay", "barrier")] != [x for x in want[q] if x[0] != "barrier"]:
            raise AssertionError(f"{q}: without its delays, the sequence differs from the plain circuit's")
    if "swap" in hc.circuit.count_ops():
        raise AssertionError("a SWAP was inserted")


def native_json(hc: L.HardwareCircuit) -> list[dict[str, Any]]:
    """The serialized instructions as plain JSON: name, locus and args, in order."""
    return [{"name": i.name, "locus": list(i.locus), "args": _plain(i.args)} for i in sent(hc)]


def _plain(args: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (float(v) if isinstance(v, float) else v) for k, v in args.items()}


def native_sha256(hc: L.HardwareCircuit) -> str:
    return hashlib.sha256(json.dumps(native_json(hc), sort_keys=True).encode()).hexdigest()


def qasm(hc: L.HardwareCircuit) -> str:
    from qiskit import qasm3

    return qasm3.dumps(hc.circuit)


# ---------------------------------------------------------------------------
# Timing: an ASAP schedule of the logical circuit with barriers as synchronisation points.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    duration_ns: float
    meas_start_ns: tuple[float, ...]  # per clbit
    meas_layer: tuple[int, ...]  # per clbit: the barrier-delimited layer it is in (0-based)
    layers: int


def schedule(hc: L.HardwareCircuit, durations: Any) -> Schedule:
    """Start times of every measurement and the circuit's duration, from ``durations`` (a ``device.Durations``).

    Each instruction starts when all its qubits are free; a barrier waits for all of its qubits.
    That is how the circuit is meant to run with barriers; the server's own schedule may differ,
    so the times are estimates.
    """
    qc = hc.logical
    name = {i: hc.placement.physical[s] for i, s in enumerate(hc.stim_qubits)}
    free = [0.0] * qc.num_qubits
    layer = 0
    starts = [0.0] * qc.num_clbits
    layers_of = [0] * qc.num_clbits
    for ci in qc.data:
        op = ci.operation.name
        qs = [qc.find_bit(q).index for q in ci.qubits]
        if op == "barrier":
            t = max(free[q] for q in qs)
            for q in qs:
                free[q] = t
            layer += 1
            continue
        t0 = max(free[q] for q in qs)
        if op == "delay":
            dt = float(ci.operation.params[0]) * {"ns": 1.0, "us": 1e3, "s": 1e9, "dt": 1.0}[ci.operation.unit]
        else:
            dt = durations.of(op, tuple(name[q] for q in qs))
        if op == "measure":
            c = qc.find_bit(ci.clbits[0]).index
            starts[c] = t0
            layers_of[c] = layer
        for q in qs:
            free[q] = t0 + dt
    return Schedule(max(free), tuple(starts), tuple(layers_of), layer + 1)


# ---------------------------------------------------------------------------
# Small Stim helpers shared by the families.
# ---------------------------------------------------------------------------


def measured_qubits(c: stim.Circuit) -> list[int]:
    return L.measured_qubits(c)


def detector_records(c: stim.Circuit) -> list[list[int]]:
    """Per detector, the absolute measurement indices it reads, most recent first."""
    out = []
    n = 0
    for inst in c.flattened():
        if inst.name in ("M", "MR"):
            n += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            out.append(sorted((n + t.value for t in inst.targets_copy()), reverse=True))
    return out


def observable_records(c: stim.Circuit) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    n = 0
    for inst in c.flattened():
        if inst.name in ("M", "MR"):
            n += len(inst.targets_copy())
        elif inst.name == "OBSERVABLE_INCLUDE":
            k = int(inst.gate_args_copy()[0])
            out.setdefault(k, []).extend(n + t.value for t in inst.targets_copy())
    return {k: sorted(v, reverse=True) for k, v in out.items()}
