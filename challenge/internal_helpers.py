"""Measurement order and hardware bit ordering, shared across both bases."""
import numpy as np
import stim

MEASURE = {"M", "MX", "MY", "MR", "MRX", "MRY"}


def get_qubit_lists(circuit: stim.Circuit) -> tuple[list, list]:
    coords = circuit.get_final_qubit_coordinates()
    data, anc = [], []
    for q in sorted(set(get_meas_order(circuit))):
        xy = coords.get(q)
        (data if xy is None or (int(xy[0]) % 2 and int(xy[1]) % 2) else anc).append(q)
    return data, anc


def get_meas_order(circuit: stim.Circuit) -> list[int]:
    return [t.value for i in circuit.flattened() if i.name in MEASURE
            for t in i.targets_copy() if t.is_qubit_target]


def counts_to_measurement_array(counts: dict, num_measurements: int,
                                 total_shots: int) -> np.ndarray:
    """Qiskit little endian -> Stim order; reject truncated or malformed results."""
    if total_shots < 0 or num_measurements < 0:
        raise ValueError("Shot and measurement counts must be nonnegative.")
    result = np.empty((total_shots, num_measurements), dtype=bool)
    row = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if len(bits) != num_measurements or set(bits) - {"0", "1"}:
            raise ValueError("Invalid measurement bitstring width or alphabet.")
        if not isinstance(count, (int, np.integer)) or count < 0 or row + count > total_shots:
            raise ValueError("Invalid counts or shot total.")
        result[row:row + count] = np.array(list(bits[::-1])) == "1"
        row += count
    if row != total_shots:
        raise ValueError("Counts do not sum to requested shots.")
    return result
