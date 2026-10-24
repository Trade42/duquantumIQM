"""
Minimal d=3 rotated surface code memory experiment in Stim (simulation only).

With-reset reference only; the hardware runs the no-reset variant.
"""

import stim
import pymatching
import numpy as np

DISTANCE = 3
ROUNDS = 3
SHOTS = 100_000


def build_circuit(p: float, rounds: int = ROUNDS) -> stim.Circuit:
    """Z-basis memory experiment with uniform circuit-level depolarizing noise."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=DISTANCE,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def logical_error_rate(circuit: stim.Circuit, shots: int = SHOTS) -> float:
    """Sample detectors, decode with MWPM, compare against true observable.

    Returns the logical error probability after `rounds` rounds (not per round).
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots, separate_observables=True)
    pred = matcher.decode_batch(dets)
    return np.mean(np.any(pred != obs, axis=1))


if __name__ == "__main__":
    c = build_circuit(p=1e-3)
    print(f"qubits={c.num_qubits}  detectors={c.num_detectors}  "
          f"observables={c.num_observables}  measurements={c.num_measurements}")
    print(f"code distance (graphlike) = "
          f"{len(c.shortest_graphlike_error())}\n")

    print(f"{'p':>8} {'P_L':>10}")
    for p in [1e-3, 3e-3, 5e-3, 1e-2]:
        print(f"{p:>8.0e} {logical_error_rate(build_circuit(p)):>10.2e}")
