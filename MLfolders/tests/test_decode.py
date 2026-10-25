"""dqiqm.decode: the measurement record's column order, m2d, the decoder, and both hardware loaders."""

import json

import numpy as np
import pytest
import stim

from dqiqm import decode
from dqiqm.decode import counts_to_measurements, detection_events, load_memory, logical_error_probability

SEED = 2026
P = 0.003


def standin(basis: str, rounds: int, p: float = P) -> stim.Circuit:
    """The M1 stand-in: Stim's generated with-reset d=3 rotated memory circuit, uniform noise p."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def record_targets(circuit: stim.Circuit) -> tuple[list[list[int]], list[list[int]]]:
    """Each detector's and each observable's measurement columns: walk the circuit, resolve rec[-k]."""
    det_cols, obs_cols = [], [[] for _ in range(circuit.num_observables)]
    m = 0
    for inst in circuit.flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            cols = [m + t.value for t in inst.targets_copy() if t.is_measurement_record_target]
            if inst.name == "DETECTOR":
                det_cols.append(cols)
            else:
                obs_cols[int(inst.gate_args_copy()[0])].extend(cols)
        else:
            m += inst.num_measurements
    assert m == circuit.num_measurements
    return det_cols, obs_cols


def reference_events(circuit: stim.Circuit, measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detectors and observables computed by hand, independently of compile_m2d_converter.

    Like m2d, each parity is compared with the circuit's noiseless reference sample.
    """
    det_cols, obs_cols = record_targets(circuit)
    ref = circuit.reference_sample().astype(np.uint8)
    rec = np.asarray(measurements, dtype=np.uint8)

    def parity(cols):
        return (rec[:, cols].sum(axis=1) + ref[cols].sum()) % 2 == 1

    return np.stack([parity(c) for c in det_cols], axis=1), np.stack([parity(c) for c in obs_cols], axis=1)


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_noiseless_record_has_no_events(basis, rounds):
    circuit = standin(basis, rounds).without_noise()
    assert (circuit.num_detectors, circuit.num_observables) == (8 * rounds, 1)
    assert circuit.num_measurements == 8 * rounds + 9
    record = circuit.compile_sampler(seed=SEED).sample(1000).astype(np.uint8)
    if basis == "X":
        assert record.any(axis=0).sum() > 8  # X basis: random first-round Z checks, so the test has teeth
    dets, obs = detection_events(circuit, record)
    assert dets.shape == (1000, 8 * rounds) and obs.shape == (1000, 1)
    assert not dets.any() and not obs.any()


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_m2d_matches_a_hand_xor_of_the_detector_targets(basis):
    circuit = standin(basis, 3, p=0.02)
    record = circuit.compile_sampler(seed=SEED).sample(2000)
    dets, obs = detection_events(circuit, record)
    want_dets, want_obs = reference_events(circuit, record)
    assert dets.any() and obs.any()
    np.testing.assert_array_equal(dets, want_dets)
    np.testing.assert_array_equal(obs, want_obs)


def test_one_flipped_measurement_fires_exactly_its_detectors():
    """Column j is Stim measurement j: flipping it fires exactly the detectors that name measurement j."""
    circuit = standin("Z", 2).without_noise()
    det_cols, obs_cols = record_targets(circuit)
    record = circuit.compile_sampler(seed=SEED).sample(1).astype(np.uint8)
    for j in range(circuit.num_measurements):
        flipped = record.copy()
        flipped[0, j] ^= 1
        dets, obs = detection_events(circuit, flipped)
        want_dets = [cols.count(j) % 2 == 1 for cols in det_cols]
        want_obs = [cols.count(j) % 2 == 1 for cols in obs_cols]
        assert dets[0].tolist() == want_dets and obs[0].tolist() == want_obs
        assert 1 <= dets.sum() <= 2  # every measurement sits in one or two detectors here


def test_detection_events_refuses_a_wrong_record():
    circuit = standin("Z", 2)
    good = np.zeros((4, circuit.num_measurements), dtype=np.uint8)
    with pytest.raises(ValueError, match="columns"):
        detection_events(circuit, good[:, :-1])
    with pytest.raises(ValueError, match="2-D"):
        detection_events(circuit, good[0])
    bad = good.copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="0/1"):
        detection_events(circuit, bad)


def test_decode_from_circuit_and_from_dem_agree_and_correct_errors():
    circuit = standin("Z", 3)
    record = circuit.compile_sampler(seed=SEED).sample(5000)
    dets, obs = detection_events(circuit, record)
    pred = decode.decode(circuit, dets)
    dem = circuit.detector_error_model(decompose_errors=True)
    np.testing.assert_array_equal(pred, decode.decode(dem, dets))
    assert pred.shape == obs.shape and pred.dtype == np.bool_
    decoded = logical_error_probability(pred, obs)
    raw = logical_error_probability(np.zeros_like(obs), obs)
    # 0.0062 decoded (31 / 5,000) and 0.064 raw at stim 1.16.0, seed 2026
    assert decoded.p < 0.05 and decoded.p < raw.p / 3


def test_decode_refuses_a_noiseless_or_mismatched_model():
    circuit = standin("Z", 3)
    dets = np.zeros((3, circuit.num_detectors), dtype=bool)
    with pytest.raises(ValueError, match="no error mechanisms"):
        decode.decode(circuit.without_noise(), dets)
    with pytest.raises(ValueError, match="must match"):
        decode.decode(standin("Z", 2), dets)
    with pytest.raises(TypeError):
        decode.decode("not a circuit", dets)


def test_logical_error_probability_counts_any_observable_and_gives_wilson():
    pred = np.array([[0, 0], [1, 0], [0, 1], [1, 1]] * 25, dtype=bool)
    obs = np.array([[0, 0], [1, 0], [0, 0], [1, 0]] * 25, dtype=bool)
    k, n, p, lo, hi = logical_error_probability(pred, obs)
    assert (k, n, p) == (50, 100, 0.5)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    zero = logical_error_probability(obs, obs)
    assert zero.k == 0 and zero.wilson_lo == 0.0 and zero.wilson_hi == pytest.approx(0.0370, abs=1e-4)


# ---------------------------------------------------------------------------
# Loaders: the bit order is pinned against real Qiskit output (Aer), and against Stim.
# ---------------------------------------------------------------------------

# Stim and Qiskit versions of one record whose five measurements are 1, 0, 0, 1, 1: no reversal
# or register shuffle maps it onto itself.
STIM_PATTERN = stim.Circuit("X 0\nM 0 1 2\nX 1\nM 1 0")
PATTERN = np.array([1, 0, 0, 1, 1], dtype=np.uint8)


def qiskit_pattern(two_registers: bool = False):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    q = QuantumRegister(3, "q")
    if two_registers:
        a, b = ClassicalRegister(3, "a"), ClassicalRegister(2, "b")
        qc = QuantumCircuit(q, a, b)
    else:
        qc = QuantumCircuit(q, ClassicalRegister(5, "c"))
    clbits = qc.clbits
    qc.x(0)
    qc.measure(q[0], clbits[0])
    qc.measure(q[1], clbits[1])
    qc.measure(q[2], clbits[2])
    qc.x(1)
    qc.measure(q[1], clbits[3])
    qc.measure(q[0], clbits[4])
    return qc


@pytest.fixture(scope="module")
def aer_results():
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=SEED)
    return {two: sim.run(qiskit_pattern(two), shots=50, memory=True).result() for two in (False, True)}


def test_stim_pattern_is_the_pattern():
    np.testing.assert_array_equal(STIM_PATTERN.compile_sampler(seed=SEED).sample(3), np.tile(PATTERN, (3, 1)))


def test_counts_loader_bit_order_by_hand():
    # rightmost character is clbit 0
    got = counts_to_measurements({"011": 2, "100": 1}, 3, shots=3)
    np.testing.assert_array_equal(got, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    # two registers a (clbits 0-2) and b (clbits 3-4): Qiskit prints "b a"
    np.testing.assert_array_equal(counts_to_measurements({"11 001": 1}, 5), [PATTERN])
    np.testing.assert_array_equal(counts_to_measurements({"0x19": 1}, 5), [PATTERN])  # 0b11001
    with pytest.raises(ValueError, match="bits"):
        counts_to_measurements({"0110": 1}, 3)
    with pytest.raises(ValueError, match="sum"):
        counts_to_measurements({"011": 2}, 3, shots=3)


@pytest.mark.parametrize("two_registers", [False, True])
def test_counts_loader_bit_order_against_qiskit_and_stim(aer_results, two_registers):
    counts = aer_results[two_registers].get_counts()
    assert len(counts) == 1
    got = counts_to_measurements(counts, STIM_PATTERN.num_measurements, shots=50)
    np.testing.assert_array_equal(got, STIM_PATTERN.compile_sampler(seed=SEED).sample(50).astype(np.uint8))


@pytest.mark.parametrize("two_registers", [False, True])
def test_memory_loader_bit_order_through_save_raw(aer_results, tmp_path, two_registers):
    from dqiqm import hw

    result = aer_results[two_registers]
    job_dir = hw.save_raw(
        result, task="decode-fit-test", device="aer", qubit_mapping=[{0: "QB1", 1: "QB2", 2: "QB3"}],
        shots=50, day="20261024", data_root=tmp_path,
    )
    bits, meta = load_memory(job_dir, 0, circuit=STIM_PATTERN)
    assert bits.dtype == np.uint8 and meta["shots"] == 50
    np.testing.assert_array_equal(bits, STIM_PATTERN.compile_sampler(seed=SEED).sample(50).astype(np.uint8))


def write_job(job_dir, arrays, *, shots, success=True):
    job_dir.mkdir(parents=True)
    meta = {"job_id": "test-job", "shots": shots, "circuits": len(arrays), "success": success}
    (job_dir / "meta.json").write_text(json.dumps(meta))
    if success:
        np.savez_compressed(job_dir / "memory.npz", **{f"c{i}": a for i, a in enumerate(arrays)})
    return job_dir


def test_memory_loader_to_detection_events_offline(tmp_path):
    """The hardware path end to end: noiseless records in memory.npz -> load -> m2d -> no events."""
    circuits = [standin("Z", r).without_noise() for r in (1, 2)]
    arrays = [c.compile_sampler(seed=SEED).sample(200).astype(np.uint8) for c in circuits]
    job = write_job(tmp_path / "job", arrays, shots=200)
    for i, circuit in enumerate(circuits):
        bits, _ = load_memory(job, i, circuit=circuit)
        np.testing.assert_array_equal(bits, arrays[i])
        dets, obs = detection_events(circuit, bits)
        assert not dets.any() and not obs.any()
    with pytest.raises(ValueError, match="measurements"):
        load_memory(job, 0, circuit=circuits[1])
    with pytest.raises(IndexError):
        load_memory(job, 2)
    with pytest.raises(ValueError, match="did not succeed"):
        load_memory(write_job(tmp_path / "failed", [], shots=200, success=False))
    with pytest.raises(ValueError, match="shots"):
        load_memory(write_job(tmp_path / "short", [arrays[0][:100]], shots=200))
