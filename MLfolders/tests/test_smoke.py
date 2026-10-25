"""Offline smoke tests of the stack: Stim + PyMatching, IQM's fake backends, Qrisp, the qubit selector."""

import numpy as np
import pymatching
import pytest
import stim

SEED = 2026


def rep_code(p: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=5,
        rounds=5,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def test_a_stim_repetition_code_decodes_to_a_plausible_ler():
    circuit = rep_code(0.02)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert len(circuit.shortest_graphlike_error()) == 5
    matcher = pymatching.Matching.from_detector_error_model(dem)
    shots = 20_000
    det, obs = circuit.compile_detector_sampler(seed=SEED).sample(shots, separate_observables=True)
    ler = np.mean(np.any(matcher.decode_batch(det) != obs, axis=1))
    raw = np.mean(obs[:, 0])  # the undecoded observable flips
    # 0.0032 (64 / 20,000) at stim 1.16.0, seed 2026; raw flips 0.086
    assert 0.001 < ler < 0.01
    assert ler < raw / 5


def test_b_noiseless_repetition_code_has_no_detection_events():
    circuit = rep_code(0.02).without_noise()
    assert (circuit.num_detectors, circuit.num_observables) == (4 * 5 + 4, 1)
    det, obs = circuit.compile_detector_sampler(seed=SEED).sample(1000, separate_observables=True)
    assert not det.any()
    assert not obs.any()


def test_c_ghz_on_iqm_fake_adonis():
    from iqm.qiskit_iqm import IQMFakeAdonis
    from qiskit import QuantumCircuit, transpile

    backend = IQMFakeAdonis()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    transpiled = transpile(qc, backend, seed_transpiler=SEED)
    assert set(transpiled.count_ops()) <= {"r", "cz", "measure", "barrier"}
    counts = backend.run(transpiled, shots=2000, seed_simulator=SEED).result().get_counts()
    assert (counts.get("000", 0) + counts.get("111", 0)) / 2000 > 0.7


def test_c_fake_backends_available():
    from iqm.qiskit_iqm import IQMFakeAdonis, IQMFakeAphrodite, IQMFakeApollo, IQMFakeDeneb
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    # Apollo and Garnet (20-qubit grids) take seconds to build, so they are only imported here.
    assert callable(IQMFakeApollo) and callable(IQMFakeGarnet)
    assert IQMFakeAdonis().num_qubits == 5  # star around QB3
    assert IQMFakeDeneb().num_qubits == 6  # star: 6 qubits around a computational resonator
    # Aphrodite is a 54-qubit square grid, the size of Emerald: 90 couplers, degree at most 4
    aphrodite = IQMFakeAphrodite()
    edges = {tuple(sorted(e)) for e in aphrodite.coupling_map.get_edges()}
    degree = np.bincount(np.array(sorted(edges)).ravel())
    assert aphrodite.num_qubits == 54 and len(edges) == 90 and degree.max() == 4


@pytest.fixture(scope="module")
def rep3_noreset():
    """Distance-3 bit-flip repetition code, 2 rounds, no ancilla reset (as on Emerald)."""
    from iqm.qrisp_iqm import DetectorExperiment
    from qrisp import QuantumArray, QuantumBool, cx, measure, parity
    from qrisp.misc.stim_tools import stim_noise

    @DetectorExperiment
    def experiment(p):
        q = QuantumArray(qtype=QuantumBool(), shape=(5,))
        data, anc = q[::2], q[1::2]
        m = []
        for _ in range(2):
            for i in range(2):
                cx(data[i], anc[i])
                stim_noise("DEPOLARIZE2", p, data[i], anc[i])
                cx(data[i + 1], anc[i])
                stim_noise("DEPOLARIZE2", p, data[i + 1], anc[i])
            stim_noise("X_ERROR", p, anc)
            m.append(measure(anc))
        stim_noise("X_ERROR", p, data)
        d = measure(data)
        # Without reset the ancilla holds s_1 ^ ... ^ s_k, so the round-k detector is m_k ^ m_(k-2).
        detectors = [parity(m[0][i], expectation=0) for i in range(2)]
        detectors += [parity(m[1][i], expectation=0) for i in range(2)]  # m_2 ^ m_0, and m_0 = 0
        # final: data parity is the last syndrome; compare with s_2 = m_2 ^ m_1
        detectors += [parity(d[i], d[i + 1], m[1][i], m[0][i], expectation=0) for i in range(2)]
        return detectors, [parity(d[0], observable=True, expectation=0)]

    return experiment


def test_d_qrisp_detector_experiment(rep3_noreset):
    from qrisp.interface import StimBackend

    noiseless = rep3_noreset.to_stim(0.0)
    assert (noiseless.num_detectors, noiseless.num_observables) == (6, 1)
    det, obs = noiseless.compile_detector_sampler(seed=SEED).sample(1000, separate_observables=True)
    assert not det.any() and not obs.any()

    noisy = rep3_noreset.to_stim(0.02)
    assert len(noisy.shortest_graphlike_error()) == 3
    ler = rep3_noreset.compute_LER(0.02, shots=2000, backend=StimBackend())
    assert 0 <= ler < 0.05  # about 0.006 at p = 0.02


def test_e_qubit_selector_imports_offline():
    # the autouse no_network fixture turns any connection attempt into an error
    import iqm.qubit_selector
    from iqm.qubit_selector.qubit_selector import CalibrationDataManager

    assert iqm.qubit_selector is not None and callable(CalibrationDataManager().get_calibration_fidelities)
