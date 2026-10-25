"""dqiqm.layout: zero-SWAP placement on IQMFakeAphrodite, the native converter, and the measurement order."""

import math

import numpy as np
import pytest
import stim
from iqm.qiskit_iqm import IQMFakeAphrodite
from qiskit_aer import AerSimulator

from dqiqm import hw
from dqiqm import layout as L

SEED = 2026


@pytest.fixture(scope="module")
def backend():
    return IQMFakeAphrodite()


@pytest.fixture(scope="module")
def arch(backend):
    return L.Architecture.from_backend(backend)


@pytest.fixture(scope="module")
def placement(arch):
    return L.placements(L.standin_circuit(3, 1, "Z"), arch)[0]


# --- the grid and the placements ----------------------------------------------------------


def test_aphrodite_grid_puts_every_coupler_on_a_unit_step(arch):
    coords = arch.coords
    assert len(coords) == 54 and len(arch.couplers) == 90
    for c in arch.couplers:
        (ra, ca), (rb, cb) = (coords[q] for q in c)
        assert abs(ra - rb) + abs(ca - cb) == 1
    # the organisers' 5x5 rows (QB41-45, ..., QB9-13) are rows of the grid, one above the other
    rows = [[f"QB{41 - 8 * r + c}" for c in range(5)] for r in range(5)]
    for r, row in enumerate(rows):
        cells = [coords[q] for q in row]
        assert {rc[0] for rc in cells} == {cells[0][0]}
        assert [rc[1] for rc in cells] == list(range(cells[0][1], cells[0][1] + 5))
        if r:
            assert cells[0][0] == coords[rows[r - 1][0]][0] - 1


def test_architecture_from_a_hw_snapshot_matches_the_backend(backend, arch):
    assert L.Architecture.from_dqa(backend.architecture.model_dump(mode="json")) == arch


def test_d3_has_92_placements_on_23_qubit_sets_and_d5_fits(arch):
    p3 = L.placements(L.standin_circuit(3, 1, "Z"), arch)
    assert len(p3) == 92
    assert len({p.qubit_set for p in p3}) == 23
    p5 = L.placements(L.standin_circuit(5, 1, "Z"), arch)
    assert len(p5) == 4 and len({p.qubit_set for p in p5}) == 1
    assert len(p5[0].qubit_set) == 49


def test_place_takes_origin_and_orientation_and_refuses_off_the_chip(arch, placement):
    src = L.standin_circuit(3, 1, "Z")
    assert L.place(src, arch, placement.origin, placement.orientation) == placement
    with pytest.raises(L.PlacementError):
        L.place(src, arch, (0, 0), 0)  # the chip's top-left corner has no qubits


def test_a_broken_coupler_removes_the_placements_that_use_it(arch, placement):
    src = L.standin_circuit(3, 1, "Z")
    a, b = (placement.physical[q] for q in L.cx_pairs(src)[0])
    broken = arch.with_couplers(arch.couplers - {frozenset((a, b))})  # same grid frame, same origins
    assert broken.coords == arch.coords
    with pytest.raises(L.PlacementError, match="not a coupler"):
        L.place(src, broken, placement.origin, placement.orientation)
    left = L.placements(src, broken)
    assert len(left) < 92
    assert not any({p.physical[x], p.physical[y]} == {a, b} for p in left for x, y in L.cx_pairs(src))
    # inferred afresh, the broken grid is still a grid, and its placements are the same ones
    fresh = L.Architecture(broken.qubits, broken.couplers)
    assert {p.mapping for p in L.placements(src, fresh)} == {p.mapping for p in left}


# --- the converter ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_prx_then_rz_decomposes_any_unitary(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    u, _ = np.linalg.qr(z)
    theta, phi, alpha = L.prx_then_rz(u)
    assert 0 <= theta <= math.pi
    assert L._same_up_to_phase(u, L._rz(alpha) @ L._prx(theta, phi))


def test_hardware_circuit_has_zero_swaps_and_every_cz_on_a_coupler(backend, arch, placement):
    src = L.standin_circuit(3, 1, "Z", reset=False)
    hc = L.to_hardware(src, backend, placement)  # runs check_hardware
    assert "swap" not in hc.circuit.count_ops()
    assert list(hc.circuit.layout.initial_index_layout(filter_ancillas=True)) == list(hc.initial_layout)
    assert sorted(hc.qubit_index_to_name.values()) == sorted(placement.qubit_set)
    sent = L.sent_instructions(hc.circuit, hc.qubit_index_to_name)
    czs = [i.locus for i in sent if i.name == "cz"]
    assert len(czs) == 24 and all(arch.coupled(*locus) for locus in czs)
    assert {i.name for i in sent} == {"prx", "cz", "measure", "barrier"}
    assert {q for i in sent for q in i.locus} == placement.qubit_set  # nothing sent to other qubits
    assert sum(i.name == "prx" for i in sent) == 38  # 8 ancillas x 2, data basis changes 22


def test_hw_records_the_placed_qubits_for_each_logical_qubit(backend, placement):
    # IQM's level-0 rebuild makes the layout's registers the circuit's own, so the serializer and
    # hw.submit's meta.json both read the layout: logical qubit i -> the qubit Stim qubit i is placed on
    hc = L.to_hardware(L.standin_circuit(3, 1, "Z", reset=False), backend, placement)
    assert hc.circuit.layout.initial_layout.get_registers() == set(hc.circuit.qregs)
    (mapping,) = hw._qubit_mappings(backend, [hc.circuit], hc.qubit_index_to_name)
    assert mapping == {i: placement.physical[q] for i, q in enumerate(hc.stim_qubits)}


def test_hardware_target_has_the_simulation_targets_gates_without_resets(backend, placement):
    for basis in "ZX":
        hard = L.to_hardware(L.standin_circuit(3, 2, basis, reset=False), backend, placement)
        sim = L.to_hardware(L.standin_circuit(3, 2, basis), backend, placement, target="simulation")
        a = L._per_qubit(L.sent_instructions(hard.circuit, hard.qubit_index_to_name))
        b = L._per_qubit(i for i in L.sent_instructions(sim.circuit, sim.qubit_index_to_name) if i.name != "reset")
        assert a == b


@pytest.mark.parametrize("basis", "ZX")
def test_each_round_reads_out_in_one_barrier_segment_of_measurements_only(basis):
    qc, _ = L.native_circuit(L.standin_circuit(3, 3, basis, reset=False))
    segments = [[]]
    for ci in qc.data:
        if ci.operation.name == "barrier":
            segments.append([])
        else:
            segments[-1].append(ci.operation.name)
    readouts = [s for s in segments if "measure" in s]
    assert len(readouts) == 3  # rounds 1, 2 and 3 with the data qubits
    assert all(set(s) == {"measure"} for s in readouts)
    assert [len(s) for s in readouts] == [8, 8, 8 + 9]


@pytest.mark.parametrize(
    "text",
    [
        "R 0\nMR 0",  # no mid-circuit reset on Emerald
        "R 0\nH 0\nR 0\nM 0",  # R after the start
        "R 0\nX_ERROR(0.1) 0\nM 0",  # noise
        "R 0\nM(0.01) 0",  # a noisy readout (contract, amended 18:15 UTC)
        "R 0\nX 0\nM 0",  # outside the contract
        "RX 0\nMX 0",  # the contract has no X-basis reset or measurement
    ],
)
def test_hardware_target_refuses_what_emerald_cannot_run(text):
    with pytest.raises(ValueError):
        L.native_circuit(stim.Circuit(text))


def test_simulation_target_strips_noise():
    noisy = stim.Circuit("R 0 1\nH 0\nDEPOLARIZE1(0.1) 0\nCX 0 1\nM(0.05) 0 1")
    clean = stim.Circuit("R 0 1\nH 0\nCX 0 1\nM 0 1")
    a, _ = L.native_circuit(noisy, target="simulation")
    b, _ = L.native_circuit(clean, target="simulation")
    assert a == b


def test_the_x_basis_standin_is_noiselessly_right():
    src = L.standin_circuit(3, 3, "X")
    assert not {"RX", "MX"} & {i.name for i in src}
    gen = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3)
    assert (src.num_measurements, src.num_detectors, src.num_observables) == (
        gen.num_measurements,
        gen.num_detectors,
        gen.num_observables,
    )
    det, obs = src.compile_detector_sampler(seed=SEED).sample(256, separate_observables=True)
    assert not det.any() and not obs.any()


# --- the measurement order ----------------------------------------------------------------


def test_bit_order_measurement_j_is_clbit_j_is_memory_column_j():
    # record order differs from qubit order: M 2 0 1 records (m2, m0, m1) = (0, 0, 1)
    qc, qubits = L.native_circuit(stim.Circuit("R 0 1 2\nX 1\nM 2 0 1"), target="simulation")
    assert qubits == (0, 1, 2)
    res = AerSimulator(seed_simulator=SEED).run(qc, shots=16, memory=True).result()
    assert res.get_counts() == {"100": 16}  # Qiskit prints clbit 0 rightmost
    bits = hw._memory_bits(res.get_memory())  # how hw.save_raw stores per-shot memory
    assert bits.tolist() == [[0, 0, 1]] * 16


def test_noiseless_aer_run_fires_no_detector_of_the_source(backend, placement):
    src = L.standin_circuit(3, 2, "Z")  # with reset: simulation target
    hc = L.to_hardware(src, backend, placement, target="simulation")
    res = AerSimulator(seed_simulator=SEED).run(hc.circuit, shots=64, memory=True).result()
    bits = hw._memory_bits(res.get_memory())
    assert bits.shape == (64, src.num_measurements)
    det, obs = src.compile_m2d_converter().convert(measurements=bits.astype(bool), separate_observables=True)
    assert not det.any() and not obs.any()
    # not a trivially quiet record: the X checks (4 per round) and the 9 data qubits are random
    assert (bits.min(axis=0) != bits.max(axis=0)).sum() == 4 * 2 + 9

    # one fault: X on the centre data qubit after the 8th TICK must fire what it fires in the source
    faulty = L.insert_pauli(src, 8, 10, "X")
    want, _ = src.compile_m2d_converter().convert(
        measurements=faulty.compile_sampler(seed=SEED).sample(1), separate_observables=True
    )
    assert want.any()
    hf = L.to_hardware(faulty, backend, placement, target="simulation")
    res = AerSimulator(seed_simulator=SEED).run(hf.circuit, shots=16, memory=True).result()
    got, _ = src.compile_m2d_converter().convert(
        measurements=hw._memory_bits(res.get_memory()).astype(bool), separate_observables=True
    )
    assert (got == want[0]).all()


@pytest.mark.parametrize("basis", "ZX")
def test_converted_circuit_is_exactly_equivalent_and_faults_fire_alike(basis):
    src = L.standin_circuit(3, 2, basis)
    native, _ = L.native_circuit(src, target="simulation")
    L.check_equivalent(native, src)
    # every single X or Z on every qubit right after the first round's measurements
    tick = [i.name for i in src.flattened() if i.name in ("TICK", "MR")].index("MR")
    m2d = src.compile_m2d_converter()
    for q in L._gate_qubits(src):
        for pauli in "XZ":
            faulty = L.insert_pauli(src, tick, q, pauli)
            want = m2d.convert(measurements=faulty.compile_sampler(seed=SEED).sample(1), separate_observables=True)
            rt, clbits = L.native_to_stim(L.native_circuit(faulty, target="simulation")[0], src)
            sample = rt.compile_sampler(seed=SEED).sample(1)
            record = np.zeros_like(sample)
            record[:, clbits] = sample
            got = m2d.convert(measurements=record, separate_observables=True)
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]), (q, pauli)
