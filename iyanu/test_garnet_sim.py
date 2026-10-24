import pytest

pytest.importorskip("iqm.qiskit_iqm")

from challenge.run_on_hardware import prepare_qiskit
from iyanu.garnet_sim import NATIVE, couplings, fake_garnet, grid_from_couplings, snapshot_from_fake


@pytest.fixture(scope="module")
def garnet():
    backend = fake_garnet()
    return backend, couplings(backend), snapshot_from_fake(backend)


def test_grid_embeds_every_coupler_as_a_unit_edge(garnet):
    backend, edges, _ = garnet
    grid = grid_from_couplings(edges)
    pos = {q: (r, c) for r, row in enumerate(grid) for c, q in enumerate(row) if q}
    assert len(pos) == backend.num_qubits == 20 and len(edges) == 30
    assert all(abs(pos[a][0]-pos[b][0])+abs(pos[a][1]-pos[b][1]) == 1 for a, b in edges)
    adjacent = {frozenset((a, b)) for a in pos for b in pos
                if abs(pos[a][0]-pos[b][0])+abs(pos[a][1]-pos[b][1]) == 1}
    assert adjacent == {frozenset(e) for e in edges}


def test_grid_refuses_a_non_grid_graph():
    with pytest.raises(ValueError):
        grid_from_couplings([("QB1", "QB2"), ("QB2", "QB3"), ("QB3", "QB1")])


def test_snapshot_round_trips_the_depolarizing_channel(garnet):
    backend, _, cal = garnet
    lam = backend.error_profile.two_qubit_gate_depolarizing_error_parameters["cz"][("QB1", "QB2")]
    assert cal.edge_error("QB1", "QB2") == pytest.approx(0.75*lam)
    assert 1.25*cal.edge_error("QB1", "QB2") == pytest.approx(15/16*lam)
    assert set(cal.qubits) == {backend.index_to_qubit_name(i) for i in range(20)}


@pytest.mark.parametrize("rounds", [1, 3])
def test_compiles_native_on_couplers_with_zero_swaps(garnet, rounds):
    backend, edges, cal = garnet
    p = prepare_qiskit(backend, 3, rounds, "Z", calibration=cal, grid=grid_from_couplings(edges))
    c = p["circuit"]
    assert set(c.count_ops()) <= NATIVE
    assert p["report"]["routing_swaps"] == 0 and p["report"]["physical_qubits"] == 17
    names = {frozenset(e) for e in edges}
    for inst in c.data:
        if inst.operation.name == "cz":
            pair = frozenset(backend.index_to_qubit_name(c.find_bit(q).index) for q in inst.qubits)
            assert pair in names
