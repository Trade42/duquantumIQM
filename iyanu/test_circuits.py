from collections import Counter

import pytest

pytest.importorskip("iqm.qiskit_iqm")

from challenge.build_emerald_qubit_rotated import interaction_pairs
from iyanu.circuits import VARIANTS, prepare, vanilla_circuit
from iyanu.garnet_sim import couplings, fake_garnet, grid_from_couplings, snapshot_from_fake


@pytest.fixture(scope="module")
def garnet():
    b = fake_garnet()
    return b, snapshot_from_fake(b), grid_from_couplings(couplings(b))


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_vanilla_noiseless_has_no_events_or_flips(basis, rounds):
    c = vanilla_circuit(rounds, basis)
    dets, obs = c.compile_detector_sampler(seed=1).sample(500, separate_observables=True)
    assert not dets.any() and not obs.any()
    assert sum(1 for i in c.flattened() if i.name == "CZ" for _ in i.targets_copy()[::2]) == 24*rounds


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_serialized_czs_match_placement(garnet, variant, basis):
    from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
    b, cal, grid = garnet
    p = prepare(b, variant, 3, 2, basis, calibration=cal, grid=grid)
    sent = serialize_instructions(p["circuit"], {i: b.index_to_qubit_name(i) for i in range(b.num_qubits)})
    got = Counter(frozenset(i.locus) for i in sent if i.name == "cz")
    m = p["placement"].stim_to_physical
    want = Counter(frozenset((m[a], m[c])) for a, c in interaction_pairs(p["stim_circuit"]))
    edges = {frozenset(e) for e in couplings(b)}
    assert got == want and set(got) <= edges
    assert {i.name for i in sent} <= {"prx", "cz", "measure", "barrier"}
    assert p["report"]["routing_swaps"] == 0
