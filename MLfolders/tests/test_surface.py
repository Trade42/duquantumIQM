"""Structural checks of dqiqm.surface: geometry, contract, counts, noiseless runs, DEM, distance.

The 10^5-shot logical-error-rate comparisons are in scripts/surface_check.py.
"""

import itertools

import numpy as np
import pytest
import stim

from dqiqm.surface import NoiseModel, geometry, memory_circuit

SEED = 2026
DEFAULT_NOISE = dict(  # challenge/surface_code.py:DEFAULT_NOISE
    after_clifford_depolarization=0.003,
    after_reset_flip_probability=0.003,
    before_measure_flip_probability=0.003,
    before_round_data_depolarization=0.003,
)
DISTANCES = (3, 5)
ROUNDS = (1, 2, 3, 5, 10)


def product(*axes):
    return list(itertools.product(*axes))


CASES = product(DISTANCES, ROUNDS, "ZX", (True, False))

STRUCTURE = {"QUBIT_COORDS", "R", "H", "CX", "M", "MR", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"}
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}


def generated(d, r, basis, **noise):
    return stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=r, **noise)


def records(circuit):
    """Each detector's and observable 0's absolute measurement indices, read from the circuit."""
    dets, obs, n = [], [], 0
    for inst in circuit.flattened():
        targets = inst.targets_copy()
        if inst.name in ("M", "MR"):
            n += len(targets)
        elif inst.name == "DETECTOR":
            dets.append([n + t.value for t in targets])
        elif inst.name == "OBSERVABLE_INCLUDE":
            assert inst.gate_args_copy() == [0.0]
            obs += [n + t.value for t in targets]
    return dets, obs


def after_ancilla_measurement(circuit, k, gate, qubits, p=1.0):
    """The circuit with `gate(p) qubits` inserted right after the round-k ancilla measurement."""
    insts = list(circuit)
    measures = [i for i, inst in enumerate(insts) if inst.name in ("M", "MR")]
    where = measures[k - 1]
    assert where != measures[-1], "round k must not be the final data measurement"
    out = stim.Circuit()
    for i, inst in enumerate(insts):
        out.append(inst)
        if i == where:
            out.append(gate, qubits, p)
    return out


def fired(circuit):
    """Coordinates of the detectors that fire, and whether the observable flips, on one shot."""
    det, obs = circuit.compile_detector_sampler(seed=SEED).sample(1, separate_observables=True)
    coords = circuit.get_detector_coordinates()
    return {tuple(int(v) for v in coords[i]) for i in np.flatnonzero(det[0])}, bool(obs[0, 0])


def gf2_rank(m):
    m = m.copy() % 2
    rank = 0
    for col in range(m.shape[1]):
        pivot = next((r for r in range(rank, m.shape[0]) if m[r, col]), None)
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
    return rank


# ─── geometry ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d", (3, 5, 7))
def test_geometry_is_stims(d):
    g = geometry(d)
    stim_circuit = generated(d, 2, "Z")
    assert dict(g.coords) == {q: (int(x), int(y)) for q, (x, y) in stim_circuit.get_final_qubit_coordinates().items()}
    assert len(g.data) == d * d and len(g.x_ancillas) == len(g.z_ancillas) == (d * d - 1) // 2
    assert all(x % 2 == 1 and y % 2 == 1 for x, y in (g.coords[q] for q in g.data))
    assert all(x % 2 == 0 and y % 2 == 0 for x, y in (g.coords[q] for q in g.ancillas))
    # Stim's X-type ancillas are the ones that get H
    h_targets = {t.value for inst in stim_circuit.flattened() if inst.name == "H" for t in inst.targets_copy()}
    assert set(g.x_ancillas) == h_targets


@pytest.mark.parametrize("d", (3, 5, 7))
def test_stabilizers_define_one_logical_qubit_of_distance_d(d):
    g = geometry(d)
    col = {q: i for i, q in enumerate(g.data)}

    def rows(ancillas):
        h = np.zeros((len(ancillas), len(g.data)), dtype=np.uint8)
        for r, a in enumerate(ancillas):
            h[r, [col[q] for q in g.support[a]]] = 1
        return h

    def vec(qubits):
        v = np.zeros(len(g.data), dtype=np.uint8)
        v[[col[q] for q in qubits]] = 1
        return v

    hx, hz = rows(g.x_ancillas), rows(g.z_ancillas)
    assert not (hx.astype(int) @ hz.T.astype(int) % 2).any()  # X and Z stabilizers commute
    assert len(g.data) - gf2_rank(hx) - gf2_rank(hz) == 1  # k = 1
    zl, xl = vec(g.logical_z), vec(g.logical_x)
    assert len(g.logical_z) == len(g.logical_x) == d
    assert not (hx.astype(int) @ zl % 2).any() and not (hz.astype(int) @ xl % 2).any()
    assert int(zl @ xl) % 2 == 1  # Z_L and X_L anticommute
    # neither logical is a product of stabilizers
    assert gf2_rank(np.vstack([hz, zl])) == gf2_rank(hz) + 1
    assert gf2_rank(np.vstack([hx, xl])) == gf2_rank(hx) + 1
    weights = {len(s) for s in g.support.values()}
    assert weights == {2, 4}


# ─── the contract ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d,r,basis,no_reset", CASES)
def test_contract_instructions_and_measurement_order(d, r, basis, no_reset):
    g = geometry(d)
    for noise in (None, DEFAULT_NOISE):
        c = memory_circuit(d, r, basis, no_reset=no_reset, noise=noise)
        names = [inst.name for inst in c]
        allowed = STRUCTURE | (NOISE if noise else set())
        assert set(names) <= allowed, set(names) - allowed
        assert "REPEAT" not in names
        first_tick = names.index("TICK")
        assert all(i < first_tick for i, n in enumerate(names) if n == "R"), "R only at the start"
        assert ("MR" in names) == (not no_reset)
        assert set(c.get_final_qubit_coordinates()) == set(g.coords)
        # measurement order: each round's ancillas ascending, then the data ascending
        measured = [t.value for inst in c if inst.name in ("M", "MR") for t in inst.targets_copy()]
        assert measured == list(sorted(g.ancillas)) * r + list(sorted(g.data))
        assert c.num_observables == 1
        assert all(len(v) == 3 for v in c.get_detector_coordinates().values())
        if noise is None:  # M(p) is noise: a hardware circuit's measurements carry no argument
            assert all(not inst.gate_args_copy() for inst in c if inst.name in ("M", "MR"))
    assert memory_circuit(d, r, basis, no_reset=no_reset) == memory_circuit(d, r, basis, no_reset, NoiseModel())


@pytest.mark.parametrize("d,r,basis,no_reset", CASES)
def test_counts_match_the_formula(d, r, basis, no_reset):
    c = memory_circuit(d, r, basis, no_reset=no_reset)
    assert c.num_detectors == r * (d * d - 1)
    assert c.num_measurements == r * (d * d - 1) + d * d
    assert c.num_observables == 1
    assert len(c.get_final_qubit_coordinates()) == 2 * d * d - 1
    # per round: basis-type in round 1 and at the end, everyone from round 2
    ts = [int(v[2]) for v in c.get_detector_coordinates().values()]
    half = (d * d - 1) // 2
    assert [ts.count(t) for t in range(r + 1)] == [half] + [2 * half] * (r - 1) + [half]


# ─── noiseless ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d,r,basis,no_reset", CASES)
def test_noiseless_circuit_fires_nothing_and_its_observable_is_zero(d, r, basis, no_reset):
    c = memory_circuit(d, r, basis, no_reset=no_reset)
    det, obs = c.compile_detector_sampler(seed=SEED).sample(1000, separate_observables=True)
    assert not det.any() and not obs.any()
    # raw parities, not relative to Stim's reference sample: every one is 0 on every shot
    raw = c.compile_sampler(seed=SEED).sample(1000).astype(np.uint8)
    dets, observable = records(c)
    assert all(not (raw[:, idx].sum(axis=1) % 2).any() for idx in dets)
    assert not (raw[:, observable].sum(axis=1) % 2).any()
    # and the random-type first-round measurements really are random
    g = geometry(d)
    other = [i for i, a in enumerate(g.ancillas) if a not in g.basis_ancillas(basis)]
    assert 0.4 < raw[:, other].mean() < 0.6


# ─── DEM and distance ────────────────────────────────────────────────────────


@pytest.mark.parametrize("d,r,basis,no_reset", CASES)
def test_dem_builds_and_distance_is_d(d, r, basis, no_reset):
    c = memory_circuit(d, r, basis, no_reset=no_reset, noise=DEFAULT_NOISE)
    dem = c.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == c.num_detectors
    assert len(dem.shortest_graphlike_error()) == d


# ─── against Stim ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d,r", product((3, 5), (1, 3, 5)))
def test_with_reset_z_is_stims_circuit(d, r):
    ours = memory_circuit(d, r, "Z", no_reset=False, noise=DEFAULT_NOISE)
    assert str(ours.flattened()) == str(generated(d, r, "Z", **DEFAULT_NOISE).flattened())


@pytest.mark.parametrize("d,r", product((3, 5), (1, 3, 5)))
def test_with_reset_x_has_stims_dem(d, r):
    ours = memory_circuit(d, r, "X", no_reset=False, noise=DEFAULT_NOISE)
    theirs = generated(d, r, "X", **DEFAULT_NOISE)
    assert "RX" not in str(ours) and "MX" not in str(ours)
    dem_ours = ours.detector_error_model(decompose_errors=True)
    assert dem_ours.approx_equals(theirs.detector_error_model(decompose_errors=True).flattened(), atol=1e-12)


@pytest.mark.parametrize("d,r,basis", product((3, 5), (1, 2, 3, 5), "ZX"))
def test_no_reset_plus_a_flip_after_each_ancilla_m_is_the_with_reset_dem(d, r, basis):
    """A flip of the live ancilla after M does what a reset error does after MR (module docstring)."""
    p = DEFAULT_NOISE["after_reset_flip_probability"]
    no_reset = memory_circuit(d, r, basis, no_reset=True, noise=DEFAULT_NOISE)
    with_reset = memory_circuit(d, r, basis, no_reset=False, noise=DEFAULT_NOISE)
    g = geometry(d)
    for k in range(r, 0, -1):  # insert from the back, so the earlier positions don't move
        no_reset = after_ancilla_measurement(no_reset, k, "X_ERROR", g.ancillas, p)
    assert no_reset.detector_error_model(decompose_errors=True).approx_equals(
        with_reset.detector_error_model(decompose_errors=True), atol=1e-12
    )
    # and without that flip they differ when there is a reset between rounds (R >= 2): the
    # no-reset circuit lacks the reset errors. With R = 1 the only MR is the last one.
    plain = memory_circuit(d, r, basis, no_reset=True, noise=DEFAULT_NOISE).detector_error_model(decompose_errors=True)
    assert plain.approx_equals(with_reset.detector_error_model(decompose_errors=True), atol=1e-12) == (r == 1)


# ─── single injected errors ──────────────────────────────────────────────────


@pytest.mark.parametrize("r,basis,no_reset", product((1, 2, 3, 4), "ZX", (True, False)))
def test_an_ancilla_flip_after_round_k_fires_rounds_k_plus_1_and_k_plus_2(r, basis, no_reset):
    """X on ancilla a right after its round-k measurement flips every later raw m (no reset) or
    the next one (reset). Either way s_(k+1) flips alone: a's detectors at t = k and k + 1 fire,
    those rounds permitting, and its final detector iff k = R - 1 and it is of the basis type."""
    g = geometry(3)
    base = memory_circuit(3, r, basis, no_reset=no_reset)
    for a in g.ancillas:
        x, y = g.coords[a]
        for k in range(1, r + 1):
            expected = {(x, y, t) for t in (k, k + 1) if t <= r - 1}
            if k == r - 1 and a in g.basis_ancillas(basis):
                expected.add((x, y, r))
            assert fired(after_ancilla_measurement(base, k, "X_ERROR", [a])) == (expected, False), (a, k)


@pytest.mark.parametrize("r,basis,no_reset", product((1, 2, 3, 4), "ZX", (True, False)))
def test_a_data_error_between_rounds_fires_its_stabilizers_once(r, basis, no_reset):
    """A data error the basis-type stabilizers see, after round k, fires those stabilizers'
    detectors at t = k (round k + 1, or the final detectors when k = R) and flips the
    observable iff the qubit is on the logical."""
    g = geometry(3)
    base = memory_circuit(3, r, basis, no_reset=no_reset)
    for q in g.data:
        for k in range(1, r + 1):
            # Z basis: an X error. X basis: a Z error; after the last round the data have had
            # their readout H, so there it is an X error.
            gate = "X_ERROR" if basis == "Z" or k == r else "Z_ERROR"
            stabs = [s for s in g.basis_ancillas(basis) if q in g.support[s]]
            expected = {(*g.coords[s], k) for s in stabs}
            got = fired(after_ancilla_measurement(base, k, gate, [q]))
            assert got == (expected, q in g.logical(basis)), (q, k)


# ─── noise ───────────────────────────────────────────────────────────────────


def test_noise_forms_agree_and_zero_noise_writes_nothing():
    c = memory_circuit(3, 3, "Z", noise=DEFAULT_NOISE)
    assert c == memory_circuit(3, 3, "Z", noise=NoiseModel.uniform(0.003)) == memory_circuit(3, 3, "Z", noise=0.003)
    assert memory_circuit(3, 3, "X", noise=NoiseModel()) == memory_circuit(3, 3, "X")
    with pytest.raises(ValueError):
        memory_circuit(3, 3, noise={"after_clifford_depolarisation": 0.003})
    with pytest.raises(ValueError):
        NoiseModel(after_clifford_depolarization=1.5)


@pytest.mark.parametrize("basis", "ZX")
def test_idle_flip_at_the_reset_rate_makes_the_variants_one_dem(basis):
    """Idle noise hits the data in both variants and the live ancillas without reset; at
    idle = reset-flip rate the two variants have the same DEM."""
    noise = NoiseModel.uniform(0.003, idle=0.003)
    a = memory_circuit(3, 4, basis, no_reset=True, noise=noise).detector_error_model(decompose_errors=True)
    b = memory_circuit(3, 4, basis, no_reset=False, noise=noise).detector_error_model(decompose_errors=True)
    assert a.approx_equals(b, atol=1e-12)
    with_reset = str(memory_circuit(3, 2, basis, no_reset=False, noise=noise))
    no_reset = str(memory_circuit(3, 2, basis, no_reset=True, noise=noise))
    g = geometry(3)
    anc = " ".join(map(str, g.ancillas))
    assert f"M {anc}\nX_ERROR(0.003) {anc}\nDEPOLARIZE1(0.003) {' '.join(map(str, g.data))}" in no_reset
    assert f"MR {anc}\nX_ERROR(0.003) {anc}\nDEPOLARIZE1(0.003) {' '.join(map(str, g.data))}" in with_reset


def test_per_qubit_and_per_pair_rates():
    g = geometry(3)
    q, a = g.data[0], g.ancillas[0]
    pair = next(p for layer in g.cx_layers for p in layer if q in p)
    noise = NoiseModel(
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.002,
        qubit_rates={q: {"before_measure_flip_probability": 0.02}, a: {"before_measure_flip_probability": 0.03}},
        pair_rates={(pair[1], pair[0]): 0.01},
    )
    c = memory_circuit(3, 1, "Z", noise=noise)
    text = str(c)
    assert f"X_ERROR(0.03) {a}" in text and f"X_ERROR(0.02) {q}" in text
    assert f"DEPOLARIZE2(0.01) {pair[0]} {pair[1]}" in text
    # the DEM sees the per-pair rate: the total probability of errors grows
    total = lambda circuit: sum(inst.args_copy()[0] for inst in circuit.detector_error_model().flattened() if inst.type == "error")
    assert total(c) > total(memory_circuit(3, 1, "Z", noise=NoiseModel(0.001, 0, 0.002, 0)))
    with pytest.raises(ValueError):
        NoiseModel(qubit_rates={q: {"readout": 0.01}})


@pytest.mark.parametrize("bad", [dict(distance=4), dict(distance=1), dict(rounds=0), dict(basis="Y")])
def test_bad_arguments_raise(bad):
    args = dict(distance=3, rounds=2, basis="Z") | bad
    with pytest.raises(ValueError):
        memory_circuit(**args)


# ─── record-only readout flips, M(p) ─────────────────────────────────────────


def symptoms(circuit):
    """The distinct symptoms of the DEM's error mechanisms: detector coordinates, plus "L0"."""
    coords = circuit.get_detector_coordinates()
    out = set()
    for inst in circuit.detector_error_model().flattened():
        if inst.type != "error":
            continue
        s = set()
        for t in inst.targets_copy():
            s.add(tuple(int(v) for v in coords[t.val]) if t.is_relative_detector_id() else "L0")
        out.add(frozenset(s))
    return out


@pytest.mark.parametrize("r,basis,no_reset", product((1, 2, 3, 5), "ZX", (True, False)))
def test_record_flips_have_exactly_the_derived_symptoms(r, basis, no_reset):
    """With only record flips, each DEM error is one flipped bit. A flipped m_k of ancilla a
    fires every detector that reads it: without reset D_k and D_(k+2), and the final detector
    when k >= R - 1; with MR, D_k and D_(k+1), and the final detector when k = R. A flipped data
    bit fires the final detectors of the basis-type stabilizers on it, and the observable if it
    is on the logical."""
    g = geometry(3)
    c = memory_circuit(3, r, basis, no_reset=no_reset, noise=NoiseModel(record_flip_probability=0.01))
    expected = set()
    for a in g.ancillas:
        x, y = g.coords[a]
        basis_type = a in g.basis_ancillas(basis)
        for k in range(1, r + 1):
            rounds = (k, k + 2) if no_reset else (k, k + 1)  # the D_j that read m_k
            s = {(x, y, j - 1) for j in rounds if j <= r and (j >= 2 or basis_type)}
            final_reads = (r - 1, r) if no_reset else (r,)
            if basis_type and k in final_reads:
                s.add((x, y, r))
            if s:  # a bit no detector reads (the other type's m_1 when R <= 2) is not an error
                expected.add(frozenset(s))
    for q in g.data:
        s = {(*g.coords[a], r) for a in g.basis_ancillas(basis) if q in g.support[a]}
        expected.add(frozenset(s | ({"L0"} if q in g.logical(basis) else set())))
    assert symptoms(c) == expected


def test_record_flips_without_reset_give_edges_two_rounds_apart():
    """The D_k - D_(k+2) edges: every ancilla has one for each k = 2 .. R - 2, and no
    record flip gives a D_k - D_(k+1) edge. With MR they are D_k - D_(k+1) instead."""
    g = geometry(3)
    r = 6
    noise = NoiseModel(record_flip_probability=0.01)
    no_reset, with_reset = (symptoms(memory_circuit(3, r, "Z", nr, noise)) for nr in (True, False))
    for a in g.ancillas:
        x, y = g.coords[a]
        for k in range(2, r - 1):
            assert frozenset({(x, y, k - 1), (x, y, k + 1)}) in no_reset
            assert frozenset({(x, y, k - 1), (x, y, k)}) not in no_reset
            assert frozenset({(x, y, k - 1), (x, y, k)}) in with_reset
            assert frozenset({(x, y, k - 1), (x, y, k + 1)}) not in with_reset


@pytest.mark.parametrize("basis,no_reset", product("ZX", (True, False)))
def test_record_flips_keep_the_contract_and_the_distance(basis, no_reset):
    g = geometry(3)
    noise = NoiseModel(
        **DEFAULT_NOISE,
        record_flip_probability=0.01,
        qubit_rates={g.ancillas[2]: {"record_flip_probability": 0.02}, g.data[4]: {"record_flip_probability": 0.0}},
    )
    c = memory_circuit(3, 3, basis, no_reset=no_reset, noise=noise)
    measured = [t.value for inst in c if inst.name in ("M", "MR") for t in inst.targets_copy()]
    assert measured == list(g.ancillas) * 3 + list(g.data)
    assert all(not inst.gate_args_copy() for inst in c if inst.name == "MR")  # no MR(p): an X_ERROR before it
    assert ("M(0.02)" in str(c)) == no_reset
    assert len(c.detector_error_model(decompose_errors=True).shortest_graphlike_error()) == 3
    # a zero rate leaves the generated() identity alone
    assert memory_circuit(3, 3, basis, no_reset, NoiseModel(**DEFAULT_NOISE, record_flip_probability=0.0)) == (
        memory_circuit(3, 3, basis, no_reset, DEFAULT_NOISE)
    )
