"""The Garnet suite: placements, counts, native gates on couplers, zero SWAPs, and noiseless runs through the hardware path."""

from __future__ import annotations

import numpy as np
import pytest

from dqiqm import decode as D
from dqiqm import hw
from dqiqm import surface as S
from dqiqm.experiments import device as dev
from dqiqm.experiments import suite as SU
from dqiqm.experiments.circuits import sent


@pytest.fixture(scope="module")
def cal():
    return dev.live_calibration()


@pytest.fixture(scope="module")
def small(cal):
    """A small suite: every family, a few rounds each."""
    return SU.build(cal, e1_rounds=(1, 2, 3), e2_rounds=(1, 3), e4_rounds=(2,), e4_delays_ns=(1000.0,),
                    e5_rounds=(2, 3), e3_random=2, e3_repeats=3, rank_shots=2000)


def test_calibration_reads_the_live_snapshot(cal):
    assert cal.calibration_set_id == "a18a947f-0408-4628-81ea-1715637e434f"
    assert len(cal.qubits) == 20 and len(cal.couplers) == 30
    assert cal.qubits["QB2"]["measure_ns"] == pytest.approx(556.0)
    assert cal.median("measure_ns") == pytest.approx(480.0)
    assert cal.qubits["QB1"]["t1_ns"] == pytest.approx(23729.5, abs=0.1)  # resonance-check: QB1 T1 23.73 us
    assert cal.cz_error("QB2", "QB5") == pytest.approx(1 - 0.995777, abs=1e-6)
    assert cal.median("p0_given_1") > 2 * cal.median("p1_given_0")  # the asymmetric readout
    with pytest.raises(NotImplementedError):
        dev.calibration_from_fidelities({})


def test_stand_in_has_the_live_noise(small, cal):
    ep = small.backend.error_profile
    assert ep.readout_errors["QB2"] == {"0": cal.qubits["QB2"]["p1_given_0"], "1": cal.qubits["QB2"]["p0_given_1"]}
    assert ep.t2s["QB1"] == pytest.approx(min(cal.qubits["QB1"]["t2_ramsey_ns"], 2 * cal.qubits["QB1"]["t1_ns"]))


def test_e1_has_one_qubit_set_and_two_orientations(small):
    e1 = small.choices["E1"]
    assert len(e1["placements"]) == 4
    assert set(e1["qubits"]) == {f"QB{i}" for i in range(1, 21)} - {"QB3", "QB17", "QB18"}
    patches = {p["patch"]: p for p in e1["placements"] if p["patch"]}
    assert set(patches) == {"pA", "pB"}
    assert set(patches["pA"]["x_ancillas"]).isdisjoint(patches["pB"]["x_ancillas"])
    assert {frozenset(patches[k]["x_ancillas"]) for k in patches} == {
        frozenset({"QB1", "QB9", "QB11", "QB20"}), frozenset({"QB5", "QB7", "QB13", "QB15"})}


def test_counts(small):
    for s in small.specs:
        r, c = s.rounds, s.stim
        fam = s.params["family"]
        if fam == "surface_memory":
            assert (c.num_detectors, c.num_measurements, c.num_observables) == (8 * r, 8 * r + 9, 1), s.name
        elif fam == "repetition":
            assert (c.num_detectors, c.num_measurements, c.num_observables) == (9 * (r + 1), 9 * r + 10, 1), s.name
        elif fam == "readout":
            assert (c.num_detectors, c.num_measurements, c.num_observables) == (20 * r, 20 * r, 0), s.name
        elif fam == "stability":
            assert (c.num_detectors, c.num_measurements, c.num_observables) == ((r + 1) + 4 * (r - 1), 5 * r + 4, 1), s.name
        assert len(s.meas_round) == c.num_measurements


def test_every_circuit_is_native_on_couplers_with_zero_swaps(small):
    allowed = {"prx", "cz", "measure", "barrier", "delay"}
    for s in small.specs:
        assert s.hardware.target == "hardware"
        assert "swap" not in s.hardware.circuit.count_ops()
        ops = sent(s.hardware)
        assert {i.name for i in ops} <= allowed, s.name
        for i in ops:
            if i.name == "cz":
                assert small.arch.coupled(*i.locus), (s.name, i.locus)
        assert sum(i.name == "cz" for i in ops) == len(s.cx_pairs) * s.rounds, s.name
        used = {q for i in ops for q in i.locus}
        assert used == set(s.physical.values()), s.name


def test_e4_has_its_delays(small):
    (s4,) = [s for s in small.of("E4") if s.params["basis"] == "Z"]
    per_q = {}
    for i in sent(s4.hardware):
        if i.name == "delay":
            per_q.setdefault(i.locus[0], []).append(i.args["duration"])
    assert set(per_q) == set(s4.physical.values())
    assert all(v == [pytest.approx(1e-6)] * (s4.rounds - 1) for v in per_q.values())


def test_noiseless_hardware_path_fires_nothing(small, tmp_path):
    """One circuit per family (X preparation and delays included): Aer, save_raw, load_memory, m2d."""
    from qiskit_aer import AerSimulator

    picks = ["E1_d3_X_r03_pB", "E1_d3_Z_r03_pA_nobar", "E2_rep10_bit_1_r03", "E2_rep10_phase_p_r03",
             "E3_repeat_chkA_k03", "E3_assign_all1", "E4_d3_X_r02_pA_t01000ns", "E5_stab_X_r03", "E5_stab_Z_r03"]
    sim = AerSimulator(method="matrix_product_state", seed_simulator=7, max_parallel_threads=1)
    for k, name in enumerate(picks):
        s = small[name]
        res = sim.run(s.hardware.circuit, shots=32, memory=True).result()
        res.job_id = f"t{k}"
        d = hw.save_raw(res, task="t", device="aer", qubit_mapping=[{}], shots=32, data_root=tmp_path)
        raw, _ = D.load_memory(d, 0, circuit=s.stim)
        dets, obs = D.detection_events(s.stim, raw)
        assert not dets.any() and not obs.any(), name
        if s.params["family"] == "readout":
            prep = np.array([int(b) for b in s.params["prep"]])
            assert (raw[:, :20] == prep).all(), name  # the X gates landed on the right qubits
        if name.startswith("E1_d3_X"):
            assert 0 < raw.mean() < 1  # not a trivially quiet record


def test_distances(small, cal):
    for name, want in (("E1_d3_Z_r03_pA", 3), ("E1_d3_X_r02_pB", 3), ("E2_rep10_bit_0_r03", 10),
                       ("E2_rep10_phase_p_r03", 10), ("E5_stab_Z_r03", 3), ("E5_stab_X_r02", 2)):
        s = small[name]
        noisy = s.noisy(S.NoiseModel.uniform(0.003))
        assert noisy.num_detectors == s.stim.num_detectors
        noisy.detector_error_model(decompose_errors=True)
        assert len(noisy.shortest_graphlike_error()) == want, name
    # without reset a record flip spans two rounds: the stability experiment's time-like distance halves
    with_records = S.NoiseModel(0.003, 0.003, 0.003, 0.003, record_flip_probability=0.01)
    assert len(small["E5_stab_Z_r03"].noisy(with_records).shortest_graphlike_error()) == 2
    assert len(small["E1_d3_Z_r03_pA"].noisy(with_records).shortest_graphlike_error()) == 3


def test_calibrated_noise_keeps_the_detectors(small, cal):
    s = small["E1_d3_Z_r02_pA"]
    nm = dev.calibrated_noise(cal, s.physical, s.roles, pairs=s.cx_pairs, x_ancillas=s.x_ancillas, idle_ns=480.0)
    noisy = s.noisy(nm)
    assert noisy.num_detectors == s.stim.num_detectors and noisy.num_measurements == s.stim.num_measurements
    assert noisy.detector_error_model(decompose_errors=True).num_errors > 0
    rates = nm.pair_rates
    assert set(rates) == set(s.cx_pairs) and all(0 < p < 0.05 for p in rates.values())


def test_calibrated_noise_counts_the_idle_once(small, cal):
    # the readout-window idle lands once per window, after the readout, and never before round 1
    s = small["E1_d3_Z_r03_pA"]
    args = dict(pairs=s.cx_pairs, x_ancillas=s.x_ancillas)
    idle = dev.calibrated_noise(cal, s.physical, s.roles, idle_ns=556.0, **args)
    bare = dev.calibrated_noise(cal, s.physical, s.roles, idle_ns=0.0, **args)
    data = [q for q, r in s.roles.items() if r == "data"]
    ancillas = [q for q, r in s.roles.items() if r == "ancilla"]
    for q in data:
        c = cal.qubits[s.physical[q]]
        pxy, pz = dev.relaxation_pauli(556.0, float(c["t1_ns"]), dev.t2_ns(c))
        assert idle.rate("before_round_data_depolarization", q) == bare.rate("before_round_data_depolarization", q)
        assert idle.rate("idle_during_measurement", q) == pytest.approx(min(pxy + pz, 0.75))
        assert bare.rate("idle_during_measurement", q) == 0
    for q in ancillas:
        c = cal.qubits[s.physical[q]]
        pxy, _ = dev.relaxation_pauli(556.0, float(c["t1_ns"]), dev.t2_ns(c))
        assert idle.rate("idle_during_measurement", q) == pytest.approx(min(pxy, 0.5))
    # in the circuit: each data qubit's first channel is the idle-free round-start one
    first: dict[int, float] = {}
    for inst in s.noisy(idle).flattened():
        if inst.name == "M":
            break
        if inst.name == "DEPOLARIZE1":
            for t in inst.targets_copy():
                if t.value in data:
                    first.setdefault(t.value, inst.gate_args_copy()[0])
    assert set(first) == set(data)
    for q in data:
        assert first[q] == pytest.approx(bare.rate("before_round_data_depolarization", q))


def test_job_plan(small):
    jobs = SU.jobs(small)
    assert jobs[0].label == "P0" and len(jobs[0].circuits) == 1 and jobs[0].shots <= 1000
    pilot = small[jobs[0].circuits[0]]
    assert (pilot.params["basis"], pilot.rounds, pilot.params["patch"], pilot.params["barriers"]) == ("Z", 3, "pA", True)
    # no barrier A/B (P1 dropped: barriers stay on); its barrier-off circuit is built but in no job
    assert [j.label for j in jobs if j.phase == "pilot"] == ["P0"]
    nobar = [s.name for s in small.of("E1") if not s.params["barriers"]]
    assert nobar and not any(n in j.circuits for j in jobs for n in nobar)
    for j in jobs:
        for n in j.circuits:
            assert small[n].hardware.target == "hardware"
        assert len(set(j.circuits)) == len(j.circuits)
    batches = {j.batch for j in jobs if j.batch}
    assert batches == {1, 2, 3}
    for b in batches:
        exps = [j.experiment for j in jobs if j.batch == b]
        assert exps.count("E1") == 2 and "E3" in exps
    est = SU.estimate(jobs[0], small)
    assert est["qpu_s"] > est["qpu_s_circuits_only"] > 0
    assert est["per_shot_overhead_s"] == pytest.approx(298e-6 + SU.ELECTRONICS_PER_SHOT_S)
    assert est["qpu_s"] == pytest.approx(est["shots"] * (est["circuit_us_mean"] * 1e-6 + 298e-6) + SU.PER_CIRCUIT_OVERHEAD_S)
