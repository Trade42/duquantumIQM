"""The Emerald design: the calibration, the stand-in, patches eA/eB, live couplers only, zero SWAPs, the hardware path, the plan."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dqiqm import decode as D
from dqiqm import hw
from dqiqm import inforecord as IR
from dqiqm.experiments import device as dev
from dqiqm.experiments import emerald as EM
from dqiqm.experiments import suite as SU
from dqiqm.experiments.circuits import sent


@pytest.fixture(scope="module")
def cal():
    return EM.live_calibration()


@pytest.fixture(scope="module")
def small(cal):
    """E1 at R = 1, 2, 3 on both patches and one E4 point: every circuit kind, cheap."""
    return EM.build(cal, e1_rounds=(1, 2, 3), e4_rounds=(2,), e4_delays_ns=(1000.0,), rank_shots=500)


def test_calibration_reads_the_live_emerald_snapshot(cal):
    assert cal.device == "emerald" and cal.calibration_set_id == EM.CALSET
    assert EM.DEFAULT_SNAPSHOT.read_bytes() == EM.SNAPSHOT_SOURCE.read_bytes()
    assert len(cal.qubits) == 54 and len(cal.couplers) == 85
    assert EM.cz_less(cal) == sorted(EM.CZ_LESS, key=lambda p: (dev.qubit_sort_key(p[0]), dev.qubit_sort_key(p[1])))
    assert cal.reset_wait_ns == pytest.approx(400_000.0)
    snap = dev.load_snapshot(EM.DEFAULT_SNAPSHOT)
    obs = {o["dut_field"]: o["value"] for part in ("calibration_set", "quality_metric_set")
           for o in (snap.get(part) or {}).get("observations", [])}
    assert cal.qubits["QB3"]["p0_given_1"] == pytest.approx(obs["metrics.ssro.measure.constant.QB3.error_1_to_0"])
    assert cal.qubits["QB3"]["p1_given_0"] == pytest.approx(obs["metrics.ssro.measure.constant.QB3.error_0_to_1"])


def test_stand_in_takes_the_live_profile(small, cal):
    ep = small.backend.error_profile
    cz = ep.two_qubit_gate_depolarizing_error_parameters["cz"]
    assert len(cz) == 90
    padded = {frozenset(k) for k, v in cz.items() if v == EM.CZ_LESS_SENTINEL}
    assert padded == {frozenset(p) for p in EM.CZ_LESS}
    q = "QB20"
    assert ep.t1s[q] == pytest.approx(cal.qubits[q]["t1_ns"])
    assert ep.readout_errors[q] == {"0": cal.qubits[q]["p1_given_0"], "1": cal.qubits[q]["p0_given_1"]}
    assert all(ep.t2s[k] <= ep.t1s[k] for k in ep.t1s)  # the MPS-safe cap
    assert EM.t2_capped(cal) == ["QB8", "QB30", "QB37", "QB39", "QB50"]


def test_patches_are_two_qubit_sets_on_live_couplers(small):
    e1 = small.choices["E1"]
    assert set(e1["patches"]) == {"eA", "eB"}
    a, b = (set(e1["patches"][k]) for k in ("eA", "eB"))
    assert len(a) == len(b) == 17 and a != b
    assert len(e1["qubit_sets"]) == 15 and len(e1["placements"]) == 60
    names = [s.name for s in small.specs]
    assert all(n.endswith(("_eA", "_eB")) or "_eA_" in n for n in names)
    assert not any("pA" in n or "pB" in n for n in names)


def test_counts(small):
    for s in small.specs:
        r = s.rounds
        assert s.stim.num_detectors == 8 * r and s.stim.num_measurements == 8 * r + 9 and s.stim.num_observables == 1
        assert s.hardware.circuit.num_clbits == s.stim.num_measurements
        assert s.params["patch"] in ("eA", "eB")
    assert {s.experiment for s in small.specs} == {"E1", "E4"}
    assert len(small.of("E1")) == 2 * 2 * 3


def test_native_ops_on_live_couplers_and_zero_swaps(small, cal):
    dead = {frozenset(p) for p in EM.CZ_LESS}
    live = {frozenset(k) for k in cal.couplers}
    for s in small.specs:
        ops = {i.name for i in sent(s.hardware)}
        assert ops <= EM.NATIVE_OPS, (s.name, ops)
        czs = [frozenset(i.locus) for i in sent(s.hardware) if i.name == "cz"]
        assert czs and all(c in live and c not in dead for c in czs), s.name
        assert "swap" not in s.hardware.circuit.count_ops()
        assert EM.check_live_couplers(s, cal)["cz"] == len(czs)
        assert s.hardware.target == "hardware"


def test_noiseless_through_the_hardware_path(small, tmp_path):
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="matrix_product_state", seed_simulator=7)
    for k, s in enumerate(small.specs):
        res = sim.run(s.hardware.circuit, shots=16, memory=True).result()
        res.job_id = f"noiseless-{k}"
        d = hw.save_raw(res, task="t", device="aer", qubit_mapping=[{}], shots=16, data_root=tmp_path)
        raw, _ = D.load_memory(d, 0, circuit=s.stim)
        dets, obs = D.detection_events(s.stim, raw)
        assert raw.shape == (16, s.stim.num_measurements)
        assert dets.sum() == 0 and obs.sum() == 0, s.name


def test_dataset_matches_the_dictionary(small, cal, tmp_path):
    s = small[EM.pilot_name(small)]
    noisy = s.noisy(EM.calibrated_noise(cal, s))
    raw = noisy.compile_sampler(seed=1).sample(200).astype(np.uint8)
    files = IR.write_circuits(tmp_path, [s], small.durations)
    labels = SU.split("EP0", 0, 200)
    ds, numbers = IR.write_dataset(
        tmp_path, s, raw, run_id="ep0_" + s.name, batch=0, job_id="stim", position_in_job=0, arch=small.arch, cal=cal,
        durations=small.durations, circuit_files=files[s.name], raw_ref={}, calibration_paths={}, decoder_circuit=noisy,
        split_labels=labels, split_info=SU.split_rule("EP0", 0, labels), simulated=True, extra={"device": EM.DEVICE})
    der, _ = IR.write_derived(tmp_path, s, "ep0_" + s.name, ds)
    assert IR.check_run(ds, der) == []
    m = json.loads((ds / "manifest.json").read_text())
    assert m["device"] == "emerald" and m["spec"]["patch"] == "eA" and m["shots"] == 200
    assert numbers["pm_failures"] < numbers["obs_flips"]


def test_priors_count_the_readout_idle_once(small, cal):
    s = small[EM.pilot_name(small)]
    base = dev.calibrated_noise(cal, s.physical, s.roles, pairs=s.cx_pairs, x_ancillas=s.x_ancillas)
    assert EM.calibrated_noise(cal, s, idle_ns=0.0).qubit_rates == base.qubit_rates
    nm = EM.calibrated_noise(cal, s, idle_ns=360.0)
    for q, r in nm.qubit_rates.items():
        c = cal.qubits[s.physical[q]]
        pxy, pz = dev.relaxation_pauli(360.0, c["t1_ns"], dev.t2_ns(c))
        assert r.get("before_round_data_depolarization") == base.qubit_rates[q].get("before_round_data_depolarization")
        want = pxy + pz if s.roles[q] == "data" else pxy
        assert r["idle_during_measurement"] == pytest.approx(want)
    # garnet-design's fix round (037ad05) made device.calibrated_noise count the idle the same way
    fixed = dev.calibrated_noise(cal, s.physical, s.roles, pairs=s.cx_pairs, idle_ns=360.0, x_ancillas=s.x_ancillas)
    assert fixed.qubit_rates == nm.qubit_rates and dict(fixed.pair_rates) == dict(nm.pair_rates)


def test_job_plan(small):
    jobs = EM.jobs(small)
    labels = [j.label for j in jobs]
    assert len(set(labels)) == len(labels) and all(lab.startswith("E") for lab in labels)
    p0 = jobs[0]
    assert (p0.label, p0.circuits, p0.shots, p0.batch) == ("EP0", ("E1_d3_Z_r03_eA",), 1000, None)
    assert {j.batch for j in jobs[1:]} == {1, 2, 3}
    for b in (1, 2, 3):
        assert sum(j.experiment == "E1" for j in jobs if j.batch == b) == 4
    assert all(j.shots * len(j.circuits) < 110_000 for j in jobs)  # smaller than Garnet's E1 jobs
    est = SU.estimate(jobs[0], small)
    assert est["per_shot_overhead_s"] == pytest.approx(400e-6)
    assert (SU.split("E-B3-1-E1", 3, 10) == "test").all() and "test" not in set(SU.split("E-B1-1-E1", 1, 100))


def test_d5_does_not_fit_on_the_live_couplers(cal):
    d5 = EM.d5_feasibility(cal)
    assert d5["live_placements"] == 0
    assert d5["all_90_placements"] == 4 and len(d5["all_90_qubit_sets"]) == 1
    assert len(d5["all_90_qubit_sets"][0]["qubits"]) == 49
    needs = {tuple(p) for o in d5["all_90_qubit_sets"][0]["orientations"] for p in o["needs_cz_less"]}
    assert needs == {("QB15", "QB23"), ("QB24", "QB32"), ("QB49", "QB50")}


def test_the_record_carries_the_device():
    root = EM.RECORD
    if not (root / "circuits" / "job_plan.json").exists():
        pytest.skip("InfoRecord/emerald not written yet")
    for p in (root / "circuits").glob("*/placement.json"):
        body = json.loads(p.read_text())
        assert body and all(v["device"] == "emerald" for v in body.values()), p
    plan = json.loads((root / "circuits" / "job_plan.json").read_text())
    assert plan["device"] == "emerald" and plan["pilot"]["band"] and plan["pilot"]["x_cols"] and plan["pilot"]["z_cols"]
    assert json.loads((root / "architecture" / "emerald.json").read_text())["device"] == "emerald"
    assert json.loads((root / "calibration" / "emerald" / "placement.json").read_text())["device"] == "emerald"
    for m in (root / "sim" / "datasets").glob("*/*/manifest.json"):
        body = json.loads(m.read_text())
        assert body["device"] == "emerald" and body["label"] == "SIMULATED"
    for f in (root / "sim" / "ler").glob("*/*/ler_fits.json"):
        body = json.loads(f.read_text())
        assert body["device"] == "emerald" and body["label"] == "SIMULATED"
