"""scripts/batch_run.py offline: the brief's table against both job plans, the guards, the sanity checks on synthetic
records, and submit, recover and resume against a fake connection. Nothing here reaches the network or builds the suite."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import stim
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate
from qiskit.result import Result

from dqiqm import hw
from dqiqm import layout as L
from dqiqm.experiments import device as dev
from dqiqm.experiments import emerald as EM

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("batch_run", REPO / "scripts" / "batch_run.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["batch_run"] = B
_spec.loader.exec_module(B)

CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "11111111-2222-4333-8444-555555555555"
LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""


def row(task, label, job_id=JOB_ID, device="garnet", cs="22 x 2000", qubits="QB1,QB2", calset=CALSET,
        time="2026-10-25T00:30:00Z"):
    return f"| {time} | {task} | {device} | {calset} | {qubits} | {cs} | {job_id} | {task} {label} E1 (22 circuits x 2000 shots) |\n"


# ---------------------------------------------------------------------------
# the allowance and the plans
# ---------------------------------------------------------------------------


def test_segments_are_the_briefs_table():
    assert B.ORDER == ("garnet-b1", "garnet-b2", "emerald-b1", "emerald-b2", "garnet-b3", "emerald-b3")
    assert sum(s.jobs for s in B.SEGMENTS.values()) == B.TOTAL_JOBS == 26
    assert sum(s.shots for s in B.SEGMENTS.values()) == B.TOTAL_SHOTS == 1_419_000
    assert [(s.jobs, s.shots) for s in B.SEGMENTS.values()] == [(4, 399_000), (4, 384_000), (4, 88_000), (6, 136_000),
                                                                (4, 324_000), (4, 88_000)]
    labels = [lab for s in B.SEGMENTS.values() for lab in s.labels]
    assert len(labels) == len(set(labels)) == 26
    assert not {"P0", "P1", "EP0"} & set(labels), "the pilots and the dropped barrier A/B are in no segment"
    assert [s.split for s in B.SEGMENTS.values()] == ["finetune+validation"] * 4 + ["test"] * 2
    assert B.SEGMENTS["emerald-b2"].arming() == {"DQ_QPU": "run", "DQ_QPU_TASK": "emerald-b2", "DQ_QPU_MAX_JOBS": "6",
                                                 "DQ_QPU_MAX_SHOTS": "136000"}


@pytest.mark.parametrize("task", B.ORDER)
def test_check_plan_matches_the_committed_job_plans(task):
    seg = B.SEGMENTS[task]
    jobs = B.check_plan(seg, B.plan_json(seg.device))
    assert [j.label for j in jobs] == list(seg.labels)
    assert sum(j.total_shots for j in jobs) == seg.shots and all(j.batch == seg.batch for j in jobs)
    assert [j.position for j in jobs] == list(range(seg.jobs))


def test_p1_is_never_submittable():
    assert "P1" not in {j["label"] for j in B.plan_json("garnet")["jobs"]}
    assert all("P1" not in s.labels for s in B.SEGMENTS.values())


def test_the_decoder_priors_are_hardware_noise():
    assert B.PRIORS == "hardware_noise (79092cf)"
    assert B.decoder_priors(None, SimpleNamespace(noisy=None, stim=stim.Circuit())) is None


def _plan_copy(device="garnet"):
    return json.loads(json.dumps(B.plan_json(device)))


@pytest.mark.parametrize("mutate, match", [
    (lambda p, i: p["jobs"].__setitem__(i, {**p["jobs"][i], "shots": 4000, "total_shots": 88000}), "allows"),
    (lambda p, i: p["jobs"].__setitem__(i, {**p["jobs"][i], "order": p["jobs"][i]["order"][:-1]}), "order has"),
    (lambda p, i: p["jobs"].__setitem__(i, {**p["jobs"][i], "batch": 2}), "batch"),
    (lambda p, i: p["jobs"].pop(i), "not in the job plan"),
    (lambda p, i: p.__setitem__("device", "emerald"), "plan is"),
])
def test_check_plan_stops_on_any_disagreement(mutate, match):
    plan = _plan_copy()
    i = next(k for k, j in enumerate(plan["jobs"]) if j["label"] == "B1-1-E3")
    mutate(plan, i)
    with pytest.raises(SystemExit, match=match):
        B.check_plan(B.SEGMENTS["garnet-b1"], plan)


def test_check_plan_against_the_rebuilt_jobs():
    seg = B.SEGMENTS["emerald-b1"]
    plan = B.plan_json("emerald")
    built = {j["label"]: SimpleNamespace(circuits=tuple(j["order"]), shots=j["shots"], batch=j["batch"]) for j in plan["jobs"]}
    assert len(B.check_plan(seg, plan, built)) == 4
    built["E-B1-2-E1"] = SimpleNamespace(circuits=tuple(reversed(built["E-B1-2-E1"].circuits)), shots=2000, batch=1)
    with pytest.raises(SystemExit, match="rebuilt"):
        B.check_plan(seg, plan, built)


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------


def test_arming_exactly_as_the_segments_row(monkeypatch):
    seg = B.SEGMENTS["garnet-b2"]
    env = {**seg.arming(), "DQ_QPU_LEDGER": hw.DEFAULT_LEDGER}
    B.check_arming(seg, env)
    for k, bad in (("DQ_QPU_TASK", "garnet-b1"), ("DQ_QPU_MAX_JOBS", "5"), ("DQ_QPU_MAX_SHOTS", "399000"), ("DQ_QPU", "")):
        with pytest.raises(SystemExit, match=k):
            B.check_arming(seg, {**env, k: bad})
    with pytest.raises(SystemExit, match="ledger"):
        B.check_arming(seg, {**env, "DQ_QPU_LEDGER": "/tmp/other.md"})


def test_segments_run_in_the_briefs_order(tmp_path):
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER)
    B.check_order(B.SEGMENTS["garnet-b1"], ledger)
    with pytest.raises(SystemExit, match="garnet-b1 comes before garnet-b2"):
        B.check_order(B.SEGMENTS["garnet-b2"], ledger)
    ledger.write_text(LEDGER_HEADER + "".join(row("garnet-b1", f"B1-{i}-E1", job_id=f"{i}" * 8) for i in range(1, 5)))
    B.check_order(B.SEGMENTS["garnet-b2"], ledger)
    with pytest.raises(SystemExit, match="garnet-b2 comes before emerald-b2"):
        B.check_order(B.SEGMENTS["emerald-b2"], ledger)
    with pytest.raises(SystemExit, match="comes after"):
        B.check_order(B.SEGMENTS["garnet-b1"], Path(_append(ledger, row("garnet-b2", "B2-1-E3", job_id="9" * 8))))


def _append(path, text):
    path.write_text(path.read_text() + text)
    return path


def test_job_rows_by_label(tmp_path):
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER + row("garnet-b1", "B1-1-E3") + row("garnet-pilot", "P0", job_id="2" * 8))
    seg = B.SEGMENTS["garnet-b1"]
    rows = B.job_rows(ledger, seg)
    assert list(rows) == ["B1-1-E3"] and rows["B1-1-E3"][0].job_id == JOB_ID
    job = B.PlannedJob("B1-2-E1", "E1", 1, ("a", "b"), 5000, 1)
    assert B.purpose(seg, job) == "garnet-b1 B1-2-E1 E1 (2 circuits x 5000 shots)"


def test_merged_names_refuses_a_conflict():
    assert B.merged_names([{3: "QB4", 1: "QB2"}, {1: "QB2", 7: "QB8"}]) == {1: "QB2", 3: "QB4", 7: "QB8"}
    with pytest.raises(SystemExit, match="index 1"):
        B.merged_names([{1: "QB2"}, {1: "QB3"}])
    with pytest.raises(SystemExit, match="one device qubit"):
        B.merged_names([{1: "QB2"}, {2: "QB2"}])


def test_find_job_dir_on_any_day(tmp_path):
    (tmp_path / "20261025-garnet-b1" / JOB_ID).mkdir(parents=True)
    (tmp_path / "20261025-garnet-b1" / JOB_ID / "meta.json").write_text("{}")
    assert B.find_job_dir(tmp_path, "garnet-b1", JOB_ID) == tmp_path / "20261025-garnet-b1" / JOB_ID
    assert B.find_job_dir(tmp_path, "garnet-b2", JOB_ID) is None
    (tmp_path / "20261026-garnet-b1" / JOB_ID).mkdir(parents=True)
    (tmp_path / "20261026-garnet-b1" / JOB_ID / "meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="twice"):
        B.find_job_dir(tmp_path, "garnet-b1", JOB_ID)


# ---------------------------------------------------------------------------
# the references and the sanity checks
# ---------------------------------------------------------------------------


def memory_spec(basis="Z", rounds=2):
    """A Stim-generated d=3 rotated memory circuit standing in for a CircuitSpec: its roles from the final readout."""
    c = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", rounds=rounds, distance=3)
    last_m = [op for op in c if op.name in ("M", "MX")][-1]
    data = {t.value for t in last_m.targets_copy()}
    roles = {q: ("data" if q in data else "ancilla") for q in set(B.C.measured_qubits(c))}
    return SimpleNamespace(name=f"mem_{basis}", stim=c, roles=roles, stabilizers={}, noisy=None, params={})


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_reference_tells_random_from_deterministic_first_readouts(basis):
    spec = memory_spec(basis)
    ref = B.reference("garnet", spec, None, model=False)
    assert len(ref["first_cols"]) == 8 and len(ref["random_cols"]) == 4 and len(ref["deterministic_cols"]) == 4
    assert ref["deterministic_values"] == [0, 0, 0, 0] and not ref["noiseless_always_all_zero"]
    assert ref["model"] is None


def synthetic(ref, shots=5000, seed=7, det_fraction=0.15, ndet=24):
    rng = np.random.default_rng(seed)
    raw = (rng.random((shots, ref["measurements"])) < 0.3).astype(np.uint8)
    raw[:, ref["random_cols"]] = rng.random((shots, len(ref["random_cols"]))) < 0.5
    raw[:, ref["deterministic_cols"]] = rng.random((shots, len(ref["deterministic_cols"]))) < 0.07
    raw[:, -1] |= (rng.random(shots) < 0.999).astype(np.uint8)
    dets = (rng.random((shots, ndet)) < det_fraction).astype(np.uint8)
    return raw, dets


REF = {"circuit": "x", "measurements": 33, "detectors": 24, "first_cols": list(range(8)), "random_cols": [0, 2, 5, 7],
       "deterministic_cols": [1, 3, 4, 6], "deterministic_values": [0, 0, 0, 0], "noiseless_all_zero": 0.004,
       "noiseless_always_all_zero": False, "x_type_first_cols": [0, 2, 5, 7], "z_type_first_cols": [1, 3, 4, 6],
       "model": {"detection_fraction": 0.17, "all_zero": 0.0}}


def verdict(checks):
    return {c["check"]: c["pass"] for c in checks}


def test_sanity_passes_a_plausible_record():
    raw, dets = synthetic(REF)
    checks = B.sanity(raw, dets, REF, shots=5000, physical=["QB1"], planned=["QB1"])
    assert all(verdict(checks).values()), checks
    det = next(c for c in checks if c["check"] == "detection fraction")
    assert det["info"]["model_with_idle"] == 0.17 and 0.8 < det["info"]["ratio_to_model"] < 1.0


def test_sanity_fails_each_broken_record():
    raw, dets = synthetic(REF)
    v = verdict(B.sanity(raw[:4000], dets[:4000], REF, shots=5000))
    assert not v["measurement shape"] and not v["detection fraction"]
    v = verdict(B.sanity(np.zeros_like(raw), dets, REF, shots=5000))
    assert not v["all-zero rows"] and not v["first readouts, random"]
    v = verdict(B.sanity(1 - raw, dets, REF, shots=5000))
    assert not v["first readouts, deterministic (misread)"]
    bad = raw.copy()
    bad[:, REF["random_cols"]] = 0
    assert not verdict(B.sanity(bad, dets, REF, shots=5000))["first readouts, random"]
    v = verdict(B.sanity(raw, np.random.default_rng(1).integers(0, 2, dets.shape), REF, shots=5000))
    assert not v["detection fraction"]
    v = verdict(B.sanity(raw, dets, REF, shots=5000, physical=["QB2"], planned=["QB1"]))
    assert not v["physical qubits (meta.json)"]


def test_detection_limit_follows_the_model():
    raw, _ = synthetic(REF)
    dets = (np.random.default_rng(3).random((5000, 24)) < 0.46).astype(np.uint8)
    assert not verdict(B.sanity(raw, dets, REF, shots=5000))["detection fraction"]
    high = {**REF, "model": {"detection_fraction": 0.41, "all_zero": 0.0}}
    assert verdict(B.sanity(raw, dets, high, shots=5000))["detection fraction"]
    dets = (np.random.default_rng(3).random((5000, 24)) < 0.5).astype(np.uint8)
    assert not verdict(B.sanity(raw, dets, high, shots=5000))["detection fraction"]


def test_all_zero_rows_allowed_where_the_noiseless_record_is_all_zero():
    ref = {**REF, "random_cols": [], "deterministic_cols": list(range(8)), "deterministic_values": [0] * 8,
           "noiseless_all_zero": 1.0, "noiseless_always_all_zero": True, "model": None}
    raw = np.zeros((2000, 33), dtype=np.uint8)
    v = verdict(B.sanity(raw, np.zeros((2000, 24), dtype=np.uint8), ref, shots=2000))
    assert all(v.values())


def test_sanity_line():
    raw, dets = synthetic(REF)
    res = [{"pass": True, "checks": B.sanity(raw, dets, REF, shots=5000)}]
    line = B.sanity_line(res)
    assert line.startswith("1/1 circuits pass; detection fraction 0.1")


# ---------------------------------------------------------------------------
# submit, recover and resume, against a fake connection
# ---------------------------------------------------------------------------


def small_frozen(tmp_path, n_circuits=2):
    """A Frozen with ``n_circuits`` 2-qubit native circuits on one Emerald coupler (no suite build)."""
    fake = EM._aphrodite()
    a, b = next(iter(sorted(EM.fake_loci(fake).values())))
    ia, ib = fake.qubit_name_to_index(a), fake.qubit_name_to_index(b)
    qc = QuantumCircuit(2, 2)
    qc.append(RGate(np.pi / 2, np.pi / 2), [0])
    qc.cz(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    placed = transpile(qc, backend=fake, initial_layout=[ia, ib], optimization_level=0, seed_transpiler=2026)
    names = {ia: a, ib: b}
    stim_c = stim.Circuit("H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]")
    fz = object.__new__(B.Frozen)
    fz.device = "emerald"
    fz.conf = B.DeviceConf("emerald", tmp_path / "record", CALSET, tmp_path / "snap.json")
    sha = B.instructions_sha256(L.sent_instructions(placed, names))
    fz.suite, fz.placement = {}, {}
    for i in range(n_circuits):
        name = f"c{i}"
        fz.suite[name] = SimpleNamespace(name=name, stim=stim_c, hardware=SimpleNamespace(circuit=placed, qubit_index_to_name=names))
        fz.placement[name] = {"native_sha256": sha, "placement": {"0": a, "1": b}}
    refs = {n: {"circuit": n, "measurements": 2, "detectors": 1, "first_cols": [], "random_cols": [], "deterministic_cols": [],
                "deterministic_values": [], "noiseless_all_zero": 0.5, "noiseless_always_all_zero": False,
                "x_type_first_cols": [], "z_type_first_cols": [], "model": None} for n in fz.suite}
    job = B.PlannedJob("E-B1-1-E1", "E1", 1, tuple(fz.suite), 1000, 0)
    return fz, job, refs, sorted({a, b}, key=dev.qubit_sort_key)


def fake_result(shots, circuits):
    return Result.from_dict({
        "backend_name": "emerald", "backend_version": "", "qobj_id": "", "job_id": JOB_ID, "success": True, "date": "2026-10-25",
        "results": [{"shots": shots, "success": True,
                     "data": {"memory": ["00", "11"] * (shots // 2), "counts": {"00": shots // 2, "11": shots // 2}, "metadata": {}},
                     "header": {"name": None}, "calibration_set_id": UUID(CALSET)} for _ in range(circuits)],
        "timeline": [{"status": "received", "timestamp": "2026-10-25T00:30:00Z"},
                     {"status": "execution_started", "timestamp": "2026-10-25T00:30:01Z"},
                     {"status": "execution_ended", "timestamp": "2026-10-25T00:30:03Z"}],
    })


@pytest.fixture
def armed(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER)
    for k, v in B.SEGMENTS["emerald-b1"].arming().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))
    return ledger


def fake_connect(monkeypatch, ledger, raw_root, *, fail=False, refuse=False):
    calls = []

    class FakeJob:
        def __init__(self, rr):
            self.rr = rr

        def job_id(self):
            return JOB_ID

        def result(self, **kwargs):
            assert JOB_ID in ledger.read_text(), "the ledger row comes before the result"
            if fail:
                raise TimeoutError("the wait for the result timed out")
            return fake_result(self.rr.shots, len(self.rr.circuits))

    def connect(device, calibration_set_id=None, *, read_only):
        assert not read_only and device == "emerald" and calibration_set_id == CALSET
        if refuse:
            raise ConnectionError("no route to the server")

        class Backend:
            architecture = SimpleNamespace(calibration_set_id=UUID(CALSET))

            def create_run_request(self, circuits, shots, qubit_index_to_name=None, **options):
                calls.append(("create", len(circuits), shots, dict(qubit_index_to_name)))
                return SimpleNamespace(circuits=[SimpleNamespace(instructions=L.sent_instructions(c, qubit_index_to_name))
                                                 for c in circuits], shots=shots)

            def submit_run_request(self, run_request):
                manifests = list(Path(raw_root).glob("*-emerald-b1/E-B1-1-E1.json"))
                assert manifests and json.loads(manifests[0].read_text())["job_id"] is None, "the manifest comes first"
                calls.append(("submit", run_request.shots))
                return FakeJob(run_request)

        return Backend()

    monkeypatch.setattr(hw, "_connect", connect)
    return calls


def test_submit_one_job_then_the_checks(monkeypatch, tmp_path, armed, capsys):
    fz, job, refs, qubits = small_frozen(tmp_path)
    seg = B.SEGMENTS["emerald-b1"]
    raw_root = tmp_path / "raw"
    calls = fake_connect(monkeypatch, armed, raw_root)
    res = B.submit_job(fz, seg, job, calset=CALSET, ledger=armed, raw_root=raw_root, refs=refs, timeout=10)
    assert res["ok"], res
    assert [c[0] for c in calls] == ["create", "submit"] and calls[0][1:3] == (2, 1000)
    (r,) = hw.read_ledger(armed)
    assert (r.task, r.device, r.calibration_set_id, r.qubits, r.circuits, r.shots, r.job_id) == (
        "emerald-b1", "emerald", CALSET, ",".join(qubits), 2, 1000, JOB_ID)
    assert B.label_of(r) == "E-B1-1-E1"
    man = json.loads(Path(res["manifest"]).read_text())
    assert man["job_id"] == JOB_ID and man["ok"] and [o["circuit"] for o in man["order"]] == ["c0", "c1"]
    assert all(v["ok"] for v in man["ledger_check"].values()) and all(man["meta_check"].values())
    assert [s["pass"] for s in man["sanity"]] == [True, True]
    meta = json.loads((Path(res["path"]) / "meta.json").read_text())
    assert meta["physical_qubits"] == [qubits, qubits] and meta["circuits"] == 2
    assert f"JOB ID {JOB_ID}" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no resubmission"):
        B.submit_job(fz, seg, job, calset=CALSET, ledger=armed, raw_root=raw_root, refs=refs, timeout=10)
    assert [c[0] for c in calls] == ["create", "submit"]


def test_a_broken_record_is_not_ok(monkeypatch, tmp_path, armed):
    fz, job, refs, _ = small_frozen(tmp_path)
    refs = {n: {**r, "deterministic_cols": [0], "deterministic_values": [1]} for n, r in refs.items()}
    fake_connect(monkeypatch, armed, tmp_path / "raw")
    res = B.submit_job(fz, B.SEGMENTS["emerald-b1"], job, calset=CALSET, ledger=armed, raw_root=tmp_path / "raw", refs=refs,
                       timeout=10)
    assert not res["ok"] and "0/2 circuits pass" in res["sanity"]


def test_a_failure_after_the_job_id_is_recovered_read_only(monkeypatch, tmp_path, armed, capsys):
    fz, job, refs, qubits = small_frozen(tmp_path)
    seg = B.SEGMENTS["emerald-b1"]
    raw_root = tmp_path / "raw"
    fake_connect(monkeypatch, armed, raw_root, fail=True)
    res = B.submit_job(fz, seg, job, calset=CALSET, ledger=armed, raw_root=raw_root, refs=refs, timeout=10)
    assert not res["ok"] and res["job_id"] == JOB_ID and "TimeoutError" in res["error"]
    assert f"recover --segment emerald-b1 --job-id {JOB_ID}" in capsys.readouterr().out
    assert B.find_job_dir(raw_root, seg.task, JOB_ID) is None

    def no_connect(*a, **k):
        raise AssertionError("recover must not connect to submit")

    monkeypatch.setattr(hw, "_connect", no_connect)
    monkeypatch.setattr(fz, "jobs", lambda s: [job], raising=False)
    pulled = []
    monkeypatch.setattr(hw, "retrieve", lambda job_id, device, timeout=None: pulled.append((job_id, device)) or fake_result(1000, 2))
    got, path = B.recover_job(fz, seg, JOB_ID, ledger=armed, raw_root=raw_root, timeout=10)
    assert pulled == [(JOB_ID, "emerald")] and got == job
    day = hw.read_ledger(armed)[0].time[:10].replace("-", "")
    assert path == raw_root / f"{day}-emerald-b1" / JOB_ID
    meta = json.loads((path / "meta.json").read_text())
    assert meta["physical_qubits"] == [qubits, qubits] and meta["purpose"] == B.purpose(seg, job)
    res = B.finish_job(fz, seg, job, path, calset=CALSET, ledger=armed, refs=refs,
                       manifest=B.manifest_path(raw_root, seg, job, day))
    assert res["ok"], res
    with pytest.raises(SystemExit, match="never overwritten"):
        B.recover_job(fz, seg, JOB_ID, ledger=armed, raw_root=raw_root, timeout=10)
    with pytest.raises(SystemExit, match="rows"):
        B.recover_job(fz, seg, "22222222-2222-4333-8444-555555555555", ledger=armed, raw_root=raw_root, timeout=10)


def test_resume_checks_a_saved_job_before_moving_on(monkeypatch, tmp_path, armed):
    fz, job, refs, _ = small_frozen(tmp_path)
    seg = B.SEGMENTS["emerald-b1"]
    raw_root = tmp_path / "raw"
    assert B.resume_state(fz, seg, job, ledger=armed, raw_root=raw_root, refs=refs) == "new"
    fake_connect(monkeypatch, armed, raw_root, fail=True)
    B.submit_job(fz, seg, job, calset=CALSET, ledger=armed, raw_root=raw_root, refs=refs, timeout=10)
    assert B.resume_state(fz, seg, job, ledger=armed, raw_root=raw_root, refs=refs) == "raw missing"
    monkeypatch.setattr(fz, "jobs", lambda s: [job], raising=False)
    monkeypatch.setattr(hw, "retrieve", lambda job_id, device, timeout=None: fake_result(1000, 2))
    B.recover_job(fz, seg, JOB_ID, ledger=armed, raw_root=raw_root, timeout=10)  # saved, not checked
    bad = {n: {**r, "deterministic_cols": [0], "deterministic_values": [1]} for n, r in refs.items()}
    assert B.resume_state(fz, seg, job, ledger=armed, raw_root=raw_root, refs=bad) == "failed"
    man = next(raw_root.glob("*-emerald-b1/E-B1-1-E1.json"))
    man.write_text(json.dumps({**json.loads(man.read_text()), "ok": None}))
    assert B.resume_state(fz, seg, job, ledger=armed, raw_root=raw_root, refs=refs) == "done"
    assert B.resume_state(fz, seg, job, ledger=armed, raw_root=raw_root, refs=bad) == "done"  # checked once, OK


def test_no_job_id_means_nothing_was_spent(monkeypatch, tmp_path, armed, capsys):
    fz, job, refs, _ = small_frozen(tmp_path)
    fake_connect(monkeypatch, armed, tmp_path / "raw", refuse=True)
    res = B.submit_job(fz, B.SEGMENTS["emerald-b1"], job, calset=CALSET, ledger=armed, raw_root=tmp_path / "raw", refs=refs,
                       timeout=10)
    assert not res["ok"] and res["job_id"] is None and "ConnectionError" in res["error"]
    assert hw.read_ledger(armed) == []
    out = capsys.readouterr().out
    assert "NO JOB ID" in out and "recover" not in out


def test_hw_submit_still_refuses_beyond_the_segments_allowance(monkeypatch, tmp_path, armed):
    fz, job, refs, _ = small_frozen(tmp_path)
    monkeypatch.setenv("DQ_QPU_MAX_SHOTS", "1999")
    fake_connect(monkeypatch, armed, tmp_path / "raw")
    res = B.submit_job(fz, B.SEGMENTS["emerald-b1"], job, calset=CALSET, ledger=armed, raw_root=tmp_path / "raw", refs=refs,
                       timeout=10)
    assert not res["ok"] and "QPUGuardError" in res["error"] and hw.read_ledger(armed) == []


def test_cmd_submit_refuses_unarmed():
    with pytest.raises(SystemExit, match="DQ_QPU"):
        B.cmd_submit(SimpleNamespace(segment="garnet-b1", snapshot=Path("x.json"), timeout=None))


def test_meta_checks_name_every_field(tmp_path):
    fz, job, _, qubits = small_frozen(tmp_path)
    meta = {"qubit_mapping": [{str(k): v for k, v in sorted(m.items())} for m in fz.mappings(job)],
            "physical_qubits": [qubits, qubits], "calibration_set_id": CALSET, "shots": 1000, "circuits": 2, "success": True,
            "timeline": [{"status": "received"}]}
    assert all(B.meta_checks(meta, fz, job, CALSET).values())
    assert not B.meta_checks({**meta, "physical_qubits": [qubits, ["QB50", "QB51"]]}, fz, job, CALSET)["physical_qubits"]


# ---------------------------------------------------------------------------
# the records' helpers
# ---------------------------------------------------------------------------


def test_timeline_numbers():
    tl = [{"status": "received", "timestamp": "2026-10-25T00:00:00Z"},
          {"status": "compilation_started", "timestamp": "2026-10-25T00:00:01Z"},
          {"status": "compilation_ended", "timestamp": "2026-10-25T00:00:02Z"},
          {"status": "execution_started", "timestamp": "2026-10-25T00:00:12Z"},
          {"status": "execution_ended", "timestamp": "2026-10-25T00:00:42Z"},
          {"status": "completed", "timestamp": "2026-10-25T00:00:44Z"}]
    t = B.timeline_numbers(tl, shots=100_000)
    assert (t["queue_s"], t["compile_s"], t["wait_s"], t["execution_s"], t["post_s"], t["total_s"]) == (12, 1, 10, 30, 2, 44)
    assert t["execution_per_shot_us"] == pytest.approx(300.0)
    assert B.timeline_numbers([], shots=10) == {"statuses": []}


@pytest.mark.parametrize("params, exp, want", [
    ({"basis": "Z", "patch": "pA"}, "E1", "Z_pA_b2"),
    ({"basis": "X", "patch": "eB"}, "E1", "X_eB_b2"),
    ({"basis": "X", "patch": "pA", "delay_ns": 5000.0}, "E4", "X_pA_t5000ns_b2"),
    ({"kind": "phase", "prep": "+"}, "E2", "phase_p_b2"),
    ({"kind": "bit", "prep": "1"}, "E2", "bit_1_b2"),
    ({"kind": "Z"}, "E5", "Z_b2"),
])
def test_ler_groups_are_the_dry_runs(params, exp, want):
    assert B.ler_group(SimpleNamespace(experiment=exp, params=params), 2)[1] == want


def test_ler_group_skips_e3():
    assert B.ler_group(SimpleNamespace(experiment="E3", params={}), 1) is None


def test_e3_features_pool_per_qubit(tmp_path):
    der = tmp_path / "derived" / "E3"
    for run, prep in (("b1_E3_assign_all0", 0), ("b1_E3_assign_all1", 1)):
        (der / run).mkdir(parents=True)
        with open(der / run / "assignment.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qb", "prep", "layer", "p_wrong", "p_one", "shots"])
            w.writeheader()
            for q, p in (("QB1", 0.01 if prep == 0 else 0.05), ("QB2", 0.02 if prep == 0 else 0.10)):
                w.writerow({"qb": q, "prep": prep, "layer": 0, "p_wrong": p, "p_one": p if prep == 0 else 1 - p, "shots": 2000})
                w.writerow({"qb": q, "prep": prep, "layer": 1, "p_wrong": 0.5, "p_one": 0.5, "shots": 2000})
    (der / "b1_E3_repeat_all1_k15").mkdir()
    with open(der / "b1_E3_repeat_all1_k15" / "assignment.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["qb", "prep", "layer", "p_wrong", "p_one", "shots"]).writeheader()
    with open(der / "b1_E3_repeat_all1_k15" / "readout_flips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qb", "prep", "repeats", "p_det_bulk", "record_flip", "state_flip", "p_wrong_first",
                                          "p_wrong_last", "shots"])
        w.writeheader()
        w.writerow({"qb": "QB1", "prep": 1, "repeats": 15, "p_det_bulk": 0.08, "record_flip": 0.03, "state_flip": 0.02,
                    "p_wrong_first": 0.05, "p_wrong_last": 0.2, "shots": 2000})
    (der / "b1_E3_repeat_chkB_k15").mkdir()
    with open(der / "b1_E3_repeat_chkB_k15" / "assignment.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["qb", "prep", "layer", "p_wrong", "p_one", "shots"]).writeheader()
    with open(der / "b1_E3_repeat_chkB_k15" / "readout_flips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qb", "prep", "repeats", "p_det_bulk", "record_flip", "state_flip", "p_wrong_first",
                                          "p_wrong_last", "shots"])
        w.writeheader()
        w.writerow({"qb": "QB1", "prep": 1, "repeats": 15, "p_det_bulk": 0.5, "record_flip": 0.9, "state_flip": 0.9,
                    "p_wrong_first": 0.5, "p_wrong_last": 0.5, "shots": 2000})
    s = B.e3_features(tmp_path, 1)
    rows = {r["qb"]: r for r in s["rows"]}
    assert rows["QB1"]["p1_given_0"] == pytest.approx(0.01) and rows["QB2"]["p0_given_1"] == pytest.approx(0.10)
    assert rows["QB1"]["record_flip_prep1"] == 0.03 and rows["QB2"]["state_flip_prep1"] is None
    assert s["worst"]["p0_given_1"] == (pytest.approx(0.10), "QB2")
    assert (der / "b1_pooled" / "readout_features.csv").exists()


def test_x_vs_z_pairs_the_pools():
    fit = lambda e, lo, hi: {"eps": e, "eps_ci_bootstrap": [lo, hi]}  # noqa: E731
    fits = {"X_pA_test_b3": {"fixed_A": fit(0.06, 0.05, 0.07), "free_A": fit(0.05, 0.04, 0.06)},
            "Z_pA_test_b3": {"fixed_A": fit(0.03, 0.025, 0.035), "free_A": fit(0.045, 0.03, 0.055)}}
    (row,) = B.x_vs_z(fits)
    assert row["pool"] == "pA_test_b3" and row["fixed_A"]["ratio_X_over_Z"] == pytest.approx(2.0)
    assert not row["fixed_A"]["intervals_overlap"] and row["free_A"]["intervals_overlap"]


def test_pool_sums_the_same_round_counts():
    pts = [{"rounds": 3, "failures": 10, "shots": 100}, {"rounds": 3, "failures": 5, "shots": 100},
           {"rounds": 1, "failures": 1, "shots": 50}]
    r, k, n = B.pool(pts)
    assert r.tolist() == [1, 3] and k.tolist() == [1, 15] and n.tolist() == [50, 200]
