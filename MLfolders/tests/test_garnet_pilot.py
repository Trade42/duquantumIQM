"""scripts/garnet_pilot.py offline: the stop/go checks on synthetic arrays, the guards, and the submit and recover
paths against a fake connection. Nothing here reaches the network or builds the suite."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate
from qiskit.result import Result

from dqiqm import hw
from dqiqm import layout as L
from dqiqm.experiments import device as dev

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("garnet_pilot", REPO / "scripts" / "garnet_pilot.py")
GP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GP)

BAND = json.loads(GP.JOB_PLAN.read_text())["pilot"]["band"]
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "11111111-2222-4333-8444-555555555555"
LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""


def plausible(seed=2026, shots=1000, det_fraction=0.15):
    """A record and detection events that pass every band: X-type round-1 bits random, Z-type mostly 0."""
    rng = np.random.default_rng(seed)
    raw = (rng.random((shots, 33)) < 0.2).astype(np.uint8)
    raw[:, BAND["x_cols"]] = rng.random((shots, 4)) < 0.5
    raw[:, BAND["z_cols"]] = rng.random((shots, 4)) < 0.06
    raw[:, -1] = 1  # no all-zero row
    dets = (rng.random((shots, 24)) < det_fraction).astype(np.uint8)
    return raw, dets


def verdicts(checks):
    return {c["check"]: c["pass"] for c in checks}


# ---------------------------------------------------------------------------
# stop/go
# ---------------------------------------------------------------------------


def test_band_is_the_plans():
    assert BAND["x_cols"] == [0, 2, 5, 7] and BAND["z_cols"] == [1, 3, 4, 6]
    assert BAND["columns"] == 33 and BAND["all_zero_rows_max_per_1000"] == 5
    assert BAND["x_ancilla_m1_one"] == [0.4, 0.6] and BAND["z_ancilla_m1_one_max"] == 0.3
    assert BAND["detection_fraction"] == pytest.approx([0.05179, 0.34496], abs=1e-5)


def test_stop_go_passes_plausible_data():
    raw, dets = plausible()
    checks = GP.stop_go(raw, dets, BAND)
    assert [c["check"] for c in checks] == ["memory shape", "all-zero rows", "X-type round-1 bits, pooled mean",
                                            "Z-type round-1 bits, pooled mean", "detection fraction"]
    assert all(verdicts(checks).values()), checks


def test_stop_go_reports_the_values():
    raw, dets = plausible()
    v = {c["check"]: c["value"] for c in GP.stop_go(raw, dets, BAND)}
    assert v["X-type round-1 bits, pooled mean"] == pytest.approx(raw[:, [0, 2, 5, 7]].mean())
    assert v["Z-type round-1 bits, pooled mean"] == pytest.approx(raw[:, [1, 3, 4, 6]].mean())
    assert v["detection fraction"] == pytest.approx(dets.mean())
    assert v["all-zero rows"] == 0


@pytest.mark.parametrize("zeros, ok", [(5, True), (6, False)])
def test_all_zero_rows_at_most_5(zeros, ok):
    raw, dets = plausible()
    raw[:zeros] = 0
    assert verdicts(GP.stop_go(raw, dets, BAND))["all-zero rows"] is ok


@pytest.mark.parametrize("p, ok", [(0.5, True), (0.35, False), (0.65, False), (0.0, False), (1.0, False)])
def test_x_type_bits_must_be_random(p, ok):
    raw, dets = plausible()
    raw[:, BAND["x_cols"]] = np.random.default_rng(7).random((1000, 4)) < p
    assert verdicts(GP.stop_go(raw, dets, BAND))["X-type round-1 bits, pooled mean"] is ok


@pytest.mark.parametrize("p, ok", [(0.05, True), (0.25, True), (0.35, False), (0.5, False)])
def test_z_type_bits_must_be_mostly_0(p, ok):
    raw, dets = plausible()
    raw[:, BAND["z_cols"]] = np.random.default_rng(8).random((1000, 4)) < p
    assert verdicts(GP.stop_go(raw, dets, BAND))["Z-type round-1 bits, pooled mean"] is ok


@pytest.mark.parametrize("f, ok", [(0.17, True), (0.06, True), (0.03, False), (0.5, False)])
def test_detection_fraction_band(f, ok):
    raw, _ = plausible()
    _, dets = plausible(det_fraction=f)
    assert verdicts(GP.stop_go(raw, dets, BAND))["detection fraction"] is ok


@pytest.mark.parametrize("shape", [(999, 33), (1000, 32), (1000,)])
def test_wrong_shape_fails_everything(shape):
    raw = np.ones(shape, dtype=np.uint8)
    checks = GP.stop_go(raw, np.zeros((1000, 24)), BAND)
    assert not any(verdicts(checks).values())
    assert all(c["value"] is None for c in checks[1:])


def test_a_swapped_bit_order_fails():
    """Qiskit's bitstring order left in place puts the data readout where the round-1 bits should be."""
    raw, dets = plausible()
    raw[:, -9:] = 0  # the final data readout of a |0> memory: mostly 0
    flipped = raw[:, ::-1]
    assert not all(verdicts(GP.stop_go(flipped, dets, BAND)).values())


# ---------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------


def test_timeline_numbers():
    tl = [{"source": "s", "status": s, "timestamp": t} for s, t in (
        ("received", "2026-10-24T21:00:00Z"), ("compilation_started", "2026-10-24T21:00:01Z"),
        ("compilation_ended", "2026-10-24T21:00:02.5Z"), ("execution_started", "2026-10-24T21:00:10Z"),
        ("execution_ended", "2026-10-24T21:00:10.6Z"), ("completed", "2026-10-24T21:00:11+00:00"))]
    out = GP.timeline_numbers(tl, shots=1000, circuit_ns=2340.0, reset_wait_ns=298_000.0)
    assert out["statuses"][0] == "received" and out["total_s"] == pytest.approx(11.0)
    assert out["queue_s"] == pytest.approx(10.0) and out["compile_s"] == pytest.approx(1.5)
    assert out["execution_s"] == pytest.approx(0.6)
    assert out["per_shot_s"] == pytest.approx(600e-6)
    assert out["planned_per_shot_s"] == pytest.approx(300.34e-6)
    assert out["per_shot_beyond_plan_s"] == pytest.approx(600e-6 - 300.34e-6)
    assert out["ratio_to_plan"] == pytest.approx(600 / 300.34)


def test_timeline_without_execution_entries():
    out = GP.timeline_numbers([{"status": "received", "timestamp": "2026-10-24T21:00:00Z"}], shots=1000,
                              circuit_ns=2340.0, reset_wait_ns=298_000.0)
    assert "execution_s" not in out and "per_shot_s" not in out and out["planned_execution_s"] == pytest.approx(0.30034)


# ---------------------------------------------------------------------------
# the patch and the guards
# ---------------------------------------------------------------------------


def test_patch_from_placement():
    qubits, couplers = GP.patch_from_placement(GP.placement_entry())
    assert qubits == list(GP.PATCH_QUBITS) and len(qubits) == 17
    assert len(couplers) == 24 and len(set(couplers)) == 24


def test_patch_gates_on_the_committed_snapshot():
    arch = dev.load_snapshot(dev.DEFAULT_SNAPSHOT)["architecture"]
    qubits, couplers = GP.patch_from_placement(GP.placement_entry())
    assert GP.patch_gates(arch, qubits, couplers) == {"prx": [], "measure": [], "cz": []}
    impl = arch["gates"]["cz"]["implementations"]
    for info in impl.values():
        info["loci"] = [lc for lc in info["loci"] if set(lc) != set(couplers[0])]
    arch["gates"]["measure"]["implementations"]["constant"]["loci"].remove(["QB7"])
    missing = GP.patch_gates(arch, qubits, couplers)
    assert missing == {"prx": [], "measure": ["QB7"], "cz": ["-".join(couplers[0])]}


ARMED = {"DQ_QPU": "run", "DQ_QPU_TASK": "garnet-pilot", "DQ_QPU_MAX_JOBS": "1", "DQ_QPU_MAX_SHOTS": "1000"}


def test_arming_exactly_as_granted():
    GP.check_arming(ARMED)
    for k, bad in (("DQ_QPU", "no"), ("DQ_QPU_TASK", "other"), ("DQ_QPU_MAX_JOBS", "2"), ("DQ_QPU_MAX_SHOTS", "2000")):
        with pytest.raises(SystemExit, match=k):
            GP.check_arming({**ARMED, k: bad})
    with pytest.raises(SystemExit, match="ledger"):
        GP.check_arming({**ARMED, "DQ_QPU_LEDGER": "/tmp/elsewhere.md"})


def test_existing_jobs(tmp_path):
    assert GP.existing_jobs(tmp_path) == []
    (tmp_path / "20261024-garnet-pilot").mkdir()
    assert GP.existing_jobs(tmp_path) == []
    (tmp_path / "20261024-garnet-pilot" / JOB_ID).mkdir()
    assert GP.existing_jobs(tmp_path) == [tmp_path / "20261024-garnet-pilot" / JOB_ID]
    (tmp_path / "20261024-other").mkdir()
    assert len(GP.existing_jobs(tmp_path)) == 1


def test_instructions_sha256_is_native_sha256():
    from dqiqm.experiments import circuits as C

    p = small_pilot()
    hc = SimpleNamespace(circuit=p.hc.circuit, qubit_index_to_name=p.qubit_index_to_name)
    sent = L.sent_instructions(hc.circuit, hc.qubit_index_to_name)
    assert GP.instructions_sha256(sent) == C.native_sha256(hc)


# ---------------------------------------------------------------------------
# submit and recover, against a fake connection
# ---------------------------------------------------------------------------


def small_pilot():
    """A 2-qubit native circuit placed on IQMFakeGarnet's QB1, QB2, standing in for Pilot (no suite build)."""
    qc = QuantumCircuit(2, 2)
    qc.append(RGate(np.pi / 2, np.pi / 2), [0])
    qc.cz(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    b = IQMFakeGarnet()
    placed = transpile(qc, backend=b, initial_layout=[0, 1], optimization_level=0, seed_transpiler=2026)
    names = {0: "QB1", 1: "QB2"}
    return SimpleNamespace(hc=SimpleNamespace(circuit=placed), qubit_index_to_name=names,
                           qubit_mapping=hw._qubit_mappings(None, [placed], names), sent=[["QB1", "QB2"]])


def fake_result(shots, *, timeline=True):
    return Result.from_dict({
        "backend_name": "garnet", "backend_version": "", "qobj_id": "", "job_id": JOB_ID, "success": True,
        "date": "2026-10-24",
        "results": [{"shots": shots, "success": True,
                     "data": {"memory": ["01", "10"] * (shots // 2), "counts": {"01": shots // 2, "10": shots // 2},
                              "metadata": {}},
                     "header": {"name": "pilot"}, "calibration_set_id": UUID(CALSET)}],
        "timeline": [SimpleNamespace(status="received", timestamp="2026-10-24T21:00:00Z")] if timeline else [],
    })


class FakeJob:
    def __init__(self, ledger, shots, fail):
        self.ledger, self.shots, self.fail = ledger, shots, fail

    def job_id(self):
        return JOB_ID

    def result(self, **kwargs):
        assert JOB_ID in self.ledger.read_text(), "the ledger row comes before the result"
        if self.fail:
            raise TimeoutError("the wait for the result timed out")
        return fake_result(self.shots)


@pytest.fixture
def armed(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER)
    for k, v in ARMED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))
    return ledger


def fake_connect(monkeypatch, ledger, *, fail=False, refuse=False):
    calls = []

    def connect(device, calibration_set_id=None, *, read_only):
        assert not read_only and device == "garnet" and calibration_set_id == CALSET
        if refuse:
            raise ConnectionError("no route to the server")

        class Backend:
            architecture = SimpleNamespace(calibration_set_id=UUID(CALSET))

            def create_run_request(self, circuits, shots, qubit_index_to_name=None, **options):
                calls.append(("create", shots, dict(qubit_index_to_name)))
                return SimpleNamespace(circuits=[SimpleNamespace(instructions=L.sent_instructions(c, qubit_index_to_name))
                                                 for c in circuits], shots=shots)

            def submit_run_request(self, run_request):
                calls.append(("submit", run_request.shots))
                return FakeJob(ledger, run_request.shots, fail)

        return Backend()

    monkeypatch.setattr(hw, "_connect", connect)
    return calls


def test_submit_once_then_checks_the_ledger_row_and_meta(monkeypatch, tmp_path, armed, capsys):
    p = small_pilot()
    calls = fake_connect(monkeypatch, armed)
    res = GP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert res["ok"], res
    assert [c[0] for c in calls] == ["create", "submit"] and calls[0] == ("create", 1000, {0: "QB1", 1: "QB2"})
    (row,) = hw.read_ledger(armed)
    assert (row.task, row.device, row.calibration_set_id, row.qubits, row.circuits, row.shots, row.job_id) == (
        "garnet-pilot", "garnet", CALSET, "QB1,QB2", 1, 1000, JOB_ID)
    assert all(v["ok"] for v in res["ledger_check"].values()) and all(res["meta_check"].values())
    job_dir = Path(res["path"])
    assert job_dir.parent.name.endswith("-garnet-pilot") and job_dir.name == JOB_ID
    assert np.load(job_dir / "memory.npz")["c0"].shape == (1000, 2)
    assert f"JOB ID {JOB_ID}" in capsys.readouterr().out
    # a second submission is refused before any connection: the ledger row, and the raw job, both stop it
    with pytest.raises(SystemExit, match="already has a garnet-pilot row"):
        GP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    armed.write_text(LEDGER_HEADER)
    with pytest.raises(SystemExit, match="already holds"):
        GP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert [c[0] for c in calls] == ["create", "submit"]


def test_a_failure_after_the_job_id_is_recovered_read_only(monkeypatch, tmp_path, armed, capsys):
    p = small_pilot()
    fake_connect(monkeypatch, armed, fail=True)
    res = GP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert not res["ok"] and res["job_id"] == JOB_ID and "TimeoutError" in res["error"]
    out = capsys.readouterr().out
    assert f"JOB ID {JOB_ID}" in out and f"recover --job-id {JOB_ID}" in out
    assert GP.existing_jobs(tmp_path / "raw") == []  # save_raw never ran

    def no_connect(*a, **k):
        raise AssertionError("recover must not connect to submit")

    monkeypatch.setattr(hw, "_connect", no_connect)
    pulled = []
    monkeypatch.setattr(hw, "retrieve", lambda job_id, device, timeout=None: pulled.append((job_id, device)) or fake_result(1000))
    path = GP.recover_pilot(p, JOB_ID, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert pulled == [(JOB_ID, "garnet")]
    meta = json.loads((path / "meta.json").read_text())
    day = hw.read_ledger(armed)[0].time[:10].replace("-", "")
    assert path == tmp_path / "raw" / f"{day}-garnet-pilot" / JOB_ID
    assert meta["physical_qubits"] == [["QB1", "QB2"]] and meta["qubit_mapping"] == [{"0": "QB1", "1": "QB2"}]
    assert meta["shots"] == 1000 and meta["task"] == "garnet-pilot" and meta["calibration_set_id"] == CALSET
    with pytest.raises(SystemExit, match="never overwritten"):
        GP.recover_pilot(p, JOB_ID, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    with pytest.raises(SystemExit, match="rows"):
        GP.recover_pilot(p, "22222222-2222-4333-8444-555555555555", ledger=armed, data_root=tmp_path / "x", timeout=10)


def test_no_job_id_means_nothing_was_spent(monkeypatch, tmp_path, armed, capsys):
    fake_connect(monkeypatch, armed, refuse=True)
    res = GP.submit_pilot(small_pilot(), calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert not res["ok"] and res["job_id"] is None and "ConnectionError" in res["error"]
    assert hw.read_ledger(armed) == []
    out = capsys.readouterr().out
    assert "NO JOB ID" in out and "recover" not in out


def test_submit_refuses_unarmed(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="DQ_QPU"):
        GP.cmd_submit(SimpleNamespace(snapshot=dev.DEFAULT_SNAPSHOT, timeout=None))
