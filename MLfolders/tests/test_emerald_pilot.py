"""scripts/emerald_pilot.py offline: the stop/go checks on synthetic arrays, MINOR 3's qubit checks, the guards and the
allowance, and the submit and recover paths against a fake connection. Nothing here reaches the network or builds the
suite."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate
from qiskit.result import Result

from dqiqm import hw
from dqiqm import layout as L
from dqiqm.experiments import device as dev
from dqiqm.experiments import emerald as EM

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("emerald_pilot", REPO / "scripts" / "emerald_pilot.py")
EP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EP)

BAND = EP.pilot_band()
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "11111111-2222-4333-8444-555555555555"
GARNET_ROW = ("| 2026-10-24T22:20:36Z | garnet-pilot | garnet | a18a947f-0408-4628-81ea-1715637e434f | "
              "QB1,QB2,QB4,QB5,QB6,QB7,QB8,QB9,QB10,QB11,QB12,QB13,QB14,QB15,QB16,QB19,QB20 | 1 x 1000 | "
              "01a17000-6394-755f-a58b-b4f9c01893b0 | garnet-pilot P0 E1_d3_Z_r03_pA |\n")
LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""
EA = ["QB13", "QB14", "QB19", "QB20", "QB21", "QB22", "QB27", "QB28", "QB29", "QB30", "QB31", "QB36", "QB37", "QB38",
      "QB39", "QB44", "QB45"]


def plausible(seed=2026, shots=1000, det_fraction=0.12):
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
# the plan's band and the patch
# ---------------------------------------------------------------------------


def test_band_is_the_plans():
    plan = json.loads(EP.JOB_PLAN.read_text())["pilot"]
    assert plan["circuit"] == EP.CIRCUIT == "E1_d3_Z_r03_eA" and plan["device"] == "emerald"
    assert BAND["x_cols"] == [0, 2, 5, 7] and BAND["z_cols"] == [1, 3, 4, 6]
    assert BAND["columns"] == 33 and BAND["shots"] == 1000 and BAND["all_zero_rows_max_per_1000"] == 5
    assert BAND["x_ancilla_m1_one"] == [0.4, 0.6] and BAND["z_ancilla_m1_one_max"] == 0.3
    assert BAND["detection_fraction"] == pytest.approx([0.040208, 0.231876], abs=1e-6)
    assert plan["calibrated_stim_with_idle"]["detection_fraction"] == pytest.approx(0.115938, abs=1e-6)


def test_patch_is_ea():
    assert list(EP.PATCH_QUBITS) == EA
    qubits, couplers = EP.patch_from_placement(EP.placement_entry())
    assert qubits == EA and len(couplers) == 24 and len(set(couplers)) == 24
    plan = json.loads(EP.JOB_PLAN.read_text())
    assert plan["choices"]["E1"]["patches"]["eA"] == EA
    assert not {frozenset(c) for c in couplers} & {frozenset(c) for c in EM.CZ_LESS}


def test_patch_gates_on_the_committed_snapshot():
    """Emerald's cz has two implementations and per-locus overrides: _served reads the one that runs."""
    arch = dev.load_snapshot(EM.DEFAULT_SNAPSHOT)["architecture"]
    qubits, couplers = EP.patch_from_placement(EP.placement_entry())
    assert EP.patch_gates(arch, qubits, couplers) == {"prx": [], "measure": [], "cz": []}
    for info in arch["gates"]["cz"]["implementations"].values():
        info["loci"] = [lc for lc in info["loci"] if set(lc) != set(couplers[0])]
    arch["gates"]["measure"]["implementations"]["constant"]["loci"].remove(["QB21"])
    missing = EP.patch_gates(arch, qubits, couplers)
    assert missing == {"prx": [], "measure": ["QB21"], "cz": ["-".join(couplers[0])]}
    # a CZ-less coupler is not served
    assert not EP._served(dev.load_snapshot(EM.DEFAULT_SNAPSHOT)["architecture"], "cz", ("QB6", "QB7"))


# ---------------------------------------------------------------------------
# stop/go
# ---------------------------------------------------------------------------


def test_stop_go_passes_plausible_data():
    raw, dets = plausible()
    checks = EP.stop_go(raw, dets, BAND)
    assert [c["check"] for c in checks] == ["memory shape", "all-zero rows", "X-type round-1 bits, pooled mean",
                                            "Z-type round-1 bits, pooled mean", "detection fraction"]
    assert all(verdicts(checks).values()), checks


def test_stop_go_reports_the_values():
    raw, dets = plausible()
    v = {c["check"]: c["value"] for c in EP.stop_go(raw, dets, BAND)}
    assert v["X-type round-1 bits, pooled mean"] == pytest.approx(raw[:, [0, 2, 5, 7]].mean())
    assert v["Z-type round-1 bits, pooled mean"] == pytest.approx(raw[:, [1, 3, 4, 6]].mean())
    assert v["detection fraction"] == pytest.approx(dets.mean())
    assert v["all-zero rows"] == 0


@pytest.mark.parametrize("zeros, ok", [(5, True), (6, False)])
def test_all_zero_rows_at_most_5(zeros, ok):
    raw, dets = plausible()
    raw[:zeros] = 0
    assert verdicts(EP.stop_go(raw, dets, BAND))["all-zero rows"] is ok


@pytest.mark.parametrize("p, ok", [(0.5, True), (0.35, False), (0.65, False), (0.0, False), (1.0, False)])
def test_x_type_bits_must_be_random(p, ok):
    raw, dets = plausible()
    raw[:, BAND["x_cols"]] = np.random.default_rng(7).random((1000, 4)) < p
    assert verdicts(EP.stop_go(raw, dets, BAND))["X-type round-1 bits, pooled mean"] is ok


@pytest.mark.parametrize("p, ok", [(0.05, True), (0.25, True), (0.35, False), (0.5, False)])
def test_z_type_bits_must_be_mostly_0(p, ok):
    raw, dets = plausible()
    raw[:, BAND["z_cols"]] = np.random.default_rng(8).random((1000, 4)) < p
    assert verdicts(EP.stop_go(raw, dets, BAND))["Z-type round-1 bits, pooled mean"] is ok


@pytest.mark.parametrize("f, ok", [(0.12, True), (0.05, True), (0.22, True), (0.03, False), (0.26, False), (0.5, False)])
def test_detection_fraction_band(f, ok):
    raw, _ = plausible()
    _, dets = plausible(det_fraction=f)
    assert verdicts(EP.stop_go(raw, dets, BAND))["detection fraction"] is ok


@pytest.mark.parametrize("shape", [(999, 33), (1000, 32), (1000,)])
def test_wrong_shape_fails_everything(shape):
    raw = np.ones(shape, dtype=np.uint8)
    checks = EP.stop_go(raw, np.zeros((1000, 24)), BAND)
    assert not any(verdicts(checks).values())
    assert all(c["value"] is None for c in checks[1:])


def test_a_swapped_bit_order_fails():
    """Qiskit's bitstring order left in place puts the data readout where the round-1 bits should be."""
    raw, dets = plausible()
    raw[:, -9:] = 0  # the final data readout of a |0> memory: mostly 0
    flipped = raw[:, ::-1]
    assert not all(verdicts(EP.stop_go(flipped, dets, BAND)).values())


# ---------------------------------------------------------------------------
# MINOR 3: where the job ran
# ---------------------------------------------------------------------------


def _row(qubits=EA, task="emerald-pilot", device="emerald", calset=CALSET, cs="1 x 1000", job_id=JOB_ID):
    c, s = cs.split(" x ")
    return hw.LedgerRow("2026-10-24T23:00:00Z", task, device, calset, ",".join(qubits), int(c), int(s), job_id, "p")


WANT_MAP = [{"0": "QB13", "1": "QB14"}]


def _meta(phys=None):
    return {"job_id": JOB_ID, "physical_qubits": [EA] if phys is None else phys, "qubit_mapping": WANT_MAP}


def test_qubit_checks_pass_on_ea():
    checks = EP.qubit_checks(_meta(), [_row()], calset=CALSET, want_mapping=WANT_MAP)
    assert len(checks) == 3 and all(verdicts(checks).values()), checks


def test_qubit_checks_fail_on_another_valid_placement():
    """eB's circuit, or another orientation on other qubits, passes every band: only the qubits tell (MINOR 3)."""
    eb = json.loads(EP.JOB_PLAN.read_text())["choices"]["E1"]["patches"]["eB"]
    assert eb != EA and len(eb) == 17
    v = verdicts(EP.qubit_checks(_meta([eb]), [_row(eb)], calset=CALSET, want_mapping=WANT_MAP))
    assert v == {"physical qubits (meta.json) = eA's 17": False, "qubit_mapping (meta.json) = the placement": True,
                 "ledger row = eA's 17, 1 x 1000": False}


@pytest.mark.parametrize("field", ["task", "device", "calset", "cs"])
def test_qubit_checks_read_the_whole_ledger_row(field):
    bad = {"task": "garnet-pilot", "device": "garnet", "calset": "x", "cs": "1 x 2000"}[field]
    v = verdicts(EP.qubit_checks(_meta(), [_row(**{field: bad})], calset=CALSET, want_mapping=WANT_MAP))
    assert v["ledger row = eA's 17, 1 x 1000"] is False


@pytest.mark.parametrize("rows", [[], [_row(), _row()]])
def test_qubit_checks_need_exactly_one_ledger_row(rows):
    v = verdicts(EP.qubit_checks(_meta(), rows, calset=CALSET, want_mapping=WANT_MAP))
    assert v["ledger row = eA's 17, 1 x 1000"] is False


def test_qubit_checks_simulated_has_no_ledger():
    checks = EP.qubit_checks(_meta(), None, calset=CALSET, want_mapping=WANT_MAP)
    assert checks[-1]["band"] == "not applicable: simulated" and checks[-1]["value"] is None


# ---------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------


def test_timeline_numbers():
    tl = [{"source": "s", "status": s, "timestamp": t} for s, t in (
        ("received", "2026-10-24T23:00:00Z"), ("compilation_started", "2026-10-24T23:00:01Z"),
        ("compilation_ended", "2026-10-24T23:00:02.5Z"), ("execution_started", "2026-10-24T23:00:10Z"),
        ("execution_ended", "2026-10-24T23:00:10.6Z"), ("completed", "2026-10-24T23:00:11+00:00"))]
    out = EP.timeline_numbers(tl, shots=1000, circuit_ns=1848.0, reset_wait_ns=400_000.0)
    assert out["statuses"][0] == "received" and out["total_s"] == pytest.approx(11.0)
    assert out["queue_s"] == pytest.approx(10.0) and out["compile_s"] == pytest.approx(1.5)
    assert out["execution_s"] == pytest.approx(0.6)
    assert out["per_shot_s"] == pytest.approx(600e-6)
    assert out["planned_per_shot_s"] == pytest.approx(401.848e-6)
    assert out["per_shot_beyond_reset_wait_s"] == pytest.approx(200e-6)
    assert out["ratio_to_plan"] == pytest.approx(600 / 401.848)


def test_timeline_without_execution_entries():
    out = EP.timeline_numbers([{"status": "received", "timestamp": "2026-10-24T23:00:00Z"}], shots=1000,
                              circuit_ns=1848.0, reset_wait_ns=400_000.0)
    assert "execution_s" not in out and "per_shot_s" not in out and out["planned_execution_s"] == pytest.approx(0.401848)


def test_garnet_reference_reads_the_garnet_pilot():
    g = EP.garnet_reference()
    assert g["job_id"] == "01a17000-6394-755f-a58b-b4f9c01893b0"
    assert g["ratio"] == pytest.approx(0.22133 / 0.17248, abs=1e-3) and g["per_shot_s"] == pytest.approx(50.663e-6)


# ---------------------------------------------------------------------------
# the guards and the allowance
# ---------------------------------------------------------------------------


ARMED = {"DQ_QPU": "run", "DQ_QPU_TASK": "emerald-pilot", "DQ_QPU_MAX_JOBS": "1", "DQ_QPU_MAX_SHOTS": "1000"}


def test_arming_exactly_as_granted():
    EP.check_arming(ARMED)
    for k, bad in (("DQ_QPU", "no"), ("DQ_QPU_TASK", "garnet-pilot"), ("DQ_QPU_MAX_JOBS", "2"),
                   ("DQ_QPU_MAX_SHOTS", "2000")):
        with pytest.raises(SystemExit, match=k):
            EP.check_arming({**ARMED, k: bad})
    with pytest.raises(SystemExit, match="ledger"):
        EP.check_arming({**ARMED, "DQ_QPU_LEDGER": "/tmp/elsewhere.md"})


def test_the_garnet_row_does_not_count_against_emerald(tmp_path):
    """The allowance counts this task's rows only: the Garnet pilot's row leaves the whole allowance."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER + GARNET_ROW)
    out = EP.allowance_check(ledger)
    assert out["rows"] == 1 and out["task_rows"] == 0 and out["task_usage"] == [0, 0]
    with open(ledger, "a") as f:
        f.write(hw.ledger_line(task="emerald-pilot", device="emerald", calibration_set_id=CALSET, qubits=EA,
                               circuits=1, shots=1000, job_id=JOB_ID, purpose="p") + "\n")
    with pytest.raises(hw.QPUGuardError, match="emerald-pilot: 1 job"):
        EP.allowance_check(ledger)


def test_existing_jobs(tmp_path):
    assert EP.existing_jobs(tmp_path) == []
    (tmp_path / "20261024-emerald-pilot").mkdir()
    assert EP.existing_jobs(tmp_path) == []
    (tmp_path / "20261024-emerald-pilot" / JOB_ID).mkdir()
    assert EP.existing_jobs(tmp_path) == [tmp_path / "20261024-emerald-pilot" / JOB_ID]
    (tmp_path / "20261024-garnet-pilot").mkdir()
    (tmp_path / "20261024-garnet-pilot" / "x").mkdir()
    assert len(EP.existing_jobs(tmp_path)) == 1


def test_raw_root_is_the_emerald_record():
    assert EP.RAW_ROOT == REPO / "InfoRecord" / "emerald" / "raw"
    assert EP.RECORD == REPO / "InfoRecord" / "emerald" and EP.RUN_ID == "ep0_E1_d3_Z_r03_eA"


def test_instructions_sha256_is_native_sha256():
    from dqiqm.experiments import circuits as C

    p = small_pilot()
    hc = SimpleNamespace(circuit=p.hc.circuit, qubit_index_to_name=p.qubit_index_to_name)
    sent = L.sent_instructions(hc.circuit, hc.qubit_index_to_name)
    assert EP.instructions_sha256(sent) == C.native_sha256(hc)


# ---------------------------------------------------------------------------
# submit and recover, against a fake connection
# ---------------------------------------------------------------------------


def small_pilot():
    """A 2-qubit native circuit on a coupler of IQMFakeAphrodite (Emerald's names), standing in for Pilot (no suite build)."""
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
    sent = sorted({a, b}, key=dev.qubit_sort_key)
    return SimpleNamespace(hc=SimpleNamespace(circuit=placed), qubit_index_to_name=names,
                           qubit_mapping=hw._qubit_mappings(None, [placed], names), sent=[sent])


def fake_result(shots, *, timeline=True):
    return Result.from_dict({
        "backend_name": "emerald", "backend_version": "", "qobj_id": "", "job_id": JOB_ID, "success": True,
        "date": "2026-10-24",
        "results": [{"shots": shots, "success": True,
                     "data": {"memory": ["01", "10"] * (shots // 2), "counts": {"01": shots // 2, "10": shots // 2},
                              "metadata": {}},
                     "header": {"name": "pilot"}, "calibration_set_id": UUID(CALSET)}],
        "timeline": [SimpleNamespace(status="received", timestamp="2026-10-24T23:00:00Z")] if timeline else [],
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
    """Armed as the brief grants, on a ledger that already holds the Garnet pilot's row."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER + GARNET_ROW)
    for k, v in ARMED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))
    return ledger


def fake_connect(monkeypatch, ledger, *, fail=False, refuse=False):
    calls = []

    def connect(device, calibration_set_id=None, *, read_only):
        assert not read_only and device == "emerald" and calibration_set_id == CALSET
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


def _patch_qubits(monkeypatch, p):
    """The fake circuit's two qubits stand in for eA's 17 in the ledger and meta checks."""
    monkeypatch.setattr(EP, "PATCH_QUBITS", tuple(p.sent[0]))


def test_submit_once_then_checks_the_ledger_row_and_meta(monkeypatch, tmp_path, armed, capsys):
    p = small_pilot()
    _patch_qubits(monkeypatch, p)
    calls = fake_connect(monkeypatch, armed)
    res = EP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert res["ok"], res
    assert [c[0] for c in calls] == ["create", "submit"] and calls[0] == ("create", 1000, p.qubit_index_to_name)
    garnet, row = hw.read_ledger(armed)
    assert garnet.task == "garnet-pilot" and res["ledger_rows"] == 1 and res["ledger_rows_all"] == 2
    assert (row.task, row.device, row.calibration_set_id, row.qubits, row.circuits, row.shots, row.job_id) == (
        "emerald-pilot", "emerald", CALSET, ",".join(p.sent[0]), 1, 1000, JOB_ID)
    assert all(v["ok"] for v in res["ledger_check"].values()) and all(res["meta_check"].values())
    job_dir = Path(res["path"])
    assert job_dir.parent.name.endswith("-emerald-pilot") and job_dir.name == JOB_ID
    assert np.load(job_dir / "memory.npz")["c0"].shape == (1000, 2)
    assert json.loads((job_dir / "meta.json").read_text())["device"] == "emerald"
    assert f"JOB ID {JOB_ID}" in capsys.readouterr().out
    # a second submission is refused before any connection: the ledger row, and the raw job, both stop it
    with pytest.raises(SystemExit, match="already has a emerald-pilot row"):
        EP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    armed.write_text(LEDGER_HEADER + GARNET_ROW)
    with pytest.raises(SystemExit, match="already holds"):
        EP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert [c[0] for c in calls] == ["create", "submit"]


def test_submit_flags_a_job_off_ea(monkeypatch, tmp_path, armed):
    """With eA's 17 as the plan, a job sent elsewhere is reported NOT OK (the ledger row and meta.json disagree)."""
    p = small_pilot()
    fake_connect(monkeypatch, armed)
    res = EP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert not res["ok"] and not res["ledger_check"]["qubits"]["ok"] and not res["meta_check"]["physical_qubits"]


def test_a_failure_after_the_job_id_is_recovered_read_only(monkeypatch, tmp_path, armed, capsys):
    p = small_pilot()
    fake_connect(monkeypatch, armed, fail=True)
    res = EP.submit_pilot(p, calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert not res["ok"] and res["job_id"] == JOB_ID and "TimeoutError" in res["error"]
    out = capsys.readouterr().out
    assert f"JOB ID {JOB_ID}" in out and f"recover --job-id {JOB_ID}" in out
    assert EP.existing_jobs(tmp_path / "raw") == []  # save_raw never ran

    def no_connect(*a, **k):
        raise AssertionError("recover must not connect to submit")

    monkeypatch.setattr(hw, "_connect", no_connect)
    pulled = []
    monkeypatch.setattr(hw, "retrieve", lambda job_id, device, timeout=None: pulled.append((job_id, device)) or fake_result(1000))
    path = EP.recover_pilot(p, JOB_ID, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert pulled == [(JOB_ID, "emerald")]
    meta = json.loads((path / "meta.json").read_text())
    day = [r for r in hw.read_ledger(armed) if r.task == "emerald-pilot"][0].time[:10].replace("-", "")
    assert path == tmp_path / "raw" / f"{day}-emerald-pilot" / JOB_ID
    assert meta["physical_qubits"] == p.sent and meta["qubit_mapping"] == [{str(k): v for k, v in p.qubit_mapping[0].items()}]
    assert meta["shots"] == 1000 and meta["task"] == "emerald-pilot" and meta["calibration_set_id"] == CALSET
    with pytest.raises(SystemExit, match="never overwritten"):
        EP.recover_pilot(p, JOB_ID, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    with pytest.raises(SystemExit, match="rows"):
        EP.recover_pilot(p, "22222222-2222-4333-8444-555555555555", ledger=armed, data_root=tmp_path / "x", timeout=10)
    # the Garnet job is not this task's: recover refuses it
    with pytest.raises(SystemExit, match="rows"):
        EP.recover_pilot(p, "01a17000-6394-755f-a58b-b4f9c01893b0", ledger=armed, data_root=tmp_path / "y", timeout=10)


def test_no_job_id_means_nothing_was_spent(monkeypatch, tmp_path, armed, capsys):
    fake_connect(monkeypatch, armed, refuse=True)
    res = EP.submit_pilot(small_pilot(), calset=CALSET, ledger=armed, data_root=tmp_path / "raw", timeout=10)
    assert not res["ok"] and res["job_id"] is None and "ConnectionError" in res["error"]
    assert [r.task for r in hw.read_ledger(armed)] == ["garnet-pilot"]
    out = capsys.readouterr().out
    assert "NO JOB ID" in out and "recover" not in out


def test_submit_refuses_unarmed(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="DQ_QPU"):
        EP.cmd_submit(SimpleNamespace(snapshot=EM.DEFAULT_SNAPSHOT, timeout=None))
