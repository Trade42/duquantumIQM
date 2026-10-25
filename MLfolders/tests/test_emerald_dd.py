"""scripts/emerald_dd.py offline: the segments against the brief, the DD options through IQM's own create_run_request and
hw.submit, the run-request, meta.json and server checks, the guards and the order, the sanity and pilot checks, and the
A/B statistics. Nothing here reaches the network or builds the suite."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import iqm.iqm_client
import numpy as np
import pytest
from iqm.iqm_client import IQMClient
from iqm.iqm_client.models import STANDARD_DD_STRATEGY, DDMode, HeraldingMode
from iqm.qiskit_iqm import IQMFakeAdonis
from qiskit import QuantumCircuit, transpile
from qiskit.result import Result

from dqiqm import hw
from dqiqm import stats

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("emerald_dd", REPO / "scripts" / "emerald_dd.py")
EDD = importlib.util.module_from_spec(_spec)
sys.modules["emerald_dd"] = EDD  # its dataclasses look their module up there
_spec.loader.exec_module(EDD)

PILOT, MAIN = EDD.SEGMENTS["emerald-dd-pilot"], EDD.SEGMENTS["emerald-dd"]
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "11111111-2222-4333-8444-555555555555"
LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""
EA = ["QB13", "QB14", "QB19", "QB20", "QB21", "QB22", "QB27", "QB28", "QB29", "QB30", "QB31", "QB36", "QB37", "QB38",
      "QB39", "QB44", "QB45"]


def ledger_row(seg, job, *, job_id=JOB_ID, time="2026-10-25T02:00:00Z", purpose=None, qubits=EA):
    p = EDD.tagged_purpose(seg, job) if purpose is None else purpose
    return (f"| {time} | {seg.task} | emerald | {CALSET} | {','.join(qubits)} | {len(job.names)} x {job.shots} | {job_id} "
            f"| {p} |\n")


# ---------------------------------------------------------------------------
# the segments: the brief's allowance
# ---------------------------------------------------------------------------


def test_the_segments_are_the_briefs():
    assert EDD.ORDER == ("emerald-dd-pilot", "emerald-dd")
    assert EDD.check_segments(EDD.load_placement()) == []
    assert (PILOT.n_jobs, PILOT.total_shots, PILOT.max_jobs, PILOT.max_shots) == (1, 1000, 1, 1000)
    assert (MAIN.n_jobs, MAIN.total_shots, MAIN.max_jobs, MAIN.max_shots) == (6, 80_000, 6, 96_000)
    (ddp,) = PILOT.jobs
    assert (ddp.label, ddp.dd, ddp.names, ddp.shots) == ("DDP", True, ("E1_d3_X_r03_eA",), 1000)
    assert [(j.label, j.dd) for j in MAIN.jobs] == [("A1", True), ("B1", False), ("B2", False), ("A2", True),
                                                    ("E4A", True), ("E4B", False)]
    assert PILOT.arming() == {"DQ_QPU": "run", "DQ_QPU_TASK": "emerald-dd-pilot", "DQ_QPU_MAX_JOBS": "1",
                              "DQ_QPU_MAX_SHOTS": "1000"}
    assert MAIN.arming() == {"DQ_QPU": "run", "DQ_QPU_TASK": "emerald-dd", "DQ_QPU_MAX_JOBS": "6", "DQ_QPU_MAX_SHOTS": "80000"}


def test_the_circuits_are_frozen_ones_on_ea():
    placement = EDD.load_placement()
    assert sorted(EDD.E1_CIRCUITS) == sorted(f"E1_d3_{b}_r{r:02d}_eA" for b in "ZX" for r in (3, 8, 15))
    # every frozen E4 circuit at 5 us, and nothing else: there are 8, so E4's jobs are 8 x 2,000
    assert sorted(EDD.E4_CIRCUITS) == sorted(k for k in placement if k.startswith("E4_") and k.endswith("_t05000ns"))
    for name in (*EDD.E1_CIRCUITS, *EDD.E4_CIRCUITS):
        assert sorted(set(placement[name]["placement"].values()), key=lambda q: int(q[2:])) == EA
        assert placement[name]["device"] == "emerald"


def test_check_segments_flags_a_broken_abba(monkeypatch):
    jobs = list(MAIN.jobs)
    jobs[1], jobs[3] = jobs[3], jobs[1]  # on, on, off, off
    broken = EDD.Segment(MAIN.task, tuple(EDD.DDJob(j.label, j.dd, j.experiment, j.names, j.shots, i)
                                          for i, j in enumerate(jobs)), 6, 96_000)
    monkeypatch.setitem(EDD.SEGMENTS, "emerald-dd", broken)
    assert any("ABBA" in p for p in EDD.check_segments())
    over = EDD.Segment(MAIN.task, tuple(EDD.DDJob(j.label, j.dd, j.experiment, j.names, 5000, j.position)
                                        for j in MAIN.jobs), 6, 96_000)
    monkeypatch.setitem(EDD.SEGMENTS, "emerald-dd", over)
    assert any("> the brief's" in p for p in EDD.check_segments())


# ---------------------------------------------------------------------------
# the options, the purpose and the ledger tag
# ---------------------------------------------------------------------------


def test_dd_on_names_the_standard_strategy_and_off_is_the_defaults():
    on = EDD.compilation_options(True)
    assert on.dd_mode == DDMode.ENABLED and on.dd_strategy == STANDARD_DD_STRATEGY
    assert on.heralding_mode == HeraldingMode.NONE and on.active_reset_cycles is None
    assert EDD.compilation_options(False) is None
    assert EDD.options_record(False) == hw.compilation_options_record(None)
    rec = EDD.options_record(True)
    assert rec["dd_mode"] == "enabled"
    assert rec["dd_strategy"]["gate_sequences"] == [[9, "XYXYYXYX", "asap"], [5, "YXYX", "asap"], [2, "XX", "center"]]
    assert {k: v for k, v in rec.items() if k not in ("dd_mode", "dd_strategy")} == {
        k: v for k, v in EDD.options_record(False).items() if k not in ("dd_mode", "dd_strategy")}


def test_the_purpose_carries_the_label_first_and_the_tag_last():
    a1, b1 = MAIN.job("A1"), MAIN.job("B1")
    assert EDD.tagged_purpose(MAIN, a1) == "emerald-dd A1 E1 DD on (6 circuits x 2000 shots) [dd=ENABLED,dd_strategy=879fe63c]"
    assert EDD.tagged_purpose(MAIN, b1) == EDD.purpose(MAIN, b1) == "emerald-dd B1 E1 DD off (6 circuits x 2000 shots)"
    for job in (*PILOT.jobs, *MAIN.jobs):
        seg = PILOT if job.label == "DDP" else MAIN
        assert job.label == EDD.label_of(SimpleNamespace(task=seg.task, purpose=EDD.tagged_purpose(seg, job)))


def test_the_tagged_rows_parse_in_the_ledger(tmp_path):
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER + ledger_row(PILOT, PILOT.jobs[0])
                      + "".join(ledger_row(MAIN, j, job_id=f"2222222{i}-2222-4333-8444-555555555555")
                                for i, j in enumerate(MAIN.jobs)))
    rows = hw.read_ledger(ledger)
    assert len(rows) == 7 and hw.usage("emerald-dd", ledger) == (6, 80_000) and hw.usage("emerald-dd-pilot", ledger) == (1, 1000)
    assert sorted(EDD.job_rows(ledger, MAIN)) == sorted(j.label for j in MAIN.jobs)


# ---------------------------------------------------------------------------
# the options through IQM's own create_run_request and hw.submit (offline)
# ---------------------------------------------------------------------------


def native_pair() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.r(math.pi / 2, math.pi / 2, 0)
    qc.cz(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture
def adonis():
    return IQMFakeAdonis()


def offline_backend(monkeypatch, device, ledger):
    """The real ``hw._connect`` and ``IQMBackend.create_run_request`` on a fake device; only the network is stood in for."""

    class Client:
        quantum_computer_name = device.name
        create_run_request = IQMClient.create_run_request

        def __init__(self, url, quantum_computer=None, **kwargs):
            # the server layer hw makes read-only; nothing here sends a request
            self._iqm_server_client = SimpleNamespace(_send_request=lambda *a, **k: None)

        def get_dynamic_quantum_architecture(self, calibration_set_id=None):
            return device.architecture

    monkeypatch.setattr(iqm.iqm_client, "IQMClient", Client)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    real = hw._connect
    built = []

    def connect(name, calibration_set_id=None, *, read_only):
        b = real(name, calibration_set_id, read_only=read_only)
        create = b.create_run_request

        def create_run_request(*a, **k):
            built.append(create(*a, **k))
            return built[-1]

        b.create_run_request = create_run_request
        b.submit_run_request = lambda rr: FakeJob(ledger, rr)
        return b

    monkeypatch.setattr(hw, "_connect", connect)
    return built


class FakeJob:
    def __init__(self, ledger, rr):
        self.ledger, self.rr = ledger, rr

    def job_id(self):
        return JOB_ID

    def result(self, **kwargs):
        assert JOB_ID in self.ledger.read_text(), "the ledger row comes before the result"
        n = self.rr.shots
        return Result.from_dict({
            "backend_name": "emerald", "backend_version": "", "qobj_id": "", "job_id": JOB_ID, "success": True,
            "date": "2026-10-25",
            "results": [{"shots": n, "success": True, "data": {"memory": ["01", "10"] * (n // 2), "metadata": {}},
                         "header": {"name": "c"}, "calibration_set_id": UUID(CALSET)} for _ in self.rr.circuits],
            "timeline": [SimpleNamespace(status="execution_started", timestamp="2026-10-25T02:00:01Z"),
                         SimpleNamespace(status="execution_ended", timestamp="2026-10-25T02:00:02Z")],
        })


@pytest.mark.parametrize("dd", [True, False])
def test_the_dd_options_reach_the_request_meta_json_and_the_ledger(monkeypatch, tmp_path, adonis, dd):
    seg = MAIN
    job = MAIN.job("A1" if dd else "B1")
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_HEADER)
    for k, v in {"DQ_QPU": "run", "DQ_QPU_TASK": seg.task, "DQ_QPU_MAX_JOBS": "1", "DQ_QPU_MAX_SHOTS": "4"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))
    built = offline_backend(monkeypatch, adonis, ledger)
    qc = transpile(native_pair(), adonis, initial_layout=[0, 2], optimization_level=0, seed_transpiler=2026)
    sub = hw.submit([qc], device="emerald", shots=4, purpose=EDD.purpose(seg, job),
                    circuit_compilation_options=EDD.compilation_options(dd), data_root=tmp_path / "raw")
    (rr,) = built
    assert rr.dd_mode == (DDMode.ENABLED if dd else DDMode.DISABLED)
    assert rr.dd_strategy == (STANDARD_DD_STRATEGY if dd else None)
    meta = json.loads((sub.path / "meta.json").read_text())
    assert EDD._plain(meta["compilation_options"]) == EDD._plain(EDD.options_record(dd))
    (row,) = hw.read_ledger(ledger)
    assert row.purpose == meta["purpose"] == EDD.tagged_purpose(seg, job)
    assert EDD.label_of(row) == job.label


def fake_frozen(qc_names, hashes, qubits):
    """The surface of ``Frozen`` the checks read: placement hashes and qubits per circuit name."""
    placement = {n: {"native_sha256": h} for n, h in zip(qc_names, hashes)}
    return SimpleNamespace(placement=placement, planned_qubits=lambda n: list(qubits),
                           hashes=lambda job: [placement[n]["native_sha256"] for n in job.names])


def _request(monkeypatch, adonis, tmp_path, dd):
    built = offline_backend(monkeypatch, adonis, tmp_path / "unused.md")
    b = hw._connect("emerald", None, read_only=True)
    qc = transpile(native_pair(), adonis, initial_layout=[0, 2], optimization_level=0, seed_transpiler=2026)
    b.create_run_request([qc], shots=2000, qubit_index_to_name={0: "QB1", 2: "QB3"},
                         circuit_compilation_options=EDD.compilation_options(dd))
    (rr,) = built
    return rr


def test_run_request_checks_accept_each_jobs_own_options_only(monkeypatch, tmp_path, adonis):
    on_rr = _request(monkeypatch, adonis, tmp_path, True)
    off_rr = _request(monkeypatch, adonis, tmp_path, False)
    sha = hw.instructions_sha256(on_rr.circuits[0].instructions)
    assert sha == hw.instructions_sha256(off_rr.circuits[0].instructions), "DD changes no instruction"
    a1 = EDD.DDJob("A1", True, "E1", ("c0",), 2000, 0)
    b1 = EDD.DDJob("B1", False, "E1", ("c0",), 2000, 1)
    fz = fake_frozen(["c0"], [sha], ["QB1", "QB3"])
    calset = str(on_rr.calibration_set_id)
    assert EDD.run_request_checks(on_rr, fz, a1, calset)["options"]["dd_mode"] == "enabled"
    assert EDD.run_request_checks(off_rr, fz, b1, calset)["options"]["dd_mode"] == "disabled"
    with pytest.raises(AssertionError, match="options"):
        EDD.run_request_checks(off_rr, fz, a1, calset)
    with pytest.raises(AssertionError, match="options"):
        EDD.run_request_checks(on_rr, fz, b1, calset)
    with pytest.raises(AssertionError, match="hash"):
        EDD.run_request_checks(on_rr, fake_frozen(["c0"], ["0" * 64], ["QB1", "QB3"]), a1, calset)
    with pytest.raises(AssertionError, match="sent to"):
        EDD.run_request_checks(on_rr, fake_frozen(["c0"], [sha], ["QB1", "QB2"]), a1, calset)


# ---------------------------------------------------------------------------
# the server's record and meta.json
# ---------------------------------------------------------------------------


def server_rec(dd, hashes, shots=2000, calset=CALSET):
    return {"status": "completed", "compilation_options": EDD._plain(EDD.options_record(dd)), "shots": shots,
            "calibration_set_id": calset, "native_sha256": list(hashes)}


def test_server_checks():
    job = EDD.DDJob("A1", True, "E1", ("c0", "c1"), 2000, 0)
    fz = fake_frozen(["c0", "c1"], ["a" * 64, "b" * 64], EA)
    assert all(EDD.server_checks(server_rec(True, ["a" * 64, "b" * 64]), fz, job, CALSET).values())
    bad = EDD.server_checks(server_rec(False, ["a" * 64, "b" * 64]), fz, job, CALSET)
    assert not bad["dd_mode"] and not bad["dd_strategy sequences"]
    assert not EDD.server_checks(server_rec(True, ["b" * 64, "a" * 64]), fz, job, CALSET)["circuits = native_sha256"]
    assert not EDD.server_checks(server_rec(True, ["a" * 64, "b" * 64], shots=1000), fz, job, CALSET)["shots"]
    assert EDD.server_checks({"error": "ConnectionError: x"}, fz, job, CALSET) == {"server record": False}
    # the strategy as the payload gives it back: tuples, not lists, still match
    rec = server_rec(True, ["a" * 64, "b" * 64])
    rec["compilation_options"]["dd_strategy"]["gate_sequences"] = [tuple(s) for s in rec["compilation_options"]["dd_strategy"]["gate_sequences"]]
    assert EDD.server_checks(EDD._plain(rec), fz, job, CALSET)["dd_strategy sequences"]


def test_server_record_reads_retrieve_native(monkeypatch):
    params = SimpleNamespace()
    from iqm.iqm_client.models import CircuitJobParameters

    params = CircuitJobParameters(shots=1000, calibration_set_id=UUID(CALSET), dd_mode=DDMode.ENABLED,
                                  dd_strategy=STANDARD_DD_STRATEGY.model_dump())
    circ = SimpleNamespace(instructions=[SimpleNamespace(name="prx", locus=("QB1",), args={"angle": 0.5, "phase": 0.25})])
    calls = []
    monkeypatch.setattr(hw, "retrieve_native", lambda job_id, device: calls.append((job_id, device)) or SimpleNamespace(
        data=SimpleNamespace(status=SimpleNamespace(value="completed")), parameters=params, circuits=[circ]))
    rec = EDD.server_record(JOB_ID)
    assert calls == [(JOB_ID, "emerald")]
    assert rec["compilation_options"]["dd_mode"] == "enabled" and rec["shots"] == 1000 and rec["status"] == "completed"
    assert rec["compilation_options"]["dd_strategy"]["gate_sequences"] == EDD.options_record(True)["dd_strategy"]["gate_sequences"]
    assert rec["native_sha256"] == [hw.instructions_sha256(circ.instructions)]


def test_server_record_retries_then_reports(monkeypatch):
    n = []

    def fail(job_id, device):
        n.append(1)
        raise ConnectionError("no route")

    monkeypatch.setattr(hw, "retrieve_native", fail)
    rec = EDD.server_record(JOB_ID, attempts=2, wait_s=0)
    assert len(n) == 2 and rec == {"error": "ConnectionError: no route"}


def test_meta_checks_read_the_options_and_the_order():
    job = EDD.DDJob("A1", True, "E1", ("c0",), 2000, 0)
    fz = SimpleNamespace(mappings=lambda j: [{0: "QB1", 1: "QB3"}], planned_qubits=lambda n: ["QB1", "QB3"],
                         measured=lambda j: [{"0": "QB1", "1": "QB3"}], hashes=lambda j: ["a" * 64])
    meta = {"qubit_mapping": [{"0": "QB1", "1": "QB3"}], "physical_qubits": [["QB1", "QB3"]],
            "measured_qubits": [{"0": "QB1", "1": "QB3"}], "circuit_order": [{"position": 0, "name": "c0", "native_sha256": "a" * 64}],
            "compilation_options": EDD.options_record(True), "calibration_set_id": CALSET, "shots": 2000, "circuits": 1,
            "success": True, "timeline": [{"status": "created"}]}
    assert all(EDD.meta_checks(meta, fz, job, CALSET).values())
    off = dict(meta, compilation_options=EDD.options_record(False))
    assert not EDD.meta_checks(off, fz, job, CALSET)["compilation_options"]
    swapped = dict(meta, measured_qubits=[{"0": "QB3", "1": "QB1"}])
    assert not EDD.meta_checks(swapped, fz, job, CALSET)["measured_qubits"]


# ---------------------------------------------------------------------------
# the guards: arming, order, no resubmission
# ---------------------------------------------------------------------------


def test_arming_exactly_as_granted():
    for seg in (PILOT, MAIN):
        env = dict(seg.arming())
        EDD.check_arming(seg, env)
        for k in env:
            with pytest.raises(SystemExit, match=k):
                EDD.check_arming(seg, {**env, k: "7"})
        with pytest.raises(SystemExit, match="ledger"):
            EDD.check_arming(seg, {**env, "DQ_QPU_LEDGER": "/tmp/other.md"})
    with pytest.raises(SystemExit, match="DQ_QPU_MAX_SHOTS"):
        EDD.check_arming(MAIN, {**MAIN.arming(), "DQ_QPU_MAX_SHOTS": "96000"})  # the exact sum, not the cap


def _manifest(raw_root, seg, job, ok, day="20261025"):
    p = EDD.manifest_path(raw_root, seg, job, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"job_id": JOB_ID, "ok": ok}))


def test_segment_two_runs_only_after_a_passing_pilot(tmp_path):
    ledger, raw = tmp_path / "ledger.md", tmp_path / "raw"
    ledger.write_text(LEDGER_HEADER)
    EDD.check_order(PILOT, ledger, raw)
    with pytest.raises(SystemExit, match="0 of 1 jobs"):
        EDD.check_order(MAIN, ledger, raw)
    ledger.write_text(LEDGER_HEADER + ledger_row(PILOT, PILOT.jobs[0]))
    with pytest.raises(SystemExit, match="did not pass"):
        EDD.check_order(MAIN, ledger, raw)  # in the ledger, never checked
    _manifest(raw, PILOT, PILOT.jobs[0], False)
    with pytest.raises(SystemExit, match="did not pass"):
        EDD.check_order(MAIN, ledger, raw)
    _manifest(raw, PILOT, PILOT.jobs[0], True)
    EDD.check_order(MAIN, ledger, raw)
    # and once segment 2 has started, the pilot segment refuses
    ledger.write_text(ledger.read_text() + ledger_row(MAIN, MAIN.jobs[0], job_id="33333333-2222-4333-8444-555555555555"))
    with pytest.raises(SystemExit, match="comes after"):
        EDD.check_order(PILOT, ledger, raw)


def test_resume_never_resubmits(tmp_path):
    ledger, raw = tmp_path / "ledger.md", tmp_path / "raw"
    job = MAIN.job("A1")
    ledger.write_text(LEDGER_HEADER)
    assert EDD.resume_state(None, MAIN, job, ledger=ledger, raw_root=raw, refs={}, cal=None) == "new"
    ledger.write_text(LEDGER_HEADER + ledger_row(MAIN, job))
    assert EDD.resume_state(None, MAIN, job, ledger=ledger, raw_root=raw, refs={}, cal=None) == "raw missing"
    (raw / "20261025-emerald-dd" / JOB_ID).mkdir(parents=True)
    (raw / "20261025-emerald-dd" / JOB_ID / "meta.json").write_text("{}")
    _manifest(raw, MAIN, job, True)
    assert EDD.resume_state(None, MAIN, job, ledger=ledger, raw_root=raw, refs={}, cal=None) == "done"
    _manifest(raw, MAIN, job, False)
    assert EDD.resume_state(None, MAIN, job, ledger=ledger, raw_root=raw, refs={}, cal=None) == "failed"
    # a fail-closed row is a stop, never a retry
    ledger.write_text(LEDGER_HEADER + ledger_row(MAIN, job, job_id="UNKNOWN-20261025T020000Z"))
    assert EDD.resume_state(None, MAIN, job, ledger=ledger, raw_root=raw, refs={}, cal=None) == "failed"
    with pytest.raises(SystemExit, match="no resubmission"):
        EDD.submit_job(None, MAIN, job, calset=CALSET, ledger=ledger, raw_root=raw, refs={}, cal=None, timeout=1)


def test_submit_refuses_unarmed_before_anything():
    with pytest.raises(SystemExit, match="DQ_QPU"):
        EDD.cmd_submit(SimpleNamespace(segment="emerald-dd-pilot", snapshot="x.json", timeout=None))


def test_recover_reads_only_and_records_what_was_sent(monkeypatch, tmp_path):
    ledger, raw = tmp_path / "ledger.md", tmp_path / "raw"
    job = PILOT.jobs[0]
    ledger.write_text(LEDGER_HEADER + ledger_row(PILOT, job))
    monkeypatch.setattr(hw, "_connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recover must not connect")))
    pulled = []
    rr = SimpleNamespace(shots=1000, circuits=[0])
    monkeypatch.setattr(hw, "retrieve", lambda job_id, device, timeout=None: pulled.append(job_id) or FakeJob(ledger, rr).result())
    fz = SimpleNamespace(hashes=lambda j: ["a" * 64], mappings=lambda j: [{0: "QB1"}], planned_qubits=lambda n: ["QB1", "QB3"])
    j, path = EDD.recover_job(fz, PILOT, JOB_ID, ledger=ledger, raw_root=raw, timeout=1)
    assert pulled == [JOB_ID] and j is job and path == raw / "20261025-emerald-dd-pilot" / JOB_ID
    meta = json.loads((path / "meta.json").read_text())
    assert meta["compilation_options"] == EDD.options_record(True) and meta["purpose"] == EDD.tagged_purpose(PILOT, job)
    assert meta["circuit_order"] == [{"position": 0, "name": "E1_d3_X_r03_eA", "native_sha256": "a" * 64}]
    with pytest.raises(SystemExit, match="never overwritten"):
        EDD.recover_job(fz, PILOT, JOB_ID, ledger=ledger, raw_root=raw, timeout=1)
    ledger.write_text(LEDGER_HEADER + ledger_row(PILOT, job, purpose="emerald-dd-pilot DDP E1 DD on (1 circuits x 1000 shots)"))
    with pytest.raises(SystemExit, match="purpose"):
        EDD.recover_job(fz, PILOT, JOB_ID, ledger=ledger, raw_root=tmp_path / "raw2", timeout=1)


# ---------------------------------------------------------------------------
# sanity and the pilot's checks, on synthetic records
# ---------------------------------------------------------------------------


REF = {"measurements": 33, "detectors": 24, "random_cols": [1, 3, 4, 6], "deterministic_cols": [0, 2, 5, 7],
       "deterministic_values": [0, 0, 0, 0], "x_type_first_cols": [0, 2, 5, 7], "z_type_first_cols": [1, 3, 4, 6],
       "noiseless_all_zero": 0.0, "noiseless_always_all_zero": False,
       "model": {"detection_fraction": 0.1057, "all_zero": 0.0}}


def x_basis_record(seed=2026, shots=1000, det=0.13):
    rng = np.random.default_rng(seed)
    raw = (rng.random((shots, 33)) < 0.2).astype(np.uint8)
    raw[:, REF["random_cols"]] = rng.random((shots, 4)) < 0.5
    raw[:, REF["deterministic_cols"]] = rng.random((shots, 4)) < 0.05
    raw[:, -1] = 1
    dets = (rng.random((shots, 24)) < det).astype(np.uint8)
    obs = (rng.random((shots, 1)) < 0.25).astype(np.uint8)
    return raw, dets, obs


def verdicts(checks):
    return {c["check"]: c["pass"] for c in checks}


def test_sanity_passes_an_x_basis_record_and_catches_a_swap():
    raw, dets, _ = x_basis_record()
    assert all(verdicts(EDD.sanity(raw, dets, REF, shots=1000, physical=EA, planned=EA)).values())
    swapped = raw.copy()
    swapped[:, REF["random_cols"]], swapped[:, REF["deterministic_cols"]] = raw[:, REF["deterministic_cols"]], raw[:, REF["random_cols"]]
    v = verdicts(EDD.sanity(swapped, dets, REF, shots=1000))
    assert not v["first readouts, random"] and not v["first readouts, deterministic (misread)"]
    assert not verdicts(EDD.sanity(raw[:999], dets[:999], REF, shots=1000))["measurement shape"]
    assert not verdicts(EDD.sanity(np.zeros_like(raw), dets, REF, shots=1000))["all-zero rows"]


TIMELINE = [{"status": "execution_started", "timestamp": "2026-10-25T02:00:01Z"},
            {"status": "execution_ended", "timestamp": "2026-10-25T02:00:02Z"}]


@pytest.mark.parametrize("det,k,ok", [(0.13, 100, True), (0.03, 100, False), (0.25, 100, False), (0.13, 250, False),
                                      (0.13, 199, True)])
def test_pilot_checks(det, k, ok):
    raw, dets, obs = x_basis_record(det=det)
    checks = EDD.pilot_checks(raw, dets, obs, REF, pm_failures=k, timeline=TIMELINE)
    assert all(c["pass"] for c in checks) == ok, checks
    p_l = next(c for c in checks if c["check"].startswith("decoded P_L"))
    assert p_l["band"].endswith("0.1990")  # twice the batch's worst DD-off P_L, 199/2000


def test_pilot_checks_need_the_execution_span():
    raw, dets, obs = x_basis_record()
    checks = EDD.pilot_checks(raw, dets, obs, REF, pm_failures=100, timeline=TIMELINE[:1])
    assert not verdicts(checks)["timeline: execution_started and execution_ended"]


# ---------------------------------------------------------------------------
# the A/B statistics
# ---------------------------------------------------------------------------


def test_newcombe_matches_the_published_example():
    # Newcombe (1998), Stat. Med. 17:873, Table II example (a): 56/70 - 48/80, hybrid score interval 0.0524 to 0.3339
    d = EDD.newcombe(56, 70, 48, 80)
    assert d["diff"] == pytest.approx(0.2)
    assert d["ci95"] == pytest.approx([0.0524, 0.3339], abs=1e-4)


def test_pair_table_pools_abba_and_reports_the_drift():
    per_job = {"A1": (100, 2000), "A2": (120, 2000), "B1": (180, 2000), "B2": (160, 2000)}
    t = EDD.pair_table(per_job, ["A1", "A2"], ["B1", "B2"])
    assert (t["on"]["k"], t["on"]["n"], t["off"]["k"], t["off"]["n"]) == (220, 4000, 340, 4000)
    assert t["off_minus_on"]["diff"] == pytest.approx(0.03)
    lo, hi = t["off_minus_on"]["ci95"]
    assert 0 < lo < 0.03 < hi and t["off_minus_on"]["z"] > 3
    assert t["drift_on"]["pair"] == "A1 - A2" and t["drift_on"]["diff"] == pytest.approx(-0.01)
    assert t["drift_off"]["diff"] == pytest.approx(0.01)
    assert t["on"]["wilson95"] == pytest.approx(list(stats.wilson(220, 4000)))
    single = EDD.pair_table({"E4A": (10, 100), "E4B": (20, 100)}, ["E4A"], ["E4B"])
    assert "drift_on" not in single and single["off_minus_on"]["diff"] == pytest.approx(0.1)


def test_eps_difference_has_the_right_sign_and_covers_the_truth():
    rng = np.random.default_rng(7)
    rounds, n = (3, 8, 15), 4000

    def arm(eps):
        return [(r, int(rng.binomial(n, stats.p_model(eps, r))), n) for r in rounds]

    d = EDD.eps_difference(arm(0.03), arm(0.02), reps=200)
    assert d["diff"] > 0 and d["ci95_bootstrap"][0] < 0.01 < d["ci95_bootstrap"][1]
    same = EDD.eps_difference(arm(0.025), arm(0.025), reps=200)
    assert same["ci95_bootstrap"][0] < 0 < same["ci95_bootstrap"][1]


def test_stabilizer_rates_pool_each_arm_per_stabilizer():
    rows = [{"stabilizer": "Z(6,2)", "type": "Z", "ancilla_qb": "QB39", "data_qbs": "QB31;QB38", "round": "0"},
            {"stabilizer": "X(2,0)", "type": "X", "ancilla_qb": "QB14", "data_qbs": "QB13;QB22", "round": "0"},
            {"stabilizer": "Z(6,2)", "type": "Z", "ancilla_qb": "QB39", "data_qbs": "QB31;QB38", "round": "1"}]
    a1 = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    a2 = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    b1 = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    b2 = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    st = EDD.stabilizer_rates({"A1": a1, "A2": a2, "B1": b1, "B2": b2}, rows, ["A1", "A2"], ["B1", "B2"])
    z = st["Z(6,2)"]
    assert (z["type"], z["ancilla"], z["data"], z["rounds"]) == ("Z", "QB39", ["QB31", "QB38"], [0, 1])
    assert (z["on"]["events"], z["on"]["trials"], z["off"]["events"], z["off"]["trials"]) == (1, 8, 6, 8)
    assert z["on"]["per_round"] == [0.25, 0.0] and z["off"]["per_round"] == [0.75, 0.75]
    assert z["off_minus_on"] == pytest.approx(5 / 8)
    assert z["per_job"] == {"A1": 0.25, "A2": 0.0, "B1": 1.0, "B2": 0.5}
    assert st["X(2,0)"]["on"]["rate"] == 0.25 and st["X(2,0)"]["off"]["rate"] == 0.25


def test_run_ids_cannot_collide_with_the_batches():
    for seg in EDD.SEGMENTS.values():
        for job in seg.jobs:
            for name in job.names:
                rid = EDD.run_id(job, name)
                assert rid.startswith("dd_") and rid.endswith(name) and f"_{job.arm}_" in rid
