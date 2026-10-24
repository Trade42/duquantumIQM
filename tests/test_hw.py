"""The guarded hardware path, against fake backends. Nothing here reaches the network."""

import json
import re
import subprocess
import sys
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.result import Result

from dqiqm import hw

LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "11111111-2222-4333-8444-555555555555"
MEMORY = ["000", "111", "001", "111", "000"]  # qiskit order: clbit 0 is the rightmost character


def ghz3() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc


def write_ledger(path, rows=()):
    path.write_text(LEDGER_HEADER + "".join(r + "\n" for r in rows))
    return path


def row(task, circuits, shots, job="aaaaaaaa-0000-4000-8000-000000000000"):
    return f"| 2026-10-24T12:00:00Z | {task} | emerald | {CALSET} | QB1,QB2 | {circuits} x {shots} | {job} | pilot |"


def arm(monkeypatch, ledger, task="hwtask", jobs="3", shots="10000"):
    monkeypatch.setenv("DQ_QPU", "run")
    monkeypatch.setenv("DQ_QPU_TASK", task)
    monkeypatch.setenv("DQ_QPU_MAX_JOBS", jobs)
    monkeypatch.setenv("DQ_QPU_MAX_SHOTS", shots)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))


@pytest.fixture
def no_connect(monkeypatch):
    def connect(*args, **kwargs):
        raise AssertionError("the guard let a connection through")

    monkeypatch.setattr(hw, "_connect", connect)


class FakeJob:
    def __init__(self, ledger, shots):
        self.ledger = ledger
        self.shots = shots

    def job_id(self):
        return JOB_ID

    def result(self, **kwargs):
        assert JOB_ID in self.ledger.read_text(), "the ledger line must be written before waiting for results"
        return Result.from_dict(
            {
                "backend_name": "emerald",
                "backend_version": "",
                "qobj_id": "",
                "job_id": JOB_ID,
                "success": True,
                "date": "2026-10-24",
                "results": [
                    {
                        "shots": self.shots,
                        "success": True,
                        "data": {"memory": MEMORY, "counts": {"000": 2, "111": 2, "001": 1}, "metadata": {}},
                        "header": {"name": "ghz3"},
                        "calibration_set_id": UUID(CALSET),
                    }
                ],
                "timeline": [SimpleNamespace(status="created", timestamp="2026-10-24T12:00:00Z")],
            }
        )


class FakeBackend:
    architecture = SimpleNamespace(calibration_set_id=UUID(CALSET))

    def __init__(self, ledger):
        self.ledger = ledger
        self.runs = []

    def index_to_qubit_name(self, index):
        return f"QB{index + 1}"

    def run(self, circuits, shots, qubit_index_to_name=None, **options):
        self.runs.append((circuits, shots, qubit_index_to_name))
        return FakeJob(self.ledger, shots)


def test_import_makes_no_network_call_and_loads_no_iqm_module():
    code = (
        "import socket, sys\n"
        "def boom(*a, **k): raise SystemExit('network call at import')\n"
        "socket.socket.connect = boom; socket.create_connection = boom; socket.getaddrinfo = boom\n"
        "import dqiqm.hw\n"
        "assert not [m for m in sys.modules if m == 'iqm' or m.startswith('iqm.')], 'iqm imported'\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"DQ_QPU": "run"},
        {"DQ_QPU": "yes", "DQ_QPU_TASK": "hwtask"},
        {"DQ_QPU": "run", "DQ_QPU_TASK": "../escape"},
    ],
)
def test_unarmed_raises_before_any_connection(monkeypatch, tmp_path, no_connect, env):
    write_ledger(tmp_path / "ledger.md")
    monkeypatch.setenv("DQ_QPU_MAX_JOBS", "10")
    monkeypatch.setenv("DQ_QPU_MAX_SHOTS", "100000")
    monkeypatch.setenv("DQ_QPU_LEDGER", str(tmp_path / "ledger.md"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(hw.QPUGuardError, match="not armed"):
        hw.submit(ghz3(), device="emerald", shots=100, purpose="test")


@pytest.mark.parametrize(
    ("rows", "jobs", "shots", "n_circuits", "n_shots", "match"),
    [
        ([], None, None, 1, 100, "DQ_QPU_MAX_JOBS=0"),  # no allowance set at all
        ([row("hwtask", 1, 1000)], "1", "100000", 1, 100, "1 job"),  # job count used up
        ([row("hwtask", 2, 500)], "5", "1500", 1, 600, "DQ_QPU_MAX_SHOTS=1500"),  # 1000 used + 600 > 1500
        ([], "5", "1000", 3, 400, "3 x 400"),  # a batch counts circuits x shots
    ],
)
def test_over_the_allowance_raises(monkeypatch, tmp_path, no_connect, rows, jobs, shots, n_circuits, n_shots, match):
    ledger = write_ledger(tmp_path / "ledger.md", rows + [row("othertask", 50, 100000)])
    arm(monkeypatch, ledger)
    for name, value in (("DQ_QPU_MAX_JOBS", jobs), ("DQ_QPU_MAX_SHOTS", shots)):
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(hw.QPUGuardError, match=re.escape(match)):
        hw.submit([ghz3()] * n_circuits, device="emerald", shots=n_shots, purpose="test")


def test_other_tasks_do_not_count(tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md", [row("othertask", 50, 100000), row("hwtask", 1, 1000)])
    env = {"DQ_QPU_LEDGER": str(ledger), "DQ_QPU_MAX_JOBS": "2", "DQ_QPU_MAX_SHOTS": "2000"}
    assert hw.usage("hwtask", ledger) == (1, 1000)
    hw.check_allowance("hwtask", 1, 1000, env)  # exactly at the limit is allowed


def test_missing_or_unparsable_ledger_refuses(monkeypatch, tmp_path, no_connect):
    arm(monkeypatch, tmp_path / "absent.md")
    with pytest.raises(hw.QPUGuardError, match="missing"):
        hw.submit(ghz3(), device="emerald", shots=100, purpose="test")
    ledger = write_ledger(tmp_path / "ledger.md", ["| 2026-10-24 | hwtask | emerald | ? | ? | lots | ? | ? |"])
    arm(monkeypatch, ledger)
    with pytest.raises(hw.QPUGuardError, match="cannot parse"):
        hw.submit(ghz3(), device="emerald", shots=100, purpose="test")


def test_ledger_line_has_the_header_format(tmp_path):
    line = hw.ledger_line(
        task="hwtask",
        device="emerald",
        calibration_set_id=CALSET,
        qubits=["QB1", "QB2", "QB10"],
        circuits=4,
        shots=1000,
        job_id=JOB_ID,
        purpose="pilot | d=3\nr=2",
    )
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    assert len(cells) == len(hw.LEDGER_COLUMNS) == 8
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", cells[0])
    assert cells[1:7] == ["hwtask", "emerald", CALSET, "QB1,QB2,QB10", "4 x 1000", JOB_ID]
    assert cells[7] == "pilot / d=3 r=2"  # no pipe or newline can break the table
    ledger = write_ledger(tmp_path / "ledger.md", [line])
    (parsed,) = hw.read_ledger(ledger)
    assert (parsed.task, parsed.circuits, parsed.shots, parsed.job_id) == ("hwtask", 4, 1000, JOB_ID)


def test_submit_writes_ledger_first_then_meta_json_and_memory(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger, jobs="1", shots="5")
    fake = FakeBackend(ledger)
    calls = []

    def connect(device, calibration_set_id=None, *, read_only):
        calls.append((device, calibration_set_id, read_only))
        return fake

    monkeypatch.setattr(hw, "_connect", connect)
    data_root = tmp_path / "data" / "hw"
    sub = hw.submit(ghz3(), device="emerald", shots=5, purpose="unit test", calibration_set_id=CALSET, data_root=data_root)

    assert calls == [("emerald", CALSET, False)]
    assert sub.job_id == JOB_ID and sub.calibration_set_id == CALSET
    (entry,) = hw.read_ledger(ledger)
    assert (entry.task, entry.device, entry.calibration_set_id) == ("hwtask", "emerald", CALSET)
    assert (entry.qubits, entry.circuits, entry.shots, entry.job_id) == ("QB1,QB2,QB3", 1, 5, JOB_ID)

    assert sub.path.parent.parent == data_root
    assert re.fullmatch(r"\d{8}-hwtask", sub.path.parent.name) and sub.path.name == JOB_ID
    meta = json.loads((sub.path / "meta.json").read_text())
    assert meta["job_id"] == JOB_ID and meta["device"] == "emerald" and meta["calibration_set_id"] == CALSET
    assert meta["qubit_mapping"] == {"0": "QB1", "1": "QB2", "2": "QB3"}
    assert (meta["shots"], meta["circuits"], meta["success"]) == (5, 1, True)
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", meta["commit"])
    assert meta["timeline"] == ["namespace(status='created', timestamp='2026-10-24T12:00:00Z')"]
    bits = np.load(sub.path / "memory.npz")["c0"]
    assert bits.shape == (5, 3) and bits.dtype == np.uint8
    assert bits[2].tolist() == [1, 0, 0]  # "001": clbit 0 set

    # the job is in the ledger now, so the allowance of one job is spent
    with pytest.raises(hw.QPUGuardError, match="1 job"):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="unit test", data_root=data_root)


def test_retrieve_is_read_only_and_needs_no_arming(monkeypatch, tmp_path):
    calls = []

    class Backend:
        def retrieve_job(self, job_id):
            return FakeJob(write_ledger(tmp_path / "l.md", [row("t", 1, 5, job=job_id)]), 5)

    def connect(device, calibration_set_id=None, *, read_only):
        calls.append(read_only)
        return Backend()

    monkeypatch.setattr(hw, "_connect", connect)
    result = hw.retrieve(JOB_ID, "emerald")
    assert calls == [True] and result.job_id == JOB_ID
    out = hw.save_raw(result, task="review", device="emerald", qubit_mapping={0: "QB1"}, shots=5, data_root=tmp_path)
    assert (out / "meta.json").is_file() and (out / "memory.npz").is_file()


def test_read_only_backend_cannot_submit(monkeypatch):
    import iqm.iqm_client
    from iqm.qiskit_iqm import IQMFakeAdonis

    class FakeClient:
        quantum_computer_name = "fake-adonis"

        def __init__(self, url, quantum_computer=None, **kwargs):
            assert url == hw.SERVER_URL

        def get_dynamic_quantum_architecture(self, calibration_set_id=None):
            return IQMFakeAdonis().architecture

        def submit_run_request(self, *args, **kwargs):
            raise AssertionError("submitted")

        submit_circuits = submit_run_request

    monkeypatch.setattr(iqm.iqm_client, "IQMClient", FakeClient)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    backend = hw.backend("adonis")
    assert backend.num_qubits == 5
    with pytest.raises(hw.QPUGuardError, match="read-only"):
        backend.run(ghz3(), shots=10)
    with pytest.raises(hw.QPUGuardError, match="read-only"):
        backend.client.submit_run_request(None)
    with pytest.raises(hw.QPUGuardError, match="read-only"):
        backend.client.submit_circuits([])
