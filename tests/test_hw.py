"""The guarded hardware path, against fake backends. Nothing here reaches the network."""

import json
import re
import subprocess
import sys
from functools import cache
from types import SimpleNamespace
from uuid import UUID

import iqm.iqm_client
import numpy as np
import pytest
from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm import IQMFakeAdonis, IQMFakeAphrodite
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import QuantumCircuit, transpile
from qiskit.result import Result

from dqiqm import hw

LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "11111111-2222-4333-8444-555555555555"
MEMORY = ["000", "111", "001", "111", "000"]  # qiskit order: clbit 0 is the rightmost character


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def ghz3() -> QuantumCircuit:
    return ghz(3)


FAKE_DEVICES = {"adonis": IQMFakeAdonis, "aphrodite": IQMFakeAphrodite}  # aphrodite: the 54-qubit Emerald stand-in


@cache
def fake_device(name):
    return FAKE_DEVICES[name]()


@cache
def transpiled_ghz(device, n, initial_layout=None):
    return transpile(ghz(n), fake_device(device), initial_layout=initial_layout, seed_transpiler=2026)


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
    def __init__(self, ledger, shots, circuits=1):
        self.ledger = ledger
        self.shots = shots
        self.circuits = circuits

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
                ]
                * self.circuits,
                "timeline": [SimpleNamespace(status="created", timestamp="2026-10-24T12:00:00Z")],
            }
        )


class FakeBackend:
    """The backend surface hw.submit uses. create_run_request serializes as IQMBackend's does (h, cx pass as-is)."""

    architecture = SimpleNamespace(calibration_set_id=UUID(CALSET))

    def __init__(self, ledger):
        self.ledger = ledger
        self.runs = []

    def index_to_qubit_name(self, index):
        return f"QB{index + 1}"

    def create_run_request(self, circuits, shots, qubit_index_to_name=None, **options):
        names = qubit_index_to_name or {i: self.index_to_qubit_name(i) for c in circuits for i in range(c.num_qubits)}
        serialized = [SimpleNamespace(instructions=serialize_instructions(c, names, ("h", "cx"))) for c in circuits]
        return SimpleNamespace(circuits=serialized, shots=shots)

    def submit_run_request(self, run_request):
        self.runs.append(run_request)
        return FakeJob(self.ledger, run_request.shots, len(run_request.circuits))


def offline_client(device):
    """An IQMClient for a fake device, with no network: its architecture, IQM's own
    create_run_request (which validates circuits against it), and a submit_run_request that records."""

    class Client:
        quantum_computer_name = device.name
        create_run_request = IQMClient.create_run_request

        def __init__(self, url, quantum_computer=None, **kwargs):
            assert url == hw.SERVER_URL
            self.submitted = []

        def get_dynamic_quantum_architecture(self, calibration_set_id=None):
            return device.architecture

        def submit_run_request(self, run_request, use_timeslot=False):
            self.submitted.append((run_request, use_timeslot))
            return SimpleNamespace(job_id=UUID(JOB_ID))

    return Client


def submit_offline(monkeypatch, tmp_path, device, circuits):
    """hw.submit through the real _connect and IQMBackend.create_run_request, on a fake device.

    Only the network is stood in for: the client, and the backend's submit_run_request, which
    records the request it is given and returns a FakeJob.
    """
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger, jobs="1", shots=str(5 * len(circuits)))
    monkeypatch.setattr(iqm.iqm_client, "IQMClient", offline_client(fake_device(device)))
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    real_connect = hw._connect
    built, submitted = [], []

    def connect(name, calibration_set_id=None, *, read_only):
        backend = real_connect(name, calibration_set_id, read_only=read_only)
        create = backend.create_run_request

        def create_run_request(*args, **kwargs):
            built.append(create(*args, **kwargs))
            return built[-1]

        def submit_run_request(run_request):
            submitted.append(run_request)
            return FakeJob(ledger, run_request.shots, len(run_request.circuits))

        backend.create_run_request, backend.submit_run_request = create_run_request, submit_run_request
        return backend

    monkeypatch.setattr(hw, "_connect", connect)
    sub = hw.submit(circuits, device=device, shots=5, purpose="qubit record", data_root=tmp_path / "hw")
    assert len(built) == 1 and submitted[0] is built[0], "one run request, and that one is submitted"
    (entry,) = hw.read_ledger(ledger)
    return entry, sub, json.loads((sub.path / "meta.json").read_text())


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
    assert meta["qubit_mapping"] == [{"0": "QB1", "1": "QB2", "2": "QB3"}]  # no layout: qubit i is sent to QB(i+1)
    assert meta["physical_qubits"] == [["QB1", "QB2", "QB3"]]
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
    out = hw.save_raw(result, task="review", device="emerald", qubit_mapping=[{0: "QB1"}], shots=5, data_root=tmp_path)
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


@pytest.mark.parametrize(
    ("device", "n", "sent", "layout"),
    [
        ("adonis", 3, ["QB1", "QB2", "QB3"], ["QB1", "QB3", "QB2"]),
        ("aphrodite", 5, ["QB5", "QB6", "QB11", "QB12", "QB13"], ["QB11", "QB5", "QB6", "QB12", "QB13"]),
    ],
)
def test_submit_records_the_qubits_it_sends(monkeypatch, tmp_path, device, n, sent, layout):
    """A transpiled circuit: the ledger has the qubits IQM's serializer sends, and meta.json the layout."""
    circuit = transpiled_ghz(device, n)
    backend = fake_device(device)
    names = {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}
    loci = {q for inst in serialize_instructions(circuit, names) for q in inst.locus}  # what IQMBackend.run sends
    assert sorted(loci, key=hw._qubit_sort_key) == sent

    entry, sub, meta = submit_offline(monkeypatch, tmp_path, device, [circuit])

    assert entry.qubits == ",".join(sent) and sub.qubits == sent
    assert entry.calibration_set_id == str(backend.architecture.calibration_set_id)
    by_layout = {str(v): names[p] for v, p in enumerate(circuit.layout.initial_index_layout(filter_ancillas=True))}
    assert meta["qubit_mapping"] == [by_layout] and list(by_layout.values()) == layout
    assert meta["physical_qubits"] == [sent]


def test_a_batch_records_each_circuits_own_layout(monkeypatch, tmp_path):
    seeded = transpiled_ghz("adonis", 3)  # logical 0, 1, 2 on QB1, QB3, QB2
    placed = transpiled_ghz("adonis", 3, initial_layout=(3, 2, 4))  # on QB4, QB3, QB5
    entry, sub, meta = submit_offline(monkeypatch, tmp_path, "adonis", [seeded, placed])
    assert entry.qubits == "QB1,QB2,QB3,QB4,QB5" and (entry.circuits, entry.shots) == (2, 5)
    assert meta["qubit_mapping"] == [{"0": "QB1", "1": "QB3", "2": "QB2"}, {"0": "QB4", "1": "QB3", "2": "QB5"}]
    assert meta["physical_qubits"] == [["QB1", "QB2", "QB3"], ["QB3", "QB4", "QB5"]]


def test_the_submitting_backend_sends_the_request_it_is_given(monkeypatch):
    monkeypatch.setattr(iqm.iqm_client, "IQMClient", offline_client(fake_device("adonis")))
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    backend = hw._connect("adonis", read_only=False)
    run_request = backend.create_run_request([transpiled_ghz("adonis", 3)], shots=10)
    job = backend.submit_run_request(run_request)
    ((sent, use_timeslot),) = backend.client.submitted
    assert sent is run_request and use_timeslot is False
    assert job.job_id() == JOB_ID and job.circuit_metadata == [run_request.circuits[0].metadata]


def test_submit_refuses_the_timeslot_queue(no_connect):
    with pytest.raises(ValueError, match="use_timeslot"):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="test", use_timeslot=True)


def test_calibration_fidelities_come_from_the_requested_set(monkeypatch):
    asked, backends = [], []

    class Client:
        def get_quality_metric_set(self, calibration_set_id=None):
            asked.append(calibration_set_id)
            return SimpleNamespace(describes_id=UUID(CALSET), observation_set_id=UUID(JOB_ID), observations=[])

    architecture = fake_device("adonis").architecture.model_copy(update={"calibration_set_id": UUID(CALSET)})

    def backend(device, calibration_set_id=None):
        backends.append((device, calibration_set_id))
        return SimpleNamespace(architecture=architecture, client=Client())

    monkeypatch.setattr(hw, "backend", backend)
    out = hw.calibration_fidelities("adonis", CALSET)
    assert backends == [("adonis", CALSET)] and asked == [UUID(CALSET)]
    assert (out["calibration_set_id"], out["quality_metric_set_id"]) == (CALSET, JOB_ID)
    figures = {"CZ", "CLIFFORD", "1Q", "readout", "readout_qndness", "t1", "t2"}  # iqm-qubit-selector's keys
    assert set(out) == {"calibration_set_id", "quality_metric_set_id"} | figures
