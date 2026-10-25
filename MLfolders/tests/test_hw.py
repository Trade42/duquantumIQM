"""The guarded hardware path, against fake backends. Nothing here reaches the network."""

import dataclasses
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
from functools import cache
from math import pi
from types import SimpleNamespace
from uuid import UUID

import iqm.iqm_client
import numpy as np
import pytest
import requests
from iqm.iqm_client import IQMClient
from iqm.iqm_client.models import CircuitCompilationOptions, DDMode, DDStrategy, HeraldingMode
from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
from iqm.qiskit_iqm import IQMFakeAdonis, IQMFakeAphrodite, MoveGate
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMBackend
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from iqm.station_control.client.authentication import ClientConfigurationError
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.result import Result

from dqiqm import hw
from dqiqm.experiments import circuits as C

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


def native_ghz(n: int) -> QuantumCircuit:
    """GHZ in IQM's native ops: r(pi/2, pi/2) is RY(pi/2), and CX = RY(pi/2)_t . CZ . RY(-pi/2)_t."""
    qc = QuantumCircuit(n, n)
    qc.r(pi / 2, pi / 2, 0)
    for i in range(n - 1):
        qc.r(-pi / 2, pi / 2, i + 1)
        qc.cz(i, i + 1)
        qc.r(pi / 2, pi / 2, i + 1)
    qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def ghz3() -> QuantumCircuit:
    return native_ghz(3)


@pytest.fixture(autouse=True)
def default_data_root(monkeypatch, tmp_path):
    """A save without data_root lands in tmp_path, never in the repo's data/hw."""
    monkeypatch.setattr(hw, "DATA_ROOT", tmp_path / "default-data-root")


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
    """The backend surface hw.submit uses. create_run_request serializes as IQMBackend's does, and its
    request carries no option fields (as the pilot tests' fakes)."""

    architecture = SimpleNamespace(calibration_set_id=UUID(CALSET))

    def __init__(self, ledger):
        self.ledger = ledger
        self.runs = []

    def index_to_qubit_name(self, index):
        return f"QB{index + 1}"

    def create_run_request(self, circuits, shots, qubit_index_to_name=None, **options):
        names = qubit_index_to_name or {i: self.index_to_qubit_name(i) for c in circuits for i in range(c.num_qubits)}
        serialized = [SimpleNamespace(instructions=serialize_instructions(c, names)) for c in circuits]
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


def offline_backend(monkeypatch, tmp_path, device, n_circuits):
    """The real _connect and IQMBackend.create_run_request, on a fake device, armed for one job of 5 shots.

    Only the network is stood in for: the client, and the backend's submit_run_request, which
    records the request it is given and returns a FakeJob. Returns (ledger, built, submitted).
    """
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger, jobs="1", shots=str(5 * n_circuits))
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
    return ledger, built, submitted


def submit_offline(monkeypatch, tmp_path, device, circuits, **kwargs):
    """hw.submit through :func:`offline_backend`. Returns (ledger row, Submission, meta.json, the request)."""
    ledger, built, submitted = offline_backend(monkeypatch, tmp_path, device, len(circuits))
    sub = hw.submit(circuits, device=device, shots=5, purpose="qubit record", data_root=tmp_path / "hw", **kwargs)
    assert len(built) == 1 and submitted[0] is built[0], "one run request, and that one is submitted"
    (entry,) = hw.read_ledger(ledger)
    return entry, sub, json.loads((sub.path / "meta.json").read_text()), built[0]


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
    # an old-style save (no new arguments): the new fields are null, the old ones as before
    meta = json.loads((out / "meta.json").read_text())
    assert meta["measured_qubits"] is meta["circuit_order"] is meta["compilation_options"] is None
    assert (meta["circuits"], meta["circuit_names"], meta["qubit_mapping"]) == (1, [None], [{"0": "QB1"}])


def test_read_only_backend_cannot_submit(monkeypatch):
    import iqm.iqm_client
    from iqm.qiskit_iqm import IQMFakeAdonis

    class FakeServer:
        def _send_request(self, http_method, url_path, **kwargs):
            raise AssertionError(f"{http_method.__name__.upper()} {url_path} reached the HTTP layer")

    class FakeClient:
        quantum_computer_name = "fake-adonis"

        def __init__(self, url, quantum_computer=None, **kwargs):
            assert url == hw.SERVER_URL
            self._iqm_server_client = FakeServer()

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
    with pytest.raises(hw.QPUGuardError, match="read-only"):
        backend.client._iqm_server_client.submit_circuits(None)


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

    entry, sub, meta, _ = submit_offline(monkeypatch, tmp_path, device, [circuit])

    assert entry.qubits == ",".join(sent) and sub.qubits == sent
    assert entry.calibration_set_id == str(backend.architecture.calibration_set_id)
    by_layout = {str(v): names[p] for v, p in enumerate(circuit.layout.initial_index_layout(filter_ancillas=True))}
    assert meta["qubit_mapping"] == [by_layout] and list(by_layout.values()) == layout
    assert meta["physical_qubits"] == [sent]
    # unrouted: each clbit v is measured where logical qubit v was placed
    assert meta["measured_qubits"] == meta["qubit_mapping"] == [{str(k): v for k, v in sub.measured_qubits[0].items()}]


def test_a_batch_records_each_circuits_own_layout(monkeypatch, tmp_path):
    seeded = transpiled_ghz("adonis", 3)  # logical 0, 1, 2 on QB1, QB3, QB2
    placed = transpiled_ghz("adonis", 3, initial_layout=(3, 2, 4))  # on QB4, QB3, QB5
    entry, sub, meta, _ = submit_offline(monkeypatch, tmp_path, "adonis", [seeded, placed])
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


# ---------------------------------------------------------------------------
# Compilation options: recorded in meta.json, non-defaults tag the ledger purpose.
# ---------------------------------------------------------------------------

DEFAULT_OPTIONS = {
    "max_circuit_duration_over_t2": None,
    "heralding_mode": "none",
    "move_gate_validation": "strict",
    "move_gate_frame_tracking": "full",
    "active_reset_cycles": None,
    "dd_mode": "disabled",
    "dd_strategy": None,
}


def test_a_default_job_records_the_defaults_and_keeps_its_purpose(monkeypatch, tmp_path):
    entry, sub, meta, request = submit_offline(monkeypatch, tmp_path, "adonis", [transpiled_ghz("adonis", 3)])
    assert meta["compilation_options"] == sub.compilation_options == DEFAULT_OPTIONS
    assert entry.purpose == meta["purpose"] == sub.purpose == "qubit record"
    assert (request.dd_mode, request.heralding_mode, request.dd_strategy) == (DDMode.DISABLED, HeraldingMode.NONE, None)


def test_compilation_options_are_recorded_and_tag_the_ledger(monkeypatch, tmp_path):
    strategy = DDStrategy(
        gate_sequences=[(9, "XYXYYXYX", "asap"), (2, "XX", "center")], target_qubits=frozenset({"QB3", "QB1", "QB2"})
    )
    options = CircuitCompilationOptions(
        dd_mode=DDMode.ENABLED, dd_strategy=strategy, heralding_mode=HeraldingMode.ZEROS, active_reset_cycles=2
    )
    entry, sub, meta, request = submit_offline(
        monkeypatch, tmp_path, "adonis", [transpiled_ghz("adonis", 3)], circuit_compilation_options=options
    )
    # IQM's own create_run_request put the options in the request that was sent
    assert (request.dd_mode, request.dd_strategy, request.heralding_mode) == (DDMode.ENABLED, strategy, HeraldingMode.ZEROS)
    assert request.active_reset_cycles == 2
    recorded = {
        **DEFAULT_OPTIONS,
        "heralding_mode": "zeros",
        "active_reset_cycles": 2,
        "dd_mode": "enabled",
        "dd_strategy": {
            "merge_contiguous_waits": True,
            "target_qubits": ["QB1", "QB2", "QB3"],
            "skip_leading_wait": True,
            "skip_trailing_wait": True,
            "gate_sequences": [[9, "XYXYYXYX", "asap"], [2, "XX", "center"]],
        },
    }
    assert meta["compilation_options"] == sub.compilation_options == recorded
    strategy_tag = hashlib.sha256(json.dumps(recorded["dd_strategy"], sort_keys=True).encode()).hexdigest()[:8]
    tag = f"[herald=ZEROS,reset=2,dd=ENABLED,dd_strategy={strategy_tag}]"
    assert entry.purpose == meta["purpose"] == f"qubit record {tag}"  # the row parsed: the ledger stays readable
    assert (entry.circuits, entry.shots, entry.job_id) == (1, 5, JOB_ID)


def test_the_tag_names_only_what_differs_from_the_defaults():
    record = hw.compilation_options_record
    assert hw.compilation_options_tag(record(None)) == hw.compilation_options_tag(record(CircuitCompilationOptions())) == ""
    assert record(None) == DEFAULT_OPTIONS
    assert hw.compilation_options_tag(record(CircuitCompilationOptions(dd_mode=DDMode.ENABLED))) == "[dd=ENABLED]"
    t2 = CircuitCompilationOptions(max_circuit_duration_over_t2=0.0, active_reset_cycles=0)
    assert hw.compilation_options_tag(record(t2)) == "[t2_limit=0.0,reset=0]"


def test_only_circuit_compilation_options_are_accepted(no_connect):
    with pytest.raises(ValueError, match=re.escape("unsupported run option(s) ['heralding_mode']")):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="test", heralding_mode="zeros")
    with pytest.raises(TypeError, match="CircuitCompilationOptions"):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="test", circuit_compilation_options={"dd_mode": "enabled"})


def test_a_request_whose_options_differ_from_the_record_refuses(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    fake = FakeBackend(ledger)
    create = fake.create_run_request

    def create_with_dd(*args, **kwargs):
        request = create(*args, **kwargs)
        request.dd_mode = DDMode.ENABLED  # the request would send DD on, while meta.json would say off
        return request

    fake.create_run_request = create_with_dd
    monkeypatch.setattr(hw, "_connect", lambda *a, **k: fake)
    with pytest.raises(hw.QPUGuardError, match="dd_mode is 'enabled', not 'disabled'"):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="test")
    assert fake.runs == [] and hw.read_ledger(ledger) == []


# ---------------------------------------------------------------------------
# Circuit order: position, name and native_sha256 of each circuit, in submission order.
# ---------------------------------------------------------------------------


def test_the_circuit_order_is_recorded_by_position(monkeypatch, tmp_path):
    seeded = transpiled_ghz("adonis", 3)
    placed = transpiled_ghz("adonis", 3, initial_layout=(3, 2, 4))
    backend = fake_device("adonis")
    names = {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}
    # the repo's own hash of a hardware circuit (experiments.circuits.native_sha256), not hw's
    expected = [C.native_sha256(SimpleNamespace(circuit=c, qubit_index_to_name=names)) for c in (seeded, placed)]
    assert expected[0] != expected[1]

    _, sub, meta, request = submit_offline(
        monkeypatch, tmp_path, "adonis", [seeded, placed], circuit_names=["E1_a", "E1_b"], native_sha256=expected
    )
    order = [{"position": 0, "name": "E1_a", "native_sha256": expected[0]},
             {"position": 1, "name": "E1_b", "native_sha256": expected[1]}]
    assert meta["circuit_order"] == sub.circuit_order == order
    assert [hw.instructions_sha256(c.instructions) for c in request.circuits] == expected
    assert meta["circuits"] == 2 and meta["circuit_names"] == [None, None]  # unchanged: as IQM returns them


def test_without_names_the_order_takes_the_serialized_circuit_names(monkeypatch, tmp_path):
    circuit = transpiled_ghz("adonis", 3)
    _, _, meta, request = submit_offline(monkeypatch, tmp_path, "adonis", [circuit])
    assert meta["circuit_order"] == [
        {"position": 0, "name": circuit.name, "native_sha256": hw.instructions_sha256(request.circuits[0].instructions)}
    ]


def test_a_circuit_order_that_does_not_fit_refuses(monkeypatch, tmp_path, no_connect):
    arm(monkeypatch, write_ledger(tmp_path / "ledger.md"))
    with pytest.raises(ValueError, match=re.escape("circuit_names has 1 entries for 2 circuit(s)")):
        hw.submit([ghz3(), ghz3()], device="emerald", shots=5, purpose="test", circuit_names=["only-one"])
    with pytest.raises(ValueError, match=re.escape("native_sha256 has 3 entries for 2 circuit(s)")):
        hw.submit([ghz3(), ghz3()], device="emerald", shots=5, purpose="test", native_sha256=["a", "b", "c"])
    with pytest.raises(TypeError, match="not a str"):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="test", circuit_names="E1")


def test_a_wrong_native_sha256_refuses_before_submitting(monkeypatch, tmp_path):
    ledger, built, submitted = offline_backend(monkeypatch, tmp_path, "adonis", 1)
    with pytest.raises(hw.QPUGuardError, match="native_sha256 0+ given, but the run request's is [0-9a-f]{64}"):
        hw.submit([transpiled_ghz("adonis", 3)], device="adonis", shots=5, purpose="test", native_sha256=["0" * 64])
    assert len(built) == 1 and submitted == [] and hw.read_ledger(ledger) == []


# ---------------------------------------------------------------------------
# The op allowlist.
# ---------------------------------------------------------------------------


def with_reset() -> QuantumCircuit:
    qc = native_ghz(2)
    qc.reset(0)
    return qc


def with_feedback() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.r(pi, 0, 0)
    return qc


def with_move() -> QuantumCircuit:
    qc = native_ghz(2)
    qc.append(MoveGate(), [0, 1])
    return qc


def untranspiled() -> QuantumCircuit:
    return ghz(2)  # h and cx: IQM's serializer refuses them


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (with_reset, r"circuit 1 \('circuit-\d+'\) has reset x1; only barrier, cz, delay, measure, prx may be sent"),
        (with_feedback, r"circuit 1 .* has cc_prx x1"),
        (with_move, r"circuit 1 .* has move x1"),
        (untranspiled, r"circuit 1 .* cannot be serialized: Instruction 'h'"),
    ],
)
def test_the_op_allowlist_refuses_before_any_connection(monkeypatch, tmp_path, no_connect, make, match):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    with pytest.raises(hw.QPUGuardError, match="op allowlist: " + match):
        hw.submit([ghz3(), make()], device="emerald", shots=5, purpose="test")
    assert hw.read_ledger(ledger) == []


def test_native_ops_pass_the_allowlist():
    qc = native_ghz(3)
    qc.delay(100, 0, unit="ns")
    hw.check_native_ops([qc, transpiled_ghz("adonis", 3), transpiled_ghz("aphrodite", 5)])
    ops = {i.name for i in serialize_instructions(qc, {i: f"QB{i + 1}" for i in range(3)})}
    assert ops == {"prx", "cz", "barrier", "delay", "measure"} == hw.NATIVE_OPS


def test_the_built_request_is_checked_again(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    fake = FakeBackend(ledger)
    create = fake.create_run_request

    def create_with_reset(*args, **kwargs):
        request = create(*args, **kwargs)
        request.circuits[0].instructions.append(SimpleNamespace(name="reset", locus=("QB1",), args={}))
        return request

    fake.create_run_request = create_with_reset
    monkeypatch.setattr(hw, "_connect", lambda *a, **k: fake)
    with pytest.raises(hw.QPUGuardError, match="op allowlist: circuit 0 .* would send reset x1"):
        hw.submit(ghz3(), device="emerald", shots=5, purpose="test")
    assert fake.runs == [] and hw.read_ledger(ledger) == []


# ---------------------------------------------------------------------------
# The real _IQMServerClient, with only its HTTP layer stood in.
# ---------------------------------------------------------------------------

DEVICES = ("emerald", "emerald:mock", "garnet", "garnet:mock", "sirius", "sirius:mock")  # resonance-check's list


def serve(monkeypatch, devices=DEVICES, architecture=None):
    """Replace ``_IQMServerClient._send_request``: answer the GETs hw's read paths make, record every call."""
    calls = []
    listing = {"quantum_computers": [{"id": str(UUID(int=i + 1)), "alias": a, "display_name": a} for i, a in enumerate(devices)]}

    def send(self, http_method, url_path, **kwargs):
        calls.append((http_method.__name__.upper(), url_path))
        if http_method.__name__ != "get":
            raise AssertionError(f"{http_method.__name__.upper()} {url_path} reached the HTTP layer")
        if url_path == "quantum-computers":
            return SimpleNamespace(text=json.dumps(listing), status_code=200)
        if url_path.endswith("/dynamic-quantum-architecture") and architecture is not None:
            return SimpleNamespace(text=architecture.model_dump_json(), status_code=200)
        raise AssertionError(f"unexpected GET {url_path}")

    monkeypatch.setattr(_IQMServerClient, "_send_request", send)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    return calls


def test_list_devices_works_when_the_token_sees_several(monkeypatch):
    calls = serve(monkeypatch)
    with pytest.raises(ClientConfigurationError, match="Quantum computer not selected"):
        IQMClient(hw.SERVER_URL)  # the route list_devices took before (resonance-check finding 1)
    devices = hw.list_devices()
    assert [d["alias"] for d in devices] == list(DEVICES)
    assert set(devices[0]) == {"id", "alias", "display_name"}
    assert calls == [("GET", "quantum-computers")] * 2


def test_list_devices_works_with_one_device(monkeypatch):
    serve(monkeypatch, devices=("garnet",))
    assert [d["alias"] for d in hw.list_devices()] == ["garnet"]


def test_every_submit_route_of_the_read_only_backend_refuses(monkeypatch):
    """Skeleton review MINOR 7: the private and unbound routes too, on the real IQMClient and _IQMServerClient."""
    calls = serve(monkeypatch, devices=("adonis", "garnet"), architecture=fake_device("adonis").architecture)
    b = hw.backend("adonis")
    circuit = transpiled_ghz("adonis", 3)
    request = b.create_run_request([circuit], shots=10)  # building a request only reads
    server = b.client._iqm_server_client
    routes = {
        "backend.run": lambda: b.run(circuit, shots=10),
        "IQMBackend.run, unbound": lambda: IQMBackend.run(b, circuit, shots=10),
        "client.submit_run_request": lambda: b.client.submit_run_request(request),
        "client.submit_circuits": lambda: b.client.submit_circuits(request.circuits),
        "IQMClient.submit_run_request, unbound": lambda: IQMClient.submit_run_request(b.client, request),
        "IQMClient.submit_circuits, unbound": lambda: IQMClient.submit_circuits(b.client, request.circuits),
        "server.submit_circuits": lambda: server.submit_circuits(request),
        "server.submit_run": lambda: server.submit_run(None),
        "server.submit_sweep": lambda: server.submit_sweep(None),
        "server._submit_job": lambda: server._submit_job(job_type="circuit"),
        "server.cancel_job": lambda: server.cancel_job(JOB_ID),
        "server.delete_job": lambda: server.delete_job(JOB_ID),
        "_IQMServerClient.submit_circuits, unbound": lambda: _IQMServerClient.submit_circuits(server, request),
        "_IQMServerClient._submit_job, unbound": lambda: _IQMServerClient._submit_job(server, job_type="circuit"),
        "_IQMServerClient.delete_job, unbound": lambda: _IQMServerClient.delete_job(server, JOB_ID),
    }
    for label, route in routes.items():
        with pytest.raises(hw.QPUGuardError, match="read-only"):
            route()
        assert {method for method, _ in calls} == {"GET"}, label
    assert ("GET", "quantum-computers") in calls


# ---------------------------------------------------------------------------
# measured_qubits: where each memory column was measured (skeleton review MINOR 8).
# ---------------------------------------------------------------------------


def test_a_routed_circuit_records_where_each_clbit_is_measured(monkeypatch, tmp_path):
    routed = transpiled_ghz("aphrodite", 3, initial_layout=(0, 26, 53))  # far apart: routing adds SWAPs
    assert routed.count_ops().get("cz", 0) > 2, "the transpiler routed it"
    backend = fake_device("aphrodite")
    names = {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}
    initial = {str(v): names[p] for v, p in enumerate(routed.layout.initial_index_layout(filter_ancillas=True))}
    final = {str(v): names[p] for v, p in enumerate(routed.layout.final_index_layout(filter_ancillas=True))}
    assert initial == {"0": "QB1", "1": "QB27", "2": "QB54"} and final != initial

    _, _, meta, request = submit_offline(monkeypatch, tmp_path, "aphrodite", [routed])
    assert meta["qubit_mapping"] == [initial]  # the initial placement, as before
    assert meta["measured_qubits"] == [final]  # where clbit v was actually measured
    measures = {i.args["key"]: i.locus[0] for i in request.circuits[0].instructions if i.name == "measure"}
    assert sorted(measures.values(), key=hw._qubit_sort_key) == sorted(final.values(), key=hw._qubit_sort_key)


def test_measured_columns_follow_iqms_own_memory_layout():
    """Two measured registers around an unmeasured one, a bit left unmeasured, measurements out of order."""
    qc = QuantumCircuit(QuantumRegister(3), ClassicalRegister(2, "a"), ClassicalRegister(1, "unused"), ClassicalRegister(2, "b"))
    qc.r(pi / 2, pi / 2, [0, 1, 2])
    qc.measure(0, qc.cregs[2][1])  # b[1]
    qc.measure(1, qc.cregs[0][0])  # a[0]
    qc.measure(2, qc.cregs[2][0])  # b[0]
    names = {0: "QB1", 1: "QB2", 2: "QB3"}
    instructions = serialize_instructions(qc, names)
    (measured,) = hw._measured_qubits(SimpleNamespace(circuits=[SimpleNamespace(instructions=instructions)]))
    assert measured == {0: "QB2", 2: "QB3", 3: "QB1"}  # a = columns 0-1, "unused" left out, b = columns 2-3
    keys = {i.args["key"]: i.locus[0] for i in instructions if i.name == "measure"}
    for key, qubit in keys.items():
        # one shot in which only this measurement read 1, through IQMJob's own formatting and save_raw's parser
        results = {k: [[int(k == key)]] for k in keys}
        (row,) = hw._memory_bits(IQMJob._iqm_format_measurement_results(results, 1))
        assert measured[int(np.flatnonzero(row)[0])] == qubit


# ---------------------------------------------------------------------------
# The ledger lock.
# ---------------------------------------------------------------------------


def test_of_two_concurrent_submissions_only_one_fits_one_job(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger, jobs="1", shots="100")
    both_checked = threading.Barrier(2)

    class SlowBackend(FakeBackend):
        def submit_run_request(self, run_request):
            time.sleep(0.3)  # the other submission is waiting for the lock meanwhile
            return super().submit_run_request(run_request)

    fake = SlowBackend(ledger)

    def connect(*args, **kwargs):
        both_checked.wait(timeout=30)  # both have passed the first allowance check, before any line exists
        return fake

    monkeypatch.setattr(hw, "_connect", connect)
    outcomes = []

    def one(i):
        try:
            hw.submit(ghz3(), device="emerald", shots=5, purpose=f"thread {i}", data_root=tmp_path / f"hw{i}")
            outcomes.append("submitted")
        except hw.QPUGuardError as e:
            outcomes.append(str(e))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert sorted(outcomes) == ["hwtask: 1 job(s) already in the ledger; DQ_QPU_MAX_JOBS=1", "submitted"]
    assert len(fake.runs) == 1 and len(hw.read_ledger(ledger)) == 1


def test_the_ledger_lock_refuses_a_missing_ledger(tmp_path):
    with pytest.raises(hw.QPUGuardError, match="cannot be opened to lock it"), hw.ledger_lock(tmp_path / "absent.md"):
        pass
    assert not (tmp_path / "absent.md").exists()


# ---------------------------------------------------------------------------
# A submit call that raises: the fail-closed ledger row
# ---------------------------------------------------------------------------

MAIN_AT_THE_BATCH = "79092cf"
"""Local main while the Garnet and Emerald batch runs from its hw.py, appending to the same ledger file."""

UNKNOWN_ID = re.compile(r"UNKNOWN-\d{8}T\d{6}Z")


class LostReplyBackend(FakeBackend):
    """The server took the POST, then the reply was lost: submit_run_request raises."""

    def __init__(self, ledger, error):
        super().__init__(ledger)
        self.error = error

    def submit_run_request(self, run_request):
        self.runs.append(run_request)
        raise self.error


def lost_reply(monkeypatch, tmp_path, error, *, jobs="1"):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger, jobs=jobs, shots="5")
    fake = LostReplyBackend(ledger, error)
    monkeypatch.setattr(hw, "_connect", lambda *a, **k: fake)
    return ledger, fake


def submit_ghz3(tmp_path):
    return hw.submit(ghz3(), device="emerald", shots=5, purpose="unit test", data_root=tmp_path / "hw")


def test_a_submit_call_that_raises_leaves_one_fail_closed_row(monkeypatch, tmp_path):
    real_ledger = write_ledger(tmp_path / "real.md")
    arm(monkeypatch, real_ledger, jobs="1", shots="5")
    monkeypatch.setattr(hw, "_connect", lambda *a, **k: FakeBackend(real_ledger))
    submit_ghz3(tmp_path / "real")
    (real,) = hw.read_ledger(real_ledger)

    ledger, fake = lost_reply(monkeypatch, tmp_path, requests.exceptions.ReadTimeout("read timed out"))
    with pytest.raises(hw.JobMayExistError, match="may exist on Resonance") as raised:
        submit_ghz3(tmp_path)
    assert isinstance(raised.value.__cause__, requests.exceptions.ReadTimeout)
    assert not isinstance(raised.value, hw.QPUGuardError), "not a refusal: the job may have been sent"
    (row,) = hw.read_ledger(ledger)
    assert UNKNOWN_ID.fullmatch(row.job_id) and row.job_id == raised.value.ledger_job_id
    assert row.job_id == f"UNKNOWN-{row.time.replace('-', '').replace(':', '')}"
    assert (row.task, row.device, row.calibration_set_id, row.qubits, row.circuits, row.shots) == (
        real.task, real.device, real.calibration_set_id, real.qubits, real.circuits, real.shots)
    assert row.purpose == "unit test [no job id: submit raised ReadTimeout]"
    assert row.job_id in str(raised.value) and "Check Resonance's job list, read-only, before any retry" in str(raised.value)
    assert hw.usage("hwtask", ledger) == (1, 5) and not (tmp_path / "hw").exists()

    # the row consumed the allowance: a retry is refused before any connection, and the placeholder can't be re-pulled
    monkeypatch.setattr(hw, "_connect", lambda *a, **k: pytest.fail("connected"))
    with pytest.raises(hw.QPUGuardError, match=re.escape("1 job(s) already in the ledger")):
        submit_ghz3(tmp_path)
    with pytest.raises(ValueError, match="is a fail-closed ledger row, not a job id"):
        hw.retrieve(row.job_id, "emerald")
    assert len(fake.runs) == 1 and len(hw.read_ledger(ledger)) == 1


def test_a_keyboard_interrupt_in_the_submit_call_also_leaves_the_row(monkeypatch, tmp_path):
    ledger, _ = lost_reply(monkeypatch, tmp_path, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as raised:
        submit_ghz3(tmp_path)
    (row,) = hw.read_ledger(ledger)
    assert UNKNOWN_ID.fullmatch(row.job_id) and row.purpose == "unit test [no job id: submit raised KeyboardInterrupt]"
    assert any(row.job_id in note and "may exist" in note for note in raised.value.__notes__)


def fail_at(stage):
    def fail(*args, **kwargs):
        raise RuntimeError(f"{stage} failed")

    return fail


@pytest.mark.parametrize("stage", ["connect", "create_run_request", "sent_ops", "unarmed", "allowance"])
def test_no_row_for_a_failure_before_the_submit_call(monkeypatch, tmp_path, stage):
    ledger, fake = lost_reply(monkeypatch, tmp_path, AssertionError("the submit call was reached"))
    if stage == "connect":
        monkeypatch.setattr(hw, "_connect", fail_at(stage))
    elif stage == "create_run_request":
        fake.create_run_request = fail_at(stage)
    elif stage == "sent_ops":
        monkeypatch.setattr(hw, "_check_sent_ops", fail_at(stage))
    elif stage == "unarmed":
        monkeypatch.delenv("DQ_QPU")
    else:
        monkeypatch.setenv("DQ_QPU_MAX_SHOTS", "4")
    with pytest.raises((RuntimeError, hw.QPUGuardError)) as raised:
        submit_ghz3(tmp_path)
    assert not isinstance(raised.value, hw.JobMayExistError)
    assert hw.read_ledger(ledger) == [] and fake.runs == []


def main_hw(monkeypatch, tmp_path):
    """main's hw.py at MAIN_AT_THE_BATCH, loaded from git as its own module."""
    try:
        source = subprocess.run(
            ["git", "-C", str(hw.REPO_ROOT), "show", f"{MAIN_AT_THE_BATCH}:src/dqiqm/hw.py"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"main {MAIN_AT_THE_BATCH}'s hw.py is not in this checkout: {e}")
    path = tmp_path / "main_hw.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("main_hw_at_the_batch", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # its dataclasses look their module up while it runs
    spec.loader.exec_module(module)
    return module


def test_mains_parser_reads_the_fail_closed_rows(monkeypatch, tmp_path):
    """The batch runs on main's hw.py and shares the one ledger: a row it can't parse would stop every
    submission, for every task. So main's own read_ledger must take every fail-closed row."""
    main = main_hw(monkeypatch, tmp_path)
    ledger, _ = lost_reply(monkeypatch, tmp_path, requests.exceptions.ConnectionError("reset"), jobs="5")
    with pytest.raises(hw.JobMayExistError):
        submit_ghz3(tmp_path)
    # the tagged purposes of the other two paths, through the same row builder
    for purpose, error in (("C3 pilot [dd=ENABLED]", ValueError("bad JobData")),
                           ("rabi pilot [pulse,playlist=1a2b3c4d]", TypeError("cannot deepcopy"))):
        with pytest.raises(hw.JobMayExistError):
            hw._fail_closed(ledger, error, task="hwtask", device="garnet", calibration_set_id=CALSET,
                            qubits=["QB1", "QB3"], circuits=2, shots=100, purpose=purpose)
    ours, theirs = hw.read_ledger(ledger), main.read_ledger(ledger)
    assert len(ours) == len(theirs) == 3
    assert [dataclasses.astuple(r) for r in theirs] == [dataclasses.astuple(r) for r in ours]
    assert all(UNKNOWN_ID.fullmatch(r.job_id) for r in theirs)
    assert [r.purpose for r in theirs] == [
        "unit test [no job id: submit raised ConnectionError]",
        "C3 pilot [dd=ENABLED] [no job id: submit raised ValueError]",
        "rabi pilot [pulse,playlist=1a2b3c4d] [no job id: submit raised TypeError]",
    ]
    assert main.usage("hwtask", ledger) == hw.usage("hwtask", ledger) == (3, 5 + 2 * 2 * 100)
