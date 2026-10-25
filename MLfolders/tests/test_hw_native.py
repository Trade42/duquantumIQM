"""hw.submit_native, native iqm.pulse.Circuit jobs (prx_12, measure_012), against a fake IQMClient.

IQM's own ``create_run_request`` validates every circuit here, against a fake architecture with
``prx_12`` calibrated. Nothing reaches the network. Not to be used until reviewed and merged; C3 runs
only after a pilot.
"""

import json
import re
from math import pi
from types import SimpleNamespace
from typing import ClassVar
from uuid import UUID

import iqm.iqm_client
import numpy as np
import pytest
import requests
from iqm.iqm_client import IQMClient
from iqm.iqm_client.iqm_client import CircuitJob
from iqm.iqm_client.models import CircuitCompilationOptions, DDMode
from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
from iqm.iqm_server_client.models import JobStatus
from iqm.pulse import Circuit, CircuitOperation
from iqm.qiskit_iqm import IQMFakeAdonis

from dqiqm import hw

LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "44444444-2222-4333-8444-555555555555"
TIMELINE = [{"status": "created", "timestamp": "2026-10-25T12:00:00Z"}]
SHOTS = 100


@pytest.fixture(autouse=True)
def default_data_root(monkeypatch, tmp_path):
    """A save without data_root lands in tmp_path, never in the repo's data/hw."""
    monkeypatch.setattr(hw, "DATA_ROOT", tmp_path / "default-data-root")


def architecture(prx_12_on=("QB1", "QB2", "QB3", "QB4", "QB5")):
    """IQMFakeAdonis's architecture with ``prx_12`` calibrated on ``prx_12_on`` (Garnet has it on all 20)."""
    a = IQMFakeAdonis().architecture
    prx = a.gates["prx"]
    impl = next(iter(prx.implementations.values()))
    # built fresh, not model_copy'd: GateInfo caches its loci
    prx_12 = type(prx)(implementations={"__fake": type(impl)(loci=tuple((q,) for q in prx_12_on))},
                       default_implementation="__fake", override_default_implementation={})
    return a.model_copy(update={"calibration_set_id": UUID(CALSET), "gates": {**a.gates, "prx_12": prx_12}})


def op(name, *locus, **args):
    return CircuitOperation(name=name, locus=tuple(locus), args=args)


def ref2():
    """C3's ref2 shape (garnet-calib design.json): QB1 to |1>, prx_12 to |2>, then both read out."""
    return Circuit(name="C3_QB1-QB3_ref2", instructions=(
        op("prx", "QB1", angle=pi / 2, phase=0.0), op("barrier", "QB1"),
        op("prx", "QB1", angle=pi / 2, phase=0.0), op("barrier", "QB1"),
        op("prx_12", "QB1", angle=pi, phase=0.0), op("barrier", "QB1", "QB3"),
        op("measure", "QB1", key="m_QB1"), op("measure", "QB3", key="m_QB3"),
    ))


def plain():
    return Circuit(name="C3_QB1-QB3_plain_n01", instructions=(
        op("prx", "QB1", angle=pi / 2, phase=0.0), op("prx", "QB3", angle=pi / 2, phase=0.0), op("barrier", "QB1", "QB3"),
        op("cz", "QB1", "QB3"), op("barrier", "QB1", "QB3"),
        op("measure", "QB1", key="m_QB1"), op("measure", "QB3", key="m_QB3"),
    ))


def write_ledger(path, rows=()):
    path.write_text(LEDGER_HEADER + "".join(r + "\n" for r in rows))
    return path


def arm(monkeypatch, ledger, task="c3task", jobs="1", shots=str(2 * SHOTS)):
    monkeypatch.setenv("DQ_QPU", "run")
    monkeypatch.setenv("DQ_QPU_TASK", task)
    monkeypatch.setenv("DQ_QPU_MAX_JOBS", jobs)
    monkeypatch.setenv("DQ_QPU_MAX_SHOTS", shots)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))


def results_for(circuits, shots=SHOTS):
    """Per circuit, per key: shots x 1 values, ternary for QB1 (a stand-in for measure_012's 0/1/2)."""
    rng = np.random.default_rng(2026)
    return [{inst.args["key"]: rng.integers(0, 3 if inst.locus == ("QB1",) else 2, size=(shots, 1)).tolist()
             for inst in c.instructions if inst.name == "measure"} for c in circuits]


class FakeCircuitJob(CircuitJob):
    """iqm-client's own ``CircuitJob``: its ``job_id`` and ``payload()`` are the real ones. Only the two calls
    that would poll the server are stood in, ``wait_for_completion`` and ``result``."""

    def __init__(self, client, ledger, results, status):
        super().__init__(data=SimpleNamespace(id=UUID(JOB_ID), timeline=TIMELINE, status=JobStatus.WAITING),
                         _iqm_client=client)
        self.ledger, self.results, self.final_status = ledger, results, status

    def wait_for_completion(self, *, timeout_secs=None):
        if self.ledger is not None:
            assert JOB_ID in self.ledger.read_text(), "the ledger line must be written before waiting for the job"
        self.data.status = self.final_status
        return self.final_status

    def result(self):
        return self.results if self.data.status == JobStatus.COMPLETED else None


def offline_client(ledger, arch, status=JobStatus.COMPLETED, submit_raises=None):
    """An IQMClient with IQM's own create_run_request and a recording submit_run_request; no network.

    The server keeps each submitted request as its JSON, and serves it back on ``GET jobs/<id>/payload``
    only, to iqm-client's own payload getters: what a re-pull reads is what the wire carries.
    With ``submit_raises``, the server takes the request and the reply is lost: the submit call raises it.
    """
    stored = {}  # job id -> the submitted run request, as JSON

    class Server:
        get_submit_circuits_payload = _IQMServerClient.get_submit_circuits_payload
        _deserialize_response = staticmethod(_IQMServerClient._deserialize_response)
        gets: ClassVar[list] = []

        def _send_request(self, http_method, url_path, **kwargs):
            job_id = url_path.removeprefix("jobs/").removesuffix("/payload")
            if http_method.__name__ == "get" and url_path == f"jobs/{job_id}/payload" and job_id in stored:
                Server.gets.append(url_path)
                return SimpleNamespace(text=stored[job_id], status_code=200)
            raise AssertionError(f"{http_method.__name__.upper()} {url_path} reached the HTTP layer")

    class Client:
        create_run_request = IQMClient.create_run_request
        _get_submit_circuits_payload = IQMClient._get_submit_circuits_payload
        instances: ClassVar[list] = []
        server = Server

        def __init__(self, url, quantum_computer=None, **kwargs):
            assert url == hw.SERVER_URL and quantum_computer == "garnet"
            self.submitted, self.jobs = [], []
            self._iqm_server_client = Server()
            Client.instances.append(self)

        def get_dynamic_quantum_architecture(self, calibration_set_id=None):
            assert calibration_set_id in (None, UUID(CALSET))
            return arch

        def submit_run_request(self, run_request, use_timeslot=False):
            self.submitted.append((run_request, use_timeslot))
            if submit_raises is not None:
                raise submit_raises
            stored[JOB_ID] = _IQMServerClient._serialize_model(run_request)
            self.jobs.append(FakeCircuitJob(self, ledger, results_for(run_request.circuits, run_request.shots), status))
            return self.jobs[-1]

        def get_job(self, job_id):
            # as IQMClient.get_job: a new job object on this client, from the server's record
            (job,) = [j for c in Client.instances for j in c.jobs if j.job_id == job_id]
            return FakeCircuitJob(self, None, job.results, job.final_status)

    return Client


@pytest.fixture
def client(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    fake = offline_client(ledger, architecture())
    monkeypatch.setattr(iqm.iqm_client, "IQMClient", fake)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    return SimpleNamespace(ledger=ledger, cls=fake)


@pytest.fixture
def no_client(monkeypatch):
    def connect(*args, **kwargs):
        raise AssertionError("the guard let a connection through")

    monkeypatch.setattr(hw, "_connect_client", connect)


def submit(circuits, **kwargs):
    return hw.submit_native(circuits, **{"device": "garnet", "shots": SHOTS, "purpose": "C3 pilot", **kwargs})


def test_submit_native_sends_prx_12_and_saves_the_per_key_results(client, tmp_path):
    circuits = [ref2(), plain()]
    hashes = [hw.instructions_sha256(c.instructions) for c in circuits]
    sub = submit(circuits, native_sha256=hashes, data_root=tmp_path / "hw")

    ((fake,),) = [client.cls.instances]
    ((request, use_timeslot),) = fake.submitted
    assert use_timeslot is False and request.shots == SHOTS
    assert request.calibration_set_id == UUID(CALSET), "the calibration set is pinned in the request"
    prx_12 = [i for i in request.circuits[0].instructions if i.name == "prx_12"]
    assert [(i.locus, i.args) for i in prx_12] == [(("QB1",), {"angle": pi, "phase": 0.0})]

    (row,) = hw.read_ledger(client.ledger)
    assert (row.task, row.device, row.calibration_set_id, row.qubits) == ("c3task", "garnet", CALSET, "QB1,QB3")
    assert (row.circuits, row.shots, row.job_id, row.purpose) == (2, SHOTS, JOB_ID, "C3 pilot")
    assert (sub.job_id, sub.status, sub.qubits) == (JOB_ID, "completed", ["QB1", "QB3"])

    meta = json.loads((sub.path / "meta.json").read_text())
    assert sub.path.parent.name == f"{row.time[:10].replace('-', '')}-c3task" and sub.path.name == JOB_ID
    assert (meta["kind"], meta["device"], meta["calibration_set_id"], meta["shots"], meta["circuits"]) == (
        "native", "garnet", CALSET, SHOTS, 2)
    assert meta["circuit_order"] == sub.circuit_order == [
        {"position": 0, "name": "C3_QB1-QB3_ref2", "native_sha256": hashes[0]},
        {"position": 1, "name": "C3_QB1-QB3_plain_n01", "native_sha256": hashes[1]},
    ]
    assert meta["measurement_keys"] == [{"m_QB1": ["QB1"], "m_QB3": ["QB3"]}] * 2
    assert meta["physical_qubits"] == [["QB1", "QB3"]] * 2
    assert meta["compilation_options"] == hw.compilation_options_record()
    assert (meta["status"], meta["success"], meta["timeline"], meta["purpose"]) == ("completed", True, TIMELINE, "C3 pilot")
    sent = fake.jobs[0].results
    with np.load(sub.path / "measurements.npz") as npz:
        assert len(meta["arrays"]) == len(npz.files) == 4
        for entry in meta["arrays"]:
            saved = npz[entry["key"]]
            assert np.array_equal(saved, np.asarray(sent[entry["circuit"]][entry["measurement_key"]]))
            assert entry["qubits"] == [entry["measurement_key"][2:]] and entry["shape"] == [SHOTS, 1]
    assert {int(v) for e in meta["arrays"] if e["qubits"] == ["QB1"] for v in np.load(sub.path / "measurements.npz")[e["key"]].ravel()} == {0, 1, 2}

    # the allowance (one job) is spent: a second submission is refused before any connection
    with pytest.raises(hw.QPUGuardError, match="1 job"):
        submit(circuits, data_root=tmp_path / "hw")
    assert len(client.cls.instances) == 1


def test_compilation_options_go_into_the_request_and_tag_the_ledger(client, tmp_path):
    sub = submit([ref2()], circuit_compilation_options=CircuitCompilationOptions(dd_mode=DDMode.ENABLED),
                 data_root=tmp_path / "hw")
    ((request, _),) = client.cls.instances[0].submitted
    assert request.dd_mode == DDMode.ENABLED and sub.compilation_options["dd_mode"] == "enabled"
    (row,) = hw.read_ledger(client.ledger)
    assert row.purpose == sub.purpose == "C3 pilot [dd=ENABLED]"


@pytest.mark.parametrize(
    ("circuits", "error", "match"),
    [
        ([Circuit(name="r", instructions=(op("reset", "QB1"), op("measure", "QB1", key="m")))], hw.QPUGuardError,
         r"op allowlist: circuit 0 \('r'\) has reset x1; only barrier, cz, delay, measure, measure_012, prx, prx_12"),
        ([Circuit(name="cc", instructions=(op("measure", "QB1", key="m"), op("cc_prx", "QB1", angle=pi, phase=0.0,
                                                                                   feedback_qubit="QB1", feedback_key="m")))],
         hw.QPUGuardError, "has cc_prx x1"),
        ([Circuit(name="dup", instructions=(op("measure", "QB1", key="m"), op("measure", "QB3", key="m")))], ValueError,
         "measurement key 'm' is used twice"),
        (["not a circuit"], TypeError, "iqm.pulse.Circuit"),
        ([], ValueError, "no circuits"),
    ],
)
def test_bad_circuits_refuse_before_any_connection(monkeypatch, tmp_path, no_client, circuits, error, match):
    arm(monkeypatch, write_ledger(tmp_path / "ledger.md"))
    with pytest.raises(error, match=match):
        submit(circuits)


def test_unarmed_and_bad_options_refuse_before_any_connection(monkeypatch, tmp_path, no_client):
    write_ledger(tmp_path / "ledger.md")
    monkeypatch.setenv("DQ_QPU_LEDGER", str(tmp_path / "ledger.md"))
    with pytest.raises(hw.QPUGuardError, match="not armed"):
        submit([ref2()])
    arm(monkeypatch, tmp_path / "ledger.md")
    with pytest.raises(ValueError, match="use_timeslot"):
        submit([ref2()], use_timeslot=True)
    with pytest.raises(ValueError, match="unsupported option"):
        submit([ref2()], qubit_mapping={"a": "QB1"})
    with pytest.raises(ValueError, match=re.escape("native_sha256 has 2 entries for 1 circuit(s)")):
        submit([ref2()], native_sha256=["a", "b"])
    with pytest.raises(hw.QPUGuardError, match=re.escape("1 x 100 requested > DQ_QPU_MAX_SHOTS=50")):
        monkeypatch.setenv("DQ_QPU_MAX_SHOTS", "50")
        submit([ref2()])


def test_iqms_validation_refuses_prx_12_where_it_is_not_calibrated(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    fake = offline_client(ledger, architecture(prx_12_on=("QB2",)))
    monkeypatch.setattr(iqm.iqm_client, "IQMClient", fake)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    with pytest.raises(Exception, match=re.escape("('QB1',) is not allowed as locus for 'prx_12'")) as raised:
        submit([ref2()])
    assert not isinstance(raised.value, hw.QPUGuardError), "refused by IQM's validation, not by the allowlist"
    assert fake.instances[0].submitted == [] and hw.read_ledger(ledger) == []


def test_measure_012_passes_the_allowlist_but_iqm_client_35_refuses_it(client):
    c = Circuit(name="m012", instructions=(op("prx", "QB1", angle=pi, phase=0.0), op("measure_012", "QB1", key="m")))
    with pytest.raises(Exception, match="measure_012") as raised:
        submit([c])
    assert not isinstance(raised.value, hw.QPUGuardError), "refused by IQM's validation, not by the allowlist"
    assert client.cls.instances[0].submitted == [] and hw.read_ledger(client.ledger) == []


def test_a_wrong_native_sha256_refuses_before_submitting(client):
    with pytest.raises(hw.QPUGuardError, match="native_sha256 0+ given"):
        submit([ref2()], native_sha256=["0" * 64])
    assert client.cls.instances[0].submitted == [] and hw.read_ledger(client.ledger) == []


def test_a_failed_job_is_still_recorded(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    monkeypatch.setattr(iqm.iqm_client, "IQMClient", offline_client(ledger, architecture(), status=JobStatus.FAILED))
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    sub = submit([ref2()], data_root=tmp_path / "hw")
    meta = json.loads((sub.path / "meta.json").read_text())
    assert (sub.status, meta["success"], meta["arrays"]) == ("failed", False, []) and len(hw.read_ledger(ledger)) == 1
    assert not (sub.path / "measurements.npz").exists()


def test_a_re_pull_saves_as_the_fresh_job_did(client, tmp_path):
    """The re-pull reads the circuits through iqm-client's own CircuitJob.payload(), from the JSON the server
    stored: payload() hands them back as dicts, and retrieve_native rebuilds them as Circuit objects."""
    fresh = submit([ref2(), plain()], data_root=tmp_path / "fresh")
    ((request, _),) = client.cls.instances[0].submitted
    job = hw.retrieve_native(JOB_ID, "garnet")
    read_only = client.cls.instances[-1]
    assert client.cls.server.gets == [f"jobs/{JOB_ID}/payload"], "one GET, through the read-only server client"
    with pytest.raises(hw.QPUGuardError, match="read-only"):
        read_only.submit_run_request(None)
    with pytest.raises(hw.QPUGuardError, match="read-only"):
        read_only._iqm_server_client.submit_circuits(None)
    assert job.parameters.shots == SHOTS and job.parameters.calibration_set_id == UUID(CALSET)
    assert all(isinstance(c, Circuit) for c in job.circuits)
    assert [hw.instructions_sha256(c.instructions) for c in job.circuits] == [
        hw.instructions_sha256(c.instructions) for c in request.circuits] == [o["native_sha256"] for o in fresh.circuit_order]
    assert [hw._measurement_keys(c) for c in job.circuits] == [hw._measurement_keys(c) for c in request.circuits]
    path = hw.save_native_raw(
        job.results, circuits=job.circuits, job_id=JOB_ID, task="c3task", device="garnet",
        calibration_set_id=str(job.parameters.calibration_set_id), shots=job.parameters.shots, status=job.data.status,
        compilation_options=hw.compilation_options_record(job.parameters), timeline=job.data.timeline,
        purpose=fresh.purpose, day=fresh.path.parent.name.split("-")[0], data_root=tmp_path / "repull",
    )
    a, b = (json.loads((p / "meta.json").read_text()) for p in (fresh.path, path))
    assert {k for k in a if a[k] != b[k]} <= {"saved_utc"}
    with np.load(fresh.path / "measurements.npz") as x, np.load(path / "measurements.npz") as y:
        assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)

    # save_native_raw also takes payload()'s dicts as they come
    as_served, _ = read_only.get_job(UUID(JOB_ID)).payload()
    assert all(isinstance(c, dict) for c in as_served)
    again = hw.save_native_raw(job.results, circuits=as_served, job_id=JOB_ID, task="c3task", device="garnet",
                               calibration_set_id=CALSET, shots=SHOTS, status=job.data.status, data_root=tmp_path / "dicts")
    c = json.loads((again / "meta.json").read_text())
    assert (c["circuit_order"], c["measurement_keys"], c["physical_qubits"]) == (
        a["circuit_order"], a["measurement_keys"], a["physical_qubits"])


def test_a_submit_call_that_raises_leaves_one_fail_closed_row(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    fake = offline_client(ledger, architecture(), submit_raises=requests.exceptions.ReadTimeout("read timed out"))
    monkeypatch.setattr(iqm.iqm_client, "IQMClient", fake)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    with pytest.raises(hw.JobMayExistError, match="may exist on Resonance") as raised:
        submit([ref2(), plain()], data_root=tmp_path / "hw")
    assert isinstance(raised.value.__cause__, requests.exceptions.ReadTimeout)

    # the row a completed submission writes (test_submit_native_sends_prx_12_...), with the placeholder id
    (row,) = hw.read_ledger(ledger)
    assert (row.task, row.device, row.calibration_set_id, row.qubits) == ("c3task", "garnet", CALSET, "QB1,QB3")
    assert (row.circuits, row.shots) == (2, SHOTS) and hw.usage("c3task", ledger) == (1, 2 * SHOTS)
    assert re.fullmatch(r"UNKNOWN-\d{8}T\d{6}Z", row.job_id) and row.job_id == raised.value.ledger_job_id
    assert row.purpose == "C3 pilot [no job id: submit raised ReadTimeout]"
    assert not (tmp_path / "hw").exists()

    with pytest.raises(hw.QPUGuardError, match=re.escape("1 job(s) already in the ledger")):
        submit([ref2(), plain()], data_root=tmp_path / "hw")
    with pytest.raises(ValueError, match="is a fail-closed ledger row, not a job id"):
        hw.retrieve_native(row.job_id, "garnet")
    assert len(fake.instances) == 1 and len(fake.instances[0].submitted) == 1, "no second connection, no second submit"
    assert len(hw.read_ledger(ledger)) == 1
