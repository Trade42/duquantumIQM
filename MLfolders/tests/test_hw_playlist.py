"""hw.submit_playlist, the guarded pulse-level path, against a fake Pulla. Nothing here reaches the network.

Not to be used until Sal and the mentors confirm pulse-level submission rights: these tests are
the only runs it has had.
"""

import json
import re
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import requests
from exa.common.data.parameter import Parameter, Setting, Sweep
from exa.common.data.setting_node import SettingNode
from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
from iqm.iqm_server_client.models import JobStatus
from iqm.models.playlist import Playlist
from iqm.models.playlist.segment import Segment
from iqm.pulla.pulla import Pulla
from iqm.station_control.interface.models import RunDefinition
from iqm.station_control.interface.models.sweep import SweepDefinition

from dqiqm import hw

LEDGER_HEADER = """# QPU ledger

| UTC time | task | device | calibration_set_id | physical qubits | circuits x shots | job id | purpose |
|---|---|---|---|---|---|---|---|
"""
CALSET = "0f0e0d0c-0b0a-4908-8706-050403020100"
JOB_ID = "33333333-2222-4333-8444-555555555555"
TIMELINE = [{"status": "created", "timestamp": "2026-10-25T12:00:00Z"}]


@pytest.fixture(autouse=True)
def default_data_root(monkeypatch, tmp_path):
    """A save without data_root lands in tmp_path, never in the repo's data/hw."""
    monkeypatch.setattr(hw, "DATA_ROOT", tmp_path / "default-data-root")


def run_definition(*, segments=2, shots=1000, qubits=("QB10", "QB2"), spots=None, averaging_bins="shots"):
    """A minimal RunDefinition, as a compiler's final stage would shape it: the playlist, the settings
    ``SettingNode.set_shots`` writes, the qubits, and optional soft-sweep spots."""
    options = {"playlist_repeats": Setting(Parameter("playlist_repeats"), shots),
               "end_delay": Setting(Parameter("end_delay"), 300e-6)}
    if averaging_bins is not None:
        options["averaging_bins"] = Setting(Parameter("averaging_bins"), shots if averaging_bins == "shots" else averaging_bins)
    settings = SettingNode("root", subtrees={"controllers": SettingNode("controllers", subtrees={
        "options": SettingNode("options", settings=options)})})
    sweeps = [] if spots is None else [(Sweep(parameter=Parameter("flux"), data=list(np.linspace(0, 1, spots))),)]
    sweep = SweepDefinition(sweep_id=None, dut_label="dut", settings=settings, sweeps=sweeps, return_parameters=["QB2__readout"],
                            playlist=Playlist(segments=[Segment() for _ in range(segments)]))
    return RunDefinition(run_id=None, username="Pulla User", experiment_name="Pulla", experiment_label="pulla",
                         additional_run_properties={"qubits": list(qubits)}, components=list(qubits), sweep_definition=sweep)


def write_ledger(path, rows=()):
    path.write_text(LEDGER_HEADER + "".join(r + "\n" for r in rows))
    return path


def arm(monkeypatch, ledger, task="pulsetask", jobs="2", shots="2000"):
    monkeypatch.setenv("DQ_QPU", "run")
    monkeypatch.setenv("DQ_QPU_TASK", task)
    monkeypatch.setenv("DQ_QPU_MAX_JOBS", jobs)
    monkeypatch.setenv("DQ_QPU_MAX_SHOTS", shots)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))


def sweep_results(segments, shots):
    rng = np.random.default_rng(2026)
    return {"QB2__readout": [rng.integers(0, 2, size=(segments, shots)).astype(np.int8)],
            "QB10__readout": [rng.random((segments, shots)) + 1j * rng.random((segments, shots))]}


class FakePullaJob:
    """``PullaJob``'s surface: ``job_id`` is a property (qiskit's is a method), and ``data`` holds the timeline."""

    def __init__(self, ledger, status=JobStatus.COMPLETED):
        self.ledger = ledger
        self.final_status = status
        self.data = SimpleNamespace(timeline=TIMELINE, status=JobStatus.WAITING)

    @property
    def job_id(self):
        return UUID(JOB_ID)

    def wait_for_completion(self, *, timeout_secs=None):
        assert JOB_ID in self.ledger.read_text(), "the ledger line must be written before waiting for the job"
        self.data.status = self.final_status
        return self.final_status


class FakePulla:
    """The Pulla surface hw.submit_playlist uses: submit_playlist, and the server client's sweep-results getter."""

    def __init__(self, ledger, results, status=JobStatus.COMPLETED, submit_raises=None):
        self.ledger, self.results, self.status, self.submit_raises = ledger, results, status, submit_raises
        self.submitted = []
        self._iqm_server_client = SimpleNamespace(get_job_artifact_sweep_results=self.sweep_results)

    def sweep_results(self, job_id):
        assert str(job_id) == JOB_ID
        return self.results

    def submit_playlist(self, run_definition, *, context, use_timeslot=False):
        self.submitted.append((run_definition, context, use_timeslot))
        if self.submit_raises is not None:  # the server took it, then the reply was lost
            raise self.submit_raises
        return FakePullaJob(self.ledger, self.status)


@pytest.fixture
def no_pulla(monkeypatch):
    def connect(*args, **kwargs):
        raise AssertionError("the guard let a Pulla connection through")

    monkeypatch.setattr(hw, "_connect_pulla", connect)


def fake_pulla(monkeypatch, ledger, results, status=JobStatus.COMPLETED, submit_raises=None):
    fake = FakePulla(ledger, results, status, submit_raises)
    calls = []

    def connect(device, *, read_only):
        calls.append((device, read_only))
        return fake

    monkeypatch.setattr(hw, "_connect_pulla", connect)
    return fake, calls


def submit(rd, **kwargs):
    kwargs = {"context": {"k": "v"}, "device": "garnet", "calibration_set_id": CALSET, "purpose": "rabi pilot", **kwargs}
    return hw.submit_playlist(rd, **kwargs)


def test_the_shape_is_read_from_the_run_definition():
    shape = hw.playlist_shape(run_definition(segments=3, shots=500, spots=4, averaging_bins=1))
    assert (shape.segments, shape.spots, shape.circuits, shape.shots, shape.averaging_bins) == (3, 4, 12, 500, 1)
    assert shape.qubits == ["QB2", "QB10"]
    assert re.fullmatch(r"[0-9a-f]{64}", shape.playlist_sha256)
    # the hash is of the playlist only: deterministic, and it changes with the playlist
    assert hw.playlist_shape(run_definition(segments=3, shots=7)).playlist_sha256 == shape.playlist_sha256
    assert hw.playlist_shape(run_definition(segments=2)).playlist_sha256 != shape.playlist_sha256
    assert hw.playlist_shape(run_definition(averaging_bins=None)).averaging_bins is None


@pytest.mark.parametrize(
    ("rd", "match"),
    [
        (run_definition(segments=0), "no playlist segments"),
        (run_definition(shots=0), "playlist_repeats must be a positive int"),
        (run_definition(qubits=()), "names no qubits"),
    ],
)
def test_an_unreadable_run_definition_refuses_before_any_connection(monkeypatch, tmp_path, no_pulla, rd, match):
    arm(monkeypatch, write_ledger(tmp_path / "ledger.md"))
    with pytest.raises(ValueError, match=match):
        submit(rd)


def test_a_run_definition_without_shot_settings_refuses(monkeypatch, tmp_path, no_pulla):
    rd = run_definition()
    rd.sweep_definition.settings = SettingNode("root", subtrees={"controllers": SettingNode("controllers")})
    with pytest.raises(ValueError, match="cannot read settings.controllers.options.playlist_repeats"):
        submit(rd)


@pytest.mark.parametrize("env", [{}, {"DQ_QPU": "run"}, {"DQ_QPU": "yes", "DQ_QPU_TASK": "pulsetask"}])
def test_unarmed_refuses_before_any_connection(monkeypatch, tmp_path, no_pulla, env):
    write_ledger(tmp_path / "ledger.md")
    monkeypatch.setenv("DQ_QPU_MAX_JOBS", "10")
    monkeypatch.setenv("DQ_QPU_MAX_SHOTS", "100000")
    monkeypatch.setenv("DQ_QPU_LEDGER", str(tmp_path / "ledger.md"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(hw.QPUGuardError, match="not armed"):
        submit(run_definition())


@pytest.mark.parametrize(
    ("jobs", "shots", "rd", "match"),
    [
        (None, None, run_definition(), "DQ_QPU_MAX_JOBS=0"),  # no allowance at all
        # 2 segments x 3 soft-sweep spots, by 1000 repeats: 6,000 shots
        ("1", "5000", run_definition(segments=2, shots=1000, spots=3), "0 shots used + 6 x 1000 requested > DQ_QPU_MAX_SHOTS=5000"),
    ],
)
def test_over_the_allowance_refuses_before_any_connection(monkeypatch, tmp_path, no_pulla, jobs, shots, rd, match):
    arm(monkeypatch, write_ledger(tmp_path / "ledger.md"))
    for name, value in (("DQ_QPU_MAX_JOBS", jobs), ("DQ_QPU_MAX_SHOTS", shots)):
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(hw.QPUGuardError, match=re.escape(match)):
        submit(rd)


def test_the_timeslot_queue_and_unknown_options_are_refused(no_pulla):
    with pytest.raises(ValueError, match="use_timeslot"):
        submit(run_definition(), use_timeslot=True)
    with pytest.raises(ValueError, match="unsupported option"):
        submit(run_definition(), shots=10)
    with pytest.raises(ValueError, match="calibration_set_id is required"):
        submit(run_definition(), calibration_set_id="")
    with pytest.raises(ValueError, match="purpose is required"):
        submit(run_definition(), purpose=" ")


def test_submit_playlist_writes_the_ledger_first_then_the_raw_results(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    rd = run_definition(segments=2, shots=1000)
    shape = hw.playlist_shape(rd)
    results = sweep_results(2, 1000)
    fake, calls = fake_pulla(monkeypatch, ledger, results)

    sub = submit(rd, data_root=tmp_path / "hw")

    assert calls == [("garnet", False)]
    ((sent, context, use_timeslot),) = fake.submitted
    assert sent is rd and context == {"k": "v"} and use_timeslot is False
    (row,) = hw.read_ledger(ledger)
    tag = f"[pulse,playlist={shape.playlist_sha256[:8]}]"
    assert (row.task, row.device, row.calibration_set_id, row.qubits) == ("pulsetask", "garnet", CALSET, "QB2,QB10")
    assert (row.circuits, row.shots, row.job_id, row.purpose) == (2, 1000, JOB_ID, f"rabi pilot {tag}")
    assert (sub.job_id, sub.status, sub.purpose, sub.shape) == (JOB_ID, "completed", row.purpose, shape)

    assert sub.path == tmp_path / "hw" / f"{row.time[:10].replace('-', '')}-pulsetask" / JOB_ID
    meta = json.loads((sub.path / "meta.json").read_text())
    assert meta["kind"] == "playlist" and meta["playlist_sha256"] == shape.playlist_sha256
    assert (meta["job_id"], meta["task"], meta["device"], meta["calibration_set_id"]) == (JOB_ID, "pulsetask", "garnet", CALSET)
    assert (meta["segments"], meta["spots"], meta["circuits"], meta["shots"], meta["averaging_bins"]) == (2, 1, 2, 1000, 1000)
    assert meta["physical_qubits"] == ["QB2", "QB10"] and meta["purpose"] == row.purpose
    assert (meta["status"], meta["success"], meta["timeline"]) == ("completed", True, TIMELINE)
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", meta["commit"])
    with np.load(sub.path / "sweep_results.npz") as npz:
        for entry in meta["arrays"]:
            saved = npz[entry["key"]]
            assert np.array_equal(saved, results[entry["return_parameter"]][entry["spot"]])
            assert (list(saved.shape), str(saved.dtype)) == (entry["shape"], entry["dtype"])
    assert {e["return_parameter"] for e in meta["arrays"]} == set(results)

    # the ledger now holds 2000 of the 2000 shots: one more job is refused before any connection
    with pytest.raises(hw.QPUGuardError, match="2000 shots used"):
        submit(rd, data_root=tmp_path / "hw")
    assert len(fake.submitted) == 1 and calls == [("garnet", False)]


def test_a_failed_job_is_still_recorded(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    fake_pulla(monkeypatch, ledger, results=None, status=JobStatus.FAILED)
    sub = submit(run_definition(), data_root=tmp_path / "hw")
    assert sub.status == "failed" and len(hw.read_ledger(ledger)) == 1  # a failed job still counts
    meta = json.loads((sub.path / "meta.json").read_text())
    assert (meta["status"], meta["success"], meta["arrays"]) == ("failed", False, [])
    assert not (sub.path / "sweep_results.npz").exists()


def test_a_re_pull_saves_as_the_fresh_job_did(monkeypatch, tmp_path):
    """retrieve_playlist is read-only (GETs on a read-only server client), and its save matches the fresh one."""
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)
    rd = run_definition()
    results = sweep_results(2, 1000)
    fake_pulla(monkeypatch, ledger, results)
    fresh = submit(rd, data_root=tmp_path / "fresh")

    pulled = []

    def get_job(self, job_id):
        pulled.append(("job", str(job_id), self.quantum_computer))
        return SimpleNamespace(status=JobStatus.COMPLETED, timeline=TIMELINE)

    def get_results(self, job_id):
        pulled.append(("results", str(job_id)))
        return results

    def send(self, http_method, url_path, **kwargs):
        assert http_method.__name__ == "get" and url_path == "quantum-computers", (http_method, url_path)
        return SimpleNamespace(text=json.dumps({"quantum_computers": [
            {"id": str(UUID(int=1)), "alias": "garnet", "display_name": "Garnet"},
            {"id": str(UUID(int=2)), "alias": "emerald", "display_name": "Emerald"}]}), status_code=200)

    monkeypatch.setattr(_IQMServerClient, "_send_request", send)
    monkeypatch.setattr(_IQMServerClient, "get_job", get_job)
    monkeypatch.setattr(_IQMServerClient, "get_job_artifact_sweep_results", get_results)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    data, again = hw.retrieve_playlist(JOB_ID, "garnet")
    assert pulled == [("job", JOB_ID, "garnet"), ("results", JOB_ID)]
    day = fresh.path.parent.name.split("-")[0]
    path = hw.save_playlist_raw(again, job_id=JOB_ID, task="pulsetask", device="garnet", calibration_set_id=CALSET,
                                shape=hw.playlist_shape(rd), status=data.status, timeline=data.timeline,
                                purpose=fresh.purpose, day=day, data_root=tmp_path / "repull")
    a, b = (json.loads((p / "meta.json").read_text()) for p in (fresh.path, path))
    assert {k for k in a if a[k] != b[k]} <= {"saved_utc"}
    with np.load(fresh.path / "sweep_results.npz") as x, np.load(path / "sweep_results.npz") as y:
        assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
    with pytest.raises(FileExistsError):
        hw.save_playlist_raw(again, job_id=JOB_ID, task="pulsetask", device="garnet", calibration_set_id=CALSET,
                             shape=hw.playlist_shape(rd), status=data.status, day=day, data_root=tmp_path / "repull")


def test_the_read_only_pulla_cannot_submit(monkeypatch):
    """hw.pulla(device): a real Pulla class and _IQMServerClient, with only the constructor's downloads and
    the HTTP layer stood in. Every submit route refuses before the HTTP layer."""
    calls = []

    def send(self, http_method, url_path, **kwargs):
        calls.append((http_method.__name__.upper(), url_path))
        if http_method.__name__ != "get":
            raise AssertionError(f"{http_method.__name__.upper()} {url_path} reached the HTTP layer")
        return SimpleNamespace(text=json.dumps({"quantum_computers": [
            {"id": str(UUID(int=1)), "alias": "garnet", "display_name": "Garnet"},
            {"id": str(UUID(int=2)), "alias": "emerald", "display_name": "Emerald"}]}), status_code=200)

    class OfflinePulla(Pulla):
        def __init__(self, iqm_server_url=None, *, quantum_computer=None, **kwargs):
            # Pulla's own constructor also downloads the settings, chip design record and chip label
            self._iqm_server_client = _IQMServerClient(iqm_server_url, quantum_computer=quantum_computer)

    monkeypatch.setattr(_IQMServerClient, "_send_request", send)
    monkeypatch.setattr("iqm.pulla.pulla.Pulla", OfflinePulla)
    monkeypatch.setattr(hw, "_load_token", lambda: None)
    p = hw.pulla("garnet")
    rd = run_definition()
    server = p._iqm_server_client
    routes = {
        "pulla.submit_playlist": lambda: p.submit_playlist(rd, context={}),
        "Pulla.submit_playlist, unbound": lambda: Pulla.submit_playlist(p, rd, context={}),
        "server.submit_run": lambda: server.submit_run(rd),
        "_IQMServerClient.submit_run, unbound": lambda: _IQMServerClient.submit_run(server, rd),
        "server._submit_job": lambda: server._submit_job(job_type="run"),
        "server.cancel_job": lambda: server.cancel_job(JOB_ID),
    }
    for label, route in routes.items():
        with pytest.raises(hw.QPUGuardError, match="read-only"):
            route()
        assert {method for method, _ in calls} == {"GET"}, label
    assert calls == [("GET", "quantum-computers")]


def test_a_submit_call_that_raises_leaves_one_fail_closed_row(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger, jobs="1")
    rd = run_definition(segments=2, shots=1000)
    tag = f"[pulse,playlist={hw.playlist_shape(rd).playlist_sha256[:8]}]"
    # Pulla.submit_playlist deep-copies the context after the POST (pulla.py), for one
    fake, calls = fake_pulla(monkeypatch, ledger, None, submit_raises=TypeError("cannot pickle '_thread.lock' object"))
    with pytest.raises(hw.JobMayExistError, match="may exist on Resonance") as raised:
        submit(rd, data_root=tmp_path / "hw")
    assert isinstance(raised.value.__cause__, TypeError)

    # the row a completed submission writes (test_submit_playlist_writes_the_ledger_first_...), with the placeholder id
    (row,) = hw.read_ledger(ledger)
    assert (row.task, row.device, row.calibration_set_id, row.qubits) == ("pulsetask", "garnet", CALSET, "QB2,QB10")
    assert (row.circuits, row.shots) == (2, 1000) and hw.usage("pulsetask", ledger) == (1, 2000)
    assert re.fullmatch(r"UNKNOWN-\d{8}T\d{6}Z", row.job_id) and row.job_id == raised.value.ledger_job_id
    assert row.purpose == f"rabi pilot {tag} [no job id: submit raised TypeError]"
    assert not (tmp_path / "hw").exists()

    with pytest.raises(hw.QPUGuardError, match=re.escape("1 job(s) already in the ledger")):
        submit(rd, data_root=tmp_path / "hw")
    with pytest.raises(ValueError, match="is a fail-closed ledger row, not a job id"):
        hw.retrieve_playlist(row.job_id, "garnet")
    assert calls == [("garnet", False)] and len(fake.submitted) == 1 and len(hw.read_ledger(ledger)) == 1


def test_no_row_when_connecting_fails(monkeypatch, tmp_path):
    ledger = write_ledger(tmp_path / "ledger.md")
    arm(monkeypatch, ledger)

    def connect(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(hw, "_connect_pulla", connect)
    with pytest.raises(requests.exceptions.ConnectionError):
        submit(run_definition())
    assert hw.read_ledger(ledger) == []
