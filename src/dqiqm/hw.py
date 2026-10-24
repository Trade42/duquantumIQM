"""The one guarded path to IQM hardware (IQM Resonance).

This is the only module in the repo that may build a backend able to submit a job to
Resonance. Everything else transpiles against :func:`backend` (read-only: its ``run``
refuses) and submits through :func:`submit`.

Submitting
    :func:`submit` refuses, before any network call, unless the process is armed:

    * ``DQ_QPU=run`` and ``DQ_QPU_TASK=<task>``;
    * ``DQ_QPU_MAX_JOBS`` and ``DQ_QPU_MAX_SHOTS``: the task's allowance. Shots are
      total shots, circuits x shots summed over the task's jobs. A batch is one job.
      Unset means zero.

    Usage is counted from the task's earlier lines in the QPU ledger (``DQ_QPU_LEDGER``,
    default :data:`DEFAULT_LEDGER`). The run request is built once, and the ledger's
    physical qubits are read from that same request, which is then submitted. The moment
    the submission returns a job id, its ledger line is appended, before anything else.
    Then the raw result is saved under ``data/hw/<YYYYMMDD>-<task>/<job id>/``
    (:func:`save_raw`).

Reading
    :func:`backend`, :func:`retrieve`, :func:`list_devices`, :func:`architecture`,
    :func:`calibration_set_id` and :func:`calibration_fidelities` need the token but no
    arming.

The token is ``IQM_TOKEN``, from the environment or the repo's ``.env`` (loaded with
python-dotenv on first connection, never at import). This module never prints it.

Qrisp: no module builds a qrisp ``IQMBackend`` yet. Its ``batched_compute_LER`` drops
the job ids; see the repo's CLAUDE.md before adding one, and add it here.

Importing this module makes no network call and imports nothing from ``iqm``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SERVER_URL = "https://resonance.iqm.tech"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "hw"
DEFAULT_LEDGER = "/home/sal/duquantumIQM/orchestration/qpu-ledger.md"

LEDGER_COLUMNS = (
    "UTC time",
    "task",
    "device",
    "calibration_set_id",
    "physical qubits",
    "circuits x shots",
    "job id",
    "purpose",
)

_TASK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CIRCUITS_X_SHOTS_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")


class QPUGuardError(RuntimeError):
    """A hardware submission was refused: not armed, over the allowance, or no ledger."""


# ---------------------------------------------------------------------------
# Arming and allowance: pure functions of the environment and the ledger file.
# ---------------------------------------------------------------------------


def armed_task(env: Mapping[str, str] | None = None) -> str:
    """Return the armed task name, or raise :class:`QPUGuardError`."""
    env = os.environ if env is None else env
    if env.get("DQ_QPU") != "run":
        raise QPUGuardError("not armed: set DQ_QPU=run and DQ_QPU_TASK=<task> (needs a QPU allowance in the brief)")
    task = env.get("DQ_QPU_TASK", "").strip()
    if not _TASK_RE.fullmatch(task):
        raise QPUGuardError(f"not armed: DQ_QPU_TASK must be a task name, got {task!r}")
    return task


def ledger_path(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("DQ_QPU_LEDGER") or DEFAULT_LEDGER)


@dataclass(frozen=True)
class LedgerRow:
    time: str
    task: str
    device: str
    calibration_set_id: str
    qubits: str
    circuits: int
    shots: int
    job_id: str
    purpose: str

    @property
    def total_shots(self) -> int:
        return self.circuits * self.shots


def read_ledger(path: Path) -> list[LedgerRow]:
    """Parse the ledger's table rows. A row that cannot be parsed raises: the guard fails closed."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if tuple(cells) == LEDGER_COLUMNS or all(set(c) <= set("-: ") for c in cells):
            continue
        m = _CIRCUITS_X_SHOTS_RE.match(cells[5]) if len(cells) == len(LEDGER_COLUMNS) else None
        if m is None:
            raise QPUGuardError(f"{path}:{n}: cannot parse ledger row {line!r}")
        rows.append(
            LedgerRow(
                time=cells[0],
                task=cells[1],
                device=cells[2],
                calibration_set_id=cells[3],
                qubits=cells[4],
                circuits=int(m.group(1)),
                shots=int(m.group(2)),
                job_id=cells[6],
                purpose=cells[7],
            )
        )
    return rows


def usage(task: str, path: Path) -> tuple[int, int]:
    """(jobs, total shots) the ledger records for ``task``. Correction lines count too."""
    rows = [r for r in read_ledger(path) if r.task == task]
    return len(rows), sum(r.total_shots for r in rows)


def _env_int(env: Mapping[str, str], name: str) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        raise QPUGuardError(f"{name} must be an integer, got {raw!r}") from None
    return max(value, 0)


def check_allowance(task: str, circuits: int, shots: int, env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`QPUGuardError` unless one more job of ``circuits`` x ``shots`` fits the allowance."""
    env = os.environ if env is None else env
    path = ledger_path(env)
    if not path.is_file() or not os.access(path, os.W_OK):
        raise QPUGuardError(f"ledger {path} is missing or not writable; refusing to submit")
    max_jobs = _env_int(env, "DQ_QPU_MAX_JOBS")
    max_shots = _env_int(env, "DQ_QPU_MAX_SHOTS")
    used_jobs, used_shots = usage(task, path)
    if used_jobs + 1 > max_jobs:
        raise QPUGuardError(f"{task}: {used_jobs} job(s) already in the ledger; DQ_QPU_MAX_JOBS={max_jobs}")
    if used_shots + circuits * shots > max_shots:
        raise QPUGuardError(
            f"{task}: {used_shots} shots used + {circuits} x {shots} requested > DQ_QPU_MAX_SHOTS={max_shots}"
        )


def _cell(text: Any) -> str:
    return " ".join(str(text).replace("|", "/").split())


def ledger_line(
    *,
    task: str,
    device: str,
    calibration_set_id: str,
    qubits: Sequence[str],
    circuits: int,
    shots: int,
    job_id: str,
    purpose: str,
    time: dt.datetime | None = None,
) -> str:
    """One ledger row, in the column order of the ledger's header."""
    time = time or dt.datetime.now(dt.UTC)
    cells = (
        time.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        task,
        device,
        calibration_set_id,
        ",".join(qubits),
        f"{circuits} x {shots}",
        job_id,
        purpose,
    )
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def append_ledger(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Raw data.
# ---------------------------------------------------------------------------


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def _memory_bits(memory: Sequence[str]) -> np.ndarray:
    """Qiskit memory strings -> uint8 array (shots, clbits); column j is circuit clbit j."""
    rows = [s.replace(" ", "")[::-1] for s in memory]
    if not rows:
        return np.zeros((0, 0), dtype=np.uint8)
    return (np.array([list(r) for r in rows]) == "1").astype(np.uint8)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_raw(
    result: Any,
    *,
    task: str,
    device: str,
    qubit_mapping: Sequence[Mapping[int, str]],
    shots: int,
    physical_qubits: Sequence[Sequence[str]] | None = None,
    purpose: str = "",
    day: str | None = None,
    data_root: Path | None = None,
) -> Path:
    """Save a qiskit ``Result`` from IQM under ``<data_root>/<YYYYMMDD>-<task>/<job id>/``.

    Writes ``meta.json`` and, when the job succeeded, ``memory.npz``: one uint8 array
    ``c<i>`` of shape (shots, clbits) per circuit, column j being circuit clbit j. Works on
    a fresh result or on one re-pulled with :func:`retrieve`. ``qubit_mapping`` has one
    logical qubit -> device qubit map per circuit, in batch order; ``physical_qubits``, when
    known, the device qubits each circuit's instructions were sent to. ``day`` (YYYYMMDD,
    UTC) is the submission day; it defaults to today. Refuses to overwrite.
    """
    job_id = str(result.job_id)
    day = day or dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    out = Path(data_root or DATA_ROOT) / f"{day}-{task}" / job_id
    out.mkdir(parents=True, exist_ok=False)

    meta_extra = getattr(result, "_metadata", {}) or {}
    experiments = list(getattr(result, "results", []) or [])
    calsets = sorted({str(e.calibration_set_id) for e in experiments if getattr(e, "calibration_set_id", None)})
    success = bool(getattr(result, "success", False))
    if success:
        arrays = {f"c{i}": _memory_bits(result.get_memory(i)) for i in range(len(experiments))}
        np.savez_compressed(out / "memory.npz", **arrays)

    meta = {
        "job_id": job_id,
        "task": task,
        "device": device,
        "calibration_set_id": calsets[0] if len(calsets) == 1 else calsets,
        "qubit_mapping": [{str(k): v for k, v in sorted(m.items())} for m in qubit_mapping],
        "physical_qubits": None if physical_qubits is None else [list(q) for q in physical_qubits],
        "shots": shots,
        "circuits": len(experiments),
        "circuit_names": [getattr(getattr(e, "header", None), "name", None) for e in experiments],
        "success": success,
        "commit": git_head(),
        "purpose": purpose,
        "saved_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeline": _jsonable(meta_extra.get("timeline", [])),
        "memory": "memory.npz: c<i> uint8 (shots, clbits), column j = clbit j" if success else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


# ---------------------------------------------------------------------------
# Connecting. _connect is the single seam to the network; tests replace it.
# ---------------------------------------------------------------------------


def _load_token() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")  # keeps an IQM_TOKEN already in the environment


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise QPUGuardError("read-only connection: submit through dqiqm.hw.submit()")


def _connect(device: str, calibration_set_id: str | None = None, *, read_only: bool) -> Any:
    _load_token()
    from iqm.iqm_client import IQMClient
    from iqm.qiskit_iqm.iqm_provider import IQMBackend

    client = IQMClient(SERVER_URL, quantum_computer=device)
    if not read_only:
        from iqm.pulse import Circuit
        from iqm.qiskit_iqm.iqm_job import IQMJob

        class SubmittingIQMBackend(IQMBackend):
            def submit_run_request(self, run_request: Any) -> Any:
                """The second half of ``IQMBackend.run``: submit this very request (on-demand queue)."""
                job = IQMJob(self, self.client.submit_run_request(run_request, use_timeslot=False))
                job.circuit_metadata = [c.metadata if isinstance(c, Circuit) else {} for c in run_request.circuits]
                return job

        return SubmittingIQMBackend(client, calibration_set_id=calibration_set_id)

    class ReadOnlyIQMBackend(IQMBackend):
        run = _refuse

    client.submit_circuits = _refuse
    client.submit_run_request = _refuse
    return ReadOnlyIQMBackend(client, calibration_set_id=calibration_set_id)


def _qubit_sort_key(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(\D*)(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def _sent_qubits(run_request: Any) -> list[list[str]]:
    """Per circuit of a run request: the device qubits its serialized instructions act on."""
    return [
        sorted({q for inst in c.instructions for q in inst.locus}, key=_qubit_sort_key) for c in run_request.circuits
    ]


def _qubit_mappings(backend: Any, circuits: Sequence[Any], names: Mapping[int, str] | None) -> list[dict[int, str]]:
    """Per circuit: logical qubit -> device qubit name, reading the layout as IQM's serializer does.

    A transpiled circuit's layout places logical qubit v on physical qubit
    ``initial_index_layout(filter_ancillas=True)[v]``. A circuit with no layout (or one the
    serializer ignores) is sent qubit for qubit, so its qubits map through their own indices.
    """
    name = names.__getitem__ if names is not None else backend.index_to_qubit_name
    mappings = []
    for c in circuits:
        # the test serialize_instructions (iqm.qiskit_iqm.qiskit_to_iqm) uses to ignore a layout
        if c.layout is None or c.layout.initial_layout.get_registers() != set(c.qregs):
            used = sorted({c.find_bit(q).index for inst in c.data for q in inst.qubits})
            mappings.append({i: name(i) for i in used})
        else:
            mappings.append({v: name(p) for v, p in enumerate(c.layout.initial_index_layout(filter_ancillas=True))})
    return mappings


@dataclass
class Submission:
    job_id: str
    task: str
    device: str
    calibration_set_id: str
    qubits: list[str]
    qubit_mapping: list[dict[int, str]]
    circuits: int
    shots: int
    path: Path
    result: Any


def submit(
    circuits: Any,
    *,
    device: str,
    shots: int,
    purpose: str,
    calibration_set_id: str | None = None,
    qubit_index_to_name: Mapping[int, str] | None = None,
    data_root: Path | None = None,
    timeout: float | None = None,
    **run_options: Any,
) -> Submission:
    """Submit already-transpiled qiskit circuit(s) to ``device`` as one job, then save the result.

    Pass the ``calibration_set_id`` of the read-only :func:`backend` you transpiled
    against, so the job runs on the same calibration. ``run_options`` go to
    ``IQMBackend.create_run_request`` (e.g. ``circuit_compilation_options``). The job goes
    to the on-demand queue.

    If the wait for the result fails, the job is already in the ledger: re-pull it with
    :func:`retrieve` and save it with :func:`save_raw`.
    """
    circuit_list = list(circuits) if isinstance(circuits, (list, tuple)) else [circuits]
    if not circuit_list:
        raise ValueError("no circuits to submit")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError(f"shots must be a positive int, got {shots!r}")
    if not _cell(purpose):
        raise ValueError("purpose is required: it goes in the ledger")
    if "use_timeslot" in run_options:
        raise ValueError("use_timeslot: submit uses the on-demand queue only")

    task = armed_task()
    check_allowance(task, len(circuit_list), shots)
    ledger = ledger_path()

    hw_backend = _connect(device, calibration_set_id, read_only=False)
    calset = str(hw_backend.architecture.calibration_set_id)
    # Build the request once: the ledger's qubits are read from the request that is submitted.
    run_request = hw_backend.create_run_request(
        circuit_list, shots=shots, qubit_index_to_name=qubit_index_to_name, **run_options
    )
    sent = _sent_qubits(run_request)
    qubits = sorted({q for circuit_qubits in sent for q in circuit_qubits}, key=_qubit_sort_key)
    mappings = _qubit_mappings(hw_backend, circuit_list, qubit_index_to_name)

    job = hw_backend.submit_run_request(run_request)
    job_id = str(job.job_id())
    submitted = dt.datetime.now(dt.UTC)
    append_ledger(
        ledger,
        ledger_line(
            task=task,
            device=device,
            calibration_set_id=calset,
            qubits=qubits,
            circuits=len(circuit_list),
            shots=shots,
            job_id=job_id,
            purpose=purpose,
            time=submitted,
        ),
    )

    result = job.result() if timeout is None else job.result(timeout=timeout)
    path = save_raw(
        result,
        task=task,
        device=device,
        qubit_mapping=mappings,
        shots=shots,
        physical_qubits=sent,
        purpose=purpose,
        day=submitted.strftime("%Y%m%d"),
        data_root=data_root,
    )
    return Submission(job_id, task, device, calset, qubits, mappings, len(circuit_list), shots, path, result)


# ---------------------------------------------------------------------------
# Read-only helpers: need the token, not arming.
# ---------------------------------------------------------------------------


def backend(device: str, calibration_set_id: str | None = None) -> Any:
    """A read-only qiskit ``IQMBackend`` for transpiling and inspection. Its ``run`` refuses."""
    return _connect(device, calibration_set_id, read_only=True)


def retrieve(job_id: str, device: str, *, timeout: float | None = None) -> Any:
    """Re-pull a job's qiskit ``Result`` by id (waits if it is still running). Submits nothing."""
    job = _connect(device, read_only=True).retrieve_job(job_id)
    return job.result() if timeout is None else job.result(timeout=timeout)


def list_devices() -> list[dict[str, Any]]:
    """The quantum computers the server offers to this token."""
    _load_token()
    from iqm.iqm_client import IQMClient

    # IQMClient 35 has no public device list; this private call is the only route.
    client = IQMClient(SERVER_URL)
    return [qc.model_dump(mode="json") for qc in client._iqm_server_client.get_quantum_computers()]


def architecture(device: str, calibration_set_id: str | None = None) -> dict[str, Any]:
    """The dynamic quantum architecture (qubits, couplers, gates) for a calibration set."""
    return backend(device, calibration_set_id).architecture.model_dump(mode="json")


def calibration_set_id(device: str) -> str:
    """The server's current default calibration set id for ``device``."""
    return str(backend(device).architecture.calibration_set_id)


def calibration_fidelities(device: str, calibration_set_id: str | None = None) -> dict[str, Any]:
    """Per-qubit and per-pair calibration figures of one calibration set, parsed by iqm-qubit-selector.

    The figures are that calibration set's quality metrics (the current default set's, when
    ``calibration_set_id`` is None). ``calibration_set_id`` and ``quality_metric_set_id`` in
    the result are the server's record of which set they describe; the other keys are
    iqm-qubit-selector's (``CalibrationDataManager.get_calibration_fidelities``).
    """
    from iqm.qubit_selector.qubit_selector import CalibrationDataManager

    b = backend(device, calibration_set_id)
    # get_calibration_fidelities always reads the default set's metrics, so fetch this set's
    # ourselves and use its parser (private in iqm-qubit-selector 1.1.3; there is no public one).
    metrics = b.client.get_quality_metric_set(b.architecture.calibration_set_id)
    figures = CalibrationDataManager()._parse_calibration_metrics(metrics.observations, b)
    return {
        "calibration_set_id": str(metrics.describes_id),
        "quality_metric_set_id": str(metrics.observation_set_id),
        **figures,
    }
