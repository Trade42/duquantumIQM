"""The one guarded path to IQM hardware (IQM Resonance).

This is the only module in the repo that may build a backend able to submit a job to
Resonance. Everything else transpiles against :func:`backend` (read-only: its ``run``
refuses) and submits through :func:`submit`. See ``docs/hw.md``.

Submitting
    :func:`submit` refuses, before any network call, unless the process is armed:

    * ``DQ_QPU=run`` and ``DQ_QPU_TASK=<task>``;
    * ``DQ_QPU_MAX_JOBS`` and ``DQ_QPU_MAX_SHOTS``: the task's allowance. Shots are
      total shots, circuits x shots summed over the task's jobs. A batch is one job.
      Unset means zero.

    It also refuses, before any network call, a circuit whose serialized ops fall outside
    :data:`NATIVE_OPS`, and it checks the built run request again before sending it.

    Usage is counted from the task's earlier lines in the QPU ledger (``DQ_QPU_LEDGER``,
    default :data:`DEFAULT_LEDGER`). The run request is built once, and the ledger's
    physical qubits are read from that same request, which is then submitted. The moment
    the submission returns a job id, its ledger line is appended, before anything else.
    One exclusive lock on the ledger (:func:`ledger_lock`) spans a last allowance check,
    the submission and the append, so two submissions for one task cannot both pass; it
    is released before the wait for results. If the submit call itself raises, the job may
    still exist, so a fail-closed row with an ``UNKNOWN-<UTC stamp>`` job id is appended
    before the error is re-raised (:class:`JobMayExistError`); it counts against the
    allowance like any row. Then the raw result is saved under
    ``data/hw/<YYYYMMDD>-<task>/<job id>/`` (:func:`save_raw`), with the compilation
    options the request carried and each circuit's position, name and ``native_sha256``.
    A non-default compilation option tags the ledger purpose, e.g. ``[dd=ENABLED]``.

Native circuits
    :func:`submit_native` submits ``iqm.pulse.Circuit`` lists (``prx_12``, ``measure_012``) under
    the same guard, and :func:`save_native_raw` saves IQM's per-key result arrays. **Not to be
    used until reviewed and merged; C3 runs only after a pilot.**

Pulse level
    :func:`submit_playlist` submits a Pulla run definition under the same arming,
    allowance, ledger and lock, and :func:`save_playlist_raw` saves its raw sweep results.
    **Not to be used until Sal and the mentors confirm pulse-level submission rights.**
    :func:`pulla` is a read-only Pulla, for compiling.

Reading
    :func:`backend`, :func:`retrieve`, :func:`retrieve_native`, :func:`retrieve_playlist`, :func:`list_devices`,
    :func:`architecture`, :func:`calibration_set_id`, :func:`calibration_fidelities` and
    :func:`pulla` need the token but no arming.

The token is ``IQM_TOKEN``, from the environment or the repo's ``.env`` (loaded with
python-dotenv on first connection, never at import). This module never prints it.

Qrisp: no module builds a qrisp ``IQMBackend`` yet. Its ``batched_compute_LER`` drops
the job ids; see the repo's CLAUDE.md before adding one, and add it here.

Importing this module makes no network call and imports nothing from ``iqm``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import enum
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

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

NATIVE_OPS = frozenset({"prx", "cz", "measure", "barrier", "delay"})
"""The serialized ops a circuit may send (qiskit's ``r`` serializes as ``prx``). Any other op is refused."""

NATIVE_JOB_OPS = NATIVE_OPS | {"prx_12", "measure_012"}
"""What :func:`submit_native` may send: :data:`NATIVE_OPS`, the 1-2 transition ``prx_12`` and the
three-level ``measure_012`` (iqm-client 35.0.3's own validation still refuses ``measure_012``)."""

UNKNOWN_JOB_PREFIX = "UNKNOWN-"
"""The job id cell of a fail-closed ledger row, ``UNKNOWN-<UTC stamp>``: the submit call raised before a job id
came back. A real job id is a UUID, so the two cannot be confused."""

_TASK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CIRCUITS_X_SHOTS_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")


class QPUGuardError(RuntimeError):
    """A hardware submission was refused: not armed, over the allowance, no ledger, or not what may be sent."""


class JobMayExistError(RuntimeError):
    """The library's submit call raised, so no job id came back, but the job may exist on Resonance.

    Not a refusal: the request may have reached the server. A fail-closed ledger row, job id
    :attr:`ledger_job_id` (``UNKNOWN-<UTC stamp>``), already counts it against the allowance.
    Check Resonance's job list, read-only, before any retry (``docs/hw.md``).
    """

    def __init__(self, message: str, *, ledger_job_id: str) -> None:
        super().__init__(message)
        self.ledger_job_id = ledger_job_id


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


@contextlib.contextmanager
def ledger_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on the ledger file; a second holder waits until it is released.

    The ledger is opened read-only for the lock, so a missing ledger refuses instead of being
    created. Each call opens the file afresh: ``flock`` locks belong to the open file, so two
    calls conflict even within one process.
    """
    try:
        f = open(path, "rb")  # noqa: SIM115 -- closed by the with below; a missing ledger must refuse, not be created
    except OSError as e:
        raise QPUGuardError(f"ledger {path} cannot be opened to lock it; refusing to submit") from e
    with f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _tagged(purpose: str, tag: str) -> str:
    return f"{purpose} {tag}" if tag else purpose


def _fail_closed(ledger: Path, error: BaseException, *, purpose: str, **row: Any) -> NoReturn:
    """The library's submit call raised ``error``: append a fail-closed ledger row, then re-raise.

    Called inside the ledger lock, after the last allowance check. The server may have accepted
    the job before the reply was lost (a read timeout, a response the client can't parse), so
    the row is the job's as far as is known: the same task, device, calibration set, qubits and
    ``N x M``, job id ``UNKNOWN-<UTC stamp>``, purpose tagged with the error's type. It counts
    against the allowance, so a retry with the same allowance is refused. An ``Exception`` is
    re-raised as :class:`JobMayExistError`, from it; anything else (``KeyboardInterrupt``) as
    itself, with a note. If the row can't be appended, that error propagates instead.
    """
    now = dt.datetime.now(dt.UTC)
    unknown = f"{UNKNOWN_JOB_PREFIX}{now:%Y%m%dT%H%M%SZ}"
    tag = f"[no job id: submit raised {type(error).__name__}]"
    append_ledger(ledger, ledger_line(job_id=unknown, purpose=_tagged(purpose, tag), time=now, **row))
    message = (
        f"{row['task']}: the submit call raised {type(error).__name__} before a job id came back, but the job may "
        f"exist on Resonance. Ledger row {unknown} counts it against the allowance. Check Resonance's job list, "
        "read-only, before any retry (docs/hw.md, 'When the submit call raises')."
    )
    if not isinstance(error, Exception):
        error.add_note(message)
        raise error
    raise JobMayExistError(message, ledger_job_id=unknown) from error


def _refuse_unknown_job_id(job_id: Any) -> None:
    if str(job_id).strip().upper().startswith(UNKNOWN_JOB_PREFIX.rstrip("-")):
        raise ValueError(
            f"{job_id} is a fail-closed ledger row, not a job id: its submit call raised before a job id came back. "
            "Find the job, if there is one, in Resonance's job list, then re-pull it by its real id."
        )


# ---------------------------------------------------------------------------
# What a run request sends: ops, compilation options, circuit hashes.
# ---------------------------------------------------------------------------


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def instructions_sha256(instructions: Sequence[Any]) -> str:
    """sha256 of serialized instructions: JSON of each one's name, locus and args, keys sorted.

    The same hash as ``dqiqm.experiments.circuits.native_sha256`` and InfoRecord's
    ``placement.json`` ``native_sha256``, here taken of what a run request actually sends.
    """
    return _json_sha256(
        [
            {
                "name": i.name,
                "locus": list(i.locus),
                "args": {k: (float(v) if isinstance(v, float) else v) for k, v in i.args.items()},
            }
            for i in instructions
        ]
    )


def _refuse_off_allowlist(
    position: int, name: Any, instructions: Sequence[Any], where: str, allowed: frozenset[str] = NATIVE_OPS
) -> None:
    bad: dict[str, int] = {}
    for inst in instructions:
        if inst.name not in allowed:
            bad[inst.name] = bad.get(inst.name, 0) + 1
    if bad:
        listed = ", ".join(f"{op} x{n}" for op, n in sorted(bad.items()))
        raise QPUGuardError(
            f"op allowlist: circuit {position} ({name!r}) {where} {listed}; only {', '.join(sorted(allowed))} may be sent"
        )


def check_native_ops(circuits: Sequence[Any]) -> None:
    """Refuse (:class:`QPUGuardError`) a qiskit circuit whose serialized ops fall outside :data:`NATIVE_OPS`.

    Each circuit is serialized as ``IQMBackend`` serializes it (``serialize_instructions``),
    with placeholder qubit names: offline, no backend needed. A circuit the serializer cannot
    take (an op it doesn't know) is refused too.
    """
    from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions

    for i, c in enumerate(circuits):
        try:
            instructions = serialize_instructions(c, {q: f"q{q}" for q in range(c.num_qubits)})
        except (ValueError, KeyError) as e:
            raise QPUGuardError(f"op allowlist: circuit {i} ({c.name!r}) cannot be serialized: {e}") from e
        _refuse_off_allowlist(i, c.name, instructions, "has")


def _check_sent_ops(run_request: Any, allowed: frozenset[str] = NATIVE_OPS) -> None:
    for i, c in enumerate(run_request.circuits):
        _refuse_off_allowlist(i, getattr(c, "name", None), c.instructions, "would send", allowed)


# Ledger tag names for the CircuitCompilationOptions fields (iqm-client 35.0.3).
_OPTION_TAGS = {
    "max_circuit_duration_over_t2": "t2_limit",
    "heralding_mode": "herald",
    "move_gate_validation": "move_validation",
    "move_gate_frame_tracking": "move_tracking",
    "active_reset_cycles": "reset",
    "dd_mode": "dd",
    "dd_strategy": "dd_strategy",
}


def _option_value(value: Any) -> Any:
    """One compilation option as JSON: an enum's wire value; a DD strategy's fields, target qubits sorted."""
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "model_dump"):
        out = value.model_dump(mode="json")
        if isinstance(out.get("target_qubits"), list):
            out["target_qubits"] = sorted(out["target_qubits"], key=_qubit_sort_key)
        return out
    return value


def compilation_options_record(options: Any = None) -> dict[str, Any]:
    """Every ``CircuitCompilationOptions`` field of ``options`` (None: the defaults), as JSON: ``meta.json``'s
    ``compilation_options``. A ``CircuitJobParameters`` (a job's payload) gives the same fields."""
    from iqm.iqm_client.models import CircuitCompilationOptions

    if options is None:
        options = CircuitCompilationOptions()
    if not isinstance(options, CircuitCompilationOptions):
        raise TypeError(f"circuit_compilation_options must be a CircuitCompilationOptions, got {type(options).__name__}")
    return {f.name: _option_value(getattr(options, f.name)) for f in dataclasses.fields(CircuitCompilationOptions)}


def compilation_options_tag(record: Mapping[str, Any]) -> str:
    """The ledger purpose's suffix: each non-default option, e.g. ``[dd=ENABLED,herald=ZEROS]``.

    Empty for the defaults. Enum values are upper-cased; a DD strategy shows as the first 8 hex
    digits of the sha256 of its JSON (``meta.json`` holds the whole strategy).
    """
    defaults = compilation_options_record()
    parts = []
    for name, value in record.items():
        if name in defaults and value == defaults[name]:
            continue
        if name == "dd_strategy" and value is not None:
            shown = _json_sha256(value)[:8]
        elif isinstance(value, str):
            shown = value.upper()
        else:
            shown = value
        parts.append(f"{_OPTION_TAGS.get(name, name)}={shown}")
    return f"[{','.join(parts)}]" if parts else ""


def _check_request_options(run_request: Any, record: Mapping[str, Any]) -> None:
    """Refuse a run request whose own option fields differ from what ``meta.json`` will record."""
    for name, value in record.items():
        if hasattr(run_request, name) and _option_value(getattr(run_request, name)) != value:
            raise QPUGuardError(
                f"the run request's {name} is {_option_value(getattr(run_request, name))!r}, not {value!r}"
            )


def _circuit_order(
    run_request: Any, names: Sequence[str] | None, hashes: Sequence[str] | None
) -> list[dict[str, Any]]:
    """``meta.json``'s ``circuit_order``: each circuit's position in the batch, name and ``native_sha256``.

    IQM returns no circuit names (``circuit_names: [null]``), so the position is the link from
    ``memory.npz``'s ``c<i>`` to the circuit. The hash is always of the instructions sent; a
    caller's hash that differs refuses. The name is the caller's, else the serialized circuit's.
    """
    order = []
    for i, c in enumerate(run_request.circuits):
        sha = instructions_sha256(c.instructions)
        if hashes is not None and hashes[i] != sha:
            raise QPUGuardError(f"circuit {i}: native_sha256 {hashes[i]} given, but the run request's is {sha}")
        name = names[i] if names is not None else getattr(c, "name", None)
        order.append({"position": i, "name": name, "native_sha256": sha})
    return order


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


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_raw(
    result: Any,
    *,
    task: str,
    device: str,
    qubit_mapping: Sequence[Mapping[int, str]],
    shots: int,
    physical_qubits: Sequence[Sequence[str]] | None = None,
    measured_qubits: Sequence[Mapping[int, str]] | None = None,
    circuit_order: Sequence[Mapping[str, Any]] | None = None,
    compilation_options: Mapping[str, Any] | None = None,
    purpose: str = "",
    day: str | None = None,
    data_root: Path | None = None,
) -> Path:
    """Save a qiskit ``Result`` from IQM under ``<data_root>/<YYYYMMDD>-<task>/<job id>/``.

    Writes ``meta.json`` and, when the job succeeded, ``memory.npz``: one uint8 array
    ``c<i>`` of shape (shots, clbits) per circuit, column j being circuit clbit j. Works on
    a fresh result or on one re-pulled with :func:`retrieve`. ``qubit_mapping`` has one
    logical qubit -> device qubit map per circuit, in batch order: the initial placement.
    When known, ``physical_qubits`` has the device qubits each circuit's instructions were
    sent to, ``measured_qubits`` the device qubit each memory column was measured on,
    ``circuit_order`` each circuit's position, name and ``native_sha256``, and
    ``compilation_options`` every compilation option the request carried
    (:func:`compilation_options_record`); each unknown one is saved as null.
    ``day`` (YYYYMMDD, UTC) is the submission day; it defaults to today. Refuses to overwrite.
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
        "measured_qubits": None
        if measured_qubits is None
        else [{str(k): v for k, v in sorted(m.items())} for m in measured_qubits],
        "shots": shots,
        "circuits": len(experiments),
        "circuit_names": [getattr(getattr(e, "header", None), "name", None) for e in experiments],
        "circuit_order": None if circuit_order is None else [dict(c) for c in circuit_order],
        "compilation_options": None if compilation_options is None else dict(compilation_options),
        "success": success,
        "commit": git_head(),
        "purpose": purpose,
        "saved_utc": _utc_stamp(),
        "timeline": _jsonable(meta_extra.get("timeline", [])),
        "memory": "memory.npz: c<i> uint8 (shots, clbits), column j = clbit j" if success else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


# ---------------------------------------------------------------------------
# Connecting. _connect and _connect_pulla are the seams to the network; tests replace them.
# ---------------------------------------------------------------------------


def _load_token() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")  # keeps an IQM_TOKEN already in the environment


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise QPUGuardError("read-only connection: submit through dqiqm.hw.submit()")


# The _IQMServerClient calls that are not GETs (iqm-client 35.0.3, iqm_server_client.py):
# submit_sweep, submit_run and submit_circuits all post through _submit_job.
_SERVER_WRITES = ("submit_circuits", "submit_run", "submit_sweep", "_submit_job", "cancel_job", "delete_job")


def _read_only_server(server: Any) -> None:
    """Make an ``_IQMServerClient`` read-only: its writing calls refuse, and it sends GETs only."""
    for name in _SERVER_WRITES:
        setattr(server, name, _refuse)
    send = server._send_request

    def get_only(http_method: Any, url_path: str, **kwargs: Any) -> Any:
        method = getattr(http_method, "__name__", str(http_method)).upper()
        if method != "GET":
            raise QPUGuardError(f"read-only connection: refusing {method} {url_path}")
        return send(http_method, url_path, **kwargs)

    server._send_request = get_only


def _connect_client(device: str, *, read_only: bool) -> Any:
    """An ``IQMClient`` on ``device``. Read-only: its submitting calls and its server's writing calls refuse."""
    _load_token()
    from iqm.iqm_client import IQMClient

    client = IQMClient(SERVER_URL, quantum_computer=device)
    if read_only:
        client.submit_circuits = _refuse
        client.submit_run_request = _refuse
        # The server layer too, so the unbound IQMClient.submit_* and the private routes refuse as well.
        _read_only_server(client._iqm_server_client)
    return client


def _connect(device: str, calibration_set_id: str | None = None, *, read_only: bool) -> Any:
    from iqm.qiskit_iqm.iqm_provider import IQMBackend

    client = _connect_client(device, read_only=read_only)
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

    return ReadOnlyIQMBackend(client, calibration_set_id=calibration_set_id)


def _connect_pulla(device: str, *, read_only: bool) -> Any:
    """A Pulla on ``device``. Read-only: its ``submit_playlist`` and its server's writing calls refuse."""
    _load_token()
    from iqm.pulla.pulla import Pulla

    p = Pulla(SERVER_URL, quantum_computer=device)
    if read_only:
        p.submit_playlist = _refuse
        _read_only_server(p._iqm_server_client)
    return p


def _qubit_sort_key(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(\D*)(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def _sent_qubits(run_request: Any) -> list[list[str]]:
    """Per circuit of a run request: the device qubits its serialized instructions act on."""
    return [
        sorted({q for inst in c.instructions for q in inst.locus}, key=_qubit_sort_key) for c in run_request.circuits
    ]


def _measured_qubits(run_request: Any) -> list[dict[int, str]]:
    """Per circuit of a run request: memory column -> the device qubit measured into it.

    Read from each measurement instruction's locus and key, ``<creg>_<len>_<creg index>_<bit>``
    (iqm.qiskit_iqm's ``MeasurementKey``). ``IQMJob`` lays out each shot's bits register by
    register, in register order, leaving out a register with no measurement at all; so a
    clbit's column is its index in its register plus the lengths of the measured registers
    before it. That is the column of :func:`save_raw`'s ``memory.npz``.
    """
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    measured = []
    for c in run_request.circuits:
        keyed = []
        for inst in c.instructions:
            if inst.name in ("measure", "measurement"):  # the names iqm_client's validation treats as measurements
                if len(inst.locus) != 1:
                    raise ValueError(f"a measurement on {len(inst.locus)} qubits has no single column: {inst!r}")
                keyed.append((MeasurementKey.from_string(inst.args["key"]), inst.locus[0]))
        lengths = dict(sorted({mk.creg_idx: mk.creg_len for mk, _ in keyed}.items()))
        offsets = {idx: sum(list(lengths.values())[:i]) for i, idx in enumerate(lengths)}
        measured.append(dict(sorted((offsets[mk.creg_idx] + mk.clbit_idx, q) for mk, q in keyed)))
    return measured


def _qubit_mappings(backend: Any, circuits: Sequence[Any], names: Mapping[int, str] | None) -> list[dict[int, str]]:
    """Per circuit: logical qubit -> device qubit name, its initial placement, read as IQM's serializer does.

    A transpiled circuit's layout places logical qubit v on physical qubit
    ``initial_index_layout(filter_ancillas=True)[v]``. A circuit with no layout (or one the
    serializer ignores) is sent qubit for qubit, so its qubits map through their own indices.
    A routed circuit's SWAPs move its qubits afterwards: :func:`_measured_qubits` has where
    each clbit is measured.
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
    measured_qubits: list[dict[int, str]]
    circuit_order: list[dict[str, Any]]
    compilation_options: dict[str, Any]
    purpose: str
    circuits: int
    shots: int
    path: Path
    result: Any


def _names_or_none(label: str, given: Sequence[str] | None, n: int) -> list[str] | None:
    if given is None:
        return None
    if isinstance(given, str):
        raise TypeError(f"{label} must be a sequence with one entry per circuit, not a str")
    values = [str(v) for v in given]
    if len(values) != n:
        raise ValueError(f"{label} has {len(values)} entries for {n} circuit(s)")
    return values


def submit(
    circuits: Any,
    *,
    device: str,
    shots: int,
    purpose: str,
    calibration_set_id: str | None = None,
    qubit_index_to_name: Mapping[int, str] | None = None,
    circuit_compilation_options: Any = None,
    circuit_names: Sequence[str] | None = None,
    native_sha256: Sequence[str] | None = None,
    data_root: Path | None = None,
    timeout: float | None = None,
    **run_options: Any,
) -> Submission:
    """Submit already-transpiled qiskit circuit(s) to ``device`` as one job, then save the result.

    Pass the ``calibration_set_id`` of the read-only :func:`backend` you transpiled
    against, so the job runs on the same calibration. ``circuit_compilation_options`` (an
    ``iqm.iqm_client.models.CircuitCompilationOptions``; None means its defaults) goes to
    ``IQMBackend.create_run_request``; ``meta.json`` records every field of it, and each
    non-default one tags the ledger purpose. No other run option is accepted. The job goes
    to the on-demand queue.

    ``circuit_names`` and ``native_sha256``, when given, have one entry per circuit, in
    batch order; ``meta.json``'s ``circuit_order`` records them by position. A
    ``native_sha256`` that is not the hash of what the request sends refuses.

    Every check that needs no network runs first: the inputs, the op allowlist, the
    arming and the allowance. If the wait for the result fails, the job is already in the
    ledger: re-pull it with :func:`retrieve` and save it with :func:`save_raw`. If the submit
    call itself raises, a fail-closed ``UNKNOWN-<UTC stamp>`` row is appended and
    :class:`JobMayExistError` raised: check Resonance's job list before any retry.
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
    if run_options:
        raise ValueError(
            f"unsupported run option(s) {sorted(run_options)}: pass compilation options as circuit_compilation_options"
        )
    names = _names_or_none("circuit_names", circuit_names, len(circuit_list))
    hashes = _names_or_none("native_sha256", native_sha256, len(circuit_list))
    options = compilation_options_record(circuit_compilation_options)
    tagged = _tagged(purpose, compilation_options_tag(options))
    check_native_ops(circuit_list)

    task = armed_task()
    check_allowance(task, len(circuit_list), shots)  # before any connection
    ledger = ledger_path()

    hw_backend = _connect(device, calibration_set_id, read_only=False)
    calset = str(hw_backend.architecture.calibration_set_id)
    # Build the request once: the ledger's qubits are read from the request that is submitted.
    run_request = hw_backend.create_run_request(
        circuit_list,
        shots=shots,
        qubit_index_to_name=qubit_index_to_name,
        circuit_compilation_options=circuit_compilation_options,
    )
    _check_sent_ops(run_request)
    _check_request_options(run_request, options)
    order = _circuit_order(run_request, names, hashes)
    sent = _sent_qubits(run_request)
    measured = _measured_qubits(run_request)
    qubits = sorted({q for circuit_qubits in sent for q in circuit_qubits}, key=_qubit_sort_key)
    mappings = _qubit_mappings(hw_backend, circuit_list, qubit_index_to_name)

    # The allowance is checked again, then the job submitted and its line appended, under one
    # exclusive lock on the ledger: a concurrent submission for this task waits here, then
    # sees this job's line. The lock spans the submission (its job id is the line's), and is
    # released before the wait for results. If the submit call raises, the job may still exist:
    # _fail_closed appends its row, with an UNKNOWN job id, before re-raising.
    with ledger_lock(ledger):
        check_allowance(task, len(circuit_list), shots)
        try:
            job = hw_backend.submit_run_request(run_request)
            job_id = str(job.job_id())
        except BaseException as e:  # noqa: BLE001 -- the job may exist: recorded, then re-raised
            _fail_closed(ledger, e, task=task, device=device, calibration_set_id=calset, qubits=qubits,
                         circuits=len(circuit_list), shots=shots, purpose=tagged)
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
                purpose=tagged,
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
        measured_qubits=measured,
        circuit_order=order,
        compilation_options=options,
        purpose=tagged,
        day=submitted.strftime("%Y%m%d"),
        data_root=data_root,
    )
    return Submission(
        job_id=job_id,
        task=task,
        device=device,
        calibration_set_id=calset,
        qubits=qubits,
        qubit_mapping=mappings,
        measured_qubits=measured,
        circuit_order=order,
        compilation_options=options,
        purpose=tagged,
        circuits=len(circuit_list),
        shots=shots,
        path=path,
        result=result,
    )


# ---------------------------------------------------------------------------
# Native circuits (iqm.pulse.Circuit lists): prx_12 and measure_012, which the qiskit
# serializer cannot send. Not to be used until reviewed and merged; C3 runs only after a pilot.
# ---------------------------------------------------------------------------


def _native_circuit(circuit: Any) -> Any:
    """An ``iqm.pulse.Circuit``, rebuilt from the dict ``CircuitJob.payload()`` returns for one (iqm-client
    35.0.3 dumps the stored request with ``model_dump``); a ``Circuit`` passes through unchanged."""
    from iqm.pulse import Circuit, CircuitOperation

    if isinstance(circuit, Circuit):
        return circuit
    return Circuit(
        name=circuit["name"],
        instructions=tuple(CircuitOperation(**i) for i in circuit["instructions"]),
        metadata=circuit.get("metadata"),
    )


def _measurement_keys(circuit: Any) -> dict[str, list[str]]:
    """A native circuit's measurement keys -> the device qubits each one measures, in instruction order."""
    keys: dict[str, list[str]] = {}
    for inst in circuit.instructions:
        if inst.name in ("measure", "measure_012"):
            key = inst.args["key"]
            if key in keys:
                raise ValueError(f"circuit {circuit.name!r}: measurement key {key!r} is used twice")
            keys[key] = list(inst.locus)
    return keys


def save_native_raw(
    results: Sequence[Mapping[str, Any]] | None,
    *,
    circuits: Sequence[Any],
    job_id: str,
    task: str,
    device: str,
    calibration_set_id: str,
    shots: int,
    status: str,
    compilation_options: Mapping[str, Any] | None = None,
    names: Sequence[str] | None = None,
    timeline: Any = (),
    purpose: str = "",
    day: str | None = None,
    data_root: Path | None = None,
) -> Path:
    """Save a native circuit job's raw results under ``<data_root>/<YYYYMMDD>-<task>/<job id>/``.

    ``results`` is IQM's per-circuit measurement results (key -> shots x locus values: 0/1,
    or 0/1/2 from ``measure_012``), ``circuits`` the ``iqm.pulse.Circuit`` list that was sent
    (the run request's, or :func:`retrieve_native`'s ``circuits``; the dicts of a bare
    ``CircuitJob.payload()`` are rebuilt as circuits too). Writes ``meta.json`` and, when there are
    results, ``measurements.npz``: one array ``a<j>`` per circuit and measurement key, as IQM
    returned it, listed in ``meta.json``'s ``arrays``. Works on a fresh job or on one re-pulled
    with :func:`retrieve_native`. Refuses to overwrite.
    """
    day = day or dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    out = Path(data_root or DATA_ROOT) / f"{day}-{task}" / str(job_id)
    circuits = [_native_circuit(c) for c in circuits]
    out.mkdir(parents=True, exist_ok=False)

    keys = [_measurement_keys(c) for c in circuits]
    index, arrays = [], {}
    for i, per_key in enumerate(results or []):
        for key, values in per_key.items():
            name = f"a{len(arrays)}"
            arrays[name] = np.asarray(values)
            index.append({"key": name, "circuit": i, "measurement_key": key, "qubits": keys[i].get(key),
                          "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)})
    if arrays:
        np.savez_compressed(out / "measurements.npz", **arrays)

    order = [
        {"position": i, "name": names[i] if names is not None else c.name, "native_sha256": instructions_sha256(c.instructions)}
        for i, c in enumerate(circuits)
    ]
    meta = {
        "job_id": str(job_id),
        "kind": "native",
        "task": task,
        "device": device,
        "calibration_set_id": calibration_set_id,
        "physical_qubits": [sorted({q for inst in c.instructions for q in inst.locus}, key=_qubit_sort_key) for c in circuits],
        "measurement_keys": keys,
        "shots": shots,
        "circuits": len(circuits),
        "circuit_order": order,
        "compilation_options": None if compilation_options is None else dict(compilation_options),
        "status": str(status),
        "success": results is not None,
        "commit": git_head(),
        "purpose": purpose,
        "saved_utc": _utc_stamp(),
        "timeline": _jsonable(timeline),
        "arrays": index,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


@dataclass
class NativeSubmission:
    job_id: str
    task: str
    device: str
    calibration_set_id: str
    qubits: list[str]
    circuit_order: list[dict[str, Any]]
    compilation_options: dict[str, Any]
    purpose: str
    circuits: int
    shots: int
    status: str
    path: Path


def submit_native(
    circuits: Any,
    *,
    device: str,
    shots: int,
    purpose: str,
    calibration_set_id: str | None = None,
    circuit_compilation_options: Any = None,
    native_sha256: Sequence[str] | None = None,
    data_root: Path | None = None,
    timeout: float | None = None,
    **options: Any,
) -> NativeSubmission:
    """Submit native ``iqm.pulse.Circuit`` objects (device qubit names, ``CircuitOperation`` instructions) as one job.

    **Not to be used until reviewed and merged; C3 runs only after a pilot.**

    For what the qiskit serializer cannot send: ``prx_12`` with ``angle``/``phase`` and
    ``measure_012`` (:data:`NATIVE_JOB_OPS`; iqm-client 35.0.3's own validation still refuses
    ``measure_012``). The guard is :func:`submit`'s: the op allowlist and the arming and allowance
    before any connection, IQM's ``create_run_request`` validation against the calibration set,
    the built request checked again, the ledger lock, the ledger line the moment the job id
    returns, and the fail-closed row if the submit call raises. The calibration set is pinned
    in the request (the server default's id, when ``calibration_set_id`` is None), so the one
    recorded is the one that runs. Raw results are
    saved by :func:`save_native_raw`; if anything fails after the ledger line, re-pull the job
    with :func:`retrieve_native` and save its ``results``, ``circuits`` and ``parameters`` with it.
    """
    from uuid import UUID

    from iqm.iqm_server_client.models import JobStatus
    from iqm.pulse import Circuit

    circuit_list = list(circuits) if isinstance(circuits, (list, tuple)) else [circuits]
    if not circuit_list:
        raise ValueError("no circuits to submit")
    if not all(isinstance(c, Circuit) for c in circuit_list):
        raise TypeError("submit_native takes iqm.pulse.Circuit objects; submit takes qiskit circuits")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError(f"shots must be a positive int, got {shots!r}")
    if not _cell(purpose):
        raise ValueError("purpose is required: it goes in the ledger")
    if "use_timeslot" in options:
        raise ValueError("use_timeslot: submit_native uses the on-demand queue only")
    if options:
        raise ValueError(f"unsupported option(s) {sorted(options)}")
    hashes = _names_or_none("native_sha256", native_sha256, len(circuit_list))
    record = compilation_options_record(circuit_compilation_options)
    tagged = _tagged(purpose, compilation_options_tag(record))
    for i, c in enumerate(circuit_list):
        _refuse_off_allowlist(i, c.name, c.instructions, "has", NATIVE_JOB_OPS)
        _measurement_keys(c)

    task = armed_task()
    check_allowance(task, len(circuit_list), shots)  # before any connection
    ledger = ledger_path()

    client = _connect_client(device, read_only=False)
    calset = client.get_dynamic_quantum_architecture(
        UUID(str(calibration_set_id)) if calibration_set_id else None
    ).calibration_set_id
    run_request = client.create_run_request(
        circuit_list, calibration_set_id=calset, shots=shots, options=circuit_compilation_options
    )
    _check_sent_ops(run_request, NATIVE_JOB_OPS)
    _check_request_options(run_request, record)
    order = _circuit_order(run_request, None, hashes)
    sent = _sent_qubits(run_request)
    qubits = sorted({q for circuit_qubits in sent for q in circuit_qubits}, key=_qubit_sort_key)

    with ledger_lock(ledger):
        check_allowance(task, len(circuit_list), shots)
        try:
            job = client.submit_run_request(run_request, use_timeslot=False)
            job_id = str(job.job_id)
        except BaseException as e:  # noqa: BLE001 -- the job may exist: recorded, then re-raised
            _fail_closed(ledger, e, task=task, device=device, calibration_set_id=str(calset), qubits=qubits,
                         circuits=len(circuit_list), shots=shots, purpose=tagged)
        submitted = dt.datetime.now(dt.UTC)
        append_ledger(
            ledger,
            ledger_line(
                task=task,
                device=device,
                calibration_set_id=str(calset),
                qubits=qubits,
                circuits=len(circuit_list),
                shots=shots,
                job_id=job_id,
                purpose=tagged,
                time=submitted,
            ),
        )

    status = job.wait_for_completion() if timeout is None else job.wait_for_completion(timeout_secs=timeout)
    results = job.result() if status == JobStatus.COMPLETED else None
    path = save_native_raw(
        results,
        circuits=run_request.circuits,
        job_id=job_id,
        task=task,
        device=device,
        calibration_set_id=str(calset),
        shots=shots,
        status=status,
        compilation_options=record,
        timeline=getattr(job.data, "timeline", []),
        purpose=tagged,
        day=submitted.strftime("%Y%m%d"),
        data_root=data_root,
    )
    return NativeSubmission(
        job_id, task, device, str(calset), qubits, order, record, tagged, len(circuit_list), shots, str(status), path
    )


@dataclass
class NativeJob:
    """A native circuit job re-pulled by id: what the server holds."""

    data: Any
    """``JobData``: status, timeline."""
    results: Any
    """Per-circuit measurement results, or None unless the job completed."""
    circuits: Any
    """The circuits the server received (the job's payload), as ``iqm.pulse.Circuit`` objects."""
    parameters: Any
    """Its ``CircuitJobParameters``: shots, calibration set, compilation options."""


def retrieve_native(job_id: str, device: str, *, timeout: float | None = None) -> NativeJob:
    """Re-pull a native circuit job by id (waits if it is still running), with the server's own record of
    the circuits and options it ran. Submits nothing. ``hw.retrieve`` can't read these jobs: their
    measurement keys are not qiskit's. An ``UNKNOWN-...`` placeholder is refused, as by :func:`retrieve`.

    ``CircuitJob.payload()`` returns the circuits as dicts; they are rebuilt as ``iqm.pulse.Circuit``
    objects, so ``save_native_raw(job.results, circuits=job.circuits, ...)`` saves what the fresh
    job's save would have.
    """
    from uuid import UUID

    from iqm.iqm_server_client.models import JobStatus

    _refuse_unknown_job_id(job_id)
    job = _connect_client(device, read_only=True).get_job(UUID(str(job_id)))
    status = job.wait_for_completion() if timeout is None else job.wait_for_completion(timeout_secs=timeout)
    circuits, parameters = job.payload()
    results = job.result() if status == JobStatus.COMPLETED else None
    return NativeJob(job.data, results, [_native_circuit(c) for c in circuits], parameters)


# ---------------------------------------------------------------------------
# Pulse level (Pulla). NOT TO BE USED until Sal and the mentors confirm pulse-level
# submission rights.
# ---------------------------------------------------------------------------


def playlist_sha256(playlist: Any) -> str:
    """sha256 of a playlist as the run definition sends it: ``pack_playlist``'s protobuf, serialized deterministically."""
    from iqm.station_control.client.serializers.playlist_serializers import pack_playlist

    return hashlib.sha256(pack_playlist(playlist).SerializeToString(deterministic=True)).hexdigest()


@dataclass(frozen=True)
class PlaylistShape:
    """What a Pulla run definition will execute, read from the run definition itself."""

    segments: int
    spots: int
    """Soft-sweep spots: the whole playlist runs once per spot."""
    shots: int
    """``settings.controllers.options.playlist_repeats``: what ``SettingNode.set_shots`` writes."""
    averaging_bins: int | None
    qubits: list[str]
    playlist_sha256: str

    @property
    def circuits(self) -> int:
        """The ledger's circuits: segments x soft-sweep spots."""
        return self.segments * self.spots


def playlist_shape(run_definition: Any) -> PlaylistShape:
    """Read a run definition's segments, soft-sweep spots, shots and qubits. Raises ``ValueError`` if it can't."""
    sweep = run_definition.sweep_definition
    playlist = sweep.playlist
    if playlist is None or not playlist.segments:
        raise ValueError("the run definition has no playlist segments to execute")
    try:
        controller_options = sweep.settings.controllers.options
        shots = controller_options.playlist_repeats.value
    except AttributeError as e:
        raise ValueError("cannot read settings.controllers.options.playlist_repeats from the run definition") from e
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"settings.controllers.options.playlist_repeats must be a positive int, got {shots!r}")
    try:
        averaging_bins = controller_options.averaging_bins.value
    except AttributeError:
        averaging_bins = None
    # as IQM's own QPU-runtime estimate counts them (station_control sweep.py, _qpu_runtime)
    spots = math.prod(len(dim[0].data) for dim in sweep.sweeps)
    qubits = list(run_definition.additional_run_properties.get("qubits") or [])
    if not qubits:
        raise ValueError("the run definition names no qubits (additional_run_properties['qubits'])")
    return PlaylistShape(
        segments=len(playlist.segments),
        spots=spots,
        shots=shots,
        averaging_bins=averaging_bins,
        qubits=sorted(qubits, key=_qubit_sort_key),
        playlist_sha256=playlist_sha256(playlist),
    )


def save_playlist_raw(
    sweep_results: Mapping[str, Sequence[Any]] | None,
    *,
    job_id: str,
    task: str,
    device: str,
    calibration_set_id: str,
    shape: PlaylistShape,
    status: str,
    timeline: Any = (),
    purpose: str = "",
    day: str | None = None,
    data_root: Path | None = None,
) -> Path:
    """Save a Pulla job's raw sweep results under ``<data_root>/<YYYYMMDD>-<task>/<job id>/``.

    Writes ``meta.json`` and, when there are results, ``sweep_results.npz``: one array
    ``a<j>`` per return parameter and soft-sweep spot, as the server returned them, listed
    in ``meta.json``'s ``arrays``. Works on a fresh job or on one re-pulled with
    :func:`retrieve_playlist`. Refuses to overwrite.
    """
    day = day or dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    out = Path(data_root or DATA_ROOT) / f"{day}-{task}" / str(job_id)
    out.mkdir(parents=True, exist_ok=False)

    index, arrays = [], {}
    for label, spots in (sweep_results or {}).items():
        for spot, array in enumerate(spots):
            key = f"a{len(arrays)}"
            arrays[key] = np.asarray(array)
            index.append({"key": key, "return_parameter": label, "spot": spot,
                          "shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)})
    if arrays:
        np.savez_compressed(out / "sweep_results.npz", **arrays)

    meta = {
        "job_id": str(job_id),
        "kind": "playlist",
        "task": task,
        "device": device,
        "calibration_set_id": calibration_set_id,
        "playlist_sha256": shape.playlist_sha256,
        "physical_qubits": shape.qubits,
        "segments": shape.segments,
        "spots": shape.spots,
        "circuits": shape.circuits,
        "shots": shape.shots,
        "averaging_bins": shape.averaging_bins,
        "status": str(status),
        "success": sweep_results is not None,
        "commit": git_head(),
        "purpose": purpose,
        "saved_utc": _utc_stamp(),
        "timeline": _jsonable(timeline),
        "arrays": index,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


@dataclass
class PlaylistSubmission:
    job_id: str
    task: str
    device: str
    calibration_set_id: str
    shape: PlaylistShape
    purpose: str
    status: str
    path: Path


def submit_playlist(
    run_definition: Any,
    *,
    context: Mapping[str, Any],
    device: str,
    calibration_set_id: str,
    purpose: str,
    data_root: Path | None = None,
    timeout: float | None = None,
    **options: Any,
) -> PlaylistSubmission:
    """Submit a Pulla run definition (a pulse-level job) to ``device`` as one job, then save its raw result.

    **Not to be used until Sal and the mentors confirm pulse-level submission rights.**
    It is tested offline only, against a fake Pulla; no pulse-level job has ever been sent.

    The guard is :func:`submit`'s: arming, the allowance, the ledger lock, the ledger line the
    moment the job id returns, and the fail-closed row if the submit call raises. The ledger's
    circuits x shots is segments x soft-sweep spots by ``playlist_repeats``, and its qubits
    ``additional_run_properties['qubits']``, all
    read from ``run_definition`` (:func:`playlist_shape`). ``calibration_set_id`` is the set the
    playlist was compiled from (a run definition doesn't carry one). The ledger purpose ends
    in ``[pulse,playlist=<first 8 hex digits of its sha256>]``. ``use_timeslot`` is always off.

    If anything fails after the ledger line, re-pull the job with :func:`retrieve_playlist`
    and save it with :func:`save_playlist_raw`.
    """
    if "use_timeslot" in options:
        raise ValueError("use_timeslot: submit_playlist uses the on-demand queue only")
    if options:
        raise ValueError(f"unsupported option(s) {sorted(options)}")
    if not _cell(purpose):
        raise ValueError("purpose is required: it goes in the ledger")
    if not _cell(calibration_set_id or ""):
        raise ValueError("calibration_set_id is required: the calibration set the playlist was compiled from")
    shape = playlist_shape(run_definition)
    tagged = _tagged(purpose, f"[pulse,playlist={shape.playlist_sha256[:8]}]")

    task = armed_task()
    check_allowance(task, shape.circuits, shape.shots)  # before any connection
    ledger = ledger_path()

    p = _connect_pulla(device, read_only=False)
    with ledger_lock(ledger):
        check_allowance(task, shape.circuits, shape.shots)
        try:
            job = p.submit_playlist(run_definition, context=dict(context), use_timeslot=False)
            job_id = str(job.job_id)
        except BaseException as e:  # noqa: BLE001 -- the job may exist: recorded, then re-raised
            _fail_closed(ledger, e, task=task, device=device, calibration_set_id=str(calibration_set_id),
                         qubits=shape.qubits, circuits=shape.circuits, shots=shape.shots, purpose=tagged)
        submitted = dt.datetime.now(dt.UTC)
        append_ledger(
            ledger,
            ledger_line(
                task=task,
                device=device,
                calibration_set_id=str(calibration_set_id),
                qubits=shape.qubits,
                circuits=shape.circuits,
                shots=shape.shots,
                job_id=job_id,
                purpose=tagged,
                time=submitted,
            ),
        )

    from iqm.iqm_server_client.models import JobStatus

    status = job.wait_for_completion() if timeout is None else job.wait_for_completion(timeout_secs=timeout)
    sweep_results = (
        p._iqm_server_client.get_job_artifact_sweep_results(job.job_id) if status == JobStatus.COMPLETED else None
    )
    path = save_playlist_raw(
        sweep_results,
        job_id=job_id,
        task=task,
        device=device,
        calibration_set_id=str(calibration_set_id),
        shape=shape,
        status=status,
        timeline=getattr(job.data, "timeline", []),
        purpose=tagged,
        day=submitted.strftime("%Y%m%d"),
        data_root=data_root,
    )
    return PlaylistSubmission(job_id, task, device, str(calibration_set_id), shape, tagged, str(status), path)


# ---------------------------------------------------------------------------
# Read-only helpers: need the token, not arming.
# ---------------------------------------------------------------------------


def backend(device: str, calibration_set_id: str | None = None) -> Any:
    """A read-only qiskit ``IQMBackend`` for transpiling and inspection. Its ``run`` refuses."""
    return _connect(device, calibration_set_id, read_only=True)


def pulla(device: str) -> Any:
    """A read-only Pulla, for compiling (``get_standard_compiler``) and inspection. It cannot submit."""
    return _connect_pulla(device, read_only=True)


def retrieve(job_id: str, device: str, *, timeout: float | None = None) -> Any:
    """Re-pull a job's qiskit ``Result`` by id (waits if it is still running). Submits nothing. An
    ``UNKNOWN-...`` placeholder from a fail-closed ledger row is refused."""
    _refuse_unknown_job_id(job_id)
    job = _connect(device, read_only=True).retrieve_job(job_id)
    return job.result() if timeout is None else job.result(timeout=timeout)


def _server_client(quantum_computer: str | None) -> Any:
    """A read-only ``_IQMServerClient``. With no ``quantum_computer`` it selects none, so it can only list them."""
    _load_token()
    from iqm.iqm_server_client.iqm_server_client import _IQMServerClient

    class UnselectedServerClient(_IQMServerClient):
        def _resolve_quantum_computer(self, user_defined_quantum_computer: str | None) -> str | None:
            # the constructor resolves the device, and raises "Quantum computer not selected" when
            # the token sees more than one; listing needs no device
            return user_defined_quantum_computer

    server = (_IQMServerClient if quantum_computer else UnselectedServerClient)(
        SERVER_URL, quantum_computer=quantum_computer
    )
    _read_only_server(server)
    return server


def retrieve_playlist(job_id: str, device: str) -> tuple[Any, Any]:
    """Re-pull a Pulla job by id: its ``JobData`` (status, timeline) and raw sweep results (None unless
    completed). Submits nothing. An ``UNKNOWN-...`` placeholder is refused, as by :func:`retrieve`."""
    from iqm.iqm_server_client.models import JobStatus

    _refuse_unknown_job_id(job_id)
    server = _server_client(device)
    data = server.get_job(job_id)
    results = server.get_job_artifact_sweep_results(job_id) if data.status == JobStatus.COMPLETED else None
    return data, results


def list_devices() -> list[dict[str, Any]]:
    """The quantum computers the server offers to this token."""
    # IQMClient 35 has no public device list, and IQMClient(SERVER_URL) with no device raises
    # once the token sees several; the server client's getter, with no device selected, works.
    return [qc.model_dump(mode="json") for qc in _server_client(None).get_quantum_computers()]


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
