"""Does our Resonance token have pulse-level access through IQM Pulla? (task pulla-check): read-only.

    uv run python scripts/pulla_check.py             # live: garnet, then emerald
    uv run python scripts/pulla_check.py --devices emerald   # live: emerald only
    uv run python scripts/pulla_check.py --report    # print the committed results: no network, no token
    uv run python scripts/pulla_check.py --compare [--doc] [--json PATH]   # offline: Emerald's priors on EP0

Nothing here can submit. Before any ``Pulla`` exists, :func:`install_guard` replaces
``Pulla.submit_playlist`` with a function that raises, disables every submitting or cancelling
method of the server client, and wraps the client's single HTTP seam
(``_IQMServerClient._send_request``) so that any method but GET raises before it is sent. The
devices are an allowlist (``garnet``, ``emerald``): no ``:mock`` and no facade. ``DQ_QPU*`` is
never set, and nothing goes in the QPU ledger.

The calls are the ones the brief allows: the ``Pulla`` constructor (GETs of the device list, the
station-control settings, the chip design record and the chip label), ``get_chip_label``,
``get_chip_topology``, ``get_calibration_stash`` and ``get_standard_compiler``, plus
``dqiqm.hw.calibration_set_id`` to name the calibration set. Circuits are compiled locally into
schedules, which are inspected and never sent. Every HTTP request is logged (method, relative
path, status) and saved with the results.

Per device, it writes under ``data/calibration/<device>/pulse/`` (never overwriting; a later run
gets a UTC stamp in the name):

* ``access.json``: whether the constructor succeeded, or its exact error class and HTTP status;
* ``<calibration_set_id>.json``: the PRX, CZ and readout implementations and their calibrated
  parameters. Garnet: per locus, for the pilot patch's 17 qubits and 24 couplers. Emerald: a
  summary (median, min, max) over every locus;
* ``compile.<calibration_set_id>.json``: per-qubit readout windows and circuits compiled locally.
  Garnet: the pilot (``pilot/E1_d3_Z_r03_pA.qasm``, with and without barriers) and the
  gate-merging test. Emerald (emerald-compile): the batch's d=3, R=3 memory circuits on patches eA
  and eB, Z and X (``InfoRecord/emerald/circuits/E1/``, read only), and the windows of all 30 of
  their qubits. Emerald's compile runs only on the calibration set those circuits were placed on
  (:data:`EMERALD_CALSET`): on any other, the parameter file records the live set and the script
  stops with exit code 2.

``--compare`` is offline: it re-decodes the Emerald pilot EP0 (job ``01a17017…``) under the
pilot's own decoder, ``device.hardware_noise`` with the 360 ns ``measure_ns`` fallback, and with
the compiled window, and prints the windows, the info dicts and the per-round tables.

The token is ``IQM_TOKEN``, loaded from the worktree's ``.env`` as ``dqiqm.hw`` does it, and read
by the IQM client from the environment. This script never reads, prints or stores it. The
outputs keep no instrument address, hostname or URL: only gate and readout observation names
and numbers, and the relative API paths requested.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dqiqm import hw

REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "data" / "calibration"
PILOT_DIR = CALIBRATION / "garnet" / "pulse" / "pilot"
PILOTS = ("E1_d3_Z_r03_pA", "E1_d3_Z_r03_pA_nobar")
DEVICES = ("garnet", "emerald")
SCHEMA = "dqiqm.pulla-check/1"

#: Emerald's committed E1 circuits (emerald-design), read only.
EMERALD_CIRCUITS = REPO / "InfoRecord" / "emerald" / "circuits" / "E1"
#: The calibration set they were placed on (``dqiqm.experiments.emerald.CALSET``): the only one compiled.
EMERALD_CALSET = "23e12f6b-c23c-4736-8736-b0142989c0e6"
EMERALD_PATCHES = ("eA", "eB")
#: The batch's d=3, R=3 memory circuits: both patches, both bases.
EMERALD_COMPILED = tuple(f"E1_d3_{basis}_r03_{patch}" for patch in EMERALD_PATCHES for basis in ("Z", "X"))
#: The readout calibration kept per qubit with Emerald's windows (``prx_12`` is the shelved readout's pulse).
READOUT_GATES = ("measure", "measure_fidelity", "prx_12")

# Gates whose calibration is kept, and the merging test's loci (on the pilot patch).
KEPT_GATES = ("prx", "cz", "measure", "measure_fidelity", "reset_wait")
MERGE_QUBIT = "QB2"
MERGE_PAIR = ("QB1", "QB2")
MERGE_NS = (1, 5, 21)

# Anything that looks like an address or a credential. A file that matches is not written.
_HOST = r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:tech|com|net|org|io|fi|local|internal)\b"
_IP = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
FORBIDDEN = re.compile(rf"://|{_IP}|[Bb]earer|IQM_TOKEN|[Aa]uthorization|[Pp]assword|{_HOST}")


class ReadOnlyViolation(RuntimeError):
    """Something tried to submit, cancel, write or send a non-GET request: pulla-check is read-only."""


# ─── the guard ───────────────────────────────────────────────────────────────

REQUESTS: list[dict[str, Any]] = []  # one entry per HTTP request: method, relative path, status
_SUBMITTING = ("submit_run", "submit_sweep", "submit_circuits", "_submit_job", "cancel_job", "delete_job")


def _refuse(what: str) -> Callable[..., Any]:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ReadOnlyViolation(f"pulla-check is read-only: {what} is disabled")

    refuse.__name__ = f"refused_{what.rsplit('.', 1)[-1]}"
    refuse.pulla_check_guard = True  # type: ignore[attr-defined]
    return refuse


def install_guard() -> None:
    """Make submission impossible in this process. Idempotent; call it before constructing ``Pulla``."""
    from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
    from iqm.pulla.pulla import Pulla

    Pulla.submit_playlist = _refuse("Pulla.submit_playlist")
    for name in _SUBMITTING:
        setattr(_IQMServerClient, name, _refuse(f"_IQMServerClient.{name}"))

    send = _IQMServerClient._send_request
    if getattr(send, "pulla_check_guard", False):
        return

    def get_only(self: Any, http_method: Any, url_path: str, **kwargs: Any) -> Any:
        method = getattr(http_method, "__name__", "?").upper()
        if method != "GET":
            raise ReadOnlyViolation(f"pulla-check is read-only: refused {method} {url_path}")
        entry: dict[str, Any] = {"method": method, "path": url_path, "status": None}
        REQUESTS.append(entry)

        def recorded(url: str, **request_kwargs: Any) -> Any:
            response = http_method(url, **request_kwargs)
            entry["status"] = response.status_code
            return response

        recorded.__name__ = http_method.__name__
        return send(self, recorded, url_path, **kwargs)

    get_only.pulla_check_guard = True  # type: ignore[attr-defined]
    _IQMServerClient._send_request = get_only


def guard_installed() -> bool:
    from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
    from iqm.pulla.pulla import Pulla

    return all(
        getattr(f, "pulla_check_guard", False)
        for f in [Pulla.submit_playlist, _IQMServerClient._send_request]
        + [getattr(_IQMServerClient, n) for n in _SUBMITTING]
    )


# ─── access ──────────────────────────────────────────────────────────────────


def _utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scrub(text: str, limit: int = 400) -> str | None:
    """An error message with anything address-like replaced, or None if something credential-like is left."""
    text = re.sub(r"\S+://\S+", "<url>", str(text))
    text = re.sub(_IP, "<ip>", text)
    text = re.sub(_HOST, "<host>", text)
    text = " ".join(text.split())[:limit]
    return None if FORBIDDEN.search(text) else text


def _status_of(exc: BaseException, requests: list[dict[str, Any]]) -> int | None:
    """The HTTP status behind an IQM client error, from the request log or the exception chain."""
    failed = [r["status"] for r in requests if r["status"] is not None and r["status"] >= 400]
    if failed:
        return failed[-1]
    e: BaseException | None = exc
    while e is not None:
        response = getattr(e, "response", None)
        if response is not None and getattr(response, "status_code", None):
            return int(response.status_code)
        e = e.__cause__ or e.__context__
    return None


def error_record(exc: BaseException, requests: list[dict[str, Any]]) -> dict[str, Any]:
    """The error class, its chain, the HTTP status and a scrubbed message: never a header or token."""
    chain = []
    e: BaseException | None = exc
    while e is not None and len(chain) < 4:
        chain.append(f"{type(e).__module__}.{type(e).__qualname__}")
        e = e.__cause__ or e.__context__
    return {
        "error_class": chain[0],
        "error_chain": chain,
        "http_status": _status_of(exc, requests),
        "error_code": getattr(exc, "error_code", None),
        "message_scrubbed": _scrub(getattr(exc, "message", None) or str(exc)),
    }


def connect(device: str) -> tuple[Any, dict[str, Any]]:
    """Construct ``Pulla`` for ``device``. Returns (pulla or None, the access record)."""
    if device not in DEVICES:
        raise ReadOnlyViolation(f"device {device!r} is not allowed: only {DEVICES} (no :mock, no facade)")
    install_guard()
    if not guard_installed():
        raise ReadOnlyViolation("the guard is not installed")
    hw._load_token()  # IQM_TOKEN into the environment, as dqiqm.hw does; the client reads it there
    from iqm.pulla.pulla import Pulla

    start = len(REQUESTS)
    record: dict[str, Any] = {"device": device, "utc": _utc(), "constructor": None}
    try:
        pulla = Pulla(hw.SERVER_URL, quantum_computer=device)
    except Exception as exc:  # noqa: BLE001 - the error class is the answer
        record["constructor"] = "failed"
        record |= error_record(exc, REQUESTS[start:])
        pulla = None
    else:
        record["constructor"] = "ok"
    record["requests"] = [dict(r) for r in REQUESTS[start:]]
    return pulla, record


# ─── the pilot circuit: OpenQASM 3 as qiskit writes our hardware circuits ───


@dataclass(frozen=True)
class Op:
    name: str  # "prx", "cz", "barrier" or "measure"
    qubits: tuple[int, ...]  # physical qubit indices ($n)
    angle: float = 0.0
    phase: float = 0.0
    clbit: int | None = None


_BINOPS = {ast.Add: float.__add__, ast.Sub: float.__sub__, ast.Mult: float.__mul__, ast.Div: float.__truediv__}


def angle(expr: str) -> float:
    """Evaluate an OpenQASM angle such as ``-pi/2`` or ``1.5707963267948966``: numbers, pi, + - * / only."""

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in ("pi", "π"):
            return math.pi
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"unsupported angle expression {expr!r}")

    return ev(ast.parse(expr.strip(), mode="eval").body)


_R = re.compile(r"r\((?P<a>[^,()]+),(?P<b>[^,()]+)\)\s+\$(?P<q>\d+)\s*;")
_CZ = re.compile(r"cz\s+\$(?P<q0>\d+)\s*,\s*\$(?P<q1>\d+)\s*;")
_BARRIER = re.compile(r"barrier\s+(?P<qs>\$\d+(?:\s*,\s*\$\d+)*)\s*;")
_MEASURE = re.compile(r"\w+\[(?P<c>\d+)\]\s*=\s*measure\s+\$(?P<q>\d+)\s*;")
_HEADER = re.compile(r'OPENQASM 3(\.0)?;|include "stdgates\.inc";|bit\[\d+\] \w+;')


def parse_qasm(text: str) -> list[Op]:
    """The native operations of a qiskit-written OpenQASM 3 hardware circuit (physical ``$n`` qubits).

    Accepts only what our hardware circuits contain: ``r(angle, phase)``, ``cz``, ``barrier`` and
    ``c[i] = measure``, plus the header and the ``gate r`` definition. Anything else raises.
    """
    ops: list[Op] = []
    depth = 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("gate "):
            depth += line.count("{") - line.count("}")
            continue
        if depth:
            depth += line.count("{") - line.count("}")
            continue
        if _HEADER.fullmatch(line):
            continue
        if m := _R.fullmatch(line):
            ops.append(Op("prx", (int(m["q"]),), angle(m["a"]), angle(m["b"])))
        elif m := _CZ.fullmatch(line):
            ops.append(Op("cz", (int(m["q0"]), int(m["q1"]))))
        elif m := _BARRIER.fullmatch(line):
            ops.append(Op("barrier", tuple(int(q) for q in re.findall(r"\$(\d+)", m["qs"]))))
        elif m := _MEASURE.fullmatch(line):
            ops.append(Op("measure", (int(m["q"]),), clbit=int(m["c"])))
        else:
            raise ValueError(f"line {n}: unsupported OpenQASM {raw!r}")
    return ops


def qubit_name(index: int) -> str:
    """IQM's physical qubit index -> name: index i is QB(i+1), as in IQMBackend.index_to_qubit_name."""
    return f"QB{index + 1}"


def op_counts(ops: Iterable[Op]) -> dict[str, int]:
    """Op counts, named as in placement.json (``r`` for prx)."""
    return dict(sorted(Counter("r" if o.name == "prx" else o.name for o in ops).items()))


def used_qubits(ops: Iterable[Op], name: Callable[[int], str] = qubit_name) -> list[str]:
    qs = {q for o in ops if o.name != "barrier" for q in o.qubits}
    return [name(q) for q in sorted(qs)]


def cz_pairs(ops: Iterable[Op], name: Callable[[int], str] = qubit_name) -> list[tuple[str, str]]:
    """The distinct couplers the CZs use, each as a (lower, higher) qubit-name pair."""
    pairs = {tuple(sorted((o.qubits[0], o.qubits[1]))) for o in ops if o.name == "cz"}
    return [(name(a), name(b)) for a, b in sorted(pairs)]


def to_circuit(ops: Iterable[Op], circuit_name: str, name: Callable[[int], str] = qubit_name) -> Any:
    """An ``iqm.pulse.Circuit`` on physical qubit names. Each measurement gets key ``c<i>`` (its clbit)."""
    from iqm.pulse import Circuit, CircuitOperation

    instructions = []
    for o in ops:
        locus = tuple(name(q) for q in o.qubits)
        if o.name == "prx":
            instructions.append(CircuitOperation("prx", locus, {"angle": o.angle, "phase": o.phase}))
        elif o.name == "measure":
            instructions.append(CircuitOperation("measure", locus, {"key": f"c{o.clbit}"}))
        else:
            instructions.append(CircuitOperation(o.name, locus, {}))
    return Circuit(circuit_name, tuple(instructions))


def load_qasm(path: Path, entry: Mapping[str, Any]) -> list[Op]:
    """A committed hardware circuit's ops, checked against its placement.json entry: sha256, op counts, qubits."""
    text, name = path.read_text(), path.stem
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != entry["qasm_sha256"]:
        raise ValueError(f"{name}.qasm sha256 {digest} != placement.json {entry['qasm_sha256']}")
    ops = parse_qasm(text)
    if op_counts(ops) != entry["ops"]:
        raise ValueError(f"{name}: parsed ops {op_counts(ops)} != placement.json {entry['ops']}")
    if set(used_qubits(ops)) != set(entry["placement"].values()):
        raise ValueError(f"{name}: qubits {used_qubits(ops)} != placement {sorted(entry['placement'].values())}")
    return ops


def load_pilot(name: str) -> tuple[list[Op], dict[str, Any]]:
    """A vendored Garnet pilot circuit and its placement.json entry; checks the file against the entry."""
    entry = json.loads((PILOT_DIR / "placement.json").read_text())["entries"][name]
    return load_qasm(PILOT_DIR / f"{name}.qasm", entry), entry


def load_emerald(name: str) -> tuple[list[Op], dict[str, Any]]:
    """An Emerald E1 circuit (``InfoRecord/emerald/circuits/E1/``) and its placement.json entry, checked."""
    entry = json.loads((EMERALD_CIRCUITS / "placement.json").read_text())[name]
    if entry["device"] != "emerald" or entry["target"] != "hardware" or entry["swaps"] != 0:
        raise ValueError(f"{name}: not a zero-SWAP Emerald hardware circuit ({entry['device']}, {entry['target']})")
    return load_qasm(EMERALD_CIRCUITS / f"{name}.qasm", entry), entry


def emerald_patches(circuits: Mapping[str, tuple[list[Op], Mapping[str, Any]]]) -> dict[str, Any]:
    """Per patch: its qubits, ancillas, data qubits and CZ couplers, the same in every one of its circuits."""
    out: dict[str, Any] = {}
    for name, (ops, entry) in circuits.items():
        patch = entry["params"]["patch"]
        roles = {entry["placement"][s]: r for s, r in entry["roles"].items()}
        row = {
            "qubits": used_qubits(ops),
            "ancillas": sorted((q for q, r in roles.items() if r == "ancilla"), key=lambda q: int(q[2:])),
            "data": sorted((q for q, r in roles.items() if r == "data"), key=lambda q: int(q[2:])),
            "couplers": [f"{a}__{b}" for a, b in cz_pairs(ops)],
        }
        seen = out.setdefault(patch, row | {"circuits": []})
        if {k: v for k, v in seen.items() if k != "circuits"} != row:
            raise ValueError(f"{name}: patch {patch}'s qubits or couplers differ between its circuits")
        seen["circuits"].append(name)
    for patch, row in out.items():
        if len(row["qubits"]) != 17 or len(row["couplers"]) != 24 or len(row["ancillas"]) != 8:
            raise ValueError(f"patch {patch}: {len(row['qubits'])} qubits, {len(row['couplers'])} couplers; expected 17 and 24")
    return out


# ─── calibrated parameters ───────────────────────────────────────────────────

_GATE_FIELD = re.compile(
    r"gates\.(?P<gate>[a-z0-9_]+)\.(?P<impl>[a-z0-9_]+)\.(?P<locus>[A-Za-z0-9-]+(?:__[A-Za-z0-9-]+)?)\."
    r"(?P<param>[A-Za-z0-9_.]+)"
)
_READOUT_FIELD = re.compile(
    r"characterization\.gate_properties\.(?P<gate>measure[a-z0-9_]*)\.(?P<impl>[a-z0-9_]+)\.(?P<locus>QB\d+)\."
    r"(?P<param>[A-Za-z0-9_]+)"
)
_DRIVE_FREQUENCY = re.compile(r"controllers\.(?P<locus>QB\d+)\.drive\.frequency")
_DEFAULT_IMPL = re.compile(r"gate_definitions\.(?P<gate>[a-z0-9_]+)\.default_implementation")
_OVERRIDE = re.compile(r"gate_definitions\.(?P<gate>[a-z0-9_]+)\.(?P<impl>[a-z0-9_]+)\.override_default_for_loci")
_NAME = re.compile(r"[a-z0-9_]*")


def _value(v: Any) -> Any:
    """A JSON-safe number or bool; an array becomes {"array_len": n}; anything else (a string) is None."""
    import numpy as np

    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return int(f) if isinstance(v, (int, np.integer)) else (f if math.isfinite(f) else None)
    if isinstance(v, (list, tuple, np.ndarray)):
        return {"array_len": int(np.size(v))}
    return None


def _locus_in(locus: str, qubits: frozenset[str] | None, couplers: frozenset[frozenset[str]] | None) -> bool:
    parts = locus.split("__")
    if len(parts) == 1:
        return qubits is None or locus in qubits
    return couplers is None or frozenset(parts) in couplers


def _override_loci(v: Any) -> list[str]:
    """Emerald's ``override_default_for_loci``: an array of 'QBa__QBb' strings (kept only if name-shaped)."""
    import numpy as np

    out = [str(x) for x in np.asarray(v).ravel()]
    return [x for x in out if re.fullmatch(r"QB\d+__QB\d+|QB\d+", x)]


def extract_parameters(
    observations: Mapping[str, Any],
    qubits: Iterable[str] | None = None,
    couplers: Iterable[Iterable[str]] | None = None,
    gates: Iterable[str] = KEPT_GATES,
) -> dict[str, Any]:
    """The gate and readout calibration of a stash, per gate, implementation, locus and parameter.

    ``observations`` maps the observation name (``dut_field``) to an object with ``.value`` and
    ``.unit``. ``qubits`` and ``couplers`` (pairs, either order) limit the loci; None keeps all.
    Only numbers and bools are kept; an array keeps its length; strings are dropped and counted.
    ``controllers.*`` is dropped except each qubit's drive frequency (the PRX carrier).
    """
    q = frozenset(qubits) if qubits is not None else None
    c = frozenset(frozenset(p) for p in couplers) if couplers is not None else None
    gates = tuple(gates)
    out: dict[str, Any] = {
        "default_implementations": {},
        "override_default_for_loci": {},
        "gates": {},
        "readout_characterization": {},
        "drive_frequency_hz": {},
        "units": {},
        "dropped_non_numeric": 0,
    }
    for field, obs in sorted(observations.items()):
        value = getattr(obs, "value", obs)
        unit = getattr(obs, "unit", "") or ""
        if m := _DEFAULT_IMPL.fullmatch(field):
            if m["gate"] in gates and isinstance(value, str) and _NAME.fullmatch(value):
                out["default_implementations"][m["gate"]] = value
            continue
        if m := _OVERRIDE.fullmatch(field):
            if m["gate"] in gates:
                loci = [x for x in _override_loci(value) if _locus_in(x, q, c)]
                out["override_default_for_loci"][f"{m['gate']}.{m['impl']}"] = loci
            continue
        if m := _GATE_FIELD.fullmatch(field):
            tree, key = out["gates"], (m["gate"], m["impl"])
        elif m := _READOUT_FIELD.fullmatch(field):
            tree, key = out["readout_characterization"], (m["gate"], m["impl"])
        elif m := _DRIVE_FREQUENCY.fullmatch(field):
            if _locus_in(m["locus"], q, c):
                out["drive_frequency_hz"][m["locus"]] = _value(value)
            continue
        else:
            continue
        if m["gate"] not in gates or not _locus_in(m["locus"], q, c):
            continue
        v = _value(value)
        if v is None:
            out["dropped_non_numeric"] += 1
            continue
        tree.setdefault(key[0], {}).setdefault(key[1], {}).setdefault(m["locus"], {})[m["param"]] = v
        if unit:
            out["units"][f"{key[0]}.{key[1]}.{m['param']}"] = unit
    return out


def summarize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Per gate, implementation and parameter: the median, min and max over loci, and the locus count."""

    def stats(by_locus: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        names = sorted({p for v in by_locus.values() for p in v})
        res = {}
        for p in names:
            xs = [v[p] for v in by_locus.values() if isinstance(v.get(p), (int, float)) and not isinstance(v[p], bool)]
            if xs:
                res[p] = {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}
        return {"loci": len(by_locus), "parameters": res}

    out = {k: v for k, v in params.items() if k not in ("gates", "readout_characterization", "drive_frequency_hz")}
    for section in ("gates", "readout_characterization"):
        out[section] = {g: {i: stats(loci) for i, loci in impls.items()} for g, impls in params[section].items()}
    out["drive_frequency_hz"] = stats({q: {"frequency": f} for q, f in params["drive_frequency_hz"].items()})
    return out


def implementation_classes(compiler: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """The pulse family behind each default and per-locus override implementation: its class and waveforms."""
    pairs = list(params["default_implementations"].items())
    pairs += [tuple(k.split(".", 1)) for k in params["override_default_for_loci"]]
    out = {}
    for gate, impl in pairs:
        op = compiler.gate_definitions.get(gate)
        cls = op.implementations.get(impl) if op is not None else None
        if cls is None:
            continue
        info = {"class": cls.__name__}
        for attr in ("wave_i", "wave_q", "coupler_wave", "qubit_wave"):
            w = getattr(cls, attr, None)
            if isinstance(w, type):
                info[attr] = w.__name__
        out[f"{gate}.{impl}"] = info
    return out


# ─── local compilation ───────────────────────────────────────────────────────


def build_compilers(pulla: Any, stash: Any) -> dict[str, Any]:
    """The standard compiler on a pinned stash, as is ("full") and without the prepended reset wait ("body")."""
    from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash

    full = pulla.get_standard_compiler(loading_rules=[LatestFromStash(stash)])
    body = pulla.get_standard_compiler(loading_rules=[LatestFromStash(stash)])
    for stage in body.pulse_stages:
        stage.passes[:] = [p for p in stage.passes if p.__name__ != "prepend_reset"]
    return {"full": full, "body": body}


def compile_schedules(compiler: Any, circuits: list[Any], **stage_options: Any) -> tuple[list[Any], dict[str, Any]]:
    """Run the circuit and pulse stages (``Compiler.compile`` without ``finalize``): the Schedules, the context.

    ``stage_options`` set stage settings, e.g. ``{"timebox_resolution.resolve_timeboxes.neighborhood": 0}``.
    """
    settings = compiler.get_settings(circuits=circuits)
    for path, value in stage_options.items():
        *parents, leaf = path.split(".")
        node = settings.stages
        for part in parents:
            node = node[part]
        setattr(node, leaf, value)  # SettingNode replaces the frozen Setting with an updated one
        if node[leaf].value != value:
            raise RuntimeError(f"stage option {path} did not take: {node[leaf].value!r}")
    context = compiler.compiler_context(None, settings)
    return compiler.run_stages(circuits, context)


def circuit_stage_ops(compiler: Any, circuit: Any) -> dict[str, int]:
    """Op counts after the circuit stage: shows the terminal ``measure`` -> ``measure_fidelity`` conversion."""
    settings = compiler.get_settings(circuits=[circuit])
    context = compiler.compiler_context(None, settings)
    out, _ = compiler.run_stages([circuit], context, stages=list(compiler.circuit_stages))
    return dict(sorted(Counter(i.name for i in out[0].instructions).items()))


def _ns(builder: Any, channel: str, samples: int) -> float:
    return round(builder.channels[channel].duration_to_seconds(samples) * 1e9, 3)


_PROBE_PROPS = (
    "sample_rate",
    "instruction_duration_granularity",
    "integration_start_dead_time",
    "integration_stop_dead_time",
)


def readout_windows(compiler: Any, qubits: Iterable[str], topology: Any) -> dict[str, Any]:
    """Per qubit: its probe line and the readout-trigger length of ``measure`` and ``measure_fidelity``.

    The length is IQM's own (``Measure_CustomWaveforms._get_readout_trigger_durations``):
    acquisition_delay + integration_length, rounded to the probe channel's granularity, plus the
    channel's integration_stop_dead_time. The probe lines' timing properties come with it.
    """
    builder = compiler.get_schedule_builder()
    per_qubit: dict[str, Any] = {}
    lines: dict[str, Any] = {}
    for q in qubits:
        channel = builder.get_probe_channel(q)
        row: dict[str, Any] = {"probe_line": topology.component_to_probe_line[q]}
        for gate in ("measure", "measure_fidelity"):
            samples, _ = builder.get_implementation(gate, (q,))._get_readout_trigger_durations()
            row[f"{gate}_window_ns"] = _ns(builder, channel, samples)
        per_qubit[q] = row
        props = builder.channels[channel]
        lines[row["probe_line"]] = {k: _value(getattr(props, k)) for k in _PROBE_PROPS if hasattr(props, k)}
    return {"qubits": per_qubit, "probe_lines": dict(sorted(lines.items()))}


def schedule_summary(schedule: Any, builder: Any) -> dict[str, Any]:
    """Duration, and per channel the instructions by type; readout triggers with start, length and acquisitions."""
    channels: dict[str, Any] = {}
    triggers = []
    total = Counter()
    end_ns = 0.0
    for ch in sorted(schedule.channels()):
        t = 0
        kinds = Counter()
        for inst in schedule[ch]:
            kind = type(inst).__name__
            if kind in ("Wait", "Block") and inst.duration == 0:
                kind += "0"
            kinds[kind] += 1
            if kind == "ReadoutTrigger":
                triggers.append(
                    {
                        "channel": ch,
                        "start_ns": _ns(builder, ch, t),
                        "duration_ns": _ns(builder, ch, inst.duration),
                        "probe_ns": _ns(builder, ch, inst.probe_pulse.duration),
                        "acquisitions": len(inst.acquisitions),
                    }
                )
            t += inst.duration
        end_ns = max(end_ns, _ns(builder, ch, t))
        channels[ch] = dict(sorted(kinds.items()))
        total.update(kinds)
    triggers.sort(key=lambda x: (x["start_ns"], x["channel"]))
    first = next(iter(schedule.channels()))
    return {
        "duration_ns": _ns(builder, first, schedule.duration),
        "end_ns": end_ns,
        "instructions": dict(sorted(total.items())),
        "channels": channels,
        "readout_triggers": triggers,
    }


def compile_one(compilers: Mapping[str, Any], circuit: Any, **stage_options: Any) -> dict[str, Any]:
    """Body and full compile of one circuit: their summaries, and the reset prefix (their difference)."""
    out: dict[str, Any] = {}
    for kind, comp in compilers.items():
        schedules, context = compile_schedules(comp, [circuit], **stage_options)
        out[kind] = schedule_summary(schedules[0], context["builder"])
    if "full" in out and "body" in out:
        out["reset_prefix_ns"] = round(out["full"]["duration_ns"] - out["body"]["duration_ns"], 3)
    return out


def merge_circuits(n: int, barriers: bool) -> dict[str, Any]:
    """N prx(pi/2) on MERGE_QUBIT and N CZ on MERGE_PAIR, each with or without a barrier between gates."""
    from iqm.pulse import Circuit, CircuitOperation

    def chain(op: CircuitOperation) -> tuple[CircuitOperation, ...]:
        out: list[CircuitOperation] = []
        for k in range(n):
            if barriers and k:
                out.append(CircuitOperation("barrier", op.locus, {}))
            out.append(op)
        return tuple(out)

    prx = CircuitOperation("prx", (MERGE_QUBIT,), {"angle": math.pi / 2, "phase": 0.0})
    cz = CircuitOperation("cz", MERGE_PAIR, {})
    tag = "bar" if barriers else "nobar"
    return {
        "prx": Circuit(f"prx_x{n}_{tag}", chain(prx)),
        "cz": Circuit(f"cz_x{n}_{tag}", chain(cz)),
    }


def merge_test(compilers: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile N consecutive gates, with and without barriers: do pulses get merged or cancelled?"""
    rows = []
    for gate in ("prx", "cz"):
        for barriers in (False, True):
            for n in MERGE_NS:
                circuit = merge_circuits(n, barriers)[gate]
                schedules, context = compile_schedules(compilers["body"], [circuit])
                s = schedule_summary(schedules[0], context["builder"])
                if gate == "prx":
                    pulses = s["channels"].get(f"{MERGE_QUBIT}__drive.awg", {}).get("IQPulse", 0)
                else:
                    pulses = sum(v.get("FluxPulse", 0) for k, v in s["channels"].items() if "flux" in k)
                first_readout = min((t["start_ns"] for t in s["readout_triggers"]), default=None)
                rows.append(
                    {
                        "gate": gate,
                        "n": n,
                        "barriers": barriers,
                        "body_ns": s["duration_ns"],
                        "gates_ns": first_readout,
                        "pulses": pulses,
                        "flux_channels": sorted(k for k in s["channels"] if "flux" in k) if gate == "cz" else None,
                        "virtual_rz": s["instructions"].get("VirtualRZ", 0),
                    }
                )
    return rows


def qiskit_merge_test() -> list[dict[str, Any]]:
    """Offline: does qiskit's transpiler (against IQMFakeGarnet) merge the same chains before IQM sees them?"""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from qiskit import QuantumCircuit, transpile

    backend = IQMFakeGarnet()
    rows = []
    for gate in ("r", "cz"):
        for barriers in (False, True):
            for n in MERGE_NS:
                qc = QuantumCircuit(2)
                for k in range(n):
                    if barriers and k:
                        qc.barrier([0] if gate == "r" else [0, 1])
                    if gate == "r":
                        qc.r(math.pi / 2, 0.0, 0)
                    else:
                        qc.cz(0, 1)
                row = {"gate": gate, "n": n, "barriers": barriers}
                for level in (0, 1, 2, 3):
                    t = transpile(qc, backend, optimization_level=level, initial_layout=[0, 1], seed_transpiler=7)
                    ops = t.count_ops()
                    row[f"level{level}"] = {k: int(ops.get(k, 0)) for k in ("r", "cz")}
                rows.append(row)
    return rows


# ─── per device ──────────────────────────────────────────────────────────────


def _versions() -> dict[str, str]:
    names = ("iqm-pulla", "iqm-pulse", "iqm-client", "iqm-station-control-client", "iqm-exa-common", "qiskit")
    return {n: importlib.metadata.version(n) for n in names}


def check_clean(obj: Any) -> str:
    """Serialize ``obj``; raise if the text holds anything address- or credential-like."""
    text = json.dumps(obj, indent=1, default=str) + "\n"
    m = FORBIDDEN.search(text)
    if m:
        raise ValueError(f"refusing to write: forbidden pattern {m.group(0)!r} in output")
    return text


def _rel(path: Path) -> str:
    """``path`` relative to the repo when it is inside it."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def save(device: str, name: str, obj: Any) -> Path:
    """Write ``data/calibration/<device>/pulse/<name>.json``, or ``<name>.<UTC stamp>.json`` if it exists."""
    out = CALIBRATION / device / "pulse"
    out.mkdir(parents=True, exist_ok=True)
    text = check_clean(obj)
    path = out / f"{name}.json"
    if path.exists():
        path = out / f"{name}.{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(path, "x") as f:  # "x": never overwrite
        f.write(text)
    return path


def _header(device: str, calset: str | None = None) -> dict[str, Any]:
    head = {"schema": SCHEMA, "device": device, "utc": _utc(), "versions": _versions(), "dqiqm_commit": hw.git_head()}
    if calset:
        head["calibration_set_id"] = calset
    return head


def pinned_stash(pulla: Any, device: str) -> tuple[str, Any]:
    """Today's default calibration set id (via dqiqm.hw), and Pulla's stash of exactly that set."""
    calset = hw.calibration_set_id(device)
    stash = pulla.get_calibration_stash(calset)
    if not stash.observations:
        raise RuntimeError(f"{device}: calibration set {calset} came back empty")
    return calset, stash


NEIGHBORHOOD0 = {"timebox_resolution.resolve_timeboxes.neighborhood": 0}


def compile_circuits(compilers: Mapping[str, Any], circuits: Mapping[str, tuple[list[Op], Mapping[str, Any]]]) -> dict[str, Any]:
    """Each circuit compiled locally: our schedule, its ops, the circuit stage's ops, body and full, neighborhood 0."""
    compiled: dict[str, Any] = {}
    for name, (ops, entry) in circuits.items():
        circuit = to_circuit(ops, name)
        compiled[name] = {
            "our_schedule_ns": entry["duration_ns"],
            "ops": op_counts(ops),
            "after_circuit_stage": circuit_stage_ops(compilers["full"], circuit),
            **compile_one(compilers, circuit),
            "body_neighborhood0_ns": compile_one({"body": compilers["body"]}, circuit, **NEIGHBORHOOD0)["body"][
                "duration_ns"
            ],
        }
    return compiled


def garnet(pulla: Any, first_request: int = 0) -> list[Path]:
    calset, stash = pinned_stash(pulla, "garnet")
    topology = pulla.get_chip_topology()
    compilers = build_compilers(pulla, stash)

    pilots = {name: load_pilot(name) for name in PILOTS}
    ops, _ = pilots[PILOTS[0]]
    qubits, pairs = used_qubits(ops), cz_pairs(ops)
    if len(qubits) != 17 or len(pairs) != 24:
        raise ValueError(f"pilot patch: {len(qubits)} qubits, {len(pairs)} couplers; expected 17 and 24")
    couplers = {f"{a}__{b}": topology.get_coupler_for(a, b) for a, b in pairs}  # raises if not a coupler

    params = extract_parameters(stash.observations, qubits, pairs)
    record = _header("garnet", calset) | {
        "scope": "per locus: the pilot patch E1_d3_Z_r03_pA",
        "chip_label": pulla.get_chip_label(),
        "qubits": qubits,
        "couplers": couplers,
        "stash_observations": len(stash.observations),
        "implementation_classes": implementation_classes(compilers["full"], params),
        **params,
    }
    saved = [save("garnet", calset, record)]

    result = _header("garnet", calset) | {
        "readout_windows": readout_windows(compilers["full"], qubits, topology),
        "pilots": compile_circuits(compilers, pilots),
        "merge_test": merge_test(compilers),
        "qiskit_merge_test": qiskit_merge_test(),
        "requests": [dict(r) for r in REQUESTS[first_request:]],
    }
    saved.append(save("garnet", f"compile.{calset}", result))
    return saved


class CalibrationSetChanged(RuntimeError):
    """Emerald's live calibration set is not the one its circuits were placed on: recorded, not compiled."""


def emerald(pulla: Any, first_request: int = 0, expected_calset: str = EMERALD_CALSET) -> list[Path]:
    """Emerald's parameter summary; then, on ``expected_calset`` only, the compile file of eA and eB's R = 3 circuits.

    The compile file has Garnet's schema: ``readout_windows`` for every qubit of eA and eB (30),
    ``pilots`` (the key Garnet's file uses; here the four ``EMERALD_COMPILED`` circuits), and the
    request log, plus ``patches`` and the per-qubit readout calibration the windows come from.
    On another calibration set the parameter file records the live set, and this raises
    :class:`CalibrationSetChanged` before any compiler is built.
    """
    calset, stash = pinned_stash(pulla, "emerald")
    compiler = pulla.get_standard_compiler(loading_rules=[])  # for the gate definitions only
    params = extract_parameters(stash.observations)
    record = _header("emerald", calset) | {
        "scope": "summary over every locus: median, min, max",
        "chip_label": pulla.get_chip_label(),
        "stash_observations": len(stash.observations),
        "implementation_classes": implementation_classes(compiler, params),
        **summarize_parameters(params),
        "requests": [dict(r) for r in REQUESTS[first_request:]],
    }
    saved = [save("emerald", calset, record)]
    if calset != expected_calset:
        raise CalibrationSetChanged(
            f"emerald's live calibration set is {calset}, not {expected_calset}, the one its E1 circuits were placed on. "
            f"Recorded in {_rel(saved[0])}; nothing compiled."
        )

    circuits = {name: load_emerald(name) for name in EMERALD_COMPILED}
    patches = emerald_patches(circuits)
    topology = pulla.get_chip_topology()
    for row in patches.values():  # raises if a pair is not a coupler of the chip
        row["couplers"] = {pair: topology.get_coupler_for(*pair.split("__")) for pair in row["couplers"]}
    qubits = sorted({q for row in patches.values() for q in row["qubits"]}, key=lambda q: int(q[2:]))
    compilers = build_compilers(pulla, stash)
    result = _header("emerald", calset) | {
        "scope": f"patches {', '.join(patches)}: {len(qubits)} qubits; the d=3, R=3 E1 circuits compiled locally",
        "patches": patches,
        "readout_windows": readout_windows(compilers["full"], qubits, topology),
        "readout_parameters": extract_parameters(stash.observations, qubits, [], gates=READOUT_GATES),
        "pilots": compile_circuits(compilers, circuits),
        "requests": [dict(r) for r in REQUESTS[first_request:]],
    }
    saved.append(save("emerald", f"compile.{calset}", result))
    return saved


# ─── report ──────────────────────────────────────────────────────────────────


_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
RESULT_FILES = {
    "access": re.compile(r"access(\.\d{8}T\d{6}Z)?\.json"),
    "parameters": re.compile(rf"{_UUID}(\.\d{{8}}T\d{{6}}Z)?\.json"),
    "compile": re.compile(rf"compile\.{_UUID}(\.\d{{8}}T\d{{6}}Z)?\.json"),
}


def newest(device: str, kind: str) -> Path | None:
    """The newest result file of ``kind`` (access, parameters, compile): by the UTC time inside it."""
    folder = CALIBRATION / device / "pulse"
    files = [p for p in folder.glob("*.json") if RESULT_FILES[kind].fullmatch(p.name)] if folder.is_dir() else []
    return max(files, key=lambda p: json.loads(p.read_text())["utc"]) if files else None


def report() -> None:
    for device in DEVICES:
        path = newest(device, "access")
        if path is None:
            print(f"{device}: no access record")
            continue
        acc = json.loads(path.read_text())
        line = f"{device}: Pulla constructor {acc['constructor']}"
        if acc["constructor"] != "ok":
            line += f" ({acc['error_class']}, HTTP {acc['http_status']})"
        print(line, "| requests:", ", ".join(f"{r['method']} {r['path']} {r['status']}" for r in acc["requests"]))
    for device in DEVICES:
        path = newest(device, "parameters")
        if path is not None:
            par = json.loads(path.read_text())
            print(f"{device} {par['calibration_set_id']}: defaults {par['default_implementations']}")
            if par["override_default_for_loci"]:
                print(f"  per-locus overrides {par['override_default_for_loci']}")
            print("  " + ", ".join(f"{k} = {v['class']}" for k, v in par["implementation_classes"].items()))
    for device in DEVICES:
        report_compile(device)


def report_compile(device: str) -> None:
    path = newest(device, "compile")
    if path is None:
        return
    res = json.loads(path.read_text())
    print(f"\n{device} calibration set {res['calibration_set_id']} ({path.relative_to(REPO)})")
    windows = res["readout_windows"]["qubits"]
    for gate in ("measure", "measure_fidelity"):
        ns = [w[f"{gate}_window_ns"] for w in windows.values()]
        print(f"  {gate} readout window over the {len(ns)} qubits: median {statistics.median(ns):.0f} ns, {min(ns):.0f}-{max(ns):.0f}")
    for patch, row in res.get("patches", {}).items():
        mid = {q: windows[q]["measure_window_ns"] for q in row["ancillas"]}
        slow = max(mid.values())
        print(f"  patch {patch}: mid-circuit window (slowest of its 8 ancillas) {slow:.0f} ns on"
              f" {', '.join(q for q, v in mid.items() if v == slow)}; terminal (slowest of its 17 qubits)"
              f" {max(windows[q]['measure_fidelity_window_ns'] for q in row['qubits']):.0f} ns")
    for name, c in res["pilots"].items():
        print(
            f"  {name}: ours {c['our_schedule_ns']:.0f} ns | compiled body {c['body']['duration_ns']:.0f} ns"
            f" ({c['body']['instructions'].get('ReadoutTrigger', 0)} readout triggers;"
            f" neighborhood 0: {c['body_neighborhood0_ns']:.0f} ns), full {c['full']['duration_ns']:.0f} ns"
            f" (reset prefix {c['reset_prefix_ns']:.0f} ns) | after circuit stage {c['after_circuit_stage']}"
        )
    if "merge_test" not in res:
        return
    print("  merge test (compiled body; pulses = drive IQPulses for prx, flux pulses for cz):")
    for r in res["merge_test"]:
        print(
            f"    {r['gate']:3s} x{r['n']:<2d} barriers={r['barriers']!s:5s} pulses={r['pulses']:3d}"
            f" gates_ns={r['gates_ns']} body_ns={r['body_ns']}"
        )
    print("  qiskit transpile (IQMFakeGarnet), gate counts at optimization levels 0-3:")
    for r in res["qiskit_merge_test"]:
        counts = " ".join(f"L{lv}:{r[f'level{lv}'][r['gate']]}" for lv in range(4))
        print(f"    {r['gate']:2s} x{r['n']:<2d} barriers={r['barriers']!s:5s} {counts}")


# ─── the Emerald comparison (offline): the pilot EP0 under the fallback and the compiled window ──

EP0 = "E1_d3_Z_r03_eA"
EP0_JOB = REPO / "InfoRecord" / "emerald" / "raw" / "20261024-emerald-pilot" / "01a17017-0aef-7503-a36d-6d3151635fdf"
EP0_DATASET = REPO / "InfoRecord" / "emerald" / "datasets" / "E1" / f"ep0_{EP0}"
EP0_CHECKS = REPO / "InfoRecord" / "emerald" / "derived" / "E1" / f"ep0_{EP0}" / "pilot_checks.json"
#: The pilot's before snapshot, which its decoder was built from.
EP0_BEFORE = CALIBRATION / "emerald" / f"{EMERALD_CALSET}.20261024T224404Z.json"
COMPARE_SHOTS, COMPARE_SEED = 100_000, 2026  # the model sample, as the pilot's record and garnet_reprior.py take it
DOC = REPO / "docs" / "pulla-check.md"
BEGIN, END = "<!-- emerald-compile:begin -->", "<!-- emerald-compile:end -->"


def detector_rounds_and_types(spec: Any) -> tuple[Any, Any]:
    """Each detector's round t (its third coordinate) and type, X or Z (its ancilla's), as garnet_reprior.py has it."""
    import numpy as np

    coords = spec.stim.get_detector_coordinates()
    t = np.array([int(coords[i][2]) for i in range(spec.stim.num_detectors)])
    xy_x = {tuple(spec.stim.get_final_qubit_coordinates()[q]) for q in spec.x_ancillas}
    typ = np.array(["X" if (coords[i][0], coords[i][1]) in xy_x else "Z" for i in range(spec.stim.num_detectors)])
    return t, typ


def per_round(dets: Any, t: Any, typ: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"all": {}, "X": {}, "Z": {}}
    for k in sorted(set(t.tolist())):
        out["all"][str(k)] = float(dets[:, t == k].mean())
        for ty in ("X", "Z"):
            m = (t == k) & (typ == ty)
            if m.any():
                out[ty][str(k)] = float(dets[:, m].mean())
    return out


def evaluate(name: str, dec: Any, spec: Any, dets_hw: Any, obs_hw: Any, t: Any, typ: Any, info: Mapping[str, Any]) -> dict[str, Any]:
    """One prior on EP0: PyMatching on the hardware shots, and the prior's own model sampled (garnet_reprior's columns)."""
    import numpy as np

    from dqiqm import decode as D
    from dqiqm import stats

    dem = dec.detector_error_model(decompose_errors=True)
    fails = D.decode(dec, dets_hw)[:, 0] != obs_hw[:, 0]
    k, n = int(fails.sum()), len(dets_hw)
    raw = dec.compile_sampler(seed=COMPARE_SEED).sample(COMPARE_SHOTS).astype(np.uint8)
    dets_m, obs_m = D.detection_events(spec.stim, raw)
    km = int((D.decode(dec, dets_m)[:, 0] != obs_m[:, 0]).sum())
    rate_m = np.clip(dets_m.mean(axis=0), 1e-9, 1 - 1e-9)
    hw_k = dets_hw.sum(axis=0)
    rounds_hw, rounds_m = per_round(dets_hw, t, typ), per_round(dets_m, t, typ)
    e, elo, ehi = (float(x[0]) for x in stats.eps_single(np.array([k]), np.array([n]), np.array([spec.rounds])))
    return {
        "name": name,
        "info": dict(info),
        "decoder_sha256": hashlib.sha256(str(dec).encode()).hexdigest(),
        "dem_errors": dem.num_errors,
        "graphlike_distance": len(dec.shortest_graphlike_error()),
        "hw_failures": k, "hw_shots": n, "hw_p_l": k / n, "hw_p_l_wilson95": [float(x) for x in stats.wilson(k, n)],
        "hw_eps_single": [e, elo, ehi],
        "model_failures": km, "model_shots": COMPARE_SHOTS, "model_p_l": km / COMPARE_SHOTS,
        "model_p_l_wilson95": [float(x) for x in stats.wilson(km, COMPARE_SHOTS)],
        "model_detection_fraction": float(dets_m.mean()),
        "per_round_model": rounds_m, "per_round_hw": rounds_hw,
        "per_round_abs_error_sum": float(sum(abs(rounds_m["all"][r] - rounds_hw["all"][r]) for r in rounds_hw["all"])),
        # the binomial log-likelihood of the hardware's per-detector counts under the model's per-detector rates
        "detector_loglik": float(np.sum(hw_k * np.log(rate_m) + (n - hw_k) * np.log(1 - rate_m))),
        "_fails": fails,
    }


def mcnemar(a: Any, b: Any) -> dict[str, Any]:
    """Failures of b only, of a only, on the same shots, and the exact two-sided McNemar p-value."""
    only_b, only_a = int((b & ~a).sum()), int((a & ~b).sum())
    m = only_a + only_b
    tail = sum(math.comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2**m if m else 1.0
    return {"only_this": only_b, "only_fallback": only_a, "mcnemar_p": min(1.0, 2 * tail)}


def compare_emerald() -> dict[str, Any]:
    """EP0 re-decoded offline under the pilot's own decoder, the 360 ns fallback (today's batch prior) and the compiled window."""
    import numpy as np

    from dqiqm import decode as D
    from dqiqm.experiments import circuits as C
    from dqiqm.experiments import device as dev
    from dqiqm.experiments import emerald as EM

    if EM.CALSET != EMERALD_CALSET:
        raise SystemExit(f"emerald.CALSET {EM.CALSET} != {EMERALD_CALSET}")
    windows = dev.compiled_readout_windows(EMERALD_CALSET, device="emerald")
    if windows is None:
        raise SystemExit(f"no Emerald compile file for {EMERALD_CALSET}: run the live script first")
    compiled = json.loads((CALIBRATION / "emerald" / "pulse" / f"compile.{EMERALD_CALSET}.json").read_text())
    # the circuits as the pilot and the batch rebuild them (the committed snapshot), checked against placement.json
    suite = EM.build(EM.live_calibration(EM.DEFAULT_SNAPSHOT), experiments=("E1",), e1_rounds=(3,))
    placement = json.loads((EMERALD_CIRCUITS / "placement.json").read_text())
    specs = {name: suite[name] for name in EMERALD_COMPILED}
    for name, spec in specs.items():
        if C.native_sha256(spec.hardware) != placement[name]["native_sha256"]:
            raise SystemExit(f"{name}: rebuilt native_sha256 differs from placement.json: not the circuit that runs")
    spec = specs[EP0]
    raw, meta = D.load_memory(EP0_JOB, 0, circuit=spec.stim)
    if not np.array_equal(raw, np.load(EP0_DATASET / "raw_meas.npy")):
        raise SystemExit("memory.npz c0 differs from the dataset's raw_meas.npy")
    dets, obs = D.detection_events(spec.stim, raw)
    if not np.array_equal(dets, np.load(EP0_DATASET / "det_events.npy")):
        raise SystemExit("detection events differ from the dataset's det_events.npy")
    cal = EM.live_calibration(EP0_BEFORE)
    if cal.calibration_set_id != meta["calibration_set_id"] or cal.calibration_set_id != EMERALD_CALSET:
        raise SystemExit(f"the before snapshot is {cal.calibration_set_id}, the job ran on {meta['calibration_set_id']}")
    record = json.loads(EP0_CHECKS.read_text())

    # the windows, per patch: the fallback is the slowest ancilla's measure duration, the probe alone
    patch_rows = {}
    for name, s in specs.items():
        patch = s.params["patch"]
        ancillas = sorted((s.physical[q] for q, r in s.roles.items() if r == "ancilla"), key=lambda q: int(q[2:]))
        fallback, _ = dev.readout_layer_ns(cal, ancillas, None)
        window, source = dev.readout_layer_ns(cal, ancillas, windows)
        terminal, _ = dev.readout_layer_ns(cal, list(s.physical.values()), windows, terminal=True)
        triggers = compiled["pilots"][name]["body"]["readout_triggers"]
        starts = sorted({x["start_ns"] for x in triggers})
        layers = [max(x["duration_ns"] for x in triggers if x["start_ns"] == st) for st in starts]
        row = {"ancillas": ancillas, "fallback_ns": fallback, "compiled_ns": window, "source": source,
               "slowest": [q for q in ancillas if windows.of(q, "measure_window_ns") == window],
               "terminal_ns": terminal, "ratio": window / fallback}
        seen = patch_rows.setdefault(patch, row | {"compiled_layers_ns": {}, "body_ns": {}, "our_schedule_ns": {}})
        if {k: seen[k] for k in row} != row:
            raise SystemExit(f"{name}: patch {patch}'s windows differ between bases")
        seen["compiled_layers_ns"][name] = layers
        seen["body_ns"][name] = compiled["pilots"][name]["body"]["duration_ns"]
        seen["our_schedule_ns"][name] = compiled["pilots"][name]["our_schedule_ns"]

    t, typ = detector_rounds_and_types(spec)
    idle = EM.readout_window_ns(spec, dev.Durations.from_calibration(cal))  # the pilot's rule: 360 ns
    anchor = dev.pre_fix_noise(EM.calibrated_noise(cal, spec, idle_ns=idle))
    nf, info_f = dev.hardware_noise(cal, spec, windows=None)
    nc, info_c = dev.hardware_noise(cal, spec)
    pilot_info = {"readout_window_ns": idle, "readout_window_source": "measure_ns (emerald.readout_window_ns)",
                  "idle_ns": idle, "idle_channel": "depolarize", "t2": "ramsey", "final_readout": "measure",
                  "idle_after_last_round": True}
    rows = [evaluate("pilot_record", spec.noisy(anchor), spec, dets, obs, t, typ, pilot_info),
            evaluate("fallback_360", spec.noisy(nf), spec, dets, obs, t, typ, info_f),
            evaluate("compiled", spec.noisy(nc), spec, dets, obs, t, typ, info_c)]
    if (rows[0]["hw_failures"], rows[0]["dem_errors"], rows[0]["model_p_l"]) != (
        record["decode"]["pm_failures"], record["decoder"]["dem_errors"], record["model"]["p_l"]
    ):
        raise SystemExit(f"pilot_record does not reproduce the pilot's decode ({rows[0]['hw_failures']} failures, "
                         f"{rows[0]['dem_errors']} DEM errors, model P_L {rows[0]['model_p_l']})")
    for x in rows:
        x["vs_fallback"] = mcnemar(rows[1]["_fails"], x["_fails"])
    for x in rows:
        del x["_fails"]
    eb = specs["E1_d3_Z_r03_eB"]
    return {"job_id": meta["job_id"], "circuit": EP0, "calibration_set_id": cal.calibration_set_id,
            "before_snapshot": str(EP0_BEFORE.relative_to(REPO)), "compile_file": windows.source,
            "patches": patch_rows, "info_eA": {"fallback_360": info_f, "compiled": info_c},
            "info_eB": {"fallback_360": dev.hardware_noise(cal, eb, windows=None)[1], "compiled": dev.hardware_noise(cal, eb)[1]},
            "rows": rows}


def compare_markdown(res: Mapping[str, Any]) -> str:
    rows, pr = res["rows"], res["patches"]
    rounds = list(rows[0]["per_round_hw"]["all"])
    hw = rows[0]["per_round_hw"]
    out = [
        BEGIN,
        f"Generated by `scripts/pulla_check.py --compare`: job `{res['job_id']}` (EP0, `{res['circuit']}`, "
        f"{rows[0]['hw_shots']:,} shots, R = 3, calibration set `{res['calibration_set_id']}`), re-decoded from its committed "
        f"`memory.npz` with PyMatching. Priors from the pilot's before snapshot (`{res['before_snapshot']}`); compiled windows "
        f"from `{res['compile_file']}`. Model columns: each prior's own noisy circuit, {COMPARE_SHOTS:,} Stim shots, seed "
        f"{COMPARE_SEED}. P_L intervals are Wilson 95%.",
        "",
        "**The readout window, per patch.** Fallback: the slowest ancilla's `measure` duration, the probe alone "
        "(`device.readout_layer_ns` without compile data). Compiled: the slowest ancilla's `measure_window_ns`. Compiled "
        "layers: each readout layer's longest trigger in the compiled body (rounds 1, 2, then the terminal layer).",
        "",
        "| patch | ancillas | fallback (ns) | compiled mid-circuit window (ns) | slowest ancilla | compiled / fallback | "
        "terminal `measure_fidelity` window, slowest of 17 (ns) | compiled layers, Z / X (ns) | our schedule → compiled body, Z / X (ns) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for patch, r in pr.items():
        names = sorted(r["compiled_layers_ns"], key=lambda n: "_X_" in n)
        layers = " / ".join(", ".join(f"{v:,.0f}" for v in r["compiled_layers_ns"][n]) for n in names)
        body = " / ".join(f"{r['our_schedule_ns'][n]:,.0f} → {r['body_ns'][n]:,.0f}" for n in names)
        slowest = "all 8" if r["slowest"] == r["ancillas"] else ", ".join(r["slowest"])
        out.append(f"| {patch} | {', '.join(r['ancillas'])} | {r['fallback_ns']:g} | **{r['compiled_ns']:,.0f}** | "
                   f"{slowest} | {r['ratio']:.2f}× | {r['terminal_ns']:,.0f} | {layers} | {body} |")
    keys = ["readout_window_ns", "readout_window_source", "delay_ns", "idle_ns", "idle_channel", "idle_frame", "t2",
            "final_readout", "idle_after_last_round", "family", "calibration_set_id"]
    out += ["", "**`device.hardware_noise(cal, spec)`'s info dict**, before (no Emerald compile file: the `measure_ns` "
            "fallback, which is what `windows=None` gives) and after:", "",
            "| key | eA before | eA after | eB before | eB after |", "|---|---|---|---|---|"]
    for k in keys:
        cells = [res[p][w].get(k) for p in ("info_eA", "info_eB") for w in ("fallback_360", "compiled")]
        out.append(f"| `{k}` | " + " | ".join(f"`{c}`" for c in cells) + " |")
    out += ["", "**EP0 decoded, and the detection fraction per detector round t** (t = 0: round 1, Z-type only; t = 3: the "
            "final data detectors). `pilot_record` is the pilot's own decoder, rebuilt (it must give the record's "
            f"{rows[0]['hw_failures']} failures, {rows[0]['dem_errors']} DEM errors and model P_L {rows[0]['model_p_l']:.5f}); "
            "`fallback_360` is today's batch prior; `compiled` is the prior with this task's compile file.", "",
            "| priors | idle (ns) | final readout | idle after last round | DEM errors | d | hardware P_L (1,000 shots, R = 3) "
            "| ε single point (R = 3) | model P_L | " + " | ".join(f"t = {r}, model" for r in rounds)
            + " | Σ\\|model − hw\\| | detector log-lik | vs fallback_360: fails only here / only there (McNemar p) |",
            "|" + "---|" * (12 + len(rounds)),
            "| **hardware** | | | | | | | | | " + " | ".join(f"**{hw['all'][r]:.4f}**" for r in rounds) + " | | | |"]
    for x in rows:
        i, m, v = x["info"], x["per_round_model"]["all"], x["vs_fallback"]
        lo, hi = x["hw_p_l_wilson95"]
        e, elo, ehi = x["hw_eps_single"]
        mlo, mhi = x["model_p_l_wilson95"]
        out.append(
            f"| `{x['name']}` | {i['idle_ns']:g} | {i['final_readout'].split(' ')[0]} | {'yes' if i['idle_after_last_round'] else 'no'} "
            f"| {x['dem_errors']} | {x['graphlike_distance']} | {x['hw_failures']}/{x['hw_shots']} = **{x['hw_p_l']:.3f}** "
            f"[{lo:.3f}, {hi:.3f}] | {e:.4f} [{elo:.4f}, {ehi:.4f}] | {x['model_p_l']:.4f} [{mlo:.4f}, {mhi:.4f}] | "
            + " | ".join(f"{m[r]:.4f}" for r in rounds)
            + f" | {x['per_round_abs_error_sum']:.3f} | {x['detector_loglik']:.0f} | "
            + ("" if x["name"] == "fallback_360" else f"{v['only_this']} / {v['only_fallback']} (p = {v['mcnemar_p']:.2f})")
            + " |")
    out += ["", "Hardware over model, per round:", "", "| priors | " + " | ".join(f"t = {r}" for r in rounds) + " |",
            "|" + "---|" * (1 + len(rounds))]
    for x in rows:
        m = x["per_round_model"]["all"]
        out.append(f"| `{x['name']}` | " + " | ".join(f"{hw['all'][r] / m[r]:.2f}×" for r in rounds) + " |")
    tk = [(ty, r) for ty in ("Z", "X") for r in hw[ty]]
    out += ["", "By stabilizer type (detection fraction):", "", "| priors | " + " | ".join(f"{ty} t = {r}" for ty, r in tk) + " |",
            "|" + "---|" * (1 + len(tk)), "| **hardware** | " + " | ".join(f"**{hw[ty][r]:.4f}**" for ty, r in tk) + " |"]
    for x in rows:
        m = x["per_round_model"]
        out.append(f"| `{x['name']}` | " + " | ".join(f"{m[ty][r]:.4f}" for ty, r in tk) + " |")
    out.append(END)
    return "\n".join(out)


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", action="store_true", help="print the committed results; no network")
    ap.add_argument("--devices", nargs="+", default=list(DEVICES), choices=DEVICES)
    ap.add_argument("--compare", action="store_true", help="offline: EP0 under the fallback and the compiled Emerald window")
    ap.add_argument("--doc", action="store_true", help=f"with --compare: write the tables into {DOC.relative_to(REPO)}")
    ap.add_argument("--json", type=Path, help="with --compare: write the full results here")
    args = ap.parse_args()
    if args.report:
        report()
        return 0
    if args.compare:
        res = compare_emerald()
        md = compare_markdown(res)
        print(md)
        if args.json:
            args.json.write_text(json.dumps(res, indent=1) + "\n")
            print(f"wrote {args.json}")
        if args.doc:
            text = DOC.read_text()
            if BEGIN not in text or END not in text:
                raise SystemExit(f"{DOC} has no {BEGIN} ... {END} section")
            head, rest = text.split(BEGIN, 1)
            DOC.write_text(head + md + rest.split(END, 1)[1])
            print(f"wrote {DOC.relative_to(REPO)}")
        return 0

    for device in args.devices:
        first = len(REQUESTS)
        pulla, record = connect(device)
        record |= {k: v for k, v in _header(device).items() if k not in record}
        print(json.dumps({k: record[k] for k in ("device", "constructor", "requests")}, indent=1))
        print("saved", _rel(save(device, "access", record)))
        if pulla is None:
            continue  # the brief: stop there for that device
        try:
            for path in (garnet if device == "garnet" else emerald)(pulla, first):
                print("saved", _rel(path))
        except CalibrationSetChanged as exc:
            print(f"STOP: {exc}")
            return 2
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
