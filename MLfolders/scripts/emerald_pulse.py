"""emerald-pulse: the Emerald pilot circuit EP0 (``E1_d3_Z_r03_eA``) as a pulse-level job we compiled ourselves.

    PYTHONHASHSEED=0 uv run python scripts/emerald_pulse.py compile        # live, read-only (pulla_check's guard): snapshot, compile, record
    PYTHONHASHSEED=0 DQ_QPU=run DQ_QPU_TASK=emerald-pulse-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \\
        uv run python scripts/emerald_pulse.py submit     # the one job: recompile, match the record, hw.submit_playlist
    uv run python scripts/emerald_pulse.py recover --job-id J   # read-only; only if submit failed after the job id
    uv run python scripts/emerald_pulse.py standin        # offline: EP0's own shots through the playlist path
    uv run python scripts/emerald_pulse.py analyze        # offline: decode the job, compare with EP0
    uv run python scripts/emerald_pulse.py payload-check  # read-only GET: the server's stored payload = what we sent

**compile** connects through ``pulla_check.connect`` (its class-level guard: ``Pulla.submit_playlist``
and every writing server call refuse, and only GETs are sent). It checks that the server's default
calibration set is :data:`CALSET`, that the live architecture serves EP0's 17 qubits and 24 couplers,
and pins Pulla's stash of that set. The circuit is the frozen EP0: rebuilt from the committed snapshot,
its ``native_sha256`` checked against ``placement.json``, and compiled from exactly the instructions EP0's
run request sent (``circuits.sent``), barriers and all, with Pulla's standard compiler at 1,000 shots.
It writes ``InfoRecord/emerald/raw/<YYYYMMDD>-emerald-pulse-pilot/compile.json`` and the serialized run
definition beside it (``run_definition.pb.gz``), and never overwrites: run again, it compares the new
compile with the record instead.

**submit** must not run in a process that installed pulla_check's guard: ``hw.submit_playlist`` would hit
the refusing class method inside its ``try`` and write a fail-closed row for a job never sent. So it
recompiles with ``hw.pulla`` (an instance-level read-only Pulla), and refuses unless the playlist, the
run definition, the readout labels and IQM's ``qpu_runtime`` estimate equal the committed record, the
estimate is at most :data:`RUNTIME_CAP_S`, the arming is exactly the brief's, the ledger has no row of the
task and the slim ``context`` deep-copies. Then it calls ``hw.submit_playlist`` once. A refusal by the
server is a result: its class, HTTP status and scrubbed message go in ``submit.json``, and nothing is
retried.

**analyze** maps the saved sweep results to EP0's 33 classical bits by readout label
(``<qubit>__c_33_0_<j>``, each label's qubit checked against the placement), decodes them with EP0's own
decoder (``pilot_record``) and with ``device.hardware_noise`` (the compiled 1,036 ns window), exactly as
``pulla_check.py --compare`` decodes EP0, and compares P_L and the per-round detection fractions with EP0.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dqiqm import decode as D
from dqiqm import hw
from dqiqm import stats
from dqiqm.experiments import circuits as C
from dqiqm.experiments import device as dev
from dqiqm.experiments import emerald as EM

REPO = Path(__file__).resolve().parents[1]


def _script(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PC = _script("pulla_check")  # importing it installs nothing: only PC.connect installs the guard
EP = _script("emerald_pilot")

TASK = "emerald-pulse-pilot"  # the brief's DQ_QPU_TASK; hw names the raw folder after it
DEVICE = "emerald"
CIRCUIT = "E1_d3_Z_r03_eA"
SHOTS = 1000
CALSET = "23e12f6b-c23c-4736-8736-b0142989c0e6"
RUNTIME_CAP_S = 10.0
ARMING = {"DQ_QPU": "run", "DQ_QPU_TASK": TASK, "DQ_QPU_MAX_JOBS": "1", "DQ_QPU_MAX_SHOTS": str(SHOTS)}
PURPOSE = f"emerald-pulse EP0 {CIRCUIT} as a locally compiled Pulla playlist"
RAW_ROOT = EM.RECORD / "raw"
PLACEMENT = EM.RECORD / "circuits" / "E1" / "placement.json"
EP0_P_L = (129, 1000)  # EP0's pilot_record decode (docs/pulla-check.md)
KEY = re.compile(r"c_33_0_(\d+)")
N_CLBITS = 33


def log(msg: str = "") -> None:
    print(msg, flush=True)


def today() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d")


def record_dir(day: str) -> Path:
    return RAW_ROOT / f"{day}-{TASK}"


def write_new(path: Path, text: str | bytes) -> None:
    """Write a file that must not exist yet (raw records are never overwritten)."""
    with open(path, "xb" if isinstance(text, bytes) else "x") as f:
        f.write(text)


def require_hash_seed() -> None:
    """The compiler's probe-all measurement takes its qubit order from a set of strings, so the playlist (and its
    hash) depends on Python's string hashing. Compile and submit both run with ``PYTHONHASHSEED=0``."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("refusing: run with PYTHONHASHSEED=0 (the playlist's hash depends on it)")


def refuse_if_armed() -> None:
    if any(k in os.environ for k in ARMING):
        raise SystemExit("refusing: DQ_QPU* is set; only `submit` may run armed")


# ---------------------------------------------------------------------------
# The frozen circuit.
# ---------------------------------------------------------------------------


def frozen() -> tuple[Any, list[Any], dict[str, Any]]:
    """EP0 rebuilt from the committed snapshot: (spec, the instructions EP0's request sent, the checks)."""
    suite = EM.build(EM.live_calibration(EM.DEFAULT_SNAPSHOT), experiments=("E1",), e1_rounds=(3,))
    spec = suite[CIRCUIT]
    entry = json.loads(PLACEMENT.read_text())[CIRCUIT]
    native = C.native_sha256(spec.hardware)
    if native != entry["native_sha256"]:
        raise SystemExit(f"FREEZE: native_sha256 {native} != placement.json {entry['native_sha256']}")
    sent = C.sent(spec.hardware)
    if hw.instructions_sha256(sent) != native:
        raise SystemExit("the sent instructions do not hash to native_sha256")
    ops, _ = PC.load_emerald(CIRCUIT)  # the committed QASM: sha256, op counts, qubits against placement.json
    if [(o.name, tuple(PC.qubit_name(q) for q in o.qubits)) for o in ops] != [(i.name, tuple(i.locus)) for i in sent]:
        raise SystemExit("the committed QASM and the sent instructions differ")
    counts = dict(sorted(Counter(i.name for i in sent).items()))
    if counts.get("measure") != N_CLBITS or counts.get("barrier", 0) == 0:
        raise SystemExit(f"unexpected ops {counts}")
    return spec, sent, {"native_sha256": native, "qasm_sha256": entry["qasm_sha256"], "ops": counts,
                        "instructions": len(sent), "swaps": entry["swaps"]}


def fresh_circuit(sent: Sequence[Any]) -> Any:
    """A new ``iqm.pulse.Circuit`` of copies: the compiler renames terminal measurements in place."""
    from iqm.pulse import Circuit

    return Circuit(CIRCUIT, tuple(copy.deepcopy(list(sent))))


def expected_qubits(spec: Any) -> list[str]:
    """The device qubit measured into each classical bit j."""
    return [spec.physical[q] for q in C.measured_qubits(spec.stim)]


# ---------------------------------------------------------------------------
# Compile, and what the run definition holds.
# ---------------------------------------------------------------------------


def compile_run(pulla: Any, stash: Any, sent: Sequence[Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Pulla's standard compiler on the pinned stash, at :data:`SHOTS` shots: (run definition, context)."""
    from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash

    compiler = pulla.get_standard_compiler(loading_rules=[LatestFromStash(stash)])
    circuit = fresh_circuit(sent)
    settings = compiler.get_settings(circuits=[circuit])
    settings.set_shots(SHOTS)
    return compiler.compile(circuits=[circuit], settings=settings)


def serialized(run_definition: Any) -> bytes:
    from iqm.station_control.client.serializers.run_serializers import serialize_run_definition

    return serialize_run_definition(run_definition).SerializeToString(deterministic=True)


def canonical_sha256(run_definition: Any) -> str:
    """sha256 of the run definition with its packed sweep definition serialized deterministically.

    ``serialize_run_definition`` packs the sweep definition into a protobuf ``Any``, whose bytes keep map
    fields in arbitrary order, so the whole message's bytes differ between processes for the same content.
    """
    from iqm.station_control.client.serializers.run_serializers import serialize_run_definition
    from iqm.station_control.client.serializers.sweep_serializers import serialize_sweep_definition

    proto = serialize_run_definition(run_definition)
    proto.ClearField("sweep_definition_payload")
    sweep = serialize_sweep_definition(run_definition.sweep_definition)
    h = hashlib.sha256(proto.SerializeToString(deterministic=True))
    h.update(sweep.SerializeToString(deterministic=True))
    return h.hexdigest()


def runtime_estimate(run_definition: Any) -> float:
    """IQM's own QPU-runtime estimate, in seconds (``station_control/interface/models/sweep.py``)."""
    return float(run_definition.sweep_definition.qpu_runtime)


def slim_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """What result post-processing reads, and nothing that might not deep-copy (Pulla copies it after the POST)."""
    slim = {"readout_metrics": context["readout_metrics"],
            "mapped_readout_keys": dict(context.get("mapped_readout_keys") or {})}
    copy.deepcopy(slim)
    return slim


def readout_map(labels: Sequence[str], qubits: Sequence[str]) -> dict[str, int]:
    """Readout label -> classical bit, for the labels ``<qubit>__c_33_0_<j>``; each qubit checked. Raises ``ValueError``."""
    out: dict[str, int] = {}
    for label in labels:
        qubit, _, key = label.partition("__")
        m = KEY.fullmatch(key)
        if m is None:
            continue
        j = int(m[1])
        if j >= len(qubits) or qubits[j] != qubit:
            raise ValueError(f"{label}: clbit {j} is measured on {qubits[j] if j < len(qubits) else '?'}, not {qubit}")
        if j in out.values():
            raise ValueError(f"{label}: clbit {j} has two labels")
        out[label] = j
    if sorted(out.values()) != list(range(len(qubits))):
        raise ValueError(f"labels cover clbits {sorted(out.values())}, not 0..{len(qubits) - 1}")
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def describe(run_definition: Any, context: Mapping[str, Any], spec: Any) -> dict[str, Any]:
    """Everything the record keeps about a compiled run definition."""
    sw = run_definition.sweep_definition
    shape = hw.playlist_shape(run_definition)
    reps = sw.settings.find_by_name("options.playlist_repeats").value
    end_delay = sw.settings.find_by_name("options.end_delay").value
    if reps != shape.shots:
        raise ValueError(f"options.playlist_repeats {reps} != controllers.options.playlist_repeats {shape.shots}")
    playlist = sw.playlist
    segment_ns = {}
    for ch, instructions in playlist.segments[0].instructions.items():
        desc = playlist.channel_descriptions[ch]
        samples = sum(desc.instruction_table[i].duration_samples for i in instructions)
        segment_ns[ch] = round(samples / desc.channel_config.sampling_rate * 1e9, 3)
    labels = list(sw.return_parameters)
    rmap = readout_map(labels, expected_qubits(spec))
    occurrences = {k: list(v) for k, v in context["readout_metrics"].integration_occurrences.items()}
    arp = run_definition.additional_run_properties
    blob = serialized(run_definition)
    return {
        "playlist_sha256": shape.playlist_sha256,
        "run_definition_sha256": canonical_sha256(run_definition),
        "run_definition_bytes": len(blob),
        "shape": {"segments": shape.segments, "spots": shape.spots, "shots": shape.shots,
                  "averaging_bins": shape.averaging_bins, "circuits": shape.circuits, "qubits": shape.qubits},
        "qpu_runtime_s": runtime_estimate(run_definition),
        "qpu_runtime_cap_s": RUNTIME_CAP_S,
        "end_delay_s": end_delay,
        "segment_duration_ns": {"max": max(segment_ns.values()), "min": min(segment_ns.values()),
                                "channels": len(segment_ns), "per_channel": dict(sorted(segment_ns.items()))},
        "return_parameters": labels,
        "readout_map": rmap,
        "integration_occurrences": dict(sorted(occurrences.items())),
        "mapped_readout_keys": dict(context.get("mapped_readout_keys") or {}),
        "couplers": list(arp.get("couplers") or []),
        "probe_lines": sorted(set((arp.get("probe_lines") or {}).values())),
        "readout_components": len(arp.get("readout_components") or []),
        "use_timeslot": False,
    }


def stash_digest(stash: Any) -> dict[str, Any]:
    obs = stash.observations
    items = sorted((str(k), repr(getattr(o, "value", o))) for k, o in obs.items())
    return {"observations": len(items), "sha256": hashlib.sha256(json.dumps(items).encode()).hexdigest()}


MATCH_KEYS = ("playlist_sha256", "run_definition_sha256", "shape", "qpu_runtime_s", "return_parameters", "readout_map",
              "integration_occurrences")


def mismatches(new: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
    return [k for k in MATCH_KEYS if new.get(k) != record.get(k)]


def check_against_iqm_post_processing(context: Mapping[str, Any], rmap: Mapping[str, int], raw: np.ndarray) -> bool:
    """IQM's own ``construct_circuit_execution_results`` on stand-in sweep results gives back ``raw``, column by clbit."""
    from iqm.cpc.compiler.post_process import PullaData, construct_circuit_execution_results

    labels = context["readout_metrics"].integration_occurrences
    fake = {lb: [raw[:, rmap[lb]].astype(np.int8) if lb in rmap else np.zeros(len(raw), np.int8)] for lb in labels}
    res = construct_circuit_execution_results(PullaData(sweep_results=fake, run_data=None), dict(context))
    (circuit,) = res.circuit_measurement_results
    got = np.zeros_like(raw)
    for key, values in circuit.items():
        m = KEY.fullmatch(key)
        if m:
            got[:, int(m[1])] = np.asarray(values)[:, 0]
    return bool(np.array_equal(got, raw))


def cmd_compile(args: argparse.Namespace) -> int:
    refuse_if_armed()
    require_hash_seed()
    spec, sent, frozen_checks = frozen()
    log("FROZEN EP0 OK: " + json.dumps(frozen_checks))
    pulla, access = PC.connect(DEVICE)  # installs the read-only guard for this whole process
    if pulla is None:
        log("PULLA CONSTRUCTOR FAILED: " + json.dumps(access))
        return 3
    calset, stash = PC.pinned_stash(pulla, DEVICE)
    out = record_dir(today())
    snapshot = {"utc": PC._utc(), "default_calibration_set_id": calset, "stash": stash_digest(stash)}  # noqa: SLF001
    if calset != CALSET:
        log(f"STOP: the live calibration set is {calset}, not {CALSET}: " + json.dumps(snapshot))
        return 2
    qubits, couplers = EP.patch_from_placement(EP.placement_entry())
    missing = EP.patch_gates(hw.architecture(DEVICE, calset), qubits, couplers)
    snapshot["patch_gates_missing"] = missing
    if any(missing.values()):
        log(f"STOP: the live architecture lacks a gate EP0 needs: {missing}")
        return 2
    rd, ctx = compile_run(pulla, stash, sent)
    info = describe(rd, ctx, spec)
    rd2, ctx2 = compile_run(pulla, stash, sent)
    again = mismatches(describe(rd2, ctx2, spec), info)
    if again:
        log(f"STOP: two compiles in one process differ in {again}")
        return 5
    slim = slim_context(ctx)
    ep0_raw, _ = D.load_memory(PC.EP0_JOB, 0, circuit=spec.stim)
    iqm_pp = check_against_iqm_post_processing(slim, info["readout_map"], ep0_raw)
    from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash

    full = pulla.get_standard_compiler(loading_rules=[LatestFromStash(stash)])
    schedule = PC.compile_one({"full": full}, fresh_circuit(sent))["full"]
    blob = gzip.compress(serialized(rd), mtime=0)
    record = PC._header(DEVICE, calset) | {  # noqa: SLF001
        "task": "emerald-pulse", "armed_task": TASK, "circuit": CIRCUIT, "shots": SHOTS,
        "snapshot": snapshot, "frozen": frozen_checks,
        "compile": {"compiler": "Pulla get_standard_compiler(loading_rules=[LatestFromStash(<calset stash>)])",
                    "settings": "get_settings(circuits=[EP0]); set_shots(1000) (averaging_bins = 1000: per-shot data)",
                    "barriers": "kept (the instructions EP0's request sent)", "stage_options": "defaults",
                    "deterministic_in_process": True, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        **info,
        "schedule_full": {k: schedule[k] for k in ("duration_ns", "instructions")} | {
            "readout_triggers": schedule["readout_triggers"]},
        "context_passed": sorted(slim), "context_deepcopy_ok": True,
        "iqm_post_processing_matches_readout_map": iqm_pp,
        "run_definition_file": "run_definition.pb.gz",
        "run_definition_gz_sha256": hashlib.sha256(blob).hexdigest(),
        "requests": [dict(r) for r in PC.REQUESTS],
    }
    if not iqm_pp:
        log("STOP: IQM's own post-processing does not give back the readout map's columns")
        return 5
    path = out / "compile.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = mismatches(record, old)
        log(f"RECORD EXISTS: {PC._rel(path)}; this compile " + ("MATCHES it" if not diff else f"DIFFERS in {diff}"))  # noqa: SLF001
        return 0 if not diff else 5
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "run_definition.pb.gz", blob)
    write_new(path, PC.check_clean(record))
    log(f"COMPILED: playlist {info['playlist_sha256']}, qpu_runtime {info['qpu_runtime_s']:.4f} s, "
        f"segment {info['segment_duration_ns']['max']:.0f} ns, {info['shape']}")
    log(f"wrote {PC._rel(path)} and run_definition.pb.gz ({len(blob)} bytes)")  # noqa: SLF001
    return 0


# ---------------------------------------------------------------------------
# Submit: the one job.
# ---------------------------------------------------------------------------


def check_arming(env: Mapping[str, str]) -> None:
    got = {k: env.get(k) for k in ARMING}
    if got != ARMING:
        raise SystemExit(f"refusing: armed {got}, the brief's arming is {ARMING}")
    if env.get("DQ_QPU_LEDGER") not in (None, "", hw.DEFAULT_LEDGER):
        raise SystemExit(f"refusing: DQ_QPU_LEDGER is {env.get('DQ_QPU_LEDGER')}, not the real ledger")


def preflight(run_definition: Any, context: Mapping[str, Any], info: Mapping[str, Any], record: Mapping[str, Any], *,
              calset_now: str, ledger: Path, raw_dir: Path) -> dict[str, Any]:
    """Every refusal that needs no network. Raises ``SystemExit`` on the first failure; nothing is sent."""
    if PC.guard_installed():
        raise SystemExit("refusing: pulla_check's class-level guard is installed in this process")
    if calset_now != CALSET or record.get("calibration_set_id") != CALSET:
        raise SystemExit(f"refusing: the live calibration set is {calset_now}, the record's {record.get('calibration_set_id')}")
    if EP.ledger_rows(ledger, TASK):
        raise SystemExit(f"refusing: the ledger already has a {TASK} row; there is no resubmission")
    if raw_dir.is_dir() and any(p.is_dir() for p in raw_dir.iterdir()):
        raise SystemExit(f"refusing: {raw_dir} already holds a job")
    diff = mismatches(info, record)
    if diff:
        raise SystemExit(f"refusing: the recompile differs from the committed record in {diff}")
    shape = info["shape"]
    if (shape["circuits"], shape["shots"], shape["averaging_bins"]) != (1, SHOTS, SHOTS):
        raise SystemExit(f"refusing: shape {shape}, want 1 x {SHOTS} per-shot")
    runtime = runtime_estimate(run_definition)
    if not runtime <= RUNTIME_CAP_S:
        raise SystemExit(f"refusing: IQM's qpu_runtime estimate is {runtime:.3f} s > the {RUNTIME_CAP_S} s cap")
    slim = slim_context(context)
    return {"qpu_runtime_s": runtime, "context": slim}


def submit_once(run_definition: Any, slim: Mapping[str, Any], *, ledger: Path, raw_root: Path,
                timeout: float | None) -> dict[str, Any]:
    """The one ``hw.submit_playlist``. Never raises: what happened is returned, refusals included."""
    error = None
    with EP.LedgerWatch(ledger, task=TASK):
        try:
            sub = hw.submit_playlist(run_definition, context=slim, device=DEVICE, calibration_set_id=CALSET,
                                     purpose=PURPOSE, data_root=raw_root, timeout=timeout)
        except BaseException as e:  # noqa: BLE001 -- a refusal is a result: recorded below
            error, sub = e, None
    rows = EP.ledger_rows(ledger, TASK)
    out: dict[str, Any] = {"utc": PC._utc(), "ledger_rows": [r.__dict__ for r in rows]}  # noqa: SLF001
    if error is not None:
        out["error"] = PC.error_record(error, [])
        if error.__cause__ is not None:
            out["cause"] = PC.error_record(error.__cause__, [])
        out["ledger_job_id"] = getattr(error, "ledger_job_id", None)
        out["ok"] = False
        return out
    out |= {"job_id": sub.job_id, "status": sub.status, "path": PC._rel(sub.path), "purpose": sub.purpose,  # noqa: SLF001
            "ok": len(rows) == 1 and rows[0].job_id == sub.job_id and rows[0].circuits == 1 and rows[0].shots == SHOTS}
    return out


def cmd_submit(args: argparse.Namespace) -> int:
    check_arming(os.environ)
    require_hash_seed()
    ledger = hw.ledger_path()
    out = record_dir(today())
    record = json.loads((out / "compile.json").read_text())
    spec, sent, _ = frozen()
    calset_now = hw.calibration_set_id(DEVICE)
    pulla = hw.pulla(DEVICE)  # read-only instance: its submit_playlist and writing server calls refuse
    stash = pulla.get_calibration_stash(CALSET)
    if stash_digest(stash) != record["snapshot"]["stash"]:
        raise SystemExit("refusing: the stash differs from the committed record's")
    rd, ctx = compile_run(pulla, stash, sent)
    info = describe(rd, ctx, spec)
    pre = preflight(rd, ctx, info, record, calset_now=calset_now, ledger=ledger, raw_dir=out)
    log(f"PREFLIGHT OK: recompile = record (playlist {info['playlist_sha256'][:16]}…), qpu_runtime "
        f"{pre['qpu_runtime_s']:.4f} s <= {RUNTIME_CAP_S} s, calset {calset_now}, no {TASK} row")
    log(f"{PC._utc()} submitting 1 playlist x {SHOTS} shots on {DEVICE}")  # noqa: SLF001
    res = submit_once(rd, pre["context"], ledger=ledger, raw_root=RAW_ROOT, timeout=args.timeout)
    if res.get("job_id"):
        submitted = gzip.compress(serialized(rd), mtime=0)  # Pulla set run_id and sweep_id when it sent it
        write_new(out / "run_definition.submitted.pb.gz", submitted)
        res["run_definition_submitted_gz_sha256"] = hashlib.sha256(submitted).hexdigest()
        res["playlist_sha256_after_submit"] = hw.playlist_sha256(rd.sweep_definition.playlist)
        try:
            payload = hw._server_client(DEVICE).get_submit_run_payload(res["job_id"])  # noqa: SLF001 -- a GET
            res["server_payload_matches_submitted"] = serialized(payload) == serialized(rd)
            res["server_payload_playlist_sha256"] = hw.playlist_sha256(payload.sweep_definition.playlist)
        except Exception as e:  # noqa: BLE001 -- recorded; the job's data is already saved
            res["server_payload_error"] = PC.error_record(e, [])
    write_new(out / "submit.json", PC.check_clean(res))
    log(("SUBMIT OK: " if res["ok"] else "SUBMIT NOT OK (stop; no retry): ") + json.dumps(res, default=str))
    return 0 if res["ok"] else 4


def cmd_recover(args: argparse.Namespace) -> int:
    refuse_if_armed()
    out = record_dir(args.day or today())
    record = json.loads((out / "compile.json").read_text())
    rows = [r for r in EP.ledger_rows(hw.ledger_path(), TASK) if r.job_id == args.job_id]
    if len(rows) != 1:
        raise SystemExit(f"refusing: {args.job_id} is not a {TASK} row of the ledger")
    data, results = hw.retrieve_playlist(args.job_id, DEVICE)
    s = record["shape"]
    shape = hw.PlaylistShape(s["segments"], s["spots"], s["shots"], s["averaging_bins"], s["qubits"], record["playlist_sha256"])
    path = hw.save_playlist_raw(results, job_id=args.job_id, task=TASK, device=DEVICE, calibration_set_id=CALSET,
                                shape=shape, status=data.status, timeline=getattr(data, "timeline", []),
                                purpose=rows[0].purpose, day=out.name.split("-", 1)[0], data_root=RAW_ROOT)
    log(f"recovered into {PC._rel(path)}")  # noqa: SLF001
    return 0


# ---------------------------------------------------------------------------
# Analysis.
# ---------------------------------------------------------------------------


def to_raw(results: Mapping[str, Any], rmap: Mapping[str, int], shots: int = SHOTS) -> np.ndarray:
    """Per-label single-shot results -> (shots, 33) uint8, column j = clbit j. Raises ``ValueError``."""
    raw = np.zeros((shots, len(rmap)), np.uint8)
    for label, j in rmap.items():
        a = np.asarray(results[label]).reshape(-1)
        if np.iscomplexobj(a) or a.size != shots:
            raise ValueError(f"{label}: {a.dtype} of size {a.size}, want {shots} thresholded shots")
        if not set(np.unique(a).tolist()) <= {0, 1}:
            raise ValueError(f"{label}: values {np.unique(a)[:5]}, want 0/1")
        raw[:, j] = a
    return raw


def load_job(job_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta = json.loads((job_dir / "meta.json").read_text())
    arrays = np.load(job_dir / "sweep_results.npz")
    return {a["return_parameter"]: arrays[a["key"]] for a in meta["arrays"] if a["spot"] == 0}, meta


def newcombe(k1: int, n1: int, k2: int, n2: int) -> list[float]:
    """95% interval of p1 - p2 (Newcombe's hybrid score interval, from two Wilson intervals)."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = stats.wilson(k1, n1), stats.wilson(k2, n2)
    d = p1 - p2
    return [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def min_detectable(p: float, n: int, z_a: float = 1.959964, z_b: float = 0.841621) -> float:
    """The difference two samples of n detect at 80% power, two-sided alpha 0.05, around p."""
    return (z_a + z_b) * math.sqrt(2 * p * (1 - p) / n)


def compare(raw_pulse: np.ndarray, spec: Any) -> dict[str, Any]:
    """The pulse job and EP0 decoded with the same two priors, and the per-round detection fractions."""
    from scipy.stats import fisher_exact

    cal = EM.live_calibration(PC.EP0_BEFORE)
    idle = EM.readout_window_ns(spec, dev.Durations.from_calibration(cal))
    anchor = dev.pre_fix_noise(EM.calibrated_noise(cal, spec, idle_ns=idle))
    nc, info_c = dev.hardware_noise(cal, spec)
    t, typ = PC.detector_rounds_and_types(spec)
    ep0_raw, _ = D.load_memory(PC.EP0_JOB, 0, circuit=spec.stim)
    data = {"EP0": ep0_raw, "pulse": raw_pulse}
    dets = {k: D.detection_events(spec.stim, v) for k, v in data.items()}
    priors = {"pilot_record": spec.noisy(anchor), "compiled": spec.noisy(nc)}
    decoded: dict[str, Any] = {}
    for prior, dec in priors.items():
        for k, (d, o) in dets.items():
            row = PC.evaluate(prior, dec, spec, d, o, t, typ, {})
            del row["_fails"]
            decoded.setdefault(prior, {})[k] = row
        a, b = decoded[prior]["pulse"], decoded[prior]["EP0"]
        k1, n1, k2, n2 = a["hw_failures"], a["hw_shots"], b["hw_failures"], b["hw_shots"]
        decoded[prior]["pulse_minus_EP0"] = {"diff": k1 / n1 - k2 / n2, "newcombe95": newcombe(k1, n1, k2, n2),
                                             "fisher_p": float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue)}
    rounds = {}
    for r in sorted(set(t.tolist())):
        m = t == r
        row = {"detectors": int(m.sum())}
        for k, (d, _) in dets.items():
            kk, nn = int(d[:, m].sum()), int(d[:, m].size)
            row[k] = {"k": kk, "n": nn, "fraction": kk / nn, "wilson95": [float(x) for x in stats.wilson(kk, nn)]}
        row["pulse_minus_EP0"] = {"diff": row["pulse"]["fraction"] - row["EP0"]["fraction"],
                                  "newcombe95": newcombe(row["pulse"]["k"], row["pulse"]["n"], row["EP0"]["k"], row["EP0"]["n"])}
        rounds[str(r)] = row
    p_pool = (decoded["pilot_record"]["pulse"]["hw_failures"] + decoded["pilot_record"]["EP0"]["hw_failures"]) / 2000
    return {"priors_info": {"pilot_record": {"idle_ns": idle}, "compiled": info_c}, "decoded": decoded,
            "per_round": rounds, "all_zero_shots": int((raw_pulse.sum(axis=1) == 0).sum()),
            "detection_fraction": {k: float(d.mean()) for k, (d, _) in dets.items()},
            "min_detectable_p_l_difference": min_detectable(p_pool, 1000),
            "min_detectable_round_fraction_difference": {
                r: min_detectable((v["pulse"]["fraction"] + v["EP0"]["fraction"]) / 2, v["pulse"]["n"])
                for r, v in rounds.items()}}


def markdown(res: Mapping[str, Any]) -> str:
    def iv(x: Sequence[float], f: str = ".3f") -> str:
        return f"[{x[0]:{f}}, {x[1]:{f}}]"

    lines = ["| prior | EP0 (circuit job) P_L | pulse job P_L | pulse − EP0 (Newcombe 95%) | Fisher p |", "|---|---|---|---|---|"]
    for prior, d in res["decoded"].items():
        e, p, x = d["EP0"], d["pulse"], d["pulse_minus_EP0"]
        lines.append(f"| `{prior}` | {e['hw_failures']}/1000 = {e['hw_p_l']:.3f} {iv(e['hw_p_l_wilson95'])} | "
                     f"{p['hw_failures']}/1000 = **{p['hw_p_l']:.3f}** {iv(p['hw_p_l_wilson95'])} | "
                     f"{x['diff']:+.3f} {iv(x['newcombe95'], '+.3f')} | {x['fisher_p']:.2f} |")
    lines += ["", "| detector round t | detectors | EP0 fraction (Wilson 95%) | pulse fraction (Wilson 95%) | pulse − EP0 (Newcombe 95%) | "
              "model, compiled prior |", "|---|---|---|---|---|---|"]
    model = res["decoded"]["compiled"]["pulse"]["per_round_model"]["all"]
    for r, v in res["per_round"].items():
        lines.append(f"| {r} | {v['detectors']} | {v['EP0']['fraction']:.4f} {iv(v['EP0']['wilson95'], '.4f')} | "
                     f"**{v['pulse']['fraction']:.4f}** {iv(v['pulse']['wilson95'], '.4f')} | "
                     f"{v['pulse_minus_EP0']['diff']:+.4f} {iv(v['pulse_minus_EP0']['newcombe95'], '+.4f')} | {model[r]:.4f} |")
    return "\n".join(lines)


def find_job(day: str | None) -> Path:
    out = record_dir(day or today())
    jobs = [p for p in out.iterdir() if p.is_dir()]
    if len(jobs) != 1:
        raise SystemExit(f"{out}: {len(jobs)} job folders, want 1")
    return jobs[0]


def cmd_analyze(args: argparse.Namespace) -> int:
    refuse_if_armed()
    job = find_job(args.day)
    record = json.loads((job.parent / "compile.json").read_text())
    results, meta = load_job(job)
    if meta["playlist_sha256"] != record["playlist_sha256"] or not meta["success"]:
        raise SystemExit(f"{job}: playlist {meta['playlist_sha256']}, success {meta['success']}")
    spec, _, _ = frozen()
    if readout_map(record["return_parameters"], expected_qubits(spec)) != {k: int(v) for k, v in record["readout_map"].items()}:
        raise SystemExit("the record's readout map does not follow from its labels and the placement")
    raw = to_raw(results, record["readout_map"])
    res = {"job_id": meta["job_id"], "device": meta["device"], "calibration_set_id": meta["calibration_set_id"],
           "shots": int(raw.shape[0]), "rounds": spec.rounds, **compare(raw, spec)}
    path = job.parent / "analysis.json"
    write_new(path, PC.check_clean(res))
    log(f"wrote {PC._rel(path)}")  # noqa: SLF001
    log(markdown(res))
    return 0


def cmd_standin(args: argparse.Namespace) -> int:
    """EP0's own shots, laid out as the playlist's sweep results, through to_raw and compare: must give EP0 back."""
    refuse_if_armed()
    spec, _, _ = frozen()
    path = record_dir(args.day or today()) / "compile.json"
    rmap = (json.loads(path.read_text())["readout_map"] if path.exists()
            else {f"{q}__c_33_0_{j}": j for j, q in enumerate(expected_qubits(spec))})
    ep0_raw, _ = D.load_memory(PC.EP0_JOB, 0, circuit=spec.stim)
    fake = {label: [ep0_raw[:, j].astype(np.int8)] for label, j in rmap.items()}
    raw = to_raw({k: v[0] for k, v in fake.items()}, rmap)
    res = compare(raw, spec)
    ok = (np.array_equal(raw, ep0_raw) and res["decoded"]["pilot_record"]["pulse"]["hw_failures"] == EP0_P_L[0]
          and res["decoded"]["pilot_record"]["pulse_minus_EP0"]["diff"] == 0)
    log(markdown(res))
    log(f"STAND-IN {'OK' if ok else 'FAILED'}: readout map from {'the record' if path.exists() else 'the placement'}; "
        f"pilot_record {res['decoded']['pilot_record']['pulse']['hw_failures']} failures, "
        f"compiled {res['decoded']['compiled']['pulse']['hw_failures']}")
    return 0 if ok else 1


def cmd_payload_check(args: argparse.Namespace) -> int:
    """Read-only: the server's stored payload of the job (a GET) against what we POSTed, canonically hashed."""
    from iqm.data_definitions.station_control.v2.run_definition_pb2 import RunDefinition as RunDefinitionProto
    from iqm.station_control.client.serializers.run_serializers import deserialize_run_definition

    refuse_if_armed()
    job = find_job(args.day)
    proto = RunDefinitionProto()
    proto.ParseFromString(gzip.decompress((job.parent / "run_definition.submitted.pb.gz").read_bytes()))
    sent = deserialize_run_definition(proto)
    server = hw._server_client(DEVICE).get_submit_run_payload(job.name)  # noqa: SLF001 -- a GET
    out = {"utc": PC._utc(), "job_id": job.name,  # noqa: SLF001
           "sent_canonical_sha256": canonical_sha256(sent), "server_canonical_sha256": canonical_sha256(server),
           "sent_playlist_sha256": hw.playlist_sha256(sent.sweep_definition.playlist),
           "server_playlist_sha256": hw.playlist_sha256(server.sweep_definition.playlist),
           "run_id_equal": str(sent.run_id) == str(server.run_id)}
    out["canonical_equal"] = out["sent_canonical_sha256"] == out["server_canonical_sha256"]
    write_new(job.parent / "server_payload_check.json", PC.check_clean(out))
    log(json.dumps(out))
    return 0 if out["canonical_equal"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compile")
    s = sub.add_parser("submit")
    s.add_argument("--timeout", type=float, default=1800.0)
    r = sub.add_parser("recover")
    r.add_argument("--job-id", required=True)
    r.add_argument("--day")
    for name in ("analyze", "standin", "payload-check"):
        a = sub.add_parser(name)
        a.add_argument("--day")
    args = ap.parse_args()
    return {"compile": cmd_compile, "submit": cmd_submit, "recover": cmd_recover, "analyze": cmd_analyze,
            "standin": cmd_standin, "payload-check": cmd_payload_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
