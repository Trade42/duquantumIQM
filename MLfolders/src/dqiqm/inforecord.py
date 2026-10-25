"""The InfoRecord: everything we collect from Garnet, laid out for an ML decoder. Format ``dqiqm.inforecord/1``.

Layout (``InfoRecord/``; the dry run writes the same layout under ``InfoRecord/sim/``)::

    README.md                       what it is, how it was made, the column dictionary
    architecture/garnet.json        qubits, couplers, grid coordinates
    calibration/garnet/<UTC>-<calset>.json   one snapshot per calibration (E6: before and after every batch)
    circuits/<experiment>/<name>.stim, <name>.qasm, placement.json; circuits/job_plan.json
    raw/<YYYYMMDD>-<task>/<job id>/  dqiqm.hw.save_raw's meta.json + memory.npz, never edited
    datasets/<experiment>/<run id>/  ML-ready, one directory per circuit and batch
    derived/<experiment>/<run id>/   temporal-correlation features of the same run
    ler/<experiment>/<run id>/       P_L against rounds and the ε_L fits, one per (basis, patch, batch)

The column dictionary (:data:`COLUMNS`, :data:`ARRAYS`, :data:`MANIFEST`) is the one place each
file's columns are defined: :func:`write_dataset` and :func:`write_derived` write from it, the
README's table is printed from it (:func:`dictionary_markdown`), and :func:`check_run` checks
every written file against it.

Conventions (shared with the laptop's ``mlDecoder.data.external_dataset``):

* ``round`` is Stim's detector time coordinate t: 0-based, the final data detectors at t = R. A
  measurement's ``round`` is the same t (round k's readout at t = k − 1, the final data readout
  at t = R);
* ``x``, ``y`` are the detector's Stim coordinates, as ``circuit.get_detector_coordinates()`` gives;
* rows of ``detectors.csv`` and columns of ``det_events.npy`` are in the circuit's detector order;
  columns of ``raw_meas.npy`` are Stim measurement j = clbit j;
* arrays are ``uint8`` 0/1. An array with more than :data:`PACK_THRESHOLD` entries is bit-packed
  along its columns with ``np.packbits(..., axis=1, bitorder="little")``; the manifest's
  ``array_info`` then says ``"packed": true`` and gives the unpacked shape. :func:`load_array`
  undoes it.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dqiqm import decode as D
from dqiqm import stats
from dqiqm import temporal as T
from dqiqm.experiments import circuits as C
from dqiqm.experiments import device as dev

FORMAT = "dqiqm.inforecord/1"
PACK_THRESHOLD = 1 << 24  # entries per array; above this, arrays are bit-packed
N_BUCKETS = 10


@dataclass(frozen=True)
class Column:
    name: str  # "<k>" stands for an integer (a family of columns)
    dtype: str  # "int", "float", "str", "bool" (0/1)
    unit: str
    description: str
    optional: bool = False  # may be empty in a row, or absent from a file

    @property
    def regex(self) -> re.Pattern:
        return re.compile("^" + re.escape(self.name).replace(re.escape("<k>"), r"\d+") + "$")


def _c(name, dtype, unit, description, optional=False) -> Column:
    return Column(name, dtype, unit, description, optional)


COLUMNS: dict[str, tuple[Column, ...]] = {
    "datasets/shots.csv": (
        _c("simulated", "bool", "", "1 for a simulated shot (InfoRecord/sim), 0 for hardware"),
        _c("job_id", "str", "", "the job that ran the shot (a sim job id starts with 'sim-')"),
        _c("batch", "int", "", "the time-separated batch (1, 2, 3; 0 for the pilot phase)"),
        _c("position_in_job", "int", "", "the circuit's place in its job's run order (0-based)"),
        _c("shot", "int", "", "the shot's index within the circuit's shots, in the order returned"),
        _c("split", "str", "", "finetune, validation or test (the fixed ML split; rule in the manifest)"),
        _c("time_bucket", "int", "", f"shot * {N_BUCKETS} // shots: blocks of consecutive shots, for drift"),
        _c("circuit", "str", "", "the circuit's name (circuits/<experiment>/<name>.stim)"),
        _c("experiment", "str", "", "E1 ... E5"),
        _c("family", "str", "", "surface_memory, repetition, readout or stability"),
        _c("rounds", "int", "", "stabilizer rounds R (readouts in a row for E3)"),
        _c("basis", "str", "", "the basis the data are prepared and read in (Z or X)"),
        _c("patch", "str", "", "pA / pB (E1 orientations), line19, all20, block3x3"),
        _c("prep", "str", "", "the prepared state: 0, 1, + or, for E3, one bit per qubit"),
        _c("delay_ns", "float", "ns", "the idle after each round's readout but the last (E4)"),
        _c("barriers", "bool", "", "1 if each layer is fenced by barriers (the default)"),
        _c("raw_weight", "int", "", "the number of 1s in the shot's measurement record"),
        _c("all_zero", "bool", "", "1 if the whole record is 0"),
        _c("total_events", "int", "", "the number of detection events"),
        _c("longest_streak", "int", "rounds", "the longest run of consecutive rounds one stabilizer fired"),
        _c("sw_t<k>", "int", "", "the syndrome weight (detection events) at detector round t = k"),
        _c("obs_flip", "bool", "", "the observable's flip (the raw logical outcome against the noiseless one)", True),
        _c("pm_pred", "bool", "", "PyMatching's predicted flip (decoder in the manifest)", True),
        _c("pm_fail", "bool", "", "1 if pm_pred != obs_flip: a logical failure of the decoder", True),
    ),
    "datasets/detectors.csv": (
        _c("index", "int", "", "the detector's index: its column in det_events.npy"),
        _c("round", "int", "", "Stim's time coordinate t (0-based; final data detectors at t = R)"),
        _c("x", "float", "", "Stim's detector x coordinate"),
        _c("y", "float", "", "Stim's detector y coordinate"),
        _c("t", "float", "", "Stim's detector t coordinate, as given (equal to round)"),
        _c("stabilizer", "str", "", "the check's label, e.g. X(2,0), ZZ(3), M(QB5)"),
        _c("type", "str", "", "the check's Pauli type, X or Z"),
        _c("ancilla_qb", "str", "", "the device qubit that measures the check"),
        _c("ancilla_stim", "int", "", "its Stim qubit index"),
        _c("data_qbs", "str", "", "the device qubits in the check's support, ';'-separated"),
        _c("record", "str", "", "the measurement indices it XORs, most recent first, ';'-separated (m_k;m_(k-2);...)"),
        _c("boundary", "str", "", "first (t = 0), second (t = 1), bulk, or final (it reads the data readout)"),
        _c("grid_row", "int", "", "the ancilla's row on the device grid (architecture/garnet.json)"),
        _c("grid_col", "int", "", "the ancilla's column on the device grid"),
    ),
    "datasets/measurements.csv": (
        _c("index", "int", "", "the measurement's index j: column j of raw_meas.npy, clbit j"),
        _c("round", "int", "", "t: round k's readout at k - 1, the final data readout at R"),
        _c("stim_qubit", "int", "", "the measured Stim qubit"),
        _c("qb", "str", "", "the measured device qubit"),
        _c("role", "str", "", "data, ancilla or readout"),
        _c("stabilizer", "str", "", "the ancilla's check label (empty for data)", True),
        _c("type", "str", "", "the check's Pauli type (empty for data)", True),
        _c("moment", "int", "", "the barrier-delimited layer (moment) of the circuit it is in (0-based)"),
        _c("t_ns", "float", "ns", "estimated start of the readout from the start of the shot (calibrated durations)"),
        _c("grid_row", "int", "", "the qubit's row on the device grid"),
        _c("grid_col", "int", "", "the qubit's column on the device grid"),
    ),
    "datasets/qubits.csv": (
        _c("qb", "str", "", "device qubit (all of the device's, used or not)"),
        _c("stim_qubit", "int", "", "its Stim qubit in this circuit (empty if unused)", True),
        _c("role", "str", "", "data, ancilla, readout, or idle (not used)"),
        _c("stabilizer", "str", "", "the check it measures, if an ancilla", True),
        _c("grid_row", "int", "", "row on the device grid"),
        _c("grid_col", "int", "", "column on the device grid"),
        _c("t1_ns", "float", "ns", "T1 at collection time"),
        _c("t2_echo_ns", "float", "ns", "T2 (echo)", True),
        _c("t2_ramsey_ns", "float", "ns", "T2 (Ramsey)", True),
        _c("p1_given_0", "float", "", "readout P(1|0) of the default measure"),
        _c("p0_given_1", "float", "", "readout P(0|1) of the default measure"),
        _c("error_1q", "float", "", "1Q gate error (RB infidelity)"),
        _c("readout_leakage", "float", "", "readout-induced leakage", True),
        _c("measure_ns", "float", "ns", "readout duration", True),
        _c("cz_errors", "str", "", "CZ error (IRB infidelity) per coupler, 'QBx=err' ';'-separated"),
        _c("cz_error_mean", "float", "", "the mean of cz_errors"),
    ),
    "derived/rate_by_round.csv": (
        _c("detector", "int", "", "detector index"),
        _c("stabilizer", "str", "", "its check"),
        _c("round", "int", "", "its t"),
        _c("rate", "float", "", "firing rate over the run's shots"),
        _c("shots", "int", "", "shots"),
    ),
    "derived/rate_by_bucket.csv": (
        _c("bucket", "int", "", "time bucket (blocks of consecutive shots)"),
        _c("shot_lo", "int", "", "first shot of the bucket"),
        _c("shot_hi", "int", "", "last shot of the bucket"),
        _c("stabilizer", "str", "", "check"),
        _c("rate", "float", "", "mean firing rate of the check's detectors in the bucket"),
        _c("shots", "int", "", "shots in the bucket"),
    ),
    "derived/pij_lags.csv": (
        _c("stabilizer_i", "str", "", "the earlier detector's check"),
        _c("stabilizer_j", "str", "", "the later detector's check"),
        _c("lag", "int", "rounds", "t_j - t_i: 0, 1 (k -> k+1) or 2 (k -> k+2)"),
        _c("pij_mean", "float", "", "mean p_ij over the pairs (Google's pairwise error probability)"),
        _c("pij_max", "float", "", "largest p_ij among the pairs"),
        _c("pairs", "int", "", "number of detector pairs averaged"),
    ),
    "derived/streaks.csv": (
        _c("stabilizer", "str", "", "check"),
        _c("length", "int", "rounds", "length of a run of consecutive rounds in which the check fired"),
        _c("count", "int", "", "number of such runs over all shots"),
    ),
    "derived/syndrome_weight.csv": (
        _c("round", "int", "", "detector round t"),
        _c("weight", "int", "", "detection events in that round"),
        _c("count", "int", "", "shots with that weight"),
    ),
    "derived/assignment.csv": (
        _c("qb", "str", "", "device qubit"),
        _c("prep", "int", "", "prepared bit"),
        _c("layer", "int", "", "readout layer (0-based)"),
        _c("p_wrong", "float", "", "P(read != prep)"),
        _c("p_one", "float", "", "P(read 1)"),
        _c("shots", "int", "", "shots"),
    ),
    "derived/readout_flips.csv": (
        _c("qb", "str", "", "device qubit"),
        _c("prep", "int", "", "prepared bit"),
        _c("repeats", "int", "", "readouts in a row"),
        _c("p_det_bulk", "float", "", "mean P(D_k), 2 <= k <= K-1"),
        _c("record_flip", "float", "", "record flips per readout: mean p_ij(D_k, D_(k+1))"),
        _c("state_flip", "float", "", "state flips per readout gap: p_det_bulk - 2 record_flip"),
        _c("p_wrong_first", "float", "", "misread fraction at the first readout"),
        _c("p_wrong_last", "float", "", "misread fraction at the last readout"),
        _c("shots", "int", "", "shots"),
    ),
    "calibration/qubits.csv": (
        _c("qb", "str", "", "device qubit"),
        _c("calibration_set_id", "str", "", "the calibration set the figures describe"),
        _c("fetched_utc", "str", "", "when the snapshot was fetched"),
        _c("grid_row", "int", "", "row on the device grid"),
        _c("grid_col", "int", "", "column on the device grid"),
        _c("t1_ns", "float", "ns", "T1 (characterization.model.<q>.t1_time)"),
        _c("t2_ramsey_ns", "float", "ns", "T2 Ramsey (.t2_time)"),
        _c("t2_echo_ns", "float", "ns", "T2 echo (.t2_echo_time)"),
        _c("p1_given_0", "float", "", "readout P(1|0): metrics.ssro.measure.constant.<q>.error_0_to_1 (not the heralded figure)"),
        _c("p0_given_1", "float", "", "readout P(0|1): ...error_1_to_0"),
        _c("readout_error", "float", "", "the mean of the two"),
        _c("error_1q", "float", "", "1Q gate error: 1 - RB fidelity of the default prx"),
        _c("readout_leakage", "float", "", "readout-induced leakage (metrics.rrc...leakage)", True),
        _c("prx_ns", "float", "ns", "prx duration"),
        _c("measure_ns", "float", "ns", "measure duration"),
    ),
    "calibration/couplers.csv": (
        _c("qb_a", "str", "", "first qubit of the coupler"),
        _c("qb_b", "str", "", "second qubit"),
        _c("calibration_set_id", "str", "", "the calibration set"),
        _c("fetched_utc", "str", "", "when the snapshot was fetched"),
        _c("cz_error_irb", "float", "", "CZ error, 1 - interleaved-RB fidelity"),
        _c("cz_error_clifford_rb", "float", "", "CZ error from Clifford RB (for reference)", True),
        _c("cz_ns", "float", "ns", "CZ duration"),
    ),
    "ler/ler_points.csv": (
        _c("experiment", "str", "", "E1, E2, E4 or E5"),
        _c("run_id", "str", "", "the LER run: basis (or kind/prep), patch, batch"),
        _c("circuit", "str", "", "circuit name"),
        _c("dataset", "str", "", "the dataset directory the point comes from"),
        _c("rounds", "int", "", "R"),
        _c("shots", "int", "", "shots n"),
        _c("failures", "int", "", "decoder failures k (pm_fail)"),
        _c("p_l", "float", "", "P_L = k / n"),
        _c("wilson_lo", "float", "", "Wilson 95% lower end"),
        _c("wilson_hi", "float", "", "Wilson 95% upper end"),
        _c("eps_single", "float", "", "eps_L from this R alone: (1 - (1 - 2 P_L)^(1/R)) / 2"),
        _c("eps_single_lo", "float", "", "its 95% lower end (Wilson, mapped)"),
        _c("eps_single_hi", "float", "", "its 95% upper end"),
        _c("raw_failures", "int", "", "failures of the 'always 0' decoder (obs_flip)"),
        _c("p_raw", "float", "", "raw_failures / n"),
    ),
}

ARRAYS: dict[str, str] = {
    "datasets/raw_meas.npy": "uint8 (shots, measurements): column j = Stim measurement j = clbit j, the raw record",
    "datasets/det_events.npy": "uint8 (shots, detectors): detection events, against the noiseless reference (Stim m2d)",
    "datasets/obs_flips.npy": "uint8 (shots, observables): observable flips, against the noiseless reference",
    "derived/pij.npy": "float32 (detectors, detectors): p_ij for every pair; the diagonal is each detector's rate",
    "derived/readout_corr.npy": "float64 (qubits, qubits): correlation of misassignments between qubits (E3)",
}

MANIFEST: dict[str, str] = {
    "format": f"'{FORMAT}'",
    "label": "SIMULATED or HARDWARE",
    "simulated": "true for the dry run",
    "experiment": "E1 ... E5",
    "circuit": "the circuit's name",
    "run_id": "this directory's name: <batch>_<circuit>",
    "batch": "the time-separated batch (0: pilot phase)",
    "job_id": "the job that ran it",
    "position_in_job": "the circuit's place in the job's run order",
    "split": "the ML split: its rule (test batch, validation fraction, seed), this run's role, and the counts",
    "spec": "what the circuit is: family, d, rounds, basis, no_reset (true), gate_basis (the Stim source's: R, H, CX, M), kind, prep, patch, delay_ns, barriers, orientation",
    "device": "the device (or the stand-in, for the dry run)",
    "calibration_set_id": "the calibration set the job ran on (the noise's source, for the dry run)",
    "calibration": "paths of the snapshots before and after the batch",
    "shots": "shots",
    "num_measurements": "columns of raw_meas",
    "num_detectors": "columns of det_events",
    "num_observables": "columns of obs_flips",
    "arrays": "{'dets': 'det_events.npy', 'obs': 'obs_flips.npy', 'raw_meas': 'raw_meas.npy'}",
    "array_info": "per file: shape (unpacked), dtype, packed, bitorder, sha256 of the file",
    "placement": "Stim qubit -> device qubit",
    "circuit_files": "the circuit's .stim and .qasm under circuits/",
    "stim_sha256": "sha256 of the Stim source's text",
    "qasm_sha256": "sha256 of the .qasm",
    "native_sha256": "sha256 of the serialized native instructions (what the server receives)",
    "raw": "the raw result it was read from: path of memory.npz and its key c<i>",
    "decoder": "the decoder of pm_pred: PyMatching on the noisy copy of the circuit that ran, and its noise",
    "duration_ns": "the circuit's estimated duration (calibrated durations, barriers as sync points)",
    "n_buckets": "time buckets in shots.csv",
    "created_utc": "when this directory was written",
    "job_timeline": "the server's status timestamps (hardware), from the raw meta.json",
    "commit": "git HEAD of the code that wrote it",
    "versions": "package versions",
    "seeds": "the seeds (simulation)",
    "simulator": "how the dry run simulated it (simulation only)",
}


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(float(v)) else repr(float(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def write_csv(path: Path, kind: str, rows: Sequence[Mapping[str, Any]], header: Sequence[str] | None = None) -> None:
    """Write ``rows`` to ``path`` with the columns of ``COLUMNS[kind]`` (``header`` expands the <k> families)."""
    header = list(header or [c.name for c in COLUMNS[kind]])
    _check_header(kind, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(r.get(h)) for h in header])


def _check_header(kind: str, header: Sequence[str]) -> None:
    cols = COLUMNS[kind]
    for h in header:
        if not any(c.regex.match(h) for c in cols):
            raise ValueError(f"{kind}: column {h!r} is not in the column dictionary")
    for c in cols:
        if not c.optional and not any(c.regex.match(h) for h in header):
            raise ValueError(f"{kind}: required column {c.name!r} is missing")


def save_array(path: Path, a: np.ndarray, *, pack_threshold: int = PACK_THRESHOLD) -> dict[str, Any]:
    """Save a 2-D 0/1 array as uint8 ``.npy``, bit-packed along axis 1 if it has more than ``pack_threshold`` entries."""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D array, got {a.shape}")
    u = a.astype(np.uint8)
    packed = u.size > pack_threshold
    np.save(path, np.packbits(u, axis=1, bitorder="little") if packed else u)
    return {"shape": list(u.shape), "dtype": "uint8", "packed": packed, "bitorder": "little" if packed else None,
            "axis": 1 if packed else None, "sha256": sha256_file(path)}


def load_array(run_dir: str | Path, key: str) -> np.ndarray:
    """``key`` ("dets", "obs" or "raw_meas") of a dataset directory, unpacked to uint8 (shots, columns)."""
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "manifest.json").read_text())
    name = m["arrays"][key]
    info = m["array_info"][name]
    a = np.load(run_dir / name)
    if info["packed"]:
        a = np.unpackbits(a, axis=1, count=info["shape"][1], bitorder=info["bitorder"])
    if list(a.shape) != info["shape"]:
        raise ValueError(f"{run_dir / name}: shape {a.shape} != manifest {info['shape']}")
    return a


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def versions() -> dict[str, str]:
    from importlib.metadata import version

    out = {}
    for pkg in ("stim", "pymatching", "numpy", "qiskit", "qiskit-aer", "iqm-client"):
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 -- a missing package is recorded, not fatal
            out[pkg] = "absent"
    return out


# ---------------------------------------------------------------------------
# Per-circuit tables.
# ---------------------------------------------------------------------------


def detector_rows(spec: C.CircuitSpec, arch: Any) -> list[dict[str, Any]]:
    meas_q = C.measured_qubits(spec.stim)
    coords = spec.stim.get_detector_coordinates()
    rows = []
    for i, rec in enumerate(C.detector_records(spec.stim)):
        anc = next(meas_q[j] for j in rec if spec.roles.get(meas_q[j]) != "data")
        st = spec.stabilizers[anc]
        t = int(round(coords[i][2]))
        has_data = any(spec.roles.get(meas_q[j]) == "data" for j in rec)
        boundary = "final" if has_data else ("first" if t == 0 else "second" if t == 1 else "bulk")
        name = spec.physical[anc]
        r, c = arch.coords[name]
        rows.append({
            "index": i, "round": t, "x": coords[i][0], "y": coords[i][1], "t": coords[i][2], "stabilizer": st.label, "type": st.pauli,
            "ancilla_qb": name, "ancilla_stim": anc,
            "data_qbs": ";".join(spec.physical[q] for q in st.support if spec.roles.get(q) == "data"),
            "record": ";".join(str(j) for j in rec), "boundary": boundary, "grid_row": r, "grid_col": c,
        })
    return rows


def measurement_rows(spec: C.CircuitSpec, arch: Any, durations: dev.Durations) -> list[dict[str, Any]]:
    sched = C.schedule(spec.hardware, durations)
    rows = []
    for j, q in enumerate(C.measured_qubits(spec.stim)):
        name = spec.physical[q]
        st = spec.stabilizers.get(q)
        r, c = arch.coords[name]
        rows.append({
            "index": j, "round": spec.meas_round[j] - 1, "stim_qubit": q, "qb": name, "role": spec.roles[q],
            "stabilizer": st.label if st else "", "type": st.pauli if st else "", "moment": sched.meas_layer[j],
            "t_ns": sched.meas_start_ns[j], "grid_row": r, "grid_col": c,
        })
    return rows


def qubit_rows(spec: C.CircuitSpec, arch: Any, cal: dev.Calibration) -> list[dict[str, Any]]:
    stim_of = {name: q for q, name in spec.physical.items()}
    rows = []
    for name in arch.qubits:
        q = stim_of.get(name)
        cq = cal.qubits[name]
        czs = {(b if a == name else a): v["cz_error"] for (a, b), v in cal.couplers.items() if name in (a, b)}
        st = spec.stabilizers.get(q) if q is not None else None
        r, c = arch.coords[name]
        rows.append({
            "qb": name, "stim_qubit": q, "role": spec.roles[q] if q is not None else "idle",
            "stabilizer": st.label if st and spec.roles.get(q) != "readout" else "",
            "grid_row": r, "grid_col": c, "t1_ns": cq["t1_ns"], "t2_echo_ns": cq.get("t2_ns"),
            "t2_ramsey_ns": cq.get("t2_ramsey_ns"), "p1_given_0": cq["p1_given_0"], "p0_given_1": cq["p0_given_1"],
            "error_1q": cq["error_1q"], "readout_leakage": cq.get("readout_leakage"), "measure_ns": cq.get("measure_ns"),
            "cz_errors": ";".join(f"{n}={e!r}" for n, e in sorted(czs.items(), key=lambda kv: dev.qubit_sort_key(kv[0]))),
            "cz_error_mean": float(np.mean(list(czs.values()))) if czs else None,
        })
    return rows


def spec_block(spec: C.CircuitSpec) -> dict[str, Any]:
    p = dict(spec.params)
    return {
        "family": p.get("family"), "d": p.get("distance"), "rounds": spec.rounds, "basis": p.get("basis"),
        "no_reset": True, "gate_basis": "CX", "kind": p.get("kind"), "prep": p.get("prep"), "patch": p.get("patch"),
        "delay_ns": p.get("delay_ns", 0.0), "barriers": p.get("barriers", True), "orientation": p.get("orientation"),
    }


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------


def write_architecture(root: Path, snap: Mapping[str, Any], arch: Any, *, source: str) -> Path:
    out = root / "architecture" / "garnet.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "format": FORMAT,
        "source": source,
        "calibration_set_id": snap.get("calibration_set_id"),
        "qubits": list(arch.qubits),
        "couplers": sorted([sorted(c, key=dev.qubit_sort_key) for c in arch.couplers],
                           key=lambda p: (dev.qubit_sort_key(p[0]), dev.qubit_sort_key(p[1]))),
        "grid": {q: list(arch.coords[q]) for q in arch.qubits},
        "grid_text": arch.grid(),
        "dynamic_architecture": snap["architecture"],
        "static_architecture": snap.get("static_architecture"),
    }
    out.write_text(json.dumps(body, indent=1) + "\n")
    return out


def write_circuits(root: Path, specs: Sequence[C.CircuitSpec], durations: dev.Durations) -> dict[str, dict[str, str]]:
    """``circuits/<experiment>/<name>.stim`` and ``.qasm`` and one ``placement.json`` per experiment. Hardware target only."""
    files: dict[str, dict[str, str]] = {}
    by_exp: dict[str, dict[str, Any]] = {}
    for s in specs:
        if s.hardware.target != "hardware":
            raise ValueError(f"{s.name}: only target='hardware' circuits go in circuits/")
        d = root / "circuits" / s.experiment
        d.mkdir(parents=True, exist_ok=True)
        stim_path, qasm_path = d / f"{s.name}.stim", d / f"{s.name}.qasm"
        stim_path.write_text(str(s.stim) + "\n")
        qasm_path.write_text(C.qasm(s.hardware) + "\n")
        files[s.name] = {"stim": _rel(stim_path, root), "qasm": _rel(qasm_path, root),
                         "qasm_sha256": sha256_file(qasm_path), "native_sha256": C.native_sha256(s.hardware)}
        ops = s.hardware.circuit.count_ops()
        by_exp.setdefault(s.experiment, {})[s.name] = {
            "target": s.hardware.target,
            "placement": {str(q): n for q, n in sorted(s.physical.items())},
            "origin": list(s.placement.origin),
            "orientation": s.placement.orientation,
            "roles": {str(q): r for q, r in sorted(s.roles.items())},
            "stabilizers": {str(q): {"label": st.label, "pauli": st.pauli, "support": list(st.support)}
                            for q, st in sorted(s.stabilizers.items())},
            "params": dict(s.params),
            "num_measurements": s.stim.num_measurements,
            "num_detectors": s.stim.num_detectors,
            "num_observables": s.stim.num_observables,
            "swaps": int(ops.get("swap", 0)),
            "ops": {k: int(v) for k, v in sorted(ops.items())},
            "duration_ns": C.schedule(s.hardware, durations).duration_ns,
            "stim_sha256": s.stim_sha256(),
            "qasm_sha256": files[s.name]["qasm_sha256"],
            "native_sha256": files[s.name]["native_sha256"],
            "checks": dict(s.checks),
        }
    for exp, body in by_exp.items():
        (root / "circuits" / exp / "placement.json").write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return files


def write_dataset(
    root: Path,
    spec: C.CircuitSpec,
    raw: np.ndarray,
    *,
    run_id: str,
    batch: int,
    job_id: str,
    position_in_job: int,
    arch: Any,
    cal: dev.Calibration,
    durations: dev.Durations,
    circuit_files: Mapping[str, str],
    raw_ref: Mapping[str, Any],
    calibration_paths: Mapping[str, str],
    decoder_circuit: Any = None,
    decoder_note: str = "",
    split_labels: Sequence[str] | None = None,
    split_info: Mapping[str, Any] | None = None,
    simulated: bool,
    extra: Mapping[str, Any] | None = None,
    pack_threshold: int = PACK_THRESHOLD,
) -> tuple[Path, dict[str, Any]]:
    """Write ``datasets/<experiment>/<run_id>/`` from one circuit's raw record. Returns (path, per-run numbers)."""
    out = root / "datasets" / spec.experiment / run_id
    out.mkdir(parents=True, exist_ok=False)
    raw = np.asarray(raw, dtype=np.uint8)
    dets, obs = D.detection_events(spec.stim, raw)
    n = len(raw)
    info = {
        "raw_meas.npy": save_array(out / "raw_meas.npy", raw, pack_threshold=pack_threshold),
        "det_events.npy": save_array(out / "det_events.npy", dets, pack_threshold=pack_threshold),
        "obs_flips.npy": save_array(out / "obs_flips.npy", obs, pack_threshold=pack_threshold),
    }
    det_rows = detector_rows(spec, arch)
    t = np.array([r["round"] for r in det_rows], dtype=int)
    labels = sorted({r["stabilizer"] for r in det_rows})
    stab = np.array([labels.index(r["stabilizer"]) for r in det_rows], dtype=int)
    w, w_rounds = T.syndrome_weight(dets, t)
    longest, _ = T.streaks(dets, t, stab)
    pred = None
    if decoder_circuit is not None and spec.stim.num_observables:
        pred = D.decode(decoder_circuit, dets)
    bucket = T.buckets(n, N_BUCKETS)
    p = spec.params
    sw_cols = [f"sw_t{r}" for r in w_rounds]
    shot_rows = []
    for i in range(n):
        row = {
            "simulated": simulated, "job_id": job_id, "batch": batch, "position_in_job": position_in_job, "shot": i,
            "split": split_labels[i] if split_labels is not None else "",
            "time_bucket": int(bucket[i]), "circuit": spec.name, "experiment": spec.experiment, "family": p["family"],
            "rounds": spec.rounds, "basis": p.get("basis"), "patch": p.get("patch"), "prep": p.get("prep"),
            "delay_ns": p.get("delay_ns", 0.0), "barriers": p.get("barriers", True), "raw_weight": int(raw[i].sum()),
            "all_zero": not raw[i].any(), "total_events": int(dets[i].sum()), "longest_streak": int(longest[i]),
        }
        row.update({c: int(w[i, j]) for j, c in enumerate(sw_cols)})
        if obs.shape[1]:
            row["obs_flip"] = bool(obs[i, 0])
            if pred is not None:
                row["pm_pred"] = bool(pred[i, 0])
                row["pm_fail"] = bool(pred[i, 0] != obs[i, 0])
        shot_rows.append(row)
    base = [c.name for c in COLUMNS["datasets/shots.csv"] if c.name != "sw_t<k>" and c.name not in ("obs_flip", "pm_pred", "pm_fail")]
    tail = ["obs_flip"] * bool(obs.shape[1]) + (["pm_pred", "pm_fail"] if pred is not None else [])
    write_csv(out / "shots.csv", "datasets/shots.csv", shot_rows, base + sw_cols + tail)
    write_csv(out / "detectors.csv", "datasets/detectors.csv", det_rows)
    write_csv(out / "measurements.csv", "datasets/measurements.csv", measurement_rows(spec, arch, durations))
    write_csv(out / "qubits.csv", "datasets/qubits.csv", qubit_rows(spec, arch, cal))
    from dqiqm.hw import git_head

    manifest = {
        "format": FORMAT,
        "label": "SIMULATED" if simulated else "HARDWARE",
        "simulated": simulated,
        "experiment": spec.experiment,
        "circuit": spec.name,
        "run_id": run_id,
        "batch": batch,
        "job_id": job_id,
        "position_in_job": position_in_job,
        "split": dict(split_info) if split_info is not None else None,
        "spec": spec_block(spec),
        "device": (extra or {}).get("device", cal.device),
        "calibration_set_id": cal.calibration_set_id,
        "calibration": dict(calibration_paths),
        "shots": n,
        "num_measurements": spec.stim.num_measurements,
        "num_detectors": spec.stim.num_detectors,
        "num_observables": spec.stim.num_observables,
        "arrays": {"dets": "det_events.npy", "obs": "obs_flips.npy", "raw_meas": "raw_meas.npy"},
        "array_info": info,
        "placement": {str(q): n_ for q, n_ in sorted(spec.physical.items())},
        "circuit_files": {"stim": circuit_files["stim"], "qasm": circuit_files["qasm"]},
        "stim_sha256": spec.stim_sha256(),
        "qasm_sha256": circuit_files["qasm_sha256"],
        "native_sha256": circuit_files["native_sha256"],
        "raw": dict(raw_ref),
        "decoder": None if pred is None else {
            "kind": "pymatching", "via": "dqiqm.decode.decode", "dem": "noisy copy of the circuit that ran (decompose_errors=True)",
            "noise": decoder_note, "num_dem_errors": int(decoder_circuit.detector_error_model(decompose_errors=True).num_errors),
        },
        "duration_ns": C.schedule(spec.hardware, durations).duration_ns,
        "n_buckets": N_BUCKETS,
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_timeline": (extra or {}).get("job_timeline"),
        "commit": git_head(),
        "versions": versions(),
        "seeds": (extra or {}).get("seeds"),
        "simulator": (extra or {}).get("simulator") if simulated else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    numbers = {
        "shots": n, "det_rate": float(dets.mean()) if dets.size else 0.0,
        "all_zero": int((~raw.any(axis=1)).sum()),
        "obs_flips": int(obs[:, 0].sum()) if obs.shape[1] else None,
        "pm_failures": int((pred[:, 0] != obs[:, 0]).sum()) if pred is not None else None,
    }
    return out, numbers


def write_derived(root: Path, spec: C.CircuitSpec, run_id: str, dataset_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Write ``derived/<experiment>/<run_id>/`` from a written dataset directory."""
    out = root / "derived" / spec.experiment / run_id
    out.mkdir(parents=True, exist_ok=False)
    dets = load_array(dataset_dir, "dets")
    raw = load_array(dataset_dir, "raw_meas")
    with open(dataset_dir / "detectors.csv") as f:
        det_rows = list(csv.DictReader(f))
    t = np.array([int(r["round"]) for r in det_rows], dtype=int)
    labels = sorted({r["stabilizer"] for r in det_rows}, key=lambda s: [r["stabilizer"] for r in det_rows].index(s))
    stab = np.array([labels.index(r["stabilizer"]) for r in det_rows], dtype=int)
    n = len(dets)
    p = T.pij(dets)
    np.save(out / "pij.npy", p.astype(np.float32))
    lab = lambda i: labels[int(i)]  # noqa: E731
    write_csv(out / "pij_lags.csv", "derived/pij_lags.csv",
              [{"stabilizer_i": lab(r["stab_i"]), "stabilizer_j": lab(r["stab_j"]), "lag": r["lag"],
                "pij_mean": r["pij_mean"], "pij_max": r["pij_max"], "pairs": r["pairs"]} for r in T.lag_table(p, t, stab)])
    write_csv(out / "rate_by_round.csv", "derived/rate_by_round.csv",
              [{"detector": r["detector"], "stabilizer": lab(r["stab"]), "round": r["round"], "rate": r["rate"],
                "shots": r["shots"]} for r in T.rate_by_round(dets, t, stab)])
    write_csv(out / "rate_by_bucket.csv", "derived/rate_by_bucket.csv",
              [{**r, "stabilizer": lab(r["stab"])} for r in T.rate_by_bucket(dets, T.buckets(n, N_BUCKETS), stab)])
    _, streak_rows = T.streaks(dets, t, stab)
    write_csv(out / "streaks.csv", "derived/streaks.csv",
              [{"stabilizer": lab(r["stab"]), "length": r["length"], "count": r["count"]} for r in streak_rows])
    w, w_rounds = T.syndrome_weight(dets, t)
    write_csv(out / "syndrome_weight.csv", "derived/syndrome_weight.csv", T.weight_distribution(w, w_rounds))
    summary: dict[str, Any] = {"format": FORMAT, "dataset": _rel(dataset_dir, root), "shots": n,
                               "det_rate_mean": float(dets.mean()) if dets.size else 0.0, **T.lag_summary(p, t, stab)}
    if spec.params["family"] == "readout":
        nq = len(spec.physical)
        prep = [int(b) for b in spec.params["prep"]]
        names = [spec.physical[q] for q in range(nq)]
        rows = []
        for k in range(spec.rounds):
            rows += [{**r, "qb": names[r["qubit"]]} for r in T.assignment(raw, prep, nq, k)]
        write_csv(out / "assignment.csv", "derived/assignment.csv", rows)
        np.save(out / "readout_corr.npy", T.readout_correlation(raw, prep, nq, 0))
        if spec.rounds > 1:
            flips = T.readout_flips(dets, raw, prep, nq, spec.rounds)
            write_csv(out / "readout_flips.csv", "derived/readout_flips.csv", [{**r, "qb": names[r["qubit"]]} for r in flips])
            summary["record_flip_mean"] = float(np.mean([r["record_flip"] for r in flips]))
            summary["state_flip_mean"] = float(np.mean([r["state_flip"] for r in flips]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return out, summary


def write_ler(root: Path, experiment: str, run_id: str, points: Sequence[Mapping[str, Any]], *, title: str,
              label: str, fit: bool = True) -> tuple[Path, dict[str, Any]]:
    """Write ``ler/<experiment>/<run_id>/``: ler_points.csv, and with ``fit`` ler_fits.json and the P_L plot."""
    out = root / "ler" / experiment / run_id
    out.mkdir(parents=True, exist_ok=False)
    pts = sorted(points, key=lambda r: r["rounds"])
    r = np.array([x["rounds"] for x in pts])
    k = np.array([x["failures"] for x in pts])
    n = np.array([x["shots"] for x in pts])
    lo, hi = stats.wilson(k, n)
    e, elo, ehi = stats.eps_single(k, n, r)
    rows = [{**x, "experiment": experiment, "run_id": run_id, "p_l": x["failures"] / x["shots"], "wilson_lo": lo[i],
             "wilson_hi": hi[i], "eps_single": e[i], "eps_single_lo": elo[i], "eps_single_hi": ehi[i],
             "p_raw": x["raw_failures"] / x["shots"]} for i, x in enumerate(pts)]
    write_csv(out / "ler_points.csv", "ler/ler_points.csv", rows)
    result: dict[str, Any] = {"format": FORMAT, "experiment": experiment, "run_id": run_id, "points": "ler_points.csv"}
    if fit and len(np.unique(r)) >= 2:
        fixed = stats.fit_eps(r, k, n, free_amplitude=False, bootstrap=1000, seed=2026)
        free = stats.fit_eps(r, k, n, free_amplitude=True, bootstrap=1000, seed=2026)
        result |= {
            "model": "2 P_L = 1 - A (1 - 2 eps_L)^R, maximum likelihood on the binomial counts (dqiqm.stats.fit_eps)",
            "fixed_A": fixed.to_dict() | {"chi2_per_dof": fixed.chi2_per_dof},
            "free_A": free.to_dict() | {"chi2_per_dof": free.chi2_per_dof},
            "plot": "ler.png",
        }
        stats.plot_fidelity_decay({label: [free, fixed]}, title=title, source=_rel(out / "ler_points.csv", root),
                                  path=out / "ler.png")
    (out / "ler_fits.json").write_text(json.dumps(result, indent=1) + "\n")
    return out, result


def write_calibration_package(out_dir: Path, cal: dev.Calibration, suite: Any, *, label: str | None = None) -> list[Path]:
    """The pre-training package of one calibration: qubits.csv, couplers.csv, durations.json, placement.json, README.md.

    Everything a laptop needs to build a calibrated noise model for the circuits that will run,
    with no network: per-qubit and per-coupler figures, durations, and each experiment's Stim
    qubit -> device qubit map with roles. Made from the snapshot alone (and the seeded placement
    ranking, which is Stim simulation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = suite.arch
    stamp = {"calibration_set_id": cal.calibration_set_id, "fetched_utc": cal.fetched_utc}
    qrows = []
    for q in arch.qubits:
        c = cal.qubits[q]
        r, col = arch.coords[q]
        qrows.append({"qb": q, **stamp, "grid_row": r, "grid_col": col, "t1_ns": c["t1_ns"], "t2_ramsey_ns": c["t2_ramsey_ns"],
                      "t2_echo_ns": c["t2_ns"], "p1_given_0": c["p1_given_0"], "p0_given_1": c["p0_given_1"],
                      "readout_error": cal.readout_error(q), "error_1q": c["error_1q"], "readout_leakage": c.get("readout_leakage"),
                      "prx_ns": c["prx_ns"], "measure_ns": c["measure_ns"]})
    write_csv(out_dir / "qubits.csv", "calibration/qubits.csv", qrows)
    crows = [{"qb_a": a, "qb_b": b, **stamp, "cz_error_irb": v["cz_error"], "cz_error_clifford_rb": v.get("cz_error_clifford_rb"),
              "cz_ns": v["cz_ns"]} for (a, b), v in sorted(cal.couplers.items(), key=lambda kv: (dev.qubit_sort_key(kv[0][0]), dev.qubit_sort_key(kv[0][1])))]
    write_csv(out_dir / "couplers.csv", "calibration/couplers.csv", crows)
    dur = suite.durations
    (out_dir / "durations.json").write_text(json.dumps({
        **stamp, "units": "ns", "prx_ns": dict(dur.prx_ns), "measure_ns": dict(dur.measure_ns),
        "cz_ns": {f"{a}-{b}": v for (a, b), v in sorted(dur.cz_ns.items())}, "reset_wait_ns": cal.reset_wait_ns,
    }, indent=1) + "\n")

    def roles_of(spec):
        out = {}
        for q, role in sorted(spec.roles.items()):
            st = spec.stabilizers.get(q)
            out[str(q)] = {"qb": spec.physical[q], "role": role if role != "ancilla" else f"{st.pauli}_ancilla",
                           "stabilizer": st.label if st else None,
                           "stim_coords": list(spec.stim.get_final_qubit_coordinates()[q])}
        return out

    placements: dict[str, Any] = {**stamp, "note": (
        "Stim qubit index -> device qubit and role, for the circuits of circuits/. E1's Stim indices and coordinates are "
        "dqiqm.surface.memory_circuit's (Stim's rotated-code numbering, q = x + 7*floor(y/2) at d = 3), whose detectors are "
        "the team's make_stim_circuit's. Placements are chosen from this calibration; a new calibration set can move them.")}
    e1 = {s.params["patch"]: s for s in suite.of("E1") if s.rounds == min(x.rounds for x in suite.of("E1"))}
    placements["E1"] = {p: {"orientation": s.placement.orientation, "x_ancillas": sorted(
        (s.physical[q] for q in s.x_ancillas), key=dev.qubit_sort_key), "qubits": roles_of(s)} for p, s in sorted(e1.items())}
    placements["E1"]["ranking"] = suite.choices.get("E1")
    if suite.of("E2"):
        s = suite.of("E2")[0]
        placements["E2"] = {"line19": {"path": [s.physical[q] for q in sorted(s.physical)], "qubits": roles_of(s)},
                            "ranking": suite.choices.get("E2")}
    if suite.of("E5"):
        s = suite.of("E5")[0]
        placements["E5"] = {"block3x3": {"qubits": roles_of(s)}, "ranking": suite.choices.get("E5")}
    (out_dir / "placement.json").write_text(json.dumps(placements, indent=1, default=str) + "\n")
    (out_dir / "README.md").write_text(f"""# Garnet calibration{f' ({label})' if label else ''}: the pre-training package

Calibration set `{cal.calibration_set_id}`, fetched {cal.fetched_utc} UTC, read from `{cal.source}`
(`dqiqm.experiments.device.calibration_from_snapshot`). This is what the laptop needs to build the
team's `CalibrationSnapshot` and calibrated noise models for pre-training, before any hardware
shot exists:

| File | What |
|---|---|
| `qubits.csv` | per qubit: T1, T2 Ramsey, T2 echo, readout P(1\\|0) and P(0\\|1) of the default `measure` (not the heralded figure), the 1Q error, readout leakage, durations; times in ns |
| `couplers.csv` | per coupler: CZ error (interleaved RB), CZ error (Clifford RB), CZ duration |
| `durations.json` | prx, CZ and measure durations per qubit or coupler, and the passive-reset wait |
| `placement.json` | per experiment: Stim qubit -> device qubit, its role (data, X_ancilla, Z_ancilla) and check, and the Stim coordinates; E1's two orientations pA and pB, E2's line, E5's block |

Regenerate offline from the committed snapshot: `uv run python scripts/garnet_calibration_package.py`.
Every later snapshot (E6, before and after each batch) gets its own package next to it.
""")
    return [out_dir / f for f in ("qubits.csv", "couplers.csv", "durations.json", "placement.json", "README.md")]


# ---------------------------------------------------------------------------
# Checks and the README's dictionary.
# ---------------------------------------------------------------------------


def check_csv(path: Path, kind: str) -> list[str]:
    """Problems with one CSV against the dictionary: unknown or missing columns, empty required cells."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    problems = []
    try:
        _check_header(kind, header)
    except ValueError as e:
        problems.append(f"{path}: {e}")
    required = [i for i, h in enumerate(header) if not any(c.regex.match(h) and c.optional for c in COLUMNS[kind])]
    for n_row, row in enumerate(rows, 2):
        if len(row) != len(header):
            problems.append(f"{path}:{n_row}: {len(row)} cells for {len(header)} columns")
            break
        empty = [header[i] for i in required if row[i] == "" and header[i] not in ("stabilizer", "type", "data_qbs")]
        if empty:
            problems.append(f"{path}:{n_row}: empty required cells {empty}")
            break
    return problems


def check_run(dataset_dir: Path, derived_dir: Path | None = None) -> list[str]:
    """Every file of a dataset (and its derived) directory against the dictionary and the manifest."""
    dataset_dir = Path(dataset_dir)
    problems = []
    m = json.loads((dataset_dir / "manifest.json").read_text())
    for key in MANIFEST:
        if key not in m:
            problems.append(f"{dataset_dir}/manifest.json: missing key {key!r}")
    for key in m:
        if key not in MANIFEST:
            problems.append(f"{dataset_dir}/manifest.json: key {key!r} is not in the dictionary")
    for name in ("shots", "detectors", "measurements", "qubits"):
        problems += check_csv(dataset_dir / f"{name}.csv", f"datasets/{name}.csv")
    shapes = {"raw_meas": (m["shots"], m["num_measurements"]), "dets": (m["shots"], m["num_detectors"]),
              "obs": (m["shots"], m["num_observables"])}
    for key, shape in shapes.items():
        a = load_array(dataset_dir, key)
        if a.shape != shape:
            problems.append(f"{dataset_dir}: {key} has shape {a.shape}, expected {shape}")
        name = m["arrays"][key]
        if sha256_file(dataset_dir / name) != m["array_info"][name]["sha256"]:
            problems.append(f"{dataset_dir}/{name}: sha256 differs from the manifest")
    counts = {"shots": m["shots"], "detectors": m["num_detectors"], "measurements": m["num_measurements"]}
    for name, want in counts.items():
        with open(dataset_dir / f"{name}.csv") as f:
            got = sum(1 for _ in f) - 1
        if got != want:
            problems.append(f"{dataset_dir}/{name}.csv: {got} rows, expected {want}")
    if derived_dir is not None:
        for kind in COLUMNS:
            if kind.startswith("derived/"):
                path = Path(derived_dir) / kind.split("/", 1)[1]
                if path.exists():
                    problems += check_csv(path, kind)
        for name in ("pij_lags", "rate_by_round", "rate_by_bucket", "streaks", "syndrome_weight"):
            if not (Path(derived_dir) / f"{name}.csv").exists():
                problems.append(f"{derived_dir}: {name}.csv is missing")
        p = np.load(Path(derived_dir) / "pij.npy")
        if p.shape != (m["num_detectors"], m["num_detectors"]):
            problems.append(f"{derived_dir}/pij.npy: shape {p.shape}")
    return problems


#: Fields that change from run to run without changing the data: wall-clock times and the commit.
VOLATILE_KEYS = frozenset({"created_utc", "saved_utc", "commit", "wall_s", "seconds"})
_TIMING = re.compile(r"\b\d+(\.\d+)? s\b")


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def canonical_sha256(path: Path) -> str:
    """A file's hash without its volatile parts: JSON minus :data:`VOLATILE_KEYS`, .npz by its arrays, logs minus timings."""
    path = Path(path)
    h = hashlib.sha256()
    if path.suffix == ".json":
        h.update(json.dumps(_strip(json.loads(path.read_text())), sort_keys=True).encode())
    elif path.suffix == ".npz":  # zip members carry the write time; hash the arrays instead
        with np.load(path) as z:
            for k in sorted(z.files):
                a = z[k]
                h.update(f"{k}:{a.dtype}:{a.shape}:".encode())
                h.update(np.ascontiguousarray(a).tobytes())
    elif path.suffix == ".txt":
        h.update(_TIMING.sub("<t> s", path.read_text()).encode())
    else:
        h.update(path.read_bytes())
    return h.hexdigest()


def tree_manifest(root: Path, *, exclude: Sequence[str] = ("FILES.json",)) -> dict[str, Any]:
    """Every file under ``root`` with its size, sha256 and canonical sha256 (to check a regenerated tree)."""
    root = Path(root)
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        if rel in exclude:
            continue
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "canonical_sha256": canonical_sha256(path)}
    return {
        "format": FORMAT,
        "note": "canonical_sha256 ignores " + ", ".join(sorted(VOLATILE_KEYS)) + " in JSON, zip times in .npz and timings in logs",
        "files": len(files),
        "bytes": sum(f["bytes"] for f in files.values()),
        "tree": files,
    }


def compare_trees(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Paths whose canonical hashes differ between two :func:`tree_manifest` results, or that only one has."""
    ta, tb = a["tree"], b["tree"]
    out = [f"only in the first: {p}" for p in sorted(set(ta) - set(tb))]
    out += [f"only in the second: {p}" for p in sorted(set(tb) - set(ta))]
    out += [f"differs: {p}" for p in sorted(set(ta) & set(tb)) if ta[p]["canonical_sha256"] != tb[p]["canonical_sha256"]]
    return out


def dictionary_markdown() -> str:
    """The column dictionary as markdown tables (the README's section)."""
    lines = []
    for kind, cols in COLUMNS.items():
        lines += [f"### `{kind}`", "", "| column | type | unit | meaning |", "|---|---|---|---|"]
        lines += [f"| `{c.name}`{' (optional)' if c.optional else ''} | {c.dtype} | {c.unit} | {c.description} |" for c in cols]
        lines.append("")
    lines += ["### Arrays", "", "| file | contents |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in ARRAYS.items()]
    lines += ["", "### `datasets/manifest.json`", "", "| key | meaning |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in MANIFEST.items()]
    return "\n".join(lines) + "\n"
