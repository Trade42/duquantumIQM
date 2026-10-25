"""Real devices: InfoRecord calibration packages, the readout-window extension, the shot loader.

Three parts, all read-only on the team's ``challenge/`` package:

- **Calibration packages.** InfoRecord's ``calibration/<device>/`` (qubits.csv, couplers.csv,
  durations.json, placement.json, and the raw snapshot JSON) is vendored per calibration set
  under ``mlDecoder/calibration/<device>/<calset8>/`` (``python -m mlDecoder.hardware vendor``),
  so every spec rebuilds from the repo alone. ``snapshot`` builds the team's
  ``CalibrationSnapshot`` from it; ``qubit_map`` maps the circuit's Stim qubits to QB names by
  coordinates and checks every qubit.
- **The readout-window extension** (``readout_window``), applied after the team's ``add_noise``.
  On hardware every qubit idles through each mid-circuit readout window (about 1.1 us on
  Garnet), ancillas included, and the terminal layer (the last round's ancillas and the data)
  is one simultaneous ``measure_fidelity`` readout. The team's model idles only the qubits not
  measured in a moment, and runs the last ancilla readout and the data readout as two moments.
- **The InfoRecord loader** (``load_run``): one run directory of real (or simulated stand-in)
  shots, every file hash and the circuit's detector definitions checked against the team's
  generator. ``mlDecoder.data.inforecord_dataset`` turns a run's split into a ``Dataset``.

A device calibration is named in ``DatasetSpec.calibration`` by its full canonical string, e.g.
``inforecord:garnet:a18a947f:pA:t2=ramsey:window=1096:terminal=measure_fidelity:leak=none:readout=asym:qnd=excess``
(see ``DeviceNoise``): every choice that changes the noise is in the name. Two further
extensions, both from the calibration's own readout figures, are options:

- ``readout=asym``: each readout flips with P(1|0) or P(0|1) according to the value read, not
  their mean. Stim has no state-dependent channel, so the sampler draws the measurement record
  without readout flips and applies them afterwards (``apply_readout_flips``); the decoder
  prior's circuit keeps the team's symmetric mean.
- ``qnd=excess``: a mid-circuit readout also flips the ancilla's state with the probability its
  repeated-readout QND-ness misses (1 - qndness), less the X/Y flip the window idle already
  models. Without reset, a state flip shows as a lag-1 pair of detection events.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import stim

VENDORED = Path(__file__).resolve().parent / "calibration"
FORMAT = "dqiqm.inforecord/1"
PACKAGE_FILES = ("qubits.csv", "couplers.csv", "durations.json", "placement.json")
T2_CHOICES = ("ramsey", "echo", "mid")
TERMINAL_CHOICES = ("measure_fidelity", "measure")
LEAK_CHOICES = ("none", "readout")
READOUT_CHOICES = ("sym", "asym")
QND_CHOICES = ("none", "excess")
SPLITS = ("finetune", "validation", "test")
RESET_NS = 1.0   # The passive reset is between shots: no circuit time (see snapshot()).
MEASURE = {"M", "MX", "MY"}
# Snapshot metrics kept in the vendored metrics.json (per qubit), by their short names.
METRICS = {
    "measure_p10": "metrics.ssro.measure.constant.{q}.error_0_to_1",
    "measure_p01": "metrics.ssro.measure.constant.{q}.error_1_to_0",
    "measure_fidelity_p10": "metrics.ssro.measure_fidelity.constant.{q}.error_0_to_1",
    "measure_fidelity_p01": "metrics.ssro.measure_fidelity.constant.{q}.error_1_to_0",
    "measure_leakage": "metrics.rrc.measure.constant.{q}.leakage",
    "measure_seepage": "metrics.rrc.measure.constant.{q}.seepage",
    "measure_qndness": "metrics.rrc.measure.constant.{q}.qndness",
    "measure_fidelity_leakage": "metrics.rrc.measure_fidelity.constant.{q}.leakage",
    "measure_fidelity_qndness": "metrics.rrc.measure_fidelity.constant.{q}.qndness",
    "init_error": "characterization.model.{q}.init_error:herald=measure.constant:measure=measure.constant",
    "init_error_mf": "characterization.model.{q}.init_error:herald=measure.constant:measure=measure_fidelity.constant",
    "t1_time_s": "characterization.model.{q}.t1_time",
    "t2_time_s": "characterization.model.{q}.t2_time",
    "t2_echo_time_s": "characterization.model.{q}.t2_echo_time",
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inforecord_root() -> Path:
    """InfoRecord: MLDECODER_INFORECORD, else the nearest ancestor of this package holding an InfoRecord/ folder.

    No repo layout is assumed, so the package works wherever it sits (e.g. MLfolders/mlDecoder/).
    """
    if os.environ.get("MLDECODER_INFORECORD"):
        return Path(os.environ["MLDECODER_INFORECORD"])
    for parent in Path(__file__).resolve().parents:
        if (parent / "InfoRecord").is_dir():
            return parent / "InfoRecord"
    return Path("InfoRecord").resolve()


def device_root(device: str, root: Path | None = None) -> Path:
    """Garnet's record is InfoRecord's root; another device's is InfoRecord/<device>/."""
    root = Path(root) if root is not None else inforecord_root()
    return root if device == "garnet" else root / device


# ---------------------------------------------------------------- the calibration name

@dataclass(frozen=True)
class DeviceNoise:
    """A device calibration, its patch, and the extension's choices, parsed from a spec string."""
    device: str
    calset: str              # The calibration set id's first 8 hex digits.
    patch: str               # pA / pB (Garnet), eA / eB (Emerald): which placement maps the qubits.
    t2: str                  # ramsey (t2_time), echo (t2_echo_time) or mid (their geometric mean); capped at 2 T1.
    window_ns: float         # The readout window every qubit idles through, per mid-circuit readout layer.
    terminal: str            # measure_fidelity: the terminal layer's readout errors; measure: the mid-circuit ones.
    leak: str                # none, or readout: per-readout leakage of each ancilla, spread over its CZs.
    readout: str = "sym"     # sym (the team's mean) or asym (P(1|0) / P(0|1) by the value read, at sampling).
    qnd: str = "none"        # none, or excess: measurement-induced ancilla flips beyond the window idle.

    KEYS = ("t2", "window", "terminal", "leak", "readout", "qnd")

    @classmethod
    def parse(cls, text: str) -> "DeviceNoise":
        parts = text.split(":")
        if len(parts) != 4 + len(cls.KEYS) or parts[0] != "inforecord":
            raise ValueError(f"Device calibration must be 'inforecord:<device>:<calset8>:<patch>:t2=..:window=..:"
                             f"terminal=..:leak=..:readout=..:qnd=..', got {text!r}")
        _, device, calset, patch, *options = parts
        values = dict(option.partition("=")[::2] for option in options)
        if list(values) != list(cls.KEYS):
            raise ValueError(f"Options must be, in order, {cls.KEYS}; got {list(values)}")
        if not re.fullmatch(r"[0-9a-f]{8}", calset):
            raise ValueError(f"calset must be 8 hex digits, got {calset!r}")
        out = cls(device, calset, patch, values["t2"], float(values["window"]), values["terminal"], values["leak"],
                  values["readout"], values["qnd"])
        if (out.t2 not in T2_CHOICES or out.terminal not in TERMINAL_CHOICES or out.leak not in LEAK_CHOICES
                or out.readout not in READOUT_CHOICES or out.qnd not in QND_CHOICES):
            raise ValueError(f"Unsupported option in {text!r}")
        if not math.isfinite(out.window_ns) or out.window_ns <= 0:
            raise ValueError("window must be a positive duration in ns")
        if out.text() != text:
            raise ValueError(f"Not in canonical form: {text!r} (canonical: {out.text()!r})")
        return out

    def text(self) -> str:
        window = f"{self.window_ns:g}"
        return (f"inforecord:{self.device}:{self.calset}:{self.patch}:t2={self.t2}:window={window}"
                f":terminal={self.terminal}:leak={self.leak}:readout={self.readout}:qnd={self.qnd}")

    def slug(self) -> str:
        return (f"{self.device}-{self.calset}-{self.patch}-{self.t2}-w{self.window_ns:g}"
                f"-{'mf' if self.terminal == 'measure_fidelity' else 'm'}-{'leakro' if self.leak == 'readout' else 'noleak'}"
                f"-{self.readout}-{'qndx' if self.qnd == 'excess' else 'qnd0'}")


def device_calibration(device: str, calset: str, patch: str, *, t2: str = "ramsey", window_ns: float,
                       terminal: str = "measure_fidelity", leak: str = "none", readout: str = "sym",
                       qnd: str = "none") -> str:
    """The canonical spec string (DatasetSpec.calibration) for a device patch."""
    return DeviceNoise(device, calset[:8], patch, t2, float(window_ns), terminal, leak, readout, qnd).text()


def is_device(calibration: str | None) -> bool:
    return bool(calibration) and calibration.startswith("inforecord:")


# ---------------------------------------------------------------- vendored packages

def package_dir(device: str, calset: str) -> Path:
    return VENDORED / device / calset[:8]


@lru_cache(maxsize=None)
def package(device: str, calset: str) -> dict:
    """A vendored calibration package, its files checked against SOURCE.json's hashes."""
    directory = package_dir(device, calset)
    if not (directory / "SOURCE.json").exists():
        raise FileNotFoundError(f"No vendored package {directory}; run `python -m mlDecoder.hardware vendor "
                                f"--device {device}` with InfoRecord present.")
    source = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
    for name, digest in source["vendored_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"{directory / name}: hash mismatch against SOURCE.json")
    with (directory / "qubits.csv").open(encoding="utf-8", newline="") as stream:
        qubits = {row["qb"]: row for row in csv.DictReader(stream)}
    with (directory / "couplers.csv").open(encoding="utf-8", newline="") as stream:
        couplers = list(csv.DictReader(stream))
    load = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    calsets = {row["calibration_set_id"] for row in qubits.values()}
    if len(calsets) != 1 or not next(iter(calsets)).startswith(calset[:8]):
        raise ValueError(f"{directory}: qubits.csv is for calibration sets {calsets}, not {calset}")
    return dict(directory=directory, source=source, qubits=qubits, couplers=couplers,
                durations=load("durations.json"), placement=load("placement.json"), metrics=load("metrics.json"),
                calibration_set_id=next(iter(calsets)))


def vendor(device: str, *, root: Path | None = None, out: Path = VENDORED) -> Path:
    """Copy a device's InfoRecord calibration package into mlDecoder/calibration/<device>/<calset8>/.

    The four package files are copied byte for byte; the snapshot JSON's per-qubit metrics that
    the noise build reads (the measure_fidelity readout errors, init errors, readout leakage)
    are extracted into metrics.json. SOURCE.json records every source file's sha256.
    """
    src = device_root(device, root) / "calibration" / device
    snapshots = sorted(src.glob("*.json"))
    snapshots = [p for p in snapshots if p.name not in {"durations.json", "placement.json"}]
    with (src / "qubits.csv").open(encoding="utf-8", newline="") as stream:
        calset = next(csv.DictReader(stream))["calibration_set_id"]
    matching = [p for p in snapshots if calset in p.name]
    if len(matching) != 1:
        raise ValueError(f"Expected one snapshot JSON for {calset} in {src}, found {[p.name for p in matching]}")
    snap_path = matching[0]
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    if snap.get("calibration_set_id") != calset or snap.get("device") != device:
        raise ValueError(f"{snap_path}: snapshot is for {snap.get('device')} / {snap.get('calibration_set_id')}")
    observations = {o["dut_field"]: o["value"] for o in snap["quality_metric_set"]["observations"] if not o.get("invalid")}
    qubits = snap["architecture"]["qubits"]
    # Each device names its default readout implementations (Garnet: measure_fidelity.constant; Emerald:
    # measure_fidelity.shelved_constant); the metrics are filed under them.
    gates = snap["architecture"]["gates"]
    impl = {g: gates[g]["default_implementation"] for g in ("measure", "measure_fidelity") if g in gates}
    fields = {name: field.replace("measure_fidelity.constant", f"measure_fidelity.{impl.get('measure_fidelity', 'constant')}")
              .replace("ssro.measure.constant", f"ssro.measure.{impl.get('measure', 'constant')}")
              .replace("rrc.measure.constant", f"rrc.measure.{impl.get('measure', 'constant')}")
              for name, field in METRICS.items()}
    metrics = {q: {name: observations.get(field.format(q=q)) for name, field in fields.items()} for q in qubits}
    target = out / device / calset[:8]
    target.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_FILES:
        shutil.copyfile(src / name, target / name)
    (target / "metrics.json").write_text(json.dumps(dict(
        device=device, calibration_set_id=calset, fetched_utc=snap.get("fetched_utc_end"),
        quality_metric_set_id=snap.get("quality_metric_set_id"), fields=fields, implementations=impl,
        note="Extracted from the snapshot JSON's quality_metric_set by mlDecoder.hardware.vendor; null = not reported.",
        qubits=metrics), indent=1) + "\n", encoding="utf-8", newline="\n")
    readme = src / "README.md"
    source = dict(device=device, calibration_set_id=calset, source_directory=f"InfoRecord/{src.relative_to(src.parents[2]).as_posix()}"
                  if device != "garnet" else "InfoRecord/calibration/garnet",
                  snapshot_json=snap_path.name, source_sha256={p.name: sha256_file(p) for p in
                  [*(src / n for n in PACKAGE_FILES), snap_path] + ([readme] if readme.exists() else [])},
                  vendored_sha256={n: sha256_file(target / n) for n in (*PACKAGE_FILES, "metrics.json")})
    (target / "SOURCE.json").write_text(json.dumps(source, indent=1) + "\n", encoding="utf-8", newline="\n")
    package.cache_clear()
    return target


# ---------------------------------------------------------------- snapshot and qubit map

def _float(value) -> float | None:
    return None if value in (None, "") else float(value)


def _placement(pkg: dict, patch: str) -> dict:
    for experiment in ("E1",):
        if patch in pkg["placement"].get(experiment, {}):
            return pkg["placement"][experiment][patch]
    raise ValueError(f"Patch {patch!r} is not in the package placement ({pkg['directory']})")


def patch_roles(pkg: dict, patch: str) -> dict:
    """QB -> (role, weight): role data / X_ancilla / Z_ancilla; weight = data neighbours of an ancilla."""
    qubits = _placement(pkg, patch)["qubits"]
    data_xy = {tuple(v["stim_coords"]) for v in qubits.values() if v["role"] == "data"}
    out = {}
    for v in qubits.values():
        x, y = v["stim_coords"]
        weight = sum((x + dx, y + dy) in data_xy for dx in (-1, 1) for dy in (-1, 1)) if v["role"] != "data" else 0
        out[v["qb"]] = (v["role"], weight)
    return out


@lru_cache(maxsize=None)
def snapshot(calibration: str):
    """The team's CalibrationSnapshot for a device patch, from its vendored package.

    - 1q infidelity = error_1q (RB); CZ = cz_error_irb (interleaved RB); both average infidelities,
      which the team's model converts with (D+1)/D.
    - readout_error = the mean of P(1|0) and P(0|1) of ``measure`` (the team's readout is symmetric).
    - reset_error = the heralded init error (the passive reset_wait between shots leaves thermal
      population); a qubit without one takes the init error measured with measure_fidelity, else
      the device median (listed in metadata['imputed']).
    - t2_ns: Ramsey T2 (t2_time), echo, or their geometric mean, capped at 2 T1.
    - leakage_per_gate: 0 for leak=none. For leak=readout, each ancilla's per-readout leakage L
      spread over its w CZs per round, 1 - (1 - L)^(1/w); data qubits 0 (read only at the end).
    - durations: 1q = the longest prx, 2q = the longest CZ, measure = the readout window (so the
      team's model idles unmeasured qubits for the whole window), reset = 1 ns: the passive reset
      happens between shots (reset_wait, recorded in metadata) and takes no time inside the
      circuit, but the generator splits an X memory's initial RX (data) and R (ancillas) into two
      moments, so a reset duration would idle half the qubits. Resets only start the circuit
      (checked in readout_window); their error is reset_error.
    """
    from challenge.calibration import CalibrationSnapshot, QubitCalibration
    noise = DeviceNoise.parse(calibration)
    pkg = package(noise.device, noise.calset)
    roles = patch_roles(pkg, noise.patch)
    measured_init = [m["init_error"] for m in pkg["metrics"]["qubits"].values() if m.get("init_error") is not None]
    median_init = float(np.median(measured_init))
    qubits, imputed, excluded = {}, [], []
    required = ("t1_ns", "t2_ramsey_ns", "t2_echo_ns", "p1_given_0", "p0_given_1", "error_1q")
    for qb, row in pkg["qubits"].items():
        missing = [k for k in required if row.get(k) in (None, "")]
        if missing:
            if qb in roles:
                raise ValueError(f"{qb} of patch {noise.patch} has no {missing} in the package")
            excluded.append(dict(qb=qb, missing=missing))       # Not in the patch: left out, as the team's snapshot does.
            continue
        t1 = float(row["t1_ns"])
        ramsey, echo = float(row["t2_ramsey_ns"]), float(row["t2_echo_ns"])
        t2 = {"ramsey": ramsey, "echo": echo, "mid": math.sqrt(ramsey * echo)}[noise.t2]
        m = pkg["metrics"]["qubits"].get(qb, {})
        init = m.get("init_error")
        if init is None:
            init = m.get("init_error_mf")
            imputed.append(dict(qb=qb, field="reset_error", value=init if init is not None else median_init,
                                source="init error measured with measure_fidelity" if init is not None
                                else "device median of the measured init errors"))
            init = init if init is not None else median_init
        leak = 0.0
        role, weight = roles.get(qb, ("unused", 0))
        if noise.leak == "readout" and role != "data" and weight:
            per_readout = _float(row.get("readout_leakage")) or 0.0
            leak = 1 - (1 - per_readout) ** (1 / weight)
        qubits[qb] = QubitCalibration(single_qubit_infidelity=float(row["error_1q"]),
                                      readout_error=(float(row["p1_given_0"]) + float(row["p0_given_1"])) / 2,
                                      reset_error=float(init), t1_ns=t1, t2_ns=min(t2, 2 * t1), leakage_per_gate=leak)
    edges = {CalibrationSnapshot.edge_key(r["qb_a"], r["qb_b"]): float(r["cz_error_irb"]) for r in pkg["couplers"]
             if r["qb_a"] in qubits and r["qb_b"] in qubits and r.get("cz_error_irb") not in (None, "")}
    durations = pkg["durations"]
    if durations.get("units") != "ns":
        raise ValueError("durations.json must be in ns")
    durations_ns = {"1q": max(durations["prx_ns"].values()), "2q": max(durations["cz_ns"].values()),
                    "measure": noise.window_ns, "reset": RESET_NS}
    metadata = dict(source=f"InfoRecord {noise.device} calibration package (HARDWARE calibration set "
                    f"{pkg['calibration_set_id']}), vendored in {pkg['directory'].relative_to(VENDORED.parent).as_posix()}",
                    device=noise.device, calibration_set_id=pkg["calibration_set_id"], patch=noise.patch,
                    fetched_utc=pkg["source"].get("fetched_utc") or next(iter(pkg["qubits"].values()))["fetched_utc"],
                    t2=noise.t2, readout_window_ns=noise.window_ns, terminal_readout=noise.terminal, leak=noise.leak,
                    package_sha256=pkg["source"]["vendored_sha256"], imputed=imputed, excluded=excluded,
                    reset_wait_ns=float(durations["reset_wait_ns"]),
                    durations_note="measure = the readout window, not the probe pulse (measure_ns in durations.json)")
    return CalibrationSnapshot(qubits, edges, durations_ns, metadata)


@lru_cache(maxsize=None)
def _qubit_map(calibration: str, coords_key: tuple) -> dict:
    noise = DeviceNoise.parse(calibration)
    pkg = package(noise.device, noise.calset)
    by_xy = {}
    for v in _placement(pkg, noise.patch)["qubits"].values():
        by_xy[tuple(map(float, v["stim_coords"]))] = v["qb"]
    mapping, missing = {}, []
    for q, xy in coords_key:
        qb = by_xy.get(tuple(map(float, xy[:2])))
        if qb is None:
            missing.append((q, xy))
        mapping[q] = qb
    if missing or len(coords_key) != len(by_xy):
        raise ValueError(f"Placement {noise.patch} does not cover the circuit: {len(coords_key)} circuit qubits, "
                         f"{len(by_xy)} placed, unmatched {missing}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Placement maps two circuit qubits to one QB")
    return mapping


def qubit_map(calibration: str, circuit: stim.Circuit) -> dict:
    """Stim qubit -> QB name, matched by coordinates against the patch's placement (all must match)."""
    coords = circuit.get_final_qubit_coordinates()
    return _qubit_map(calibration, tuple(sorted((q, tuple(xy)) for q, xy in coords.items())))


def readout_errors(calibration: str, mapping: dict, *, terminal: bool) -> dict:
    """Stim qubit -> (P(1|0), P(0|1)) of a mid-circuit (measure) or terminal readout."""
    noise = DeviceNoise.parse(calibration)
    pkg = package(noise.device, noise.calset)
    out = {}
    for q, qb in mapping.items():
        if not terminal or noise.terminal == "measure":
            row = pkg["qubits"][qb]
            out[q] = (float(row["p1_given_0"]), float(row["p0_given_1"]))
        else:
            m = pkg["metrics"]["qubits"][qb]
            if m["measure_fidelity_p10"] is None or m["measure_fidelity_p01"] is None:
                raise ValueError(f"{qb}: no measure_fidelity readout errors in the package")
            out[q] = (m["measure_fidelity_p10"], m["measure_fidelity_p01"])
    return out


def terminal_readout(calibration: str, mapping: dict) -> dict:
    """Stim qubit -> symmetric readout error of the terminal layer (mean of its P(1|0) and P(0|1))."""
    return {q: (a + b) / 2 for q, (a, b) in readout_errors(calibration, mapping, terminal=True).items()}


def qnd_flip(calibration: str, mapping: dict, model) -> dict:
    """Stim qubit -> state-flip probability per mid-circuit readout beyond the window idle (qnd=excess)."""
    noise = DeviceNoise.parse(calibration)
    pkg = package(noise.device, noise.calset)
    out = {}
    for q, qb in mapping.items():
        qnd = pkg["metrics"]["qubits"][qb].get("measure_qndness")
        if qnd is None:
            raise ValueError(f"{qb}: no measure qndness in the package")
        idle = 0.0
        if model.coherence:
            px, py, _ = model.coherence_channel(q, noise.window_ns, mapping)
            idle = px + py
        out[q] = max(0.0, (1 - qnd) - idle)
    return out


def readout_flip_table(calibration: str, mapping: dict, ideal: stim.Circuit) -> tuple[np.ndarray, np.ndarray]:
    """Per measurement, in record order: (P(1|0), P(0|1)); the terminal layer takes the terminal figures."""
    from challenge.internal_helpers import get_meas_order, get_qubit_lists
    order = get_meas_order(ideal)
    data, anc = (set(v) for v in get_qubit_lists(ideal))
    mid = readout_errors(calibration, mapping, terminal=False)
    term = readout_errors(calibration, mapping, terminal=True)
    last = len(order) - len(data) - len(anc)       # The terminal layer: the last ancilla readout and the data.
    if sorted(order[last:]) != sorted(data | anc):
        raise ValueError("The circuit does not end with one ancilla readout and one data readout")
    pairs = [(term if j >= last else mid)[q] for j, q in enumerate(order)]
    return np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])


def strip_readout_flips(circuit: stim.Circuit) -> stim.Circuit:
    """The same circuit with every measurement's flip probability set to 0 (readout=asym sampling)."""
    out = stim.Circuit()
    for inst in circuit.flattened():
        out.append(inst.name, inst.targets_copy(), [0.0] if inst.name in MEASURE else inst.gate_args_copy())
    return out


def apply_readout_flips(raw: np.ndarray, p10: np.ndarray, p01: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip each recorded bit with P(1|0) if it reads 0, P(0|1) if it reads 1 (raw: (N, M) bool, in place)."""
    u = rng.random(raw.shape, dtype=np.float32)
    raw ^= np.where(raw, u < p01.astype(np.float32), u < p10.astype(np.float32))
    return raw


# ---------------------------------------------------------------- the readout-window extension

def _moments(circuit: stim.Circuit) -> list[list[stim.CircuitInstruction]]:
    moments, current = [], []
    for inst in circuit.flattened():
        current.append(inst)
        if inst.name == "TICK":
            moments.append(current)
            current = []
    if current:
        moments.append(current)
    return moments


def _measured(moment) -> set[int]:
    return {t.value for inst in moment if inst.name in MEASURE for t in inst.targets_copy() if t.is_qubit_target}


def readout_window(noisy: stim.Circuit, model, mapping: dict, calibration: str) -> stim.Circuit:
    """The team's noisy circuit with the hardware readout layers (mlDecoder extension).

    - Every mid-circuit ancilla readout: the measured ancillas also idle through the window
      (T1/T2 Pauli twirl of ``model.coherence_channel`` for ``window_ns``, applied after the
      measurement, so it flips the post-measurement state that the next round inherits). The
      unmeasured qubits already idle for the window in the team's model (measure = window).
    - The terminal layer (the last ancilla readout and the data readout, simultaneous on
      hardware): no idle on anyone, and every readout takes the terminal error (measure_fidelity).

    Models without coherence (calibrated_pauli) get the terminal readout errors only. The ideal
    circuit (``without_noise``) is unchanged.
    """
    from challenge.internal_helpers import get_qubit_lists
    noise = DeviceNoise.parse(calibration)
    data, anc = (set(v) for v in get_qubit_lists(noisy))
    moments = _moments(noisy)
    started = False
    for moment in moments:
        names = {i.name for i in moment}
        if names & {"R", "RX"} and started:
            raise ValueError("A reset after the first gate: the passive reset only starts the circuit")
        started |= bool(names & {"H", "CZ", "CX", "M", "MX"})
    meas = [i for i, m in enumerate(moments) if _measured(m)]
    if len(meas) < 2 or _measured(moments[meas[-1]]) != data or any(_measured(moments[i]) != anc for i in meas[:-1]):
        raise ValueError("Expected ancilla readout layers followed by one data readout layer")
    terminal = set(meas[-2:])
    term_p = terminal_readout(calibration, mapping)
    flips = qnd_flip(calibration, mapping, model) if noise.qnd == "excess" else {}
    out = stim.Circuit()
    for index, moment in enumerate(moments):
        measured = _measured(moment)
        if index in terminal:
            for inst in moment:
                if inst.name == "PAULI_CHANNEL_1" and not ({t.value for t in inst.targets_copy()} & measured):
                    continue                                   # No idle inside the simultaneous terminal layer.
                if inst.name in MEASURE:
                    for t in inst.targets_copy():
                        out.append(inst.name, [t], term_p[t.value])
                    continue
                out.append(inst)
            continue
        tick = moment[-1].name == "TICK"
        body = moment[:-1] if tick else moment
        for inst in body:
            out.append(inst)
        if measured and model.coherence:
            for q in sorted(measured):
                channel = model.coherence_channel(q, noise.window_ns, mapping)
                if sum(channel):
                    out.append("PAULI_CHANNEL_1", [q], channel)
        for q in sorted(measured):
            if flips.get(q):
                out.append("X_ERROR", [q], flips[q])       # Measurement-induced state flip (qnd=excess).
        if tick:
            out.append(moment[-1])
    if out.without_noise() != noisy.without_noise():
        raise AssertionError("readout_window changed the ideal circuit")
    return out


# ---------------------------------------------------------------- the InfoRecord loader

@dataclass
class Run:
    """One InfoRecord run: one circuit in one batch (or the pilot)."""
    path: Path
    manifest: dict
    rows: list[dict]                 # shots.csv, one dict per shot, in shot order.
    dets: np.ndarray                 # (N, num_detectors) uint8
    obs: np.ndarray                  # (N, num_observables) uint8
    raw_meas: np.ndarray | None      # (N, num_measurements) uint8
    detectors: list[dict]            # detectors.csv rows, in detector order.
    circuit: stim.Circuit            # The run's noise-free .stim (detectors identical to the team's).

    @property
    def spec(self) -> dict:
        return self.manifest["spec"]

    @property
    def label(self) -> str:
        return self.manifest["label"]

    @property
    def device(self) -> str:
        return self.manifest["device"]

    @property
    def split(self) -> np.ndarray:
        return np.array([r["split"] for r in self.rows])

    def describe(self) -> dict:
        m = self.manifest
        counts = {s: int((self.split == s).sum()) for s in SPLITS}
        return dict(run_id=m["run_id"], label=m["label"], device=m["device"], calibration_set_id=m["calibration_set_id"],
                    batch=m["batch"], job_id=m["job_id"], circuit=m["circuit"], patch=self.spec["patch"],
                    basis=self.spec["basis"], rounds=self.spec["rounds"], barriers=self.spec["barriers"],
                    shots=m["shots"], split_counts=counts, path=self.path.as_posix())


def _load_array(run_dir: Path, name: str, info: dict) -> np.ndarray:
    path = run_dir / name
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"{path}: sha256 mismatch against the manifest")
    array = np.load(path, allow_pickle=False)
    shape = tuple(info["shape"])
    if info.get("packed"):
        if info.get("bitorder") != "little" or info.get("axis") not in (1, None):
            raise ValueError(f"{path}: unsupported packing {info}")
        array = np.unpackbits(array, axis=1, count=shape[1], bitorder="little")
    if array.shape != shape or array.dtype != np.uint8:
        raise ValueError(f"{path}: shape {array.shape} {array.dtype}, manifest says {shape} uint8")
    if array.size and array.max() > 1:
        raise ValueError(f"{path}: not 0/1")
    return array


def _definitions(circuit: stim.Circuit) -> list:
    """Detector and observable definitions as sets of (measured qubit, occurrence)."""
    from challenge.internal_helpers import get_meas_order
    occurrence, count = [], {}
    for q in get_meas_order(circuit):
        occurrence.append((q, count.get(q, 0)))
        count[q] = count.get(q, 0) + 1
    out, n = [], 0
    for inst in circuit.flattened():
        if inst.name in {"M", "MX", "MY", "MR", "MRX", "MRY"}:
            n += sum(t.is_qubit_target for t in inst.targets_copy())
        elif inst.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            out.append((inst.name, tuple(inst.gate_args_copy()) if inst.name == "DETECTOR" else (),
                        frozenset(occurrence[n + t.value] for t in inst.targets_copy())))
    return out


def same_definitions(a: stim.Circuit, b: stim.Circuit) -> bool:
    """Same measurement order, detectors (with coordinates) and observables, as measured-qubit records."""
    from challenge.internal_helpers import get_meas_order
    return (get_meas_order(a) == get_meas_order(b) and a.get_final_qubit_coordinates() == b.get_final_qubit_coordinates()
            and _definitions(a) == _definitions(b))


def load_run(run_dir, *, record_root: Path | None = None) -> Run:
    """Load and check one InfoRecord run directory.

    Checks: the format; every array's sha256, shape and packing (``array_info``); shots.csv
    against the manifest (shot count, patch, basis, rounds, barriers; splits in finetune /
    validation / test); detectors.csv against the circuit; and the run's .stim against its
    ``stim_sha256``. ``record_root`` (where ``circuits/`` lives) defaults to the run's
    InfoRecord root (three levels up from datasets/<experiment>/<run>).
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{run_dir}: format {manifest.get('format')!r}, expected {FORMAT}")
    if manifest.get("label") not in {"HARDWARE", "SIMULATED"} or manifest["label"] == "HARDWARE" and manifest.get("simulated"):
        raise ValueError(f"{run_dir}: inconsistent label {manifest.get('label')} / simulated {manifest.get('simulated')}")
    info = manifest["array_info"]
    arrays = {key: _load_array(run_dir, name, info[name]) for key, name in manifest["arrays"].items()}
    dets, obs, raw = arrays["dets"], arrays["obs"], arrays.get("raw_meas")
    n = manifest["shots"]
    if not (len(dets) == len(obs) == n) or (raw is not None and len(raw) != n):
        raise ValueError(f"{run_dir}: array lengths disagree with shots={n}")
    if (dets.shape[1], obs.shape[1]) != (manifest["num_detectors"], manifest["num_observables"]):
        raise ValueError(f"{run_dir}: array widths disagree with the manifest")
    with (run_dir / "shots.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    spec = manifest["spec"]
    if len(rows) != n or [int(r["shot"]) for r in rows] != list(range(n)):
        raise ValueError(f"{run_dir}: shots.csv must list shots 0..{n - 1} in order")
    for key, cast in (("patch", str), ("basis", str), ("rounds", int)):
        if {cast(r[key]) for r in rows} != {cast(spec[key])}:
            raise ValueError(f"{run_dir}: shots.csv {key} disagrees with the manifest spec")
    if {bool(int(r["barriers"])) for r in rows} != {bool(spec["barriers"])}:
        raise ValueError(f"{run_dir}: shots.csv barriers disagree with the manifest spec")
    if not {r["split"] for r in rows} <= set(SPLITS):
        raise ValueError(f"{run_dir}: unknown split values {set(r['split'] for r in rows) - set(SPLITS)}")
    if "obs_flip" in rows[0] and rows[0]["obs_flip"] != "":
        if not np.array_equal(np.array([int(r["obs_flip"]) for r in rows]), obs[:, 0]):
            raise ValueError(f"{run_dir}: shots.csv obs_flip disagrees with obs_flips.npy")
    with (run_dir / "detectors.csv").open(encoding="utf-8", newline="") as stream:
        detectors = list(csv.DictReader(stream))
    root = Path(record_root) if record_root is not None else run_dir.parents[2]
    # The stand-in's sim/ tree keeps no circuits/ of its own: its circuits are the record's.
    candidates = [root / manifest["circuit_files"]["stim"], root.parent / manifest["circuit_files"]["stim"]]
    stim_path = next((c for c in candidates if c.exists()), candidates[0])
    circuit = stim.Circuit(stim_path.read_text(encoding="utf-8"))
    if hashlib.sha256(str(circuit).encode()).hexdigest() != manifest["stim_sha256"]:
        raise ValueError(f"{stim_path}: stim_sha256 mismatch against the manifest")
    if (circuit.num_detectors, circuit.num_observables) != (dets.shape[1], obs.shape[1]):
        raise ValueError(f"{run_dir}: the circuit has {circuit.num_detectors} detectors, the data {dets.shape[1]}")
    if raw is not None and raw.shape[1] != circuit.num_measurements:
        raise ValueError(f"{run_dir}: raw_meas has {raw.shape[1]} columns, the circuit {circuit.num_measurements}")
    coords = circuit.get_detector_coordinates()
    if len(detectors) != circuit.num_detectors or any(
            (int(d["index"]), float(d["x"]), float(d["y"]), float(d["t"]), int(d["round"])) !=
            (k, coords[k][0], coords[k][1], coords[k][2], int(coords[k][2])) for k, d in enumerate(detectors)):
        raise ValueError(f"{run_dir}: detectors.csv disagrees with the circuit's detector coordinates")
    if raw is not None:
        dets_raw, obs_raw = circuit.compile_m2d_converter().convert(measurements=raw.astype(bool), separate_observables=True)
        if not (np.array_equal(dets_raw.astype(np.uint8), dets) and np.array_equal(obs_raw.astype(np.uint8), obs)):
            raise ValueError(f"{run_dir}: det_events / obs_flips are not the m2d conversion of raw_meas")
    return Run(run_dir, manifest, rows, dets, obs, raw, detectors, circuit)


def find_runs(root: Path | None = None, *, device: str = "garnet", experiment: str = "E1",
              simulated: bool = False) -> list[Path]:
    """Run directories under <device record>/datasets/<experiment>/ (or sim/datasets/ when simulated)."""
    base = device_root(device, root) / ("sim" if simulated else "") / "datasets" / experiment
    return sorted(p for p in base.glob("*") if (p / "manifest.json").exists()) if base.exists() else []


def check_run_against(run: Run, ideal: stim.Circuit, calibration: str | None) -> None:
    """Refuse a run whose circuit, device or patch differs from the spec it is loaded for."""
    if not same_definitions(run.circuit, ideal):
        raise ValueError(f"{run.path}: the run's circuit differs from the team's (detectors, observables or order)")
    if is_device(calibration):
        noise = DeviceNoise.parse(calibration)
        if noise.device not in run.device.lower():
            raise ValueError(f"{run.path}: device {run.device!r} is not {noise.device!r}")
        if run.spec["patch"] != noise.patch:
            raise ValueError(f"{run.path}: patch {run.spec['patch']} is not the spec's {noise.patch}")
        if not run.manifest["calibration_set_id"].startswith(noise.calset) and run.label == "HARDWARE":
            # A later calibration set is allowed (the prior is a prior), but it is recorded.
            run.manifest.setdefault("notes", []).append(
                f"ran on calibration set {run.manifest['calibration_set_id']}, decoder prior from {noise.calset}")
        mapping = qubit_map(calibration, ideal)
        placed = {int(k): v for k, v in run.manifest["placement"].items()}
        if placed != mapping:
            raise ValueError(f"{run.path}: the run's placement differs from the package's {noise.patch} by coordinates")


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description="mlDecoder.hardware: vendor calibration packages, check InfoRecord runs")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("vendor", help="Copy a device's InfoRecord calibration package into mlDecoder/calibration/")
    v.add_argument("--device", required=True)
    v.add_argument("--root", type=Path)
    c = sub.add_parser("check", help="Load and check every run of a device (hashes, circuits, splits)")
    c.add_argument("--device", default="garnet")
    c.add_argument("--root", type=Path)
    c.add_argument("--simulated", action="store_true", help="The SIMULATED stand-in runs under sim/")
    args = parser.parse_args(argv)
    if args.command == "vendor":
        print(f"vendored {vendor(args.device, root=args.root)}")
        return 0
    runs = find_runs(args.root, device=args.device, simulated=args.simulated)
    root = device_root(args.device, args.root) / ("sim" if args.simulated else "")
    for path in runs:
        run = load_run(path, record_root=root)
        print(json.dumps(run.describe()))
    print(f"{len(runs)} runs checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
