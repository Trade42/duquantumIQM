"""IQM Garnet as the experiments see it: its grid, today's calibration, durations, and a noisy stand-in.

The calibration
    :class:`Calibration` holds per-qubit and per-coupler figures in plain units (ns, probabilities),
    read from a live snapshot (:func:`calibration_from_snapshot`) as ``scripts/resonance_check.py``
    writes it (schema ``dqiqm.calibration-snapshot/1``), or from a fake backend's error profile
    (:func:`calibration_from_error_profile`). The readout figures are the default ``measure``'s
    assignment errors, ``metrics.ssro.measure.<impl>.<q>.error_0_to_1`` / ``error_1_to_0``, not
    iqm-qubit-selector's heralded ``readout`` (``hw.calibration_fidelities``), which is optimistic.
    The terminal readout's are kept beside them (``p1_given_0_terminal``, ``p0_given_1_terminal``):
    IQM compiles a qubit's last measurement to ``measure_fidelity`` (pulla-check,
    ``docs/pulla-check.md``), and these are its unheralded errors, ``metrics.ssro.measure_fidelity.*``.
    The field choices are the resonance-check task's (``docs/resonance-check.md``, "Which field is
    which"); :func:`calibration_from_snapshot` follows its ``figures()`` (``scripts/resonance_check.py``).
    Durations are read in their unit; the snapshot gives seconds (some with an empty unit).

The stand-in device
    :func:`fake_backend` is ``IQMFakeGarnet().copy_with_error_profile(...)`` with the profile built
    from a calibration (:func:`error_profile`, following resonance-check's ``fake_comparison``): the
    live names and couplers, today's per-qubit T1, T2, 1Q and CZ errors, and the asymmetric readout.
    The T2 is the Ramsey T2 (capped at 2 T1): no circuit here has dynamical decoupling, so a qubit
    dephases as in a Ramsey experiment, not an echo one. (resonance-check's stand-in used the echo
    T2; pass ``t2="echo"`` for that.) Its noise is what Aer applies: a depolarizing channel plus thermal
    relaxation over each gate's duration, and a classical assignment error on each measurement. It
    puts no noise on ``delay`` or on idle qubits.

Decoder priors
    :func:`calibrated_noise` turns the same calibration into a :class:`dqiqm.surface.NoiseModel` on a
    circuit's Stim qubits, for the noisy copy of a circuit that PyMatching decodes with. With
    ``idle_ns=0`` it matches the stand-in (no idle noise); for hardware data, pass the idle time per
    round (the readout window, plus any delay) so T1 and T2 between gates enter the priors.
    :func:`hardware_noise` is the hardware prior in one call: the idle is the compiled readout
    window (:func:`readout_layer_ns`, from pulla-check's compile file) plus the circuit's delay, as
    a symmetric ``DEPOLARIZE1`` (the biased twirl is a switch, mapped through each data qubit's
    hardware frame, :func:`readout_frames`), and the final layer's record flips are
    ``measure_fidelity``'s.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import stim

from dqiqm.surface import NoiseModel

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
#: Per device, ``<device>/pulse/`` holds pulla-check's local compiles (``scripts/pulla_check.py``):
#: ``compile.<calibration set>.json``.
CALIBRATION_DIR = REPO_ROOT / "data" / "calibration"


def compile_dir(device: str) -> Path:
    """The folder of ``device``'s pulla-check compile files."""
    return CALIBRATION_DIR / device / "pulse"


#: Garnet's compile folder (the only one before emerald-compile); :func:`compile_dir` gives any device's.
COMPILE_DIR = compile_dir("garnet")
#: The live Garnet snapshot of 2026-10-24 19:13 UTC (calibration set a18a947f), copied byte for byte
#: from ``data/calibration/garnet/a18a947f-0408-4628-81ea-1715637e434f.20261024T191313Z.json``
#: (the resonance-check task), so the InfoRecord holds the calibration it was built from.
DEFAULT_SNAPSHOT = (
    REPO_ROOT / "InfoRecord" / "calibration" / "garnet" / "20261024T191313Z-a18a947f-0408-4628-81ea-1715637e434f.json"
)


def qubit_sort_key(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(\D*)(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def pair_key(a: str, b: str) -> tuple[str, str]:
    """A coupler as a sorted pair of qubit names (QB2 before QB10)."""
    return tuple(sorted((a, b), key=qubit_sort_key))  # type: ignore[return-value]


@dataclass(frozen=True)
class Calibration:
    """One calibration's figures. Times in ns, errors as probabilities; ``None`` where not served.

    ``qubits[q]`` has ``t1_ns``, ``t2_ns`` (echo), ``t2_ramsey_ns``, ``p1_given_0``, ``p0_given_1``,
    ``error_1q`` (RB infidelity), ``readout_leakage``, ``prx_ns`` and ``measure_ns``, and
    ``p1_given_0_terminal``, ``p0_given_1_terminal`` (the unheralded ``measure_fidelity``).
    ``couplers[(a, b)]`` (a sorted pair) has ``cz_error`` (IRB infidelity), ``cz_error_clifford_rb``
    and ``cz_ns``.
    """

    device: str
    calibration_set_id: str
    source: str
    fetched_utc: str | None
    qubits: Mapping[str, Mapping[str, float | None]]
    couplers: Mapping[tuple[str, str], Mapping[str, float | None]]
    simulated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    reset_wait_ns: float | None = None  # the device's passive-reset wait (``gates.reset_wait``), if served

    def qubit(self, q: str, key: str, default: float | None = None) -> float | None:
        v = self.qubits.get(q, {}).get(key)
        return default if v is None else v

    def cz_error(self, a: str, b: str) -> float:
        return float(self.couplers[pair_key(a, b)]["cz_error"])

    def cz_ns(self, a: str, b: str) -> float:
        v = self.couplers[pair_key(a, b)].get("cz_ns")
        return float(v) if v is not None else self.median("cz_ns", couplers=True)

    def readout_error(self, q: str) -> float:
        """The mean assignment error, (P(1|0) + P(0|1)) / 2."""
        return 0.5 * (float(self.qubits[q]["p1_given_0"]) + float(self.qubits[q]["p0_given_1"]))

    def terminal_readout_error(self, q: str) -> float | None:
        """The terminal readout's (``measure_fidelity``) mean assignment error, or None if not served."""
        a, b = self.qubits[q].get("p1_given_0_terminal"), self.qubits[q].get("p0_given_1_terminal")
        return None if a is None or b is None else 0.5 * (float(a) + float(b))

    def median(self, key: str, *, couplers: bool = False) -> float:
        rows = self.couplers.values() if couplers else self.qubits.values()
        return float(statistics.median(r[key] for r in rows if r.get(key) is not None))

    def to_json(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "calibration_set_id": self.calibration_set_id,
            "source": self.source,
            "reset_wait_ns": self.reset_wait_ns,
            "fetched_utc": self.fetched_utc,
            "simulated": self.simulated,
            "notes": list(self.notes),
            "units": "times in ns, errors as probabilities",
            "qubits": {q: dict(v) for q, v in sorted(self.qubits.items(), key=lambda kv: qubit_sort_key(kv[0]))},
            "couplers": {f"{a}-{b}": dict(v) for (a, b), v in sorted(self.couplers.items())},
        }


# ---------------------------------------------------------------------------
# Reading a calibration.
# ---------------------------------------------------------------------------


def load_snapshot(path: str | Path = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _cz_loci(arch: Mapping[str, Any]) -> dict[frozenset, tuple[str, str]]:
    """Each coupler -> the ordered locus the architecture lists it with."""
    out: dict[frozenset, tuple[str, str]] = {}
    for info in arch["gates"].get("cz", {}).get("implementations", {}).values():
        for a, b in info["loci"]:
            out.setdefault(frozenset((a, b)), (a, b))
    return out


def _impl(gate: Mapping[str, Any], locus: tuple[str, str]) -> str:
    """The implementation a locus runs; an override may list the pair in either order."""
    overrides = {
        frozenset(k.split(",") if isinstance(k, str) else k): v
        for k, v in gate.get("override_default_implementation", {}).items()
    }
    return overrides.get(frozenset(locus), gate["default_implementation"])


def calibration_from_snapshot(snap: Mapping[str, Any], source: str = "") -> Calibration:
    """Figures from a ``dqiqm.calibration-snapshot/1`` snapshot (resonance-check's ``figures()``, in ns)."""
    obs: dict[str, tuple[Any, str]] = {}
    for part in ("calibration_set", "quality_metric_set"):  # the metric set wins a clash
        for o in (snap.get(part) or {}).get("observations", []):
            obs[o["dut_field"]] = (o["value"], o.get("unit") or "")

    def get(*names: str) -> float | None:
        for f in names:
            for key in (f, f + ":par=d2"):
                if key in obs and isinstance(obs[key][0], (int, float)):
                    return float(obs[key][0])
        return None

    def ns(name: str) -> float | None:
        if name not in obs or not isinstance(obs[name][0], (int, float)):
            return None
        value, unit = obs[name]
        scale = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}.get(unit, 1e9 if value < 1 else 1.0)
        return float(value) * scale

    arch = snap["architecture"]
    gates = arch["gates"]
    m_impl = gates["measure"]["default_implementation"]
    mf_impl = gates.get("measure_fidelity", {}).get("default_implementation")
    p_impl = gates["prx"]["default_implementation"]
    qubits = {}
    for q in arch["qubits"]:
        f_1q = get(f"metrics.rb.prx.{p_impl}.{q}.fidelity")
        qubits[q] = {
            "t1_ns": ns(f"characterization.model.{q}.t1_time"),
            "t2_ns": ns(f"characterization.model.{q}.t2_echo_time"),
            "t2_ramsey_ns": ns(f"characterization.model.{q}.t2_time"),
            "p1_given_0": get(f"metrics.ssro.measure.{m_impl}.{q}.error_0_to_1"),
            "p0_given_1": get(f"metrics.ssro.measure.{m_impl}.{q}.error_1_to_0"),
            "error_1q": None if f_1q is None else 1 - f_1q,
            "readout_leakage": get(f"metrics.rrc.measure.{m_impl}.{q}.leakage"),
            "prx_ns": ns(f"gates.prx.{p_impl}.{q}.duration"),
            "measure_ns": ns(f"gates.measure.{m_impl}.{q}.duration"),
            "p1_given_0_terminal": get(f"metrics.ssro.measure_fidelity.{mf_impl}.{q}.error_0_to_1") if mf_impl else None,
            "p0_given_1_terminal": get(f"metrics.ssro.measure_fidelity.{mf_impl}.{q}.error_1_to_0") if mf_impl else None,
        }
    couplers = {}
    for (a, b) in _cz_loci(arch).values():
        impl = _impl(gates["cz"], (a, b))
        f_irb = get(f"metrics.irb.cz.{impl}.{a}__{b}.fidelity", f"metrics.irb.cz.{impl}.{b}__{a}.fidelity")
        f_clf = get(f"metrics.rb.clifford.uz_cz.{a}__{b}.fidelity", f"metrics.rb.clifford.uz_cz.{b}__{a}.fidelity")
        couplers[pair_key(a, b)] = {
            "cz_error": None if f_irb is None else 1 - f_irb,
            "cz_error_clifford_rb": None if f_clf is None else 1 - f_clf,
            "cz_ns": ns(f"gates.cz.{impl}.{a}__{b}.duration") or ns(f"gates.cz.{impl}.{b}__{a}.duration"),
        }
    for key, c in couplers.items():
        if c["cz_error"] is None:  # no IRB figure: fall back on Clifford RB
            c["cz_error"] = c["cz_error_clifford_rb"]
    waits = [ns(f"gates.reset_wait.reset_wait.{q}.duration") for q in arch["qubits"]]
    waits = [w for w in waits if w is not None]
    return Calibration(
        device=snap.get("device", "garnet"),
        calibration_set_id=str(snap["calibration_set_id"]),
        source=source or "live snapshot",
        fetched_utc=snap.get("fetched_utc_end"),
        qubits=qubits,
        couplers=couplers,
        reset_wait_ns=max(waits) if waits else None,
    )


def calibration_from_error_profile(ep: Any, *, device: str = "IQMFakeGarnet") -> Calibration:
    """Figures from a fake backend's ``IQMErrorProfile`` (depolarizing parameters -> benchmarked infidelities)."""
    prx = ep.single_qubit_gate_depolarizing_error_parameters["prx"]
    qubits = {
        q: {
            "t1_ns": float(ep.t1s[q]),
            "t2_ns": float(ep.t2s[q]),
            "t2_ramsey_ns": float(ep.t2s[q]),
            "p1_given_0": float(ep.readout_errors[q]["0"]),
            "p0_given_1": float(ep.readout_errors[q]["1"]),
            "error_1q": float(prx[q]) / 2,  # an Aer depolarizing parameter p on D = 2 levels: infidelity p/2
            "readout_leakage": None,
            "prx_ns": float(ep.single_qubit_gate_durations["prx"]),
            "measure_ns": None,
            "p1_given_0_terminal": None,
            "p0_given_1_terminal": None,
        }
        for q in ep.t1s
    }
    couplers = {
        pair_key(a, b): {
            "cz_error": 3 * float(p) / 4,  # D = 4: infidelity 3p/4
            "cz_error_clifford_rb": None,
            "cz_ns": float(ep.two_qubit_gate_durations["cz"]),
        }
        for (a, b), p in ep.two_qubit_gate_depolarizing_error_parameters["cz"].items()
    }
    return Calibration(
        device=device,
        calibration_set_id="00000000-0000-0000-0000-000000000000",
        source=f"error profile {getattr(ep, 'name', '')}".strip(),
        fetched_utc=None,
        qubits=qubits,
        couplers=couplers,
        simulated=True,
    )


def calibration_from_fidelities(fidelities: Mapping[str, Any]) -> Calibration:
    """Not supported: ``hw.calibration_fidelities`` lacks the default ``measure``'s assignment errors.

    Its ``readout`` is the heralded ``measure_fidelity`` figure (optimistic), and it serves no
    durations, so a calibration built from it would mislead every prior. Take a full snapshot
    (``scripts/resonance_check.py`` writes one) and use :func:`calibration_from_snapshot`.
    """
    raise NotImplementedError(
        "calibration_fidelities() carries the heralded readout and no durations; "
        "load a resonance-check snapshot with calibration_from_snapshot() instead"
    )


def live_calibration(path: str | Path = DEFAULT_SNAPSHOT) -> Calibration:
    path = Path(path)
    try:
        rel = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(path)
    return calibration_from_snapshot(load_snapshot(path), source=rel)


# ---------------------------------------------------------------------------
# The stand-in device: IQMFakeGarnet with a calibration's noise.
# ---------------------------------------------------------------------------


def t2_ns(c: Mapping[str, float | None], t2: str = "ramsey") -> float:
    """A qubit's T2 in ns: Ramsey (the default: no dynamical decoupling here) or echo, capped at 2·T1."""
    if t2 not in ("ramsey", "echo"):
        raise ValueError(f"t2 must be 'ramsey' or 'echo', not {t2!r}")
    first, second = ("t2_ramsey_ns", "t2_ns") if t2 == "ramsey" else ("t2_ns", "t2_ramsey_ns")
    value = c.get(first) or c.get(second)
    return float(min(value, 2 * c["t1_ns"]))


def error_profile(
    cal: Calibration, *, t2: str = "ramsey", ordered_loci: Mapping[frozenset, tuple[str, str]] | None = None
) -> Any:
    """An ``IQMErrorProfile`` from a calibration (resonance-check's ``fake_comparison``).

    A benchmarked infidelity r becomes the depolarizing parameter r·D/(D − 1) (2r for one qubit,
    4r/3 for a CZ). T2 is :func:`t2_ns` (Ramsey by default), capped at 2·T1 so Aer's relaxation
    channel is physical.
    The fake also adds thermal relaxation over each gate's duration, so the gate noise is
    slightly above the benchmarked figure.
    """
    from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile

    q = cal.qubits
    loci = ordered_loci or _fake_loci()
    return IQMErrorProfile(
        t1s={k: float(v["t1_ns"]) for k, v in q.items()},
        t2s={k: t2_ns(v, t2) for k, v in q.items()},
        single_qubit_gate_depolarizing_error_parameters={"prx": {k: 2 * float(v["error_1q"]) for k, v in q.items()}},
        two_qubit_gate_depolarizing_error_parameters={
            "cz": {loci[frozenset(k)]: 4 / 3 * float(v["cz_error"]) for k, v in cal.couplers.items()}
        },
        single_qubit_gate_durations={"prx": cal.median("prx_ns")},
        two_qubit_gate_durations={"cz": cal.median("cz_ns", couplers=True)},
        readout_errors={k: {"0": float(v["p1_given_0"]), "1": float(v["p0_given_1"])} for k, v in q.items()},
        name=f"{cal.device} {cal.calibration_set_id}",
    )


def _fake_loci() -> dict[frozenset, tuple[str, str]]:
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    loci: dict[frozenset, tuple[str, str]] = {}
    for info in IQMFakeGarnet().architecture.gates["cz"].implementations.values():
        for a, b in info.loci:
            loci[frozenset((a, b))] = (a, b)
    return loci


def fake_backend(cal: Calibration | None = None, *, t2: str = "ramsey") -> Any:
    """``IQMFakeGarnet()``, with ``cal``'s noise when given (the live calibration by default elsewhere)."""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    fake = IQMFakeGarnet()
    return fake if cal is None else fake.copy_with_error_profile(error_profile(cal, t2=t2))


# ---------------------------------------------------------------------------
# Durations.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Durations:
    """Gate and readout durations in ns, per qubit and per coupler where the calibration has them."""

    prx_ns: Mapping[str, float]
    cz_ns: Mapping[tuple[str, str], float]
    measure_ns: Mapping[str, float]
    source: str

    @classmethod
    def from_calibration(cls, cal: Calibration, *, measure_ns_default: float | None = None) -> Durations:
        prx_default = cal.median("prx_ns")
        meas = [v["measure_ns"] for v in cal.qubits.values() if v.get("measure_ns") is not None]
        if meas:
            m_default = float(statistics.median(meas))
        elif measure_ns_default is not None:
            m_default = float(measure_ns_default)
        else:
            raise ValueError(f"{cal.source}: no measure durations; pass measure_ns_default")
        return cls(
            prx_ns={q: float(v.get("prx_ns") or prx_default) for q, v in cal.qubits.items()},
            cz_ns={k: float(v.get("cz_ns") or cal.median("cz_ns", couplers=True)) for k, v in cal.couplers.items()},
            measure_ns={q: float(v.get("measure_ns") or m_default) for q, v in cal.qubits.items()},
            source=cal.source,
        )

    def of(self, op: str, qubits: tuple[str, ...]) -> float:
        if op == "r":
            return self.prx_ns[qubits[0]]
        if op == "cz":
            return self.cz_ns[pair_key(*qubits)]
        if op == "measure":
            return self.measure_ns[qubits[0]]
        raise ValueError(f"no duration for {op}")


# ---------------------------------------------------------------------------
# Compiled readout windows (pulla-check).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutWindows:
    """Per qubit, ``measure_window_ns`` and ``measure_fidelity_window_ns``, from a pulla-check compile file.

    A window is what IQM's compiler gives one readout: acquisition delay + integration + dead time.
    A readout layer lasts as long as its slowest qubit's (each probe line's trigger does).
    """

    calibration_set_id: str | None
    source: str
    qubits: Mapping[str, Mapping[str, float]]
    device: str | None = None  # the device the compile file names

    def of(self, q: str, key: str) -> float | None:
        v = self.qubits.get(q, {}).get(key)
        return None if v is None else float(v)


def load_readout_windows(path: str | Path) -> ReadoutWindows:
    path = Path(path)
    d = json.loads(path.read_text())
    try:
        source = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(path)
    return ReadoutWindows(d.get("calibration_set_id"), source, {q: dict(v) for q, v in d["readout_windows"]["qubits"].items()},
                          d.get("device"))


def compiled_readout_windows(
    calibration_set_id: str, directory: str | Path | None = None, *, device: str | None = None
) -> ReadoutWindows | None:
    """The compile file's windows for one calibration set, or None if pulla-check compiled none for it.

    The file is ``compile.<calibration_set_id>.json`` in ``directory``, else in ``device``'s folder
    (:func:`compile_dir`), else in whichever device's folder holds it (a calibration set id is one
    device's). With ``device``, a file that names another device is refused.
    """
    name = f"compile.{calibration_set_id}.json"
    if directory is not None:
        found = [Path(directory) / name]
    elif device is not None:
        found = [compile_dir(device) / name]
    else:
        found = sorted(CALIBRATION_DIR.glob(f"*/pulse/{name}"))
        if len(found) > 1:
            raise ValueError(f"{name} is in more than one device's folder: {', '.join(str(p) for p in found)}")
    found = [p for p in found if p.exists()]
    if not found:
        return None
    path = found[0]
    w = load_readout_windows(path)
    if w.calibration_set_id is None:
        raise ValueError(f"{path} names no calibration set, so it can't be checked against {calibration_set_id}")
    if w.calibration_set_id != calibration_set_id:
        raise ValueError(f"{path} is for calibration set {w.calibration_set_id}, not {calibration_set_id}")
    if device is not None and w.device not in (None, device):
        raise ValueError(f"{path} is a {w.device} compile, not {device}'s")
    return w


def readout_layer_ns(
    cal: Calibration,
    qubits: Iterable[str],
    windows: ReadoutWindows | None,
    *,
    terminal: bool = False,
) -> tuple[float, str]:
    """One readout layer's length in ns, and where it came from.

    The slowest compiled window among the layer's measured ``qubits``: ``measure_window_ns`` for a
    mid-circuit layer, ``measure_fidelity_window_ns`` for the terminal one (IQM compiles a qubit's
    last measurement to ``measure_fidelity``). Without compile data for every qubit, the old rule,
    the slowest ``measure`` duration, which is the probe alone (about half the window), logged.

    Only mid-circuit layers become idle priors: in the surface code the terminal layer reads every
    qubit, so nothing waits through it (``surface.memory_circuit`` puts no idle after the last
    round). The terminal window is for schedules and durations.
    """
    qubits = list(qubits)
    key = "measure_fidelity_window_ns" if terminal else "measure_window_ns"
    missing = [q for q in qubits if windows is None or windows.of(q, key) is None]
    if windows is not None and not missing:
        return max(windows.of(q, key) for q in qubits), f"compiled {key} ({windows.source})"
    log.warning("no compiled %s for %s (%s): the readout layer falls back on the slowest measure duration",
                key, ", ".join(missing), cal.calibration_set_id)
    return max(float(cal.qubits[q]["measure_ns"]) for q in qubits), "measure_ns (no compile data: probe only)"


# ---------------------------------------------------------------------------
# Decoder priors: a calibration as Stim noise on a circuit's qubits.
# ---------------------------------------------------------------------------


def relaxation_pauli(t_ns: float, t1_ns: float, t2_ns: float) -> tuple[float, float]:
    """The Pauli twirl of T1/T2 decay over ``t_ns``: (p_x + p_y, p_z)."""
    if t_ns <= 0:
        return 0.0, 0.0
    a = 1 - math.exp(-t_ns / t1_ns)
    b = 1 - math.exp(-t_ns / min(t2_ns, 2 * t1_ns))
    pxy = a / 2
    pz = max(b / 2 - a / 4, 0.0)
    return pxy, pz


# ---------------------------------------------------------------------------
# The data qubits' frame in the readout window, read from the circuit that runs.
# ---------------------------------------------------------------------------

#: A frame: each physical Pauli (in the readout window of the native circuit) -> the Stim Pauli it acts as.
Frame = Mapping[str, str]
#: The six permutations of X, Y, Z, the identity first, then the swaps: a single-qubit Clifford permutes the Paulis by one.
_PERMUTATIONS: tuple[dict[str, str], ...] = tuple(
    dict(zip("XYZ", p)) for p in sorted(itertools.permutations("XYZ"), key=lambda p: -sum(a == b for a, b in zip("XYZ", p))))
#: Per circuit family: does its builder put an idle after the last round, and does it mark its final readout layer?
_FAMILY_END = {"surface_memory": (False, True), "repetition": (True, False), "stability": (True, False)}


def _prx(theta: float, phi: float) -> stim.Circuit:
    """IQM's ``r(theta, phi)`` (prx) on qubit 0 as Stim gates, up to a global phase. It must be a Clifford."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    u = np.array([[c, -1j * np.exp(-1j * phi) * s], [-1j * np.exp(1j * phi) * s, c]])
    return stim.Tableau.from_unitary_matrix(u, endian="little").to_circuit()


def _measured(c: stim.Circuit) -> list[list[int]]:
    """Each DETECTOR's, then each observable's, absolute measurement indices in ``c`` (observables last, by index)."""
    dets, obs, n = [], {}, 0
    for inst in c.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            n += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            dets.append([n + t.value for t in inst.targets_copy()])
        elif inst.name == "OBSERVABLE_INCLUDE":
            obs.setdefault(int(inst.gate_args_copy()[0]), []).extend(n + t.value for t in inst.targets_copy())
    return dets + [obs[k] for k in sorted(obs)]


def _signature(c: stim.Circuit) -> tuple[frozenset[int], frozenset[int]]:
    """The detectors and observables that ``c``'s one error mechanism flips (empty if it flips none)."""
    errors = [e for e in c.detector_error_model().flattened() if e.type == "error"]
    if not errors:
        return frozenset(), frozenset()
    (e,) = errors
    ts = e.targets_copy()
    return (frozenset(t.val for t in ts if t.is_relative_detector_id()),
            frozenset(t.val for t in ts if t.is_logical_observable_id()))


def _native_pieces(spec: Any) -> list[stim.Circuit]:
    """What the server receives for ``spec`` as a Stim circuit with the model's detectors, cut at each mid-circuit window.

    The instructions are IQM's serializer's (``circuits.sent``), on device qubits mapped back to Stim
    qubits through the placement: ``prx`` becomes its Clifford, ``cz`` a CZ, ``measure`` an M (key
    ``c_<n>_0_<j>`` is the model's measurement j); ``barrier`` and ``delay`` are no-ops. The cuts fall
    right after each mid-circuit readout layer (a maximal run of ``measure``). Refuses unless every
    such layer is fenced by barriers over the data (else the compiler may schedule a data gate inside
    the window, and the frame is not the instruction order's), and unless, with a delay, every data
    qubit's delay comes before its next gate (so the delay shares the window's frame). The joined
    pieces must be the model circuit: every detector and observable deterministic, with the model's
    noiseless values.
    """
    from dqiqm.experiments.circuits import sent  # local: qiskit loads only when a frame is wanted

    to_stim = {name: s for s, name in spec.physical.items()}
    data_phys = {name for name, s in to_stim.items() if spec.roles.get(s) == "data"}
    insts = list(sent(spec.hardware))
    names = [i.name for i in insts]
    phys = [set(i.locus) for i in insts]
    layers, i = [], 0
    while i < len(insts):
        if names[i] == "measure":
            j = i
            while j < len(insts) and names[j] == "measure":
                j += 1
            layers.append((i, j))
            i = j
        else:
            i += 1
    if len(layers) != spec.rounds:
        raise ValueError(f"{spec.name}: {len(layers)} readout layers in the native circuit, not one per round ({spec.rounds})")
    mid = layers[:-1]
    for k, (a, b) in enumerate(mid, 1):
        fenced = a > 0 and b < len(insts) and all(names[x] == "barrier" and data_phys <= phys[x] for x in (a - 1, b))
        if not fenced:
            raise ValueError(f"{spec.name}: readout layer {k} is not fenced by barriers over the data, so the data's frame in "
                             "its window is the compiler's schedule, not the circuit's order")
        if float(spec.params.get("delay_ns", 0.0)) > 0:
            pending, j = set(data_phys), b
            delayed: set[str] = set()
            while pending and j < len(insts):
                if names[j] == "delay":
                    delayed |= phys[j]
                elif names[j] != "barrier" and phys[j] & pending:
                    if phys[j] & pending - delayed:
                        raise ValueError(f"{spec.name}: a data qubit has a gate between readout layer {k} and its delay")
                    pending -= phys[j]
                j += 1
    cuts = {b for _, b in mid}
    pieces, cur, order = [], stim.Circuit(), []
    for x, inst in enumerate(insts):
        if x in cuts:
            pieces.append(cur)
            cur = stim.Circuit()
        qs = [to_stim[q] for q in inst.locus]
        if names[x] == "prx":
            for g in _prx(float(inst.args["angle"]), float(inst.args["phase"])):
                cur.append(g.name, [qs[t.value] for t in g.targets_copy()], g.gate_args_copy())
        elif names[x] == "cz":
            cur.append("CZ", qs)
        elif names[x] == "measure":
            cur.append("M", qs)
            order.append(int(str(inst.args["key"]).rsplit("_", 1)[1]))
        elif names[x] not in ("barrier", "delay"):
            raise ValueError(f"{spec.name}: {names[x]!r} is not a native op this translation knows")
    model = spec.stim
    if sorted(order) != list(range(model.num_measurements)):
        raise ValueError(f"{spec.name}: the native clbits are not the model's {model.num_measurements} measurements, once each")
    pos, n = {c: p for p, c in enumerate(order)}, len(order)
    groups = _measured(model)
    for g in groups[: model.num_detectors]:
        cur.append("DETECTOR", [stim.target_rec(pos[m] - n) for m in g])
    for k, g in enumerate(groups[model.num_detectors:]):
        cur.append("OBSERVABLE_INCLUDE", [stim.target_rec(pos[m] - n) for m in g], k)
    pieces.append(cur)
    native = sum(pieces, stim.Circuit())
    native.detector_error_model()  # raises on a non-deterministic detector or observable
    ref_n, ref_m = native.reference_sample(), model.reference_sample()
    for g in groups:
        if sum(ref_n[pos[m]] for m in g) % 2 != sum(ref_m[m] for m in g) % 2:
            raise ValueError(f"{spec.name}: the native circuit's noiseless detector or observable values differ from the model's")
    return pieces


def _model_pieces(spec: Any) -> list[stim.Circuit]:
    """``spec.stim`` cut where the model puts the idle: right after round k's readout, for k = 1 ... rounds − 1."""
    want = [sum(1 for r in spec.meas_round if r <= k) for k in range(1, spec.rounds)]
    pieces, cur, n = [], stim.Circuit(), 0
    for inst in spec.stim.flattened():
        cur.append(inst)
        if stim.gate_data(inst.name).produces_measurements:
            n += len(inst.targets_copy())
            if want and n >= want[0]:
                if n != want[0]:
                    raise ValueError(f"{spec.name}: one measurement instruction spans rounds")
                pieces.append(cur)
                cur, want = stim.Circuit(), want[1:]
    return pieces + [cur]


def readout_frames(spec: Any) -> dict[int, dict[str, str]]:
    """Per data qubit (Stim index), the frame it idles in through the mid-circuit readout windows on hardware.

    Each value maps a physical Pauli in the window to the Stim Pauli at the model's idle point it
    acts as. It is derived, not assumed: a physical X, Y or Z is injected in the native circuit
    (:func:`_native_pieces`) right after readout layer k, its detector-and-observable signature is
    compared with Stim's X, Y and Z right after round k's readout in ``spec.stim``, for every window,
    and the permutation consistent with all of them is kept (the identity when it is). The compiler
    holds some data qubits in a rotated frame there (the next round's leading prx is moved before
    the readout): on those, the dominant dephasing is a Stim X. Refuses if a physical Pauli acts as
    no single Stim Pauli, or the frame changes between windows (a per-qubit channel can't say so).
    """
    data = sorted(q for q, r in spec.roles.items() if r == "data")
    if spec.rounds < 2:
        return {q: dict(_PERMUTATIONS[0]) for q in data}
    native, model = _native_pieces(spec), _model_pieces(spec)

    def joined(pieces, k, q, pauli):
        return sum(pieces[:k], stim.Circuit()) + stim.Circuit(f"{pauli}_ERROR(0.1) {q}") + sum(pieces[k:], stim.Circuit())

    frames = {}
    for q in data:
        allowed = set(range(len(_PERMUTATIONS)))
        for k in range(1, spec.rounds):
            stim_sig = {s: _signature(joined(model, k, q, s)) for s in "XYZ"}
            for p in "XYZ":
                sig = _signature(joined(native, k, q, p))
                acts_as = {s for s in "XYZ" if stim_sig[s] == sig}
                if not acts_as:
                    raise ValueError(f"{spec.name}: a physical {p} on data qubit {q} in readout window {k} acts as no single Stim Pauli")
                allowed &= {i for i, perm in enumerate(_PERMUTATIONS) if perm[p] in acts_as}
        if not allowed:
            raise ValueError(f"{spec.name}: data qubit {q}'s frame changes between readout windows")
        frames[q] = dict(_PERMUTATIONS[min(allowed)])
    return frames


def calibrated_noise(
    cal: Calibration,
    physical: Mapping[int, str],
    roles: Mapping[int, str],
    *,
    pairs: Iterable[tuple[int, int]] = (),
    idle_ns: float = 0.0,
    x_ancillas: Iterable[int] = (),
    idle_channel: str = "depolarize",
    t2: str = "ramsey",
    final_readout: bool = False,
    frames: Mapping[int, Frame] | None = None,
) -> NoiseModel:
    """A ``NoiseModel`` on Stim qubits ``physical`` (Stim qubit -> device qubit), from ``cal``.

    Each rate is a Stim Pauli probability:

    * a CX on (a, b): ``DEPOLARIZE2`` at 5/4 of the CZ's IRB infidelity (D = 4), plus each
      qubit's T1/T2 twirl over the CZ's duration;
    * an ancilla H: ``DEPOLARIZE1`` at 3/2 of the RB infidelity (D = 2), plus the twirl over a prx;
    * a data qubit, at the start of each round: its two prx per round (the H around each CX target
      in the native circuit, which Stim's CX does not carry);
    * a Z-type ancilla, before its measurement: the X/Y part of its two prx per round;
    * every readout: a record flip at the mean assignment error (P(1|0) + P(0|1)) / 2; with
      ``final_readout``, the final layer's (the last round's ancillas and the data) at the
      terminal readout's, ``measure_fidelity``, where the calibration serves it (else ``measure``'s,
      logged);
    * the readout window (``idle_ns``, when given), once per mid-circuit window, as
      ``idle_during_measurement``: on a data qubit the whole twirl (X/Y plus Z), on an ancilla only
      its X/Y part, which is what flips a live ancilla. ``idle_channel="pauli"`` gives the data the
      twirl as it is, ``PAULI_CHANNEL_1(p_xy/2, p_xy/2, p_z)`` (``idle_pauli_*``), instead of a
      ``DEPOLARIZE1`` of the same total, which spreads the Z-dominated twirl evenly over X, Y and Z.
      The twirl is physical: ``frames`` (:func:`readout_frames`) maps it into Stim's frame per data
      qubit. Without it the channel is in Stim's frame, which is right for a Stim simulation and
      wrong on hardware, where the compiler holds some data qubits in a rotated frame through the
      window (:func:`hardware_noise` passes it). ``surface`` puts no idle after the last round,
      whose ancillas are read in the data's layer.

    ``roles`` maps each Stim qubit to ``"data"``, ``"ancilla"`` or ``"readout"``. ``x_ancillas`` are
    the ancillas that carry their own H in Stim. ``idle_ns=0`` matches the stand-in, which has no
    idle noise. T2 is the Ramsey T2 by default (:func:`t2_ns`), as in the stand-in: no circuit here
    has dynamical decoupling. ``t2="echo"`` takes the echo T2, everywhere T2 enters.
    """
    if idle_channel not in ("depolarize", "pauli"):
        raise ValueError(f"idle_channel must be 'depolarize' or 'pauli', not {idle_channel!r}")
    t2_of = lambda c: t2_ns(c, t2)  # noqa: E731
    no_terminal = []
    qubit_rates: dict[int, dict[str, float]] = {}
    for s, name in physical.items():
        c = cal.qubits[name]
        t1, t2q = float(c["t1_ns"]), t2_of(c)
        pxy_g, pz_g = relaxation_pauli(float(c["prx_ns"] or cal.median("prx_ns")), t1, t2q)
        p1 = 1.5 * float(c["error_1q"]) + pxy_g + pz_g
        pxy_i, pz_i = relaxation_pauli(idle_ns, t1, t2q)
        rates = {
            "after_clifford_depolarization": min(p1, 0.75),
            "record_flip_probability": cal.readout_error(name),
        }
        if final_readout:
            p_final = cal.terminal_readout_error(name)
            if p_final is None:
                no_terminal.append(name)
            else:
                rates["final_record_flip_probability"] = p_final
        role = roles.get(s, "data")
        if role == "data":
            rates["before_round_data_depolarization"] = min(2 * p1, 0.75)
        elif role == "ancilla" and s not in set(x_ancillas):
            rates["before_measure_flip_probability"] = min(2 * (2 / 3) * p1, 0.5)
        if idle_ns > 0:
            if role == "data" and idle_channel == "pauli":
                twirl = {"X": pxy_i / 2, "Y": pxy_i / 2, "Z": pz_i}  # physical
                frame = (frames or {}).get(s, _PERMUTATIONS[0])
                rates |= {f"idle_pauli_{frame[p].lower()}": v for p, v in twirl.items()}
            else:
                rates["idle_during_measurement"] = min(pxy_i + pz_i, 0.75) if role == "data" else min(pxy_i, 0.5)
        qubit_rates[s] = rates
    if no_terminal:
        log.warning("%s: no measure_fidelity readout errors for %s; their final readout keeps measure's",
                    cal.calibration_set_id, ", ".join(no_terminal))
    pair_rates = {}
    for a, b in pairs:
        na, nb = physical[a], physical[b]
        t_cz = cal.cz_ns(na, nb)
        extra = 0.0
        for n in (na, nb):
            c = cal.qubits[n]
            pxy, pz = relaxation_pauli(t_cz, float(c["t1_ns"]), t2_of(c))
            extra += pxy + pz
        pair_rates[(a, b)] = min(1.25 * cal.cz_error(na, nb) + extra, 15 / 16)
    return NoiseModel(qubit_rates=qubit_rates, pair_rates=pair_rates)


def hardware_noise(
    cal: Calibration,
    spec: Any,
    *,
    windows: ReadoutWindows | None | str = "auto",
    t2: str = "ramsey",
    idle_channel: str = "depolarize",
    delay_ns: float | None = None,
) -> tuple[NoiseModel, dict[str, Any]]:
    """The hardware decoder prior for one circuit (a ``CircuitSpec``), and a record of how it was built.

    :func:`calibrated_noise` with the ``idle_channel`` and ``t2`` given, and ``idle_ns`` = the
    mid-circuit readout layer (:func:`readout_layer_ns` over the circuit's ancillas, which every
    mid-circuit layer measures) + ``delay_ns`` (the spec's ``delay_ns`` by default: E4's delay
    follows every readout but the last, as the idle does). ``windows="auto"`` reads pulla-check's
    compile file for ``cal``'s device and calibration set (:func:`compiled_readout_windows`, in
    ``data/calibration/<cal.device>/pulse/``); ``None`` forces the ``measure_ns`` fallback.

    The idle is a symmetric ``DEPOLARIZE1`` by default, which is the same in every frame.
    ``idle_channel="pauli"`` is the biased twirl, mapped per data qubit through the frame it idles
    in on hardware (:func:`readout_frames`, derived from ``spec.hardware``); on the pilot it fits the
    detection statistics best and decodes worst (``docs/garnet-design.md``), so it is a switch,
    not the default.

    What the circuit's builder does at the end follows its family (``spec.params["family"]``): a
    surface memory (E1, E4) has no idle after the last round and marks its final readout layer,
    which gets ``measure_fidelity``'s errors; the repetition (E2) and stability (E5) builders still
    idle after their last round and don't mark it, so their final readout keeps ``measure``'s. The
    info dict records which. Other families have no rounds to idle in, and are refused.
    """
    family = spec.params.get("family")
    if family not in _FAMILY_END:
        raise ValueError(f"hardware_noise covers the {', '.join(_FAMILY_END)} families, not {family!r}")
    idle_last, marks_final = _FAMILY_END[family]
    if isinstance(windows, str):
        if windows != "auto":
            raise ValueError(f"windows must be 'auto', None or a mapping, not {windows!r}")
        windows = compiled_readout_windows(cal.calibration_set_id, device=cal.device)
    ancillas = [spec.physical[q] for q, r in spec.roles.items() if r == "ancilla"]
    window, source = readout_layer_ns(cal, ancillas, windows)
    delay = float(spec.params.get("delay_ns", 0.0) if delay_ns is None else delay_ns)
    frames = readout_frames(spec) if idle_channel == "pauli" else None
    nm = calibrated_noise(cal, spec.physical, spec.roles, pairs=spec.cx_pairs, idle_ns=window + delay,
                          x_ancillas=spec.x_ancillas, idle_channel=idle_channel, t2=t2, final_readout=marks_final,
                          frames=frames)
    terminal = all(cal.terminal_readout_error(spec.physical[q]) is not None for q in spec.physical)
    if not marks_final:
        final = f"measure (the {family} builder does not mark its final readout layer)"
    else:
        final = "measure_fidelity (unheralded)" if terminal else "measure (no measure_fidelity errors served)"
    info = {
        "calibration_set_id": cal.calibration_set_id,
        "family": family,
        "readout_window_ns": window,
        "readout_window_source": source,
        "delay_ns": delay,
        "idle_ns": window + delay,
        "idle_channel": idle_channel,
        "idle_frame": None if frames is None else {
            str(q): "".join(f[p] for p in "XYZ") for q, f in frames.items() if f != _PERMUTATIONS[0]},
        "t2": t2,
        "final_readout": final,
        "idle_after_last_round": idle_last,
    }
    return nm, info


def pre_fix_noise(nm: NoiseModel) -> NoiseModel:
    """``nm`` with the idle after the last round put back, as every decoder made before garnet-timing had it.

    Re-deriving a record made before that fix (the Garnet and Emerald pilots') needs it, or the
    decoder is not the one the record was decoded with: ``scripts/garnet_pilot.py analyze`` should
    decode with ``pre_fix_noise(calibrated_noise(...))`` (it rebuilds the pilot's decoder byte for
    byte, 209 failures) and ``scripts/emerald_pilot.py analyze`` with
    ``pre_fix_noise(emerald.calibrated_noise(...))``.
    """
    return dataclasses.replace(nm, idle_after_last_round=True)
