"""First contact with IQM Resonance (task resonance-check): read-only, and no job of any kind.

    uv run python scripts/resonance_check.py             # live: snapshot garnet and emerald, then analyse
    uv run python scripts/resonance_check.py --offline   # analyse the committed snapshots: no network, no token

Live, it calls only dqiqm.hw's read-only helpers (``list_devices``, ``architecture``,
``calibration_set_id``, ``calibration_fidelities``, ``backend``) and, on the read-only backend's
client, the GET calls behind them: the quality-metric set, the calibration set, the static
architecture, health and about. Nothing here can submit: ``hw.backend`` refuses ``run`` and
``submit_*``, and this script never arms ``hw.submit``.

Each device's snapshot goes to ``data/calibration/<device>/<calibration_set_id>.json``, or to
``<calibration_set_id>.<UTC stamp>.json`` when that file exists: a snapshot is never overwritten.
It holds the architecture, the parsed fidelities, every quality-metric observation, and the
calibration-set observations that describe the qubits (``characterization.*``,
``gate_definitions.*`` and every ``*.duration``); the control-electronics settings are counted,
not kept. No token or header goes in it.

The analysis, run on the fresh snapshots (live) or the newest committed ones (``--offline``):

1. the key calibration numbers per device: median and worst T1, T2, readout and CZ errors;
2. zero-SWAP placements of the d = 3 rotated surface code (``dqiqm.surface.geometry(3)``: 17
   qubits, 24 ancilla-data pairs) in each device's calibrated CZ graph, by a networkx subgraph
   monomorphism, ranked by the calibration score below;
3. ``IQMFakeGarnet`` (iqm-client 35.0.3) against the live Garnet snapshot.

Placement score (lower is better), for one round plus the final data readout::

    S = sum over the 24 CZs used of -ln F_CZ
      + sum over the 17 qubits of [ -ln(1 - e_readout) + t_round / T1 ]
    t_round = 4 t_CZ + 10 t_prx + t_measure   (device medians; a CX is H.CZ.H)

F_CZ is the interleaved-RB CZ fidelity (Clifford-RB if it is missing), e_readout the
assignment error of the default ``measure``. A missing figure counts as the device's worst.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dqiqm import hw

REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "data" / "calibration"
DEVICES = ("garnet", "emerald")
SCHEMA = "dqiqm.calibration-snapshot/1"
COST_WORDS = re.compile(r"credit|cost|price|quota|balance|billing|budget", re.IGNORECASE)
# challenge/iqm_utils/QPURepetitionCode.ipynb, repetition_circuit_generator(): Garnet qubit indices
REPETITION_PATH = (17, 18, 19, 15, 16, 11, 6, 5, 10, 9, 8, 13, 12, 7, 2, 3, 0, 1, 4)


# ─── live: snapshots through dqiqm.hw ────────────────────────────────────────


def _redact(text: Any) -> str:
    """The text, with the token removed should it ever appear (it should not)."""
    text = str(text)
    token = os.environ.get("IQM_TOKEN")
    return text.replace(token, "<redacted>") if token else text


def _utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _call(log: list[dict], name: str, fn, *args):
    """Run one read-only call; log whether it worked, or its exact error. Returns the value or None."""
    try:
        value = fn(*args)
    except Exception as e:  # recorded verbatim, then the next call runs
        log.append({"call": name, "ok": False, "error": f"{type(e).__module__}.{type(e).__name__}: {_redact(e)}"})
        return None
    log.append({"call": name, "ok": True})
    return value


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json") if model is not None else None


def _versions() -> dict[str, str]:
    return {p: importlib.metadata.version(p) for p in ("iqm-client", "iqm-qubit-selector", "qiskit", "networkx")}


def fetch_device_list(log: list[dict]) -> list[dict]:
    """``hw.list_devices()``; if it fails, the same private getter through a connection ``hw`` made."""
    devices = _call(log, "hw.list_devices()", hw.list_devices)
    if devices is not None:
        return devices
    # IQMClient(SERVER_URL) without a device fails once the token sees more than one; the
    # getter itself is fine. A read-only backend for any device reaches it.
    b = _call(log, 'hw.backend("garnet")', hw.backend, "garnet")
    if b is None:
        return []
    qcs = _call(
        log,
        'hw.backend("garnet").client._iqm_server_client.get_quantum_computers()',
        b.client._iqm_server_client.get_quantum_computers,
    )
    return [_dump(q) for q in qcs or []]


def _keep_calibration_field(field: str) -> bool:
    return field.startswith(("characterization.", "gate_definitions.")) or field.endswith(".duration")


def fetch_snapshot(device: str, log_devices: list[dict]) -> dict:
    log: list[dict] = []
    start = _utc()
    arch = _call(log, f'hw.architecture("{device}")', hw.architecture, device)
    if arch is None:
        return {"device": device, "helpers": log, "fetched_utc_start": start}
    calset = arch["calibration_set_id"]
    default_now = _call(log, f'hw.calibration_set_id("{device}")', hw.calibration_set_id, device)
    fidelities = _call(
        log, f'hw.calibration_fidelities("{device}", calset)', hw.calibration_fidelities, device, calset
    )
    b = _call(log, f'hw.backend("{device}", calset)', hw.backend, device, calset)
    qms = cs = sqa = health = about = None
    if b is not None:
        pinned = b.architecture.calibration_set_id
        qms = _call(log, "client.get_quality_metric_set(calset)", b.client.get_quality_metric_set, pinned)
        cs = _call(log, "client.get_calibration_set(calset)", b.client.get_calibration_set, pinned)
        sqa = _call(log, "client.get_static_quantum_architecture()", b.client.get_static_quantum_architecture)
        health = _call(log, "client.get_health()", b.client.get_health)
        about = _call(log, "client.get_about()", b.client.get_about)

    def observation_set(model, keep=None) -> dict | None:
        if model is None:
            return None
        d = _dump(model)
        d.pop("observation_ids", None)  # the ids are in the observations themselves
        obs = d.pop("observations")
        kept = [o for o in obs if keep is None or keep(o["dut_field"])]
        d["observations_total"] = len(obs)
        d["observations_kept"] = len(kept)
        d["observations"] = kept
        return d

    calibration_set = observation_set(cs, _keep_calibration_field)
    if calibration_set is not None:
        calibration_set["kept_rule"] = "characterization.*, gate_definitions.* and *.duration; controllers.* and pulse settings dropped"
    return {
        "schema": SCHEMA,
        "device": device,
        "server": hw.SERVER_URL,
        "fetched_utc_start": start,
        "fetched_utc_end": _utc(),
        "dqiqm_commit": hw.git_head(),
        "versions": _versions(),
        "calibration_set_id": calset,
        "default_calibration_set_id_at_fetch": default_now,
        "quality_metric_set_id": None if qms is None else str(qms.observation_set_id),
        "helpers": log,
        "device_list_calls": log_devices,
        "architecture": arch,
        "static_architecture": _dump(sqa),
        "calibration_fidelities": fidelities,
        "quality_metric_set": observation_set(qms),
        "calibration_set": calibration_set,
        "health": health,
        "about": about,
    }


def save_snapshot(snap: dict) -> Path:
    """Write ``<device>/<calset>.json``, or ``<calset>.<stamp>.json`` if that exists. Never overwrites."""
    text = json.dumps(snap, indent=1, sort_keys=False) + "\n"
    token = os.environ.get("IQM_TOKEN")
    if token and token in text:
        raise RuntimeError("the token is in the snapshot; refusing to write it")
    if re.search(r"[Bb]earer ", text):
        raise RuntimeError("an authorization header is in the snapshot; refusing to write it")
    out = CALIBRATION / snap["device"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{snap['calibration_set_id']}.json"
    if path.exists():
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = out / f"{snap['calibration_set_id']}.{stamp}.json"
    with open(path, "x") as f:  # "x": fails rather than overwrite
        f.write(text)
    return path


def cost_fields(value: Any, prefix: str = "") -> list[str]:
    """Every key path, anywhere in ``value``, whose name looks like a cost, credit or quota."""
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if COST_WORDS.search(str(k)):
                found.append(f"{path} = {v!r}")
            found += cost_fields(v, path)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found += cost_fields(v, f"{prefix}[{i}]")
    return found


# ─── analysis: works on a snapshot dict, offline ─────────────────────────────


def _locus_key(locus) -> tuple[str, ...]:
    return tuple(locus.split(",")) if isinstance(locus, str) else tuple(locus)


def _impl(gate: dict, locus: tuple[str, ...]) -> str:
    """The implementation a locus runs. An override may list a CZ's pair in either order (Emerald does)."""
    overrides = {frozenset(_locus_key(k)): v for k, v in gate.get("override_default_implementation", {}).items()}
    return overrides.get(frozenset(locus), gate["default_implementation"])


def couplers(arch: dict) -> list[tuple[str, str]]:
    """The calibrated CZ couplers: every locus of any CZ implementation, as sorted pairs."""
    pairs = set()
    for info in arch["gates"].get("cz", {}).get("implementations", {}).values():
        for a, b in info["loci"]:
            pairs.add(tuple(sorted((a, b), key=hw._qubit_sort_key)))
    return sorted(pairs, key=lambda p: (hw._qubit_sort_key(p[0]), hw._qubit_sort_key(p[1])))


def _cz_loci(arch: dict) -> dict[frozenset, tuple[str, str]]:
    """Each coupler -> the locus (ordered pair) the architecture lists it with."""
    out = {}
    for info in arch["gates"].get("cz", {}).get("implementations", {}).values():
        for a, b in info["loci"]:
            out.setdefault(frozenset((a, b)), (a, b))
    return out


def figures(snap: dict) -> dict:
    """Per qubit and per coupler: the numbers that matter, from the raw observations, in SI-free units.

    Times in microseconds, durations in nanoseconds, errors as probabilities.
    """
    obs: dict[str, tuple[float, str]] = {}
    for part in ("calibration_set", "quality_metric_set"):  # metrics win a clash
        for o in (snap.get(part) or {}).get("observations", []):
            obs[o["dut_field"]] = (o["value"], o.get("unit") or "")

    def get(*fields: str) -> float | None:
        for f in fields:
            for key in (f, f + ":par=d2"):
                if key in obs and isinstance(obs[key][0], (int, float)):
                    return float(obs[key][0])
        return None

    def micro(field: str) -> float | None:
        if field not in obs or not isinstance(obs[field][0], (int, float)):
            return None
        value, unit = obs[field]
        return float(value) * {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}.get(unit, 1e6 if value < 1 else 1.0)

    arch = snap["architecture"]
    gates = arch["gates"]
    m_impl = gates["measure"]["default_implementation"]
    p_impl = gates["prx"]["default_implementation"]
    qubits = {}
    for q in arch["qubits"]:
        e_1q = get(f"metrics.rb.prx.{p_impl}.{q}.fidelity")
        dur_p, dur_m = get(f"gates.prx.{p_impl}.{q}.duration"), get(f"gates.measure.{m_impl}.{q}.duration")
        qubits[q] = {
            "t1_us": micro(f"characterization.model.{q}.t1_time"),
            "t2_ramsey_us": micro(f"characterization.model.{q}.t2_time"),
            "t2_echo_us": micro(f"characterization.model.{q}.t2_echo_time"),
            "p1_given_0": get(f"metrics.ssro.measure.{m_impl}.{q}.error_0_to_1"),
            "p0_given_1": get(f"metrics.ssro.measure.{m_impl}.{q}.error_1_to_0"),
            "readout_error": get(f"metrics.ssro.measure.{m_impl}.{q}.assignment_error"),
            "error_1q": None if e_1q is None else 1 - e_1q,
            "readout_leakage": get(f"metrics.rrc.measure.{m_impl}.{q}.leakage"),
            "readout_seepage": get(f"metrics.rrc.measure.{m_impl}.{q}.seepage"),
            "readout_qndness": get(f"metrics.rrc.measure.{m_impl}.{q}.qndness"),
            "prx_ns": None if dur_p is None else dur_p * 1e9,
            "measure_ns": None if dur_m is None else dur_m * 1e9,
        }
    pairs = {}
    for key, (a, b) in _cz_loci(arch).items():
        impl = _impl(gates["cz"], (a, b))
        f_irb = get(f"metrics.irb.cz.{impl}.{a}__{b}.fidelity", f"metrics.irb.cz.{impl}.{b}__{a}.fidelity")
        f_clf = get(f"metrics.rb.clifford.uz_cz.{a}__{b}.fidelity", f"metrics.rb.clifford.uz_cz.{b}__{a}.fidelity")
        dur = get(f"gates.cz.{impl}.{a}__{b}.duration", f"gates.cz.{impl}.{b}__{a}.duration")
        pairs[tuple(sorted((a, b), key=hw._qubit_sort_key))] = {
            "implementation": impl,
            "cz_error_irb": None if f_irb is None else 1 - f_irb,
            "cz_error_clifford_rb": None if f_clf is None else 1 - f_clf,
            "cz_ns": None if dur is None else dur * 1e9,
        }
    return {"qubits": qubits, "couplers": pairs}


def _stat(values: dict, key: str, worst=max) -> dict | None:
    vals = {k: v[key] for k, v in values.items() if v.get(key) is not None}
    if not vals:
        return None
    w = worst(vals, key=vals.get)
    return {"n": len(vals), "median": statistics.median(vals.values()), "worst": vals[w], "worst_at": w}


def key_numbers(fig: dict) -> dict:
    q, c = fig["qubits"], fig["couplers"]
    return {
        "T1 (us)": _stat(q, "t1_us", min),
        "T2 echo (us)": _stat(q, "t2_echo_us", min),
        "T2 Ramsey (us)": _stat(q, "t2_ramsey_us", min),
        "readout assignment error": _stat(q, "readout_error"),
        "P(1|0)": _stat(q, "p1_given_0"),
        "P(0|1)": _stat(q, "p0_given_1"),
        "1Q error (RB)": _stat(q, "error_1q"),
        "readout leakage": _stat(q, "readout_leakage"),
        "CZ error (IRB)": _stat(c, "cz_error_irb"),
        "CZ error (Clifford RB)": _stat(c, "cz_error_clifford_rb"),
        "prx (ns)": _stat(q, "prx_ns"),
        "CZ (ns)": _stat(c, "cz_ns"),
        "measure (ns)": _stat(q, "measure_ns"),
    }


def code_graph():
    import networkx as nx

    from dqiqm.surface import geometry

    g = geometry(3)
    graph = nx.Graph()
    for a in g.ancillas:
        for q in g.support[a]:
            graph.add_edge(a, q)
    return g, graph


def placements(snap: dict, fig: dict) -> dict:
    """Every zero-SWAP placement of the d = 3 code in the calibrated CZ graph, ranked by score."""
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher

    g, code = code_graph()
    assert code.number_of_nodes() == 17 and code.number_of_edges() == 24
    device = nx.Graph(couplers(snap["architecture"]))
    device.add_nodes_from(snap["architecture"]["qubits"])
    q, c = fig["qubits"], fig["couplers"]

    def worst(values, key, fn=max):
        vals = [v[key] for v in values if v.get(key) is not None]
        return fn(vals) if vals else None

    def median(values, key):
        vals = [v[key] for v in values if v.get(key) is not None]
        return statistics.median(vals) if vals else None

    t_cz, t_prx, t_meas = median(c.values(), "cz_ns"), median(q.values(), "prx_ns"), median(q.values(), "measure_ns")
    t_round_us = (4 * t_cz + 10 * t_prx + t_meas) / 1e3
    worst_cz = worst(c.values(), "cz_error_irb") or worst(c.values(), "cz_error_clifford_rb")
    worst_ro, worst_t1 = worst(q.values(), "readout_error"), worst(q.values(), "t1_us", min)
    missing = {
        "couplers without IRB (Clifford RB used)": sum(v["cz_error_irb"] is None for v in c.values()),
        "couplers without any CZ figure (worst used)": sum(
            v["cz_error_irb"] is None and v["cz_error_clifford_rb"] is None for v in c.values()
        ),
        "qubits without readout error (worst used)": sum(v["readout_error"] is None for v in q.values()),
        "qubits without T1 (worst used)": sum(v["t1_us"] is None for v in q.values()),
    }

    def cz_err(pair) -> float:
        v = c[tuple(sorted(pair, key=hw._qubit_sort_key))]
        for key in ("cz_error_irb", "cz_error_clifford_rb"):
            if v[key] is not None:
                return v[key]
        return worst_cz

    def ro_err(qb) -> float:
        return worst_ro if q[qb]["readout_error"] is None else q[qb]["readout_error"]

    def t1(qb) -> float:
        return worst_t1 if q[qb]["t1_us"] is None else q[qb]["t1_us"]

    groups: dict[tuple, dict] = {}
    n_maps = 0
    for mono in GraphMatcher(device, code).subgraph_monomorphisms_iter():
        n_maps += 1
        place = {v: k for k, v in mono.items()}  # code qubit -> device qubit
        edges = frozenset(frozenset((place[a], place[d])) for a, d in code.edges)
        data = frozenset(place[d] for d in g.data)
        key = (data, edges)
        if key not in groups:
            cz_term = sum(-math.log(1 - cz_err(tuple(e))) for e in edges)
            ro_term = sum(-math.log(1 - ro_err(place[v])) for v in code.nodes)
            t1_term = sum(t_round_us / t1(place[v]) for v in code.nodes)
            groups[key] = {
                "score": cz_term + ro_term + t1_term,
                "terms": {"cz": cz_term, "readout": ro_term, "t1": t1_term},
                "qubits": sorted(data | {place[a] for a in g.ancillas}, key=hw._qubit_sort_key),
                "data": sorted(data, key=hw._qubit_sort_key),
                "worst_cz": max((cz_err(tuple(e)), "-".join(sorted(e, key=hw._qubit_sort_key))) for e in edges),
                "orientations": [],
            }
        groups[key]["orientations"].append(
            {
                "x_ancillas": sorted((place[a] for a in g.x_ancillas), key=hw._qubit_sort_key),
                "z_ancillas": sorted((place[a] for a in g.z_ancillas), key=hw._qubit_sort_key),
                "map": {f"{g.coords[v][0]},{g.coords[v][1]}": place[v] for v in sorted(code.nodes)},
            }
        )
    ranked = sorted(groups.values(), key=lambda p: p["score"])
    degrees = Counter(dict(device.degree()).values())
    sqa = snap.get("static_architecture")
    static_count = None
    if sqa:  # every coupler of the chip, calibrated or not
        static = nx.Graph([tuple(p) for p in sqa["connectivity"]])
        static_count = sum(1 for _ in GraphMatcher(static, code).subgraph_monomorphisms_iter())
    return {
        "device_qubits": device.number_of_nodes(),
        "device_couplers": device.number_of_edges(),
        "degree_histogram": dict(sorted(degrees.items())),
        "degree4_qubits": sorted((n for n, d in device.degree() if d == 4), key=hw._qubit_sort_key),
        "t_round_us": t_round_us,
        "monomorphisms": n_maps,
        "static_couplers": None if not sqa else len(sqa["connectivity"]),
        "monomorphisms_all_couplers": static_count,
        "placements": len(ranked),
        "qubit_sets": len({frozenset(p["qubits"]) for p in ranked}),
        "missing_figures": {k: n for k, n in missing.items() if n},
        "ranked": ranked,
    }


def repetition_path(snap: dict) -> dict:
    """The mentors' Garnet repetition-code path (0-based indices): which of its links are couplers."""
    names = sorted(snap["architecture"]["qubits"], key=hw._qubit_sort_key)
    path = [names[i] for i in REPETITION_PATH]
    pairs = set(map(frozenset, couplers(snap["architecture"])))
    links = list(zip(path, path[1:]))
    missing = ["-".join(p) for p in links if frozenset(p) not in pairs]
    return {"path": path, "links": len(links), "missing": missing}


def fake_comparison(snap: dict, fig: dict) -> dict:
    """IQMFakeGarnet (iqm-client 35.0.3, a local simulator) against the live Garnet snapshot."""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile
    from iqm.station_control.interface.models import StaticQuantumArchitecture

    fake = IQMFakeGarnet()
    ep = fake.error_profile
    fake_arch = fake.architecture
    fake_pairs = set()
    for info in fake_arch.gates["cz"].implementations.values():
        fake_pairs |= {frozenset(locus) for locus in info.loci}
    live_pairs = set(map(frozenset, couplers(snap["architecture"])))
    q, c = fig["qubits"], fig["couplers"]

    def ratio_stats(pairs):
        r = [live / f for live, f in pairs if live and f]
        return {"n": len(r), "median_live_over_fake": statistics.median(r), "min": min(r), "max": max(r)} if r else None

    out = {
        "fake_qubits": len(fake_arch.qubits),
        "live_qubits": len(snap["architecture"]["qubits"]),
        "same_qubit_names": set(fake_arch.qubits) == set(snap["architecture"]["qubits"]),
        "fake_couplers": len(fake_pairs),
        "live_couplers": len(live_pairs),
        "same_couplers": fake_pairs == live_pairs,
        "couplers_only_in_fake": sorted("-".join(sorted(p, key=hw._qubit_sort_key)) for p in fake_pairs - live_pairs),
        "couplers_only_live": sorted("-".join(sorted(p, key=hw._qubit_sort_key)) for p in live_pairs - fake_pairs),
        "median_fake": {
            "T1 (us)": statistics.median(ep.t1s.values()) / 1e3,
            "T2 (us)": statistics.median(ep.t2s.values()) / 1e3,
            "P(1|0)": statistics.median(r["0"] for r in ep.readout_errors.values()),
            "P(0|1)": statistics.median(r["1"] for r in ep.readout_errors.values()),
            "1Q depolarizing param": statistics.median(ep.single_qubit_gate_depolarizing_error_parameters["prx"].values()),
            "CZ depolarizing param": statistics.median(ep.two_qubit_gate_depolarizing_error_parameters["cz"].values()),
            "prx (ns)": ep.single_qubit_gate_durations["prx"],
            "CZ (ns)": ep.two_qubit_gate_durations["cz"],
        },
        "per_qubit_ratio": {
            "T1": ratio_stats([(q[k]["t1_us"], ep.t1s[k] / 1e3) for k in ep.t1s if k in q]),
            "T2 echo": ratio_stats([(q[k]["t2_echo_us"], ep.t2s[k] / 1e3) for k in ep.t2s if k in q]),
            "T2 Ramsey": ratio_stats([(q[k]["t2_ramsey_us"], ep.t2s[k] / 1e3) for k in ep.t2s if k in q]),
            "P(1|0)": ratio_stats([(q[k]["p1_given_0"], ep.readout_errors[k]["0"]) for k in ep.readout_errors if k in q]),
            "P(0|1)": ratio_stats([(q[k]["p0_given_1"], ep.readout_errors[k]["1"]) for k in ep.readout_errors if k in q]),
        },
    }
    # benchmarked infidelity of a depolarizing parameter p on D levels: p (D - 1) / D
    out["per_qubit_ratio"]["1Q error vs p/2"] = ratio_stats(
        [(q[k]["error_1q"], p / 2) for k, p in ep.single_qubit_gate_depolarizing_error_parameters["prx"].items() if k in q]
    )
    out["per_qubit_ratio"]["CZ IRB error vs 3p/4"] = ratio_stats(
        [
            (c.get(tuple(sorted(k, key=hw._qubit_sort_key)), {}).get("cz_error_irb"), 3 * p / 4)
            for k, p in ep.two_qubit_gate_depolarizing_error_parameters["cz"].items()
        ]
    )
    if snap.get("static_architecture"):
        sqa = StaticQuantumArchitecture(**snap["static_architecture"])
        out["validate_compatible_architecture(live SQA)"] = bool(fake.validate_compatible_architecture(sqa))
        out["dut_label"] = {"fake": fake._IQMFakeBackend__sqa.dut_label, "live": sqa.dut_label}

    # The fake's topology with today's live noise: the stand-in for offline dataset work. A
    # benchmarked infidelity r maps to a depolarizing parameter r*D/(D-1) (D = 2 or 4), which
    # slightly overstates the noise, as the fake adds thermal relaxation on top.
    try:
        live_ep = IQMErrorProfile(
            t1s={k: v["t1_us"] * 1e3 for k, v in q.items()},
            t2s={k: min(v["t2_echo_us"] or v["t2_ramsey_us"], 2 * v["t1_us"]) * 1e3 for k, v in q.items()},
            single_qubit_gate_depolarizing_error_parameters={"prx": {k: 2 * v["error_1q"] for k, v in q.items()}},
            two_qubit_gate_depolarizing_error_parameters={
                "cz": {
                    _cz_loci(snap["architecture"])[frozenset(k)]: 4 / 3 * (v["cz_error_irb"] or v["cz_error_clifford_rb"])
                    for k, v in c.items()
                }
            },
            single_qubit_gate_durations={"prx": statistics.median(v["prx_ns"] for v in q.values())},
            two_qubit_gate_durations={"cz": statistics.median(v["cz_ns"] for v in c.values())},
            readout_errors={k: {"0": v["p1_given_0"], "1": v["p0_given_1"]} for k, v in q.items()},
            name=f"garnet {snap['calibration_set_id']}",
        )
        fake.copy_with_error_profile(live_ep)
        out["fake_with_live_profile"] = "builds: IQMFakeGarnet().copy_with_error_profile(<live snapshot>) works"
    except Exception as e:
        out["fake_with_live_profile"] = f"fails: {type(e).__name__}: {e}"
    return out


def aphrodite_comparison(snap: dict, fig: dict) -> dict:
    """IQMFakeAphrodite, the team's offline stand-in for Emerald (STATE.md), against the live Emerald."""
    from iqm.qiskit_iqm.fake_backends import IQMFakeAphrodite

    fake = IQMFakeAphrodite()
    ep = fake.error_profile
    fake_pairs = set()
    for info in fake.architecture.gates["cz"].implementations.values():
        fake_pairs |= {frozenset(locus) for locus in info.loci}
    static = {frozenset(p) for p in (snap.get("static_architecture") or {}).get("connectivity", [])}
    live = set(map(frozenset, couplers(snap["architecture"])))
    q = fig["qubits"]
    return {
        "same_qubit_names": set(fake.architecture.qubits) == set(snap["architecture"]["qubits"]),
        "fake_couplers": len(fake_pairs),
        "same_as_static_couplers": fake_pairs == static,
        "same_as_calibrated_couplers": fake_pairs == live,
        "fake_only_vs_calibrated": sorted("-".join(sorted(p, key=hw._qubit_sort_key)) for p in fake_pairs - live),
        "median T1 (us) fake / live": (
            statistics.median(ep.t1s.values()) / 1e3,
            statistics.median(v["t1_us"] for v in q.values() if v["t1_us"]),
        ),
    }


# ─── printing ────────────────────────────────────────────────────────────────


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_device(snap: dict, path: Path | str) -> dict:
    fig = figures(snap)
    print(f"\n=== {snap['device']}: calibration set {snap['calibration_set_id']} ({path})")
    print(f"fetched {snap['fetched_utc_start']} .. {snap['fetched_utc_end']}, dqiqm {snap['dqiqm_commit'][:12]}")
    print(f"quality-metric set {snap['quality_metric_set_id']}; server default at fetch: {snap['default_calibration_set_id_at_fetch']}")
    for h in snap["helpers"]:
        print(f"  {'OK  ' if h['ok'] else 'FAIL'} {h['call']}" + ("" if h["ok"] else f"  ->  {h['error']}"))
    arch = snap["architecture"]
    print(f"qubits {len(arch['qubits'])}, calibrated CZ couplers {len(couplers(arch))}")
    for gate in ("measure", "measure_012", "prx_12", "cc_prx", "reset_wait"):
        info = arch["gates"].get(gate)
        if info:
            loci = {tuple(l) for impl in info["implementations"].values() for l in impl["loci"]}
            print(f"  gate {gate}: default {info['default_implementation']}, {len(loci)} loci")
    for part in ("quality_metric_set", "calibration_set"):
        s = snap.get(part) or {}
        print(f"  {part}: {s.get('observations_kept')} of {s.get('observations_total')} observations kept")
    print(f"  {'figure':26s} {'n':>3s} {'median':>10s} {'worst':>10s}  at")
    for name, s in key_numbers(fig).items():
        if s:
            print(f"  {name:26s} {s['n']:3d} {_fmt(s['median']):>10s} {_fmt(s['worst']):>10s}  {s['worst_at']}")
        else:
            print(f"  {name:26s}   - (not served)")
    return fig


def print_placements(res: dict, top: int = 3) -> None:
    print(
        f"d=3 placement: device {res['device_qubits']} qubits / {res['device_couplers']} couplers, "
        f"degrees {res['degree_histogram']}, degree-4 {res['degree4_qubits']}"
    )
    print(
        f"  {res['monomorphisms']} monomorphisms -> {res['placements']} placements (data set + CZs used) "
        f"on {res['qubit_sets']} qubit set(s); t_round = {res['t_round_us']:.3f} us; missing {res['missing_figures'] or 'none'}"
    )
    if res["static_couplers"] is not None:
        print(
            f"  on all {res['static_couplers']} couplers of the static architecture (calibrated or not): "
            f"{res['monomorphisms_all_couplers']} monomorphisms"
        )
    for i, p in enumerate(res["ranked"][:top], 1):
        t = p["terms"]
        xz = len({frozenset(o["x_ancillas"]) for o in p["orientations"]})
        print(
            f"  #{i} score {p['score']:.4f} (CZ {t['cz']:.4f} + readout {t['readout']:.4f} + T1 {t['t1']:.4f}); "
            f"worst CZ {p['worst_cz'][1]} {p['worst_cz'][0]:.4f}; {len(p['orientations'])} orientation(s), "
            f"{xz} distinct X/Z assignment(s)"
        )
        print(f"     qubits {','.join(p['qubits'])}")
        print(f"     data   {','.join(p['data'])}")
        o = p["orientations"][0]
        print(f"     X anc  {','.join(o['x_ancillas'])}   Z anc {','.join(o['z_ancillas'])}  (first orientation)")


def print_fake(cmp: dict) -> None:
    print("\n=== IQMFakeGarnet vs live garnet")
    for k in ("fake_qubits", "live_qubits", "same_qubit_names", "fake_couplers", "live_couplers", "same_couplers"):
        print(f"  {k}: {cmp[k]}")
    if cmp["couplers_only_in_fake"] or cmp["couplers_only_live"]:
        print(f"  only in fake: {cmp['couplers_only_in_fake']}; only live: {cmp['couplers_only_live']}")
    if "validate_compatible_architecture(live SQA)" in cmp:
        print(
            f"  validate_compatible_architecture(live SQA): {cmp['validate_compatible_architecture(live SQA)']} "
            f"(dut_label fake {cmp['dut_label']['fake']}, live {cmp['dut_label']['live']})"
        )
    print("  fake medians: " + ", ".join(f"{k} {_fmt(v)}" for k, v in cmp["median_fake"].items()))
    for k, s in cmp["per_qubit_ratio"].items():
        if s:
            print(f"  live/fake {k:20s}: median {s['median_live_over_fake']:.3f} (min {s['min']:.3f}, max {s['max']:.3f}, n {s['n']})")
    print(f"  {cmp['fake_with_live_profile']}")


def newest_snapshot(device: str) -> Path | None:
    files = sorted((CALIBRATION / device).glob("*.json"))
    if not files:
        return None
    return max(files, key=lambda p: json.loads(p.read_text())["fetched_utc_start"])


def analyse(snaps: dict[str, tuple[dict, Path | str]]) -> None:
    figs = {}
    for device, (snap, path) in snaps.items():
        if "architecture" not in snap:
            print(f"\n=== {device}: no architecture; nothing to analyse")
            continue
        figs[device] = print_device(snap, path)
        print_placements(placements(snap, figs[device]))
        if device == "garnet":
            r = repetition_path(snap)
            print(f"mentors' repetition-code path ({len(r['path'])} qubits): {r['links']} links, missing couplers {r['missing'] or 'none'}")
    if "garnet" in figs:
        print_fake(fake_comparison(snaps["garnet"][0], figs["garnet"]))
    if "emerald" in figs:
        print("\n=== IQMFakeAphrodite (the team's Emerald stand-in) vs live emerald")
        for k, v in aphrodite_comparison(snaps["emerald"][0], figs["emerald"]).items():
            print(f"  {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="analyse the newest committed snapshots; no network")
    ap.add_argument("--snapshot", type=Path, action="append", default=[], help="analyse this snapshot file (offline)")
    args = ap.parse_args()

    if args.offline or args.snapshot:
        paths = args.snapshot or [p for p in map(newest_snapshot, DEVICES) if p]
        snaps = {}
        for p in paths:
            snap = json.loads(Path(p).read_text())
            p = Path(p).resolve()
            snaps[snap["device"]] = (snap, p.relative_to(REPO) if p.is_relative_to(REPO) else p)
        if not snaps:
            print("no snapshots under data/calibration/", file=sys.stderr)
            return 1
        analyse(snaps)
        return 0

    log: list[dict] = []
    devices = fetch_device_list(log)
    aliases = [d["alias"] for d in devices]
    print("device list:", ", ".join(f"{d['alias']} ({d['display_name']})" for d in devices) or "none")
    for h in log:
        print(f"  {'OK  ' if h['ok'] else 'FAIL'} {h['call']}" + ("" if h["ok"] else f"  ->  {h['error']}"))
    if not devices:
        print("no device list: authentication or connection failed", file=sys.stderr)
        return 1

    snaps = {}
    for device in DEVICES:
        if device not in aliases:
            print(f"{device}: not listed for this token")
            continue
        snap = fetch_snapshot(device, log)
        snap["device_list"] = devices
        if "architecture" not in snap:
            print(f"{device}: {snap['helpers']}")
            continue
        path = save_snapshot(snap)
        snaps[device] = (snap, path.relative_to(REPO))
        print(f"{device}: wrote {path.relative_to(REPO)} ({path.stat().st_size / 1e6:.2f} MB)")
        found = cost_fields({"health": snap["health"], "about": snap["about"], "device_list": devices})
        print(f"{device}: cost/credit/quota fields in health, about and the device list: {found or 'none'}")
    analyse(snaps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
