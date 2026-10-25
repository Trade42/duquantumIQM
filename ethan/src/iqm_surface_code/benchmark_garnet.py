"""Calibration, native placement and result helpers for Garnet memory."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import time
import json
import math
from pathlib import Path

def read_token(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Credential file is empty.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    if isinstance(value, dict):
        value = next((value[k] for k in ("token", "IQM_TOKEN", "api_key", "apikey", "access_token")
                      if value.get(k)), None)
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value.strip()):
        raise ValueError("Expected a raw IQM token, JSON string, or JSON token field.")
    return value.strip()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def calibration_from_observations(architecture, quality, pulse):
    """Read active IQM implementation metrics, explicit units, invalid flags.

    No leakage conversion: repeated-readout leakage is not per-gate leakage.
    Decoder uses calibrated Pauli gate/readout weights, without T1/T2 idle
    channels, since server pulse timing differs from logical Stim TICK timing.
    """
    from .calibration import CalibrationSnapshot, QubitCalibration
    gates = architecture["gates"]
    # Per-locus implementation overrides need their own observation selection.
    if any(g.get("override_default_implementation") for g in gates.values()):
        raise ValueError("Per-locus implementation overrides need explicit calibration selection.")
    active = {k: v["default_implementation"] for k, v in gates.items()}
    metrics = {o["dut_field"]: o for o in quality["observations"] if not o.get("invalid", False)}
    durations = {o["dut_field"]: o for o in pulse["observations"] if not o.get("invalid", False)}
    scales = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.}

    def seconds_ns(observation, *, blank_seconds=False):
        unit = observation["unit"] or ("s" if blank_seconds else "")
        if unit not in scales:
            raise ValueError("Unknown calibration time unit.")
        return float(observation["value"]) * scales[unit]

    def worst_fidelity(prefix):
        values = [float(v["value"]) for k, v in metrics.items()
                  if k == prefix or k.startswith(prefix + ":par=")]
        if not values:
            raise ValueError(f"Missing calibration metric: {prefix}")
        return min(values)

    qubits = {}
    for q in architecture["qubits"]:
        sq = worst_fidelity(f"metrics.rb.prx.{active['prx']}.{q}.fidelity")
        ro = worst_fidelity(f"metrics.ssro.measure.{active['measure']}.{q}.fidelity")
        t1 = seconds_ns(metrics[f"characterization.model.{q}.t1_time"])
        t2 = seconds_ns(metrics[f"characterization.model.{q}.t2_time"])
        qubits[q] = QubitCalibration(1-sq, 1-ro, .002, t1, t2)
    edges = {}
    for pair in gates["cz"]["implementations"][active["cz"]]["loci"]:
        a, b = pair
        prefixes = [f"metrics.irb.cz.{active['cz']}.{x}__{y}.fidelity" for x, y in ((a,b),(b,a))]
        values = [worst_fidelity(p) for p in prefixes
                  if any(k == p or k.startswith(p + ":par=") for k in metrics)]
        if not values:
            raise ValueError(f"Missing native CZ calibration for {a}, {b}.")
        edges[CalibrationSnapshot.edge_key(a,b)] = 1-min(values)
    timing = {}
    for key, gate in (("1q", "prx"), ("2q", "cz"), ("measure", "measure"), ("reset", "reset_wait")):
        prefix = f"gates.{gate}.{active[gate]}."
        observations = [o for k, o in durations.items()
                        if k.startswith(prefix) and k.endswith(".duration") and "QB" in k]
        if not observations:
            raise ValueError(f"Missing calibrated {gate} duration.")
        # IQM pulse parameter schemas define these durations in seconds even
        # when an observation's unit string is empty (iqm-pulse 14.0.3).
        timing[key] = max(seconds_ns(o, blank_seconds=True) for o in observations)
    return CalibrationSnapshot(qubits, edges, timing, dict(
        source="Live IQM Garnet quality metrics and pulse calibration",
        calibration_set_id=architecture["calibration_set_id"],
        timing_policy="maximum active-implementation duration; blank duration units use IQM pulse schema seconds",
        assumed_reset_error=.002, reset_error_note="Uncalibrated initialization prior; not measured reset fidelity",
        leakage_note="Readout leakage/QND observations retained separately; not interpreted as gate leakage",
        decoder_model="calibrated gate/readout Pauli approximation; no timed idle model or DD reweighting"))


def select_native_layout(circuit, couplings, physical_qubits, calibration=None):
    """Enumerate exact graph embeddings on Garnet's small 20-qubit graph."""
    import networkx as nx
    from .build_emerald_qubit_rotated import interaction_pairs, validate_layout
    hardware = nx.Graph()
    hardware.add_nodes_from(physical_qubits)
    hardware.add_edges_from(couplings)
    required = nx.Graph()
    required.add_nodes_from(circuit.get_final_qubit_coordinates())
    required.add_edges_from(interaction_pairs(circuit))
    if len(hardware) > 20:
        raise ValueError("This exhaustive benchmark placement is limited to <=20 physical qubits.")
    one, measured, pairs = Counter(), Counter(), Counter(interaction_pairs(circuit))
    for inst in circuit.flattened():
        targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name in {"H", "X", "Y", "Z", "S", "S_DAG"}:
            one.update(targets)
        elif inst.name in {"M", "MX"}:
            measured.update(targets)
    candidates = []
    for embedding in nx.algorithms.isomorphism.GraphMatcher(hardware, required).subgraph_monomorphisms_iter():
        mapping = {q: physical for physical, q in embedding.items()}
        validate_layout(circuit, mapping, couplings)
        score = 0.
        if calibration:
            score = sum(one[q] * -math.log1p(-calibration.qubits[p].single_qubit_infidelity)
                        + measured[q] * -math.log1p(-calibration.qubits[p].readout_error)
                        for q, p in mapping.items())
            score += sum(n * -math.log1p(-calibration.edge_error(mapping[a], mapping[b]))
                         for (a,b), n in pairs.items())
        candidates.append((score, tuple(mapping[q] for q in sorted(mapping)), mapping))
    if not candidates:
        raise ValueError("No native surface-code embedding in live Garnet topology.")
    score, _, mapping = min(candidates, key=lambda v: v[:2])
    return mapping, dict(candidates=len(candidates), score=score,
                         objective="gate/readout negative-log-fidelity proxy; no fitted LER optimization")


def analyze_counts(counts, ideal, decoder, *, expected_shots, memory=None):
    import numpy as np
    from .decoding import decode_syndromes, make_matcher
    from .extract_syndromes import extract_syndromes
    from .internal_helpers import counts_to_measurement_array
    start = time.perf_counter()
    raw = counts_to_measurement_array(counts, ideal.num_measurements, expected_shots)
    if memory is not None:
        # IQM supplies the original acquisition order as Qiskit memory. Counts
        # alone group identical records and are unsuitable for temporal studies.
        records = [record.replace(" ", "") for record in memory]
        normalized_counts = Counter()
        for record, count in counts.items():
            normalized_counts[record.replace(" ", "")] += count
        if Counter(records) != normalized_counts:
            raise ValueError("Original shot memory and aggregate counts differ.")
        raw = np.asarray([[bit == "1" for bit in record[::-1]] for record in records],dtype=bool)
    syndromes = extract_syndromes(raw, ideal, print_summary=False)
    syndromes["stim_circuit"] = ideal
    convert_s = time.perf_counter() - start
    start = time.perf_counter()
    matcher = make_matcher(decoder)
    build_s = time.perf_counter() - start
    result = decode_syndromes(syndromes, decoder, matcher=matcher)
    # Warmed decode timing excludes parsing, graph construction and scoring.
    dets = np.asarray(syndromes["det_events"], dtype=bool)
    matcher.decode_batch(dets, enable_correlations=True)
    timings = []
    for _ in range(15):
        start = time.perf_counter()
        matcher.decode_batch(dets, enable_correlations=True)
        timings.append(time.perf_counter()-start)
    median = float(np.median(timings))
    return dict(**result.to_dict(), detector_event_fraction=float(dets.mean()),
        shot_order="original acquisition order" if memory is not None else "counts-expanded grouping",
        timing=dict(convert_s=convert_s, matcher_build_s=build_s, decode_batch_median_s=median,
                    decode_repeats=len(timings), decode_shots_per_second=expected_shots/median)), raw, syndromes


def server_timing(timeline):
    entries = {(e["source"],e["status"]):datetime.fromisoformat(e["timestamp"])
               for e in timeline}
    result = {}
    for phase in ("validation","compilation","execution","post_processing"):
        start, stop = (("iqm-station-control",f"{phase}_{edge}") for edge in ("started","ended"))
        if start in entries and stop in entries:
            result[phase+"_s"] = (entries[stop]-entries[start]).total_seconds()
    return result
