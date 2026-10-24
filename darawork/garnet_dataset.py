"""Simulated IQM Garnet d=3 no-reset memory dataset for ML decoders. No QPU access.

One Parquet file per (model, basis, rounds), one row per shot: raw ancilla
records, per-round syndromes, detection events, temporal/spatial correlation
features, correlated-MWPM baseline outputs and the true logical flip.

Example: python -m darawork.garnet_dataset --shots 20000 --rounds 1 2 3 4 5 6 8 10

Hardware (real Garnet, same columns): dry run unless --submit; --max-shots caps
the job's total shots before any network call; job ids go to the ledger at once.
  python -m darawork.garnet_dataset --hardware --rounds 1 --bases Z --shots 1000 --submit
  python -m darawork.garnet_dataset --retrieve <job id>
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import numbers
import os
from pathlib import Path
import zlib
import numpy as np
import rustworkx as rx
import stim

from challenge.surface_code import make_stim_circuit
from challenge.noise_models import NoiseModel, add_noise
from challenge.calibration import CalibrationSnapshot, QubitCalibration
from challenge.build_emerald_qubit_rotated import interaction_pairs, validate_layout
from challenge.decoding import make_matcher, wilson_interval
from challenge.leakage import sample_leakage

SERVER_URL = "https://resonance.iqm.tech"
GARNET_MAX_SHOTS_PER_CIRCUIT, GARNET_MAX_CIRCUITS = 20000, 100
HW_DIR, FAKE_DIR = Path("darawork/data/garnet_hw"), Path("results/garnet_fake")

# Crystal-20 couplers, from iqm-client 35.0.3 IQMFakeGarnet (index_to_qubit_name).
GARNET_EDGES = [("QB1", "QB2"), ("QB1", "QB4"), ("QB2", "QB5"), ("QB3", "QB4"), ("QB4", "QB5"),
    ("QB5", "QB6"), ("QB6", "QB7"), ("QB8", "QB3"), ("QB8", "QB9"), ("QB8", "QB13"), ("QB9", "QB4"),
    ("QB9", "QB10"), ("QB9", "QB14"), ("QB10", "QB5"), ("QB10", "QB11"), ("QB10", "QB15"),
    ("QB11", "QB6"), ("QB11", "QB12"), ("QB12", "QB7"), ("QB13", "QB14"), ("QB14", "QB15"),
    ("QB16", "QB11"), ("QB16", "QB15"), ("QB16", "QB17"), ("QB16", "QB20"), ("QB17", "QB12"),
    ("QB18", "QB14"), ("QB18", "QB19"), ("QB19", "QB15"), ("QB19", "QB20")]
# IQM Resonance Garnet page, medians, read 24 Oct 2026.
GARNET_MEDIANS = dict(prx_fidelity=0.9994, cz_fidelity=0.9950, t1_us=37.08,
                      t2_ramsey_us=9.08, t2_echo_us=19.90)


def garnet_snapshot(*, readout_error, t2_us, durations_ns, names=None, couplers=None):
    """Every qubit/coupler at the published median; readout and durations are ASSUMED.

    names/couplers default to Garnet's; pass others for a larger hypothetical
    chip with Garnet-median qubits (d=5/7 benchmarks).
    """
    q = QubitCalibration(1 - GARNET_MEDIANS["prx_fidelity"], readout_error, 0.0,
                         GARNET_MEDIANS["t1_us"] * 1e3, t2_us * 1e3)
    qubits = {n: q for n in (names or [f"QB{i}" for i in range(1, 21)])}
    edges = {CalibrationSnapshot.edge_key(a, b): 1 - GARNET_MEDIANS["cz_fidelity"] for a, b in (couplers or GARNET_EDGES)}
    return CalibrationSnapshot(qubits, edges, dict(durations_ns), dict(
        source="IQM Resonance Garnet page medians, 24 Oct 2026; uniform across qubits",
        assumed=dict(readout_error=readout_error, durations_ns=dict(durations_ns), reset_error=0.0)))


def garnet_placements(ideal):
    """Every zero-SWAP embedding of the circuit's interaction graph into Crystal 20."""
    g, h = rx.PyGraph(), rx.PyGraph()
    names = sorted({q for e in GARNET_EDGES for q in e}, key=lambda s: int(s[2:]))
    gi = {q: g.add_node(q) for q in names}
    for a, b in GARNET_EDGES:
        g.add_edge(gi[a], gi[b], None)
    pairs = {tuple(sorted(p)) for p in interaction_pairs(ideal)}
    hi = {q: h.add_node(q) for q in sorted({q for p in pairs for q in p})}
    for a, b in pairs:
        h.add_edge(hi[a], hi[b], None)
    found = []
    for m in rx.vf2_mapping(g, h, subgraph=True, induced=False, id_order=True):
        layout = {h[j]: g[i] for i, j in m.items()}
        validate_layout(ideal, layout, GARNET_EDGES)
        found.append(dict(sorted(layout.items())))
    if not found:
        raise ValueError("No zero-SWAP Garnet embedding for this circuit.")
    return sorted(found, key=lambda m: [int(v[2:]) for v in m.values()])


def models(cal_ramsey, cal_echo, leakage):
    base = dict(idle=0)
    return {"garnet_pauli": NoiseModel(name="garnet_pauli", calibration=cal_ramsey, **base),
            "garnet_coherence": NoiseModel(name="garnet_coherence", calibration=cal_ramsey, coherence=True, **base),
            "garnet_coherence_echo": NoiseModel(name="garnet_coherence_echo", calibration=cal_echo, coherence=True, **base),
            "garnet_leakage": NoiseModel(name="garnet_leakage", calibration=cal_ramsey, coherence=True,
                                         leakage=leakage, seepage=0.01, leakage_spread=0.02, **base)}


def layout_info(ideal, basis):
    """Stabilizer grid, per-measurement (qubit, round), and stabilizer adjacency."""
    coords = ideal.get_final_qubit_coordinates()
    dets = ideal.get_detector_coordinates()
    by_xy = {}
    for k, c in dets.items():
        by_xy.setdefault((c[0], c[1]), []).append((int(c[2]), k))
    anc_at = {tuple(coords[q][:2]): q for q in coords}
    # Memory-basis stabilizers are deterministic from round 0; the others are not.
    first = {xy: min(t for t, _ in v) for xy, v in by_xy.items()}
    stabs = []
    for xy in sorted(by_xy, key=lambda xy: (xy[1], xy[0])):
        if xy not in anc_at:  # Terminal detectors sit on ancilla coordinates too.
            raise ValueError(f"Detector at {xy} has no ancilla.")
        kind = basis if first[xy] == 0 else ("X" if basis == "Z" else "Z")
        stabs.append(dict(xy=xy, kind=kind, qubit=anc_at[xy], name=f"{kind}{int(xy[0])}_{int(xy[1])}",
                          dets=dict(sorted(by_xy[xy]))))
    meas, seen = [], Counter()
    for inst in ideal.flattened():
        if inst.name in {"M", "MX", "MR", "MRX"}:
            for t in inst.targets_copy():
                meas.append((t.value, seen[t.value]))
                seen[t.value] += 1
    touches = {}
    for a, b in interaction_pairs(ideal):
        touches.setdefault(a, set()).add(b)
        touches.setdefault(b, set()).add(a)
    adj = [(i, j) for i in range(len(stabs)) for j in range(i + 1, len(stabs))
           if touches[stabs[i]["qubit"]] & touches[stabs[j]["qubit"]]]
    return stabs, meas, adj


def features(raw, dets, obs, ideal, basis, rounds, matcher, matcher_uncorr):
    stabs, meas, adj = layout_info(ideal, basis)
    shots, T, S = len(dets), rounds + 1, len(stabs)
    cols, doc = {}, {}

    def put(name, value, text):
        cols[name] = value
        doc.setdefault(name.split("__")[0], text)

    # Detection events on a (shot, stabilizer, t) grid; absent detectors stay 0.
    grid, mask = np.zeros((shots, S, T), bool), np.zeros((S, T), bool)
    for s, st in enumerate(stabs):
        for t, k in st["dets"].items():
            grid[:, s, t], mask[s, t] = dets[:, k], True
            put(f"det__{st['name']}_t{t}", dets[:, k], "detection event (stabilizer, round t); no-reset: m_t xor m_(t-2)")
    # Raw records and per-round syndromes s_t = m_t xor m_(t-1) for each ancilla.
    anc = {st["qubit"]: st for st in stabs}
    prev = {}
    for j, (q, r) in enumerate(meas):
        if q in anc:
            name = anc[q]["name"]
            put(f"meas__{name}_r{r}", raw[:, j], "raw ancilla measurement; accumulates without reset")
            put(f"syn__{name}_r{r}", raw[:, j] ^ prev.get(q, False), "per-round syndrome m_r xor m_(r-1)")
            prev[q] = raw[:, j]
        else:
            x, y = ideal.get_final_qubit_coordinates()[q][:2]
            put(f"data__q{int(x)}_{int(y)}", raw[:, j], "final data-qubit measurement")
    g = grid.astype(np.int16)
    # Per-stabilizer temporal statistics.
    for s, st in enumerate(stabs):
        row, n = g[:, s, :], st["name"]
        fired = row.any(1)
        idx = np.arange(T)
        put(f"fires__{n}", row.sum(1), "count of detection events on this stabilizer")
        put(f"first__{n}", np.where(fired, np.argmax(row, 1), -1), "first round fired, -1 if never")
        put(f"last__{n}", np.where(fired, T - 1 - np.argmax(row[:, ::-1], 1), -1), "last round fired, -1 if never")
        put(f"lag1__{n}", (row[:, 1:] * row[:, :-1]).sum(1), "pairs fired in rounds t, t+1")
        put(f"lag2__{n}", (row[:, 2:] * row[:, :-2]).sum(1),
            "pairs fired in rounds t, t+2: no-reset readout-error signature")
        run = np.zeros(shots, np.int16); best = np.zeros(shots, np.int16)
        for t in idx:
            run = (run + 1) * row[:, t]; best = np.maximum(best, run)
        put(f"maxrun__{n}", best, "longest streak of consecutive rounds fired")
    # Per-round statistics.
    xs = np.array([st["kind"] == "X" for st in stabs])
    a, b = np.array([i for i, _ in adj]), np.array([j for _, j in adj])
    w = g.sum(1)
    for t in range(T):
        put(f"w__t{t}", w[:, t], "detection events in round t")
        put(f"wX__t{t}", g[:, xs, t].sum(1), "X-stabilizer events in round t")
        put(f"wZ__t{t}", g[:, ~xs, t].sum(1), "Z-stabilizer events in round t")
        put(f"space__t{t}", (g[:, a, t] * g[:, b, t]).sum(1),
            "adjacent stabilizer pairs firing together in round t (data-error signature)")
        if t + 1 < T:
            put(f"diag__t{t}", (g[:, a, t] * g[:, b, t + 1] + g[:, b, t] * g[:, a, t + 1]).sum(1),
                "adjacent pairs at t and t+1 (mid-round / hook-error signature)")
    # Whole-shot summaries.
    tt = np.arange(T) - (T - 1) / 2
    wc = w - w.mean(1, keepdims=True)
    den = (wc ** 2).sum(1)
    put("weight_total", w.sum(1), "total detection events")
    put("weight_X", g[:, xs].sum((1, 2)), "total X-stabilizer events")
    put("weight_Z", g[:, ~xs].sum((1, 2)), "total Z-stabilizer events")
    put("lag1_total", (g[:, :, 1:] * g[:, :, :-1]).sum((1, 2)), "sum of lag1 over stabilizers")
    put("lag2_total", (g[:, :, 2:] * g[:, :, :-2]).sum((1, 2)), "sum of lag2 over stabilizers")
    put("space_total", (g[:, a] * g[:, b]).sum((1, 2)), "sum of space over rounds")
    put("diag_total", (g[:, a, :-1] * g[:, b, 1:] + g[:, b, :-1] * g[:, a, 1:]).sum((1, 2)), "sum of diag over rounds")
    put("burst_max", w.max(1), "largest per-round weight")
    put("rounds_active", (w > 0).sum(1), "rounds with at least one event")
    put("weight_trend", (w * tt).sum(1) / max((tt ** 2).sum(), 1e-9),
        "least-squares slope of per-round weight vs round (leakage grows it)")
    put("weight_autocorr1", np.where(den > 0, (wc[:, 1:] * wc[:, :-1]).sum(1) / np.where(den > 0, den, 1), 0.0),
        "lag-1 autocorrelation of per-round weight")
    # Baseline decoders and label.
    pred, wt = matcher.decode_batch(dets, return_weights=True, enable_correlations=True)
    pred_u = matcher_uncorr.decode_batch(dets)
    put("pm_pred", pred[:, 0], "correlated MWPM predicted logical flip")
    put("pm_weight", wt, "correlated MWPM solution weight (confidence proxy)")
    put("pm_uncorr_pred", pred_u[:, 0], "uncorrelated MWPM predicted logical flip")
    put("pm_disagree", pred[:, 0] != pred_u[:, 0], "correlated and uncorrelated MWPM disagree")
    put("pm_correct", pred[:, 0] == obs[:, 0], "correlated MWPM was right")
    put("obs_flip", obs[:, 0], "LABEL: true logical observable flip")
    return cols, doc


def write_table(path, raw, ideal, decoder_circuit, basis, rounds, meta, manifest):
    """Features + label for one circuit's shots -> Parquet; manifest gets a summary row."""
    import pandas as pd
    shots = len(raw)
    dets, obs = ideal.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    cols, doc = features(raw, dets, obs, ideal, basis, rounds,
                         make_matcher(decoder_circuit), make_matcher(decoder_circuit, correlated=False))
    df = pd.DataFrame({**{k: [v] * shots for k, v in meta.items()}, "shot": np.arange(shots), **cols})
    df.to_parquet(path, index=False)
    errors = int((~cols["pm_correct"]).sum())
    manifest["files"].append(dict(**meta, file=Path(path).name, shots=shots, columns=df.shape[1],
        pm_logical_error_prob=errors / shots, pm_ci95=list(wilson_interval(errors, shots)),
        mean_weight=float(cols["weight_total"].mean())))
    manifest["columns"].update(doc)
    print(f"{Path(path).name}: {df.shape[1]} columns, P_L(MWPM)={errors / shots:.4f}")


def suite_and_layout(readout_error, durations_ns, leakage):
    cal_r = garnet_snapshot(readout_error=readout_error, t2_us=GARNET_MEDIANS["t2_ramsey_us"], durations_ns=durations_ns)
    cal_e = garnet_snapshot(readout_error=readout_error, t2_us=GARNET_MEDIANS["t2_echo_us"], durations_ns=durations_ns)
    placements = garnet_placements(make_stim_circuit(3, 2, "Z", True, gate_basis="cz"))
    # Uniform calibration: every placement scores the same, so take the first.
    return models(cal_r, cal_e, leakage), placements, cal_r


def generate(out, *, shots, rounds_list, bases, model_names, seed, readout_error, durations_ns, leakage):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    suite, placements, cal = suite_and_layout(readout_error, durations_ns, leakage)
    layout = placements[0]
    manifest = dict(device="IQM Garnet (simulated, Crystal 20)", distance=3, no_reset=True, gate_basis="cz",
        qpu_jobs=0, garnet_medians=GARNET_MEDIANS, calibration_metadata=cal.metadata,
        placements_found=len(placements), layout={str(k): v for k, v in layout.items()},
        swaps=0, files=[], columns={})
    for name in model_names:
        model = suite[name]
        for basis in bases:
            for rounds in rounds_list:
                ideal = make_stim_circuit(3, rounds, basis, True, gate_basis="cz")
                circuit = add_noise(ideal, model, qubit_map=layout)
                s = zlib.crc32(f"{seed}/{name}/{basis}/{rounds}".encode())
                if model.leakage:
                    raw, _ = sample_leakage(circuit, model, shots, seed=s, qubit_map=layout)
                else:
                    raw = circuit.compile_sampler(seed=s).sample(shots)
                write_table(out / f"{name}_{basis}_r{rounds}.parquet", raw, ideal, circuit, basis, rounds,
                            dict(source="simulation", model=name, basis=basis, rounds=rounds, distance=3, seed=s),
                            manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_token(env_path=".env"):
    token = os.environ.get("IQM_TOKEN")
    if not token and Path(env_path).exists():
        for line in Path(env_path).read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "IQM_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        raise ValueError("No IQM_TOKEN in the environment or .env.")
    return token


def connect_garnet():
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(SERVER_URL, quantum_computer="garnet", token=load_token()).get_backend()


def sqrt_y_form(circuit: stim.Circuit) -> stim.Circuit:
    """Same circuit with each H as one sqrt(Y)^(+-1): half the native single-qubit gates.

    H = SQRT_Y.Z = Z.SQRT_Y_DAG (up to phase). Alternating per qubit, each leftover
    Z meets the next one through CZs, Z-basis measurements and |0> resets, all of
    which commute with Z, so the measurement record and detectors are unchanged.
    """
    out, seen = stim.Circuit(), Counter()

    def hadamard(q):
        out.append("SQRT_Y" if seen[q] % 2 == 0 else "SQRT_Y_DAG", [q])
        seen[q] += 1
    for inst in circuit.flattened():
        qs = [t.value for t in inst.targets_copy()]
        if inst.name == "H":
            for q in qs:
                hadamard(q)
        elif inst.name == "RX":
            out.append("R", qs)
            for q in qs:
                hadamard(q)
        elif inst.name == "MX":
            for q in qs:
                hadamard(q)
            out.append("M", qs)
        elif inst.name in {"R", "M", "CZ", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}:
            out.append(inst)
        else:
            raise ValueError(f"sqrt_y_form does not handle {inst.name}")
    return out


def retick(circuit: stim.Circuit, barriers: str) -> stim.Circuit:
    """TICKs become hardware barriers. all: every moment (default schedule);
    measure: only around measurements, so gates run ASAP between readouts but never
    overlap one; none: no barriers (the next round can overlap ancilla readout).
    """
    if barriers == "all":
        return circuit
    if barriers not in {"measure", "none"}:
        raise ValueError("barriers must be all, measure or none")
    out = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == "TICK":
            continue
        fence = barriers == "measure" and inst.name in {"M", "MX", "MR", "MRX"}
        if fence:
            out.append("TICK")
        out.append(inst)
        if fence:
            out.append("TICK")
    return out


def prepare_garnet(backend, rounds, basis, layout, *, sqrt_y=False, barriers="all"):
    """Compile with routing off; refuse resets, extra/non-native CZs or a changed record.

    sqrt_y: H -> one sqrt(Y) gate each (sqrt_y_form). barriers: see retick.
    Adjacent single-qubit gates are merged within barrier segments. Guards run
    afterwards; the returned ideal circuit (for decoding) has the identical record.
    """
    from qiskit import transpile
    from challenge.stim_to_qiskit import stim_to_qiskit
    ideal = make_stim_circuit(3, rounds, basis, True, gate_basis="cz")
    live = [(backend.index_to_qubit_name(a), backend.index_to_qubit_name(b))
            for a, b in backend.coupling_map.get_edges()]
    validate_layout(ideal, layout, live)  # Live couplers are authoritative.
    qc, dense, _ = stim_to_qiskit(retick(sqrt_y_form(ideal) if sqrt_y else ideal, barriers))
    # target=, not backend=: IQM's default scheduling plugin relabels qubits off the couplers.
    compiled = transpile(qc, target=backend.target, routing_method="none", optimization_level=1, seed_transpiler=7,
                         initial_layout=[backend.qubit_name_to_index(layout[q]) for q in sorted(dense)])
    if sqrt_y:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition
        compiled = PassManager([Optimize1qGatesDecomposition(target=backend.target)]).run(compiled)
    ops = compiled.count_ops()
    expected = sum(len(i.targets_copy()) // 2 for i in ideal.flattened() if i.name == "CZ")
    twoq = [i for i in compiled.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"]
    edges = {frozenset(e) for e in backend.coupling_map.get_edges()}
    if ops.get("reset", 0) or len(twoq) != expected or any(i.operation.name != "cz" for i in twoq):
        raise ValueError(f"Compilation changed the circuit: {dict(ops)}")
    if any(frozenset(compiled.find_bit(q).index for q in i.qubits) not in edges for i in twoq):
        raise ValueError("Compiled CZ is not on a live coupler.")
    if ops.get("measure", 0) != ideal.num_measurements:
        raise ValueError("Compilation changed the measurement record.")
    compiled.name = f"garnet_d3_{basis}_r{rounds}"
    return ideal, compiled


def memory_to_raw(memory, num_measurements):
    """Qiskit per-shot strings (clbit 0 rightmost) -> Stim measurement order."""
    bits = np.array([list(s.replace(" ", "")[::-1]) for s in memory])
    if bits.shape[1:] != (num_measurements,):
        raise ValueError(f"Expected {num_measurements} bits per shot, got {bits.shape}.")
    return bits == "1"


def ledger(entry, root=HW_DIR):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def live_calibration(backend):
    """Numeric calibration observations of the live device, keyed by IQM dut_field (read-only)."""
    import logging
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            found = backend.client.get_calibration_quality_metrics()
        finally:
            logging.disable(logging.NOTSET)
    flat = {}

    def walk(x):
        if hasattr(x, "dut_field"):
            if isinstance(x.value, numbers.Real) and not isinstance(x.value, bool):
                flat[x.dut_field] = float(x.value)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
    walk(dict(found) if isinstance(found, dict) else vars(found))
    return flat


def placement_cost(layout, ideal, cal, *, dd=False):
    """Expected error budget of one placement from live calibration; lower is better.

    ponytail: additive first-order budget (CZ, 1q, readout, ancilla leakage and
    non-QND readout, data dephasing during readout); a calibrated DEM sim per
    placement would rank more faithfully.
    """
    def get(key):
        if key not in cal:
            raise KeyError(f"Missing live calibration value {key}")
        return cal[key]

    def cz_fidelity(a, b):
        for x, y in ((a, b), (b, a)):
            k = f"metrics.irb.cz.slepian_crf.{x}__{y}.fidelity:par=d2"
            if k in cal:
                return cal[k]
        raise KeyError(f"No CZ fidelity for {a}-{b}")
    measured, one_q = Counter(), Counter()
    for inst in ideal.flattened():
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name in {"M", "MX"}:
            measured.update(qs)
        elif inst.name in {"H", "S", "S_DAG", "X", "Y", "Z"}:
            one_q.update(qs)
    rounds = max(measured.values())
    cost = sum(1 - cz_fidelity(layout[a], layout[b]) for a, b in interaction_pairs(ideal))
    t2 = "t2_echo_time" if dd else "t2_time"
    for q, n in measured.items():
        name = layout[q]
        cost += n * get(f"metrics.ssro.measure.constant.{name}.assignment_error")
        cost += one_q[q] * (1 - get(f"metrics.rb.prx.drag_crf_sx.{name}.fidelity:par=d2"))
        if n > 1:  # Ancilla: measured every round and never reset.
            cost += n * (get(f"metrics.rrc.measure.constant.{name}.leakage")
                         + 1 - get(f"metrics.rrc.measure.constant.{name}.qndness"))
        else:  # Data: idles through every ancilla readout.
            t = get(f"gates.measure.constant.{name}.duration")
            cost += rounds * t / get(f"characterization.model.{name}.{t2}") / 2
    return cost


def run_garnet(*, rounds_list, bases, shots, max_shots, submit, backend=None, timeout=3600,
               sqrt_y=False, barriers="all", placement="first", dd=False):
    """Dry run by default. With submit=True, one Garnet job holding every circuit.

    placement: "first" (fixed, as before) or "best" (lowest live-calibration cost).
    dd: IQM server-side dynamical decoupling.
    """
    configs = [(b, r) for b in bases for r in rounds_list]
    total = shots * len(configs)
    if not 1 <= shots <= GARNET_MAX_SHOTS_PER_CIRCUIT or len(configs) > GARNET_MAX_CIRCUITS:
        raise ValueError(f"Garnet allows 1..{GARNET_MAX_SHOTS_PER_CIRCUIT} shots and {GARNET_MAX_CIRCUITS} circuits.")
    if total > max_shots:
        raise ValueError(f"{total} total shots exceeds --max-shots {max_shots}; nothing submitted.")
    backend = backend or connect_garnet()
    candidates = garnet_placements(make_stim_circuit(3, 2, "Z", True, gate_basis="cz"))
    costs = []
    cal = live_calibration(backend) if hasattr(backend, "retrieve_job") else {}  # Real device only.
    if placement == "best":
        ref = make_stim_circuit(3, max(rounds_list), "Z", True, gate_basis="cz")
        costs = [placement_cost(c, ref, cal, dd=dd) for c in candidates]
        layout = candidates[int(np.argmin(costs))]
        print("placement costs:", [round(c, 3) for c in costs], "-> using", int(np.argmin(costs)))
    elif placement == "first":
        layout = candidates[0]
    else:
        raise ValueError("placement must be 'first' or 'best'")
    prepared = [prepare_garnet(backend, r, b, layout, sqrt_y=sqrt_y, barriers=barriers) for b, r in configs]
    for (b, r), (_, c) in zip(configs, prepared):
        print(f"{c.name}: depth {c.depth()}, {dict(c.count_ops())}")
    print(f"{len(configs)} circuits x {shots} shots = {total} total (cap {max_shots}); layout {layout}")
    if not submit:
        print("Dry run: pass --submit to send this job.")
        return None
    fake = not hasattr(backend, "retrieve_job")
    options = {"memory": True} if fake else {}
    if dd:
        if fake:
            raise ValueError("DD is a server-side option; the fake backend cannot apply it.")
        from iqm.iqm_client import CircuitCompilationOptions, DDMode
        options["circuit_compilation_options"] = CircuitCompilationOptions(dd_mode=DDMode.ENABLED)
    variant = dict(sqrt_y=sqrt_y, barriers=barriers, placement=placement, dd=dd, placement_costs=costs,
                   layout_calibration={k: v for k, v in cal.items() if any(f".{n}." in k or f".{n}__" in k or f"__{n}." in k
                                                                            for n in layout.values())})
    job = backend.run([c for _, c in prepared], shots=shots, **options)
    job_id, root = job.job_id(), FAKE_DIR if fake else HW_DIR
    ledger(dict(job_id=job_id, device=backend.name, fake=fake, shots=shots, circuits=[c.name for _, c in prepared],
                configs=configs, layout={str(k): v for k, v in layout.items()},
                sqrt_y=sqrt_y, barriers=barriers, placement=placement, dd=dd,
                submitted_utc=datetime.now(timezone.utc).isoformat()), root)
    print(f"Submitted job {job_id}; recorded in {root / 'ledger.jsonl'}")
    return save_job(job, job_id, configs, layout, fake=fake, timeout=timeout, variant=variant)


def save_job(job, job_id, configs, layout, *, fake=False, timeout=3600, variant=None):
    """Raw per-shot records first (never edited), then the feature tables."""
    result = job.result() if fake else job.result(timeout=timeout)
    out = (FAKE_DIR if fake else HW_DIR) / f"{datetime.now(timezone.utc):%Y%m%d}_{job_id}"
    out.mkdir(parents=True, exist_ok=True)
    cal_id = str(getattr(result.results[0], "calibration_set_id", None))
    suite, _, cal = suite_and_layout(0.02, {"1q": 20, "2q": 80, "measure": 700, "reset": 1000}, 1e-3)
    manifest = dict(device="IQM Garnet hardware" if not fake else "IQMFakeGarnet (local simulator)",
                    job_id=job_id, calibration_set_id=cal_id, layout={str(k): v for k, v in layout.items()},
                    decoder_model="garnet_coherence (simulated medians; not fitted to this data)",
                    variant=variant or dict(sqrt_y=False, barriers="all", placement="first", dd=False), files=[], columns={})
    for i, (basis, rounds) in enumerate(configs):
        ideal = make_stim_circuit(3, rounds, basis, True, gate_basis="cz")
        if fake:  # IQMFakeBackend drops memory=True: counts only, shot order lost.
            memory = [s for s, n in result.get_counts(i).items() for _ in range(n)]
        else:
            memory = result.get_memory(i)
        raw = memory_to_raw(memory, ideal.num_measurements)
        np.save(out / f"raw_{basis}_r{rounds}.npy", raw)
        write_table(out / f"hw_{basis}_r{rounds}.parquet", raw, ideal,
                    add_noise(ideal, suite["garnet_coherence"], qubit_map=layout), basis, rounds,
                    dict(source="hardware" if not fake else "fake_garnet", model="hardware", basis=basis,
                         rounds=rounds, distance=3, seed=-1, job_id=job_id, calibration_set_id=cal_id), manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def retrieve_garnet(job_id):
    entry = next(json.loads(l) for l in (HW_DIR / "ledger.jsonl").read_text().splitlines()
                 if json.loads(l)["job_id"] == job_id)
    layout = {int(k): v for k, v in entry["layout"].items()}
    return save_job(connect_garnet().retrieve_job(job_id), job_id, [tuple(c) for c in entry["configs"]], layout)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/garnet_d3")
    p.add_argument("--shots", type=int, default=20000)
    p.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--bases", nargs="+", default=["Z", "X"])
    p.add_argument("--models", nargs="+", default=["garnet_pauli", "garnet_coherence",
                                                   "garnet_coherence_echo", "garnet_leakage"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--readout-error", type=float, default=0.02, help="ASSUMED: not on the Resonance page")
    p.add_argument("--one-q-ns", type=float, default=20, help="ASSUMED")
    p.add_argument("--two-q-ns", type=float, default=80, help="ASSUMED")
    p.add_argument("--measure-ns", type=float, default=700, help="ASSUMED")
    p.add_argument("--leakage", type=float, default=1e-3, help="ASSUMED per participant per CZ")
    p.add_argument("--hardware", action="store_true", help="real Garnet; dry run unless --submit")
    p.add_argument("--fake", action="store_true", help="with --hardware: IQMFakeGarnet, local, no network")
    p.add_argument("--submit", action="store_true", help="actually send the job")
    p.add_argument("--max-shots", type=int, default=1000, help="cap on circuits x shots per job")
    p.add_argument("--retrieve", metavar="JOB_ID", help="re-pull a ledgered job by id")
    p.add_argument("--merge-1q", action="store_true", help="shorthand for --sqrt-y --barriers none (jobs B-D)")
    p.add_argument("--sqrt-y", action="store_true", help="H -> one sqrt(Y) gate each")
    p.add_argument("--barriers", choices=["all", "measure", "none"], default="all")
    p.add_argument("--placement", choices=["first", "best"], default="first", help="best: live-calibration cost")
    p.add_argument("--dd", action="store_true", help="IQM dynamical decoupling")
    a = p.parse_args(argv)
    if a.retrieve:
        return retrieve_garnet(a.retrieve)
    if a.hardware:
        backend = None
        if a.fake:
            from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
            backend = IQMFakeGarnet()
        return run_garnet(rounds_list=a.rounds, bases=a.bases, shots=a.shots, max_shots=a.max_shots,
                          submit=a.submit, backend=backend, sqrt_y=a.sqrt_y or a.merge_1q,
                          barriers="none" if a.merge_1q else a.barriers, placement=a.placement, dd=a.dd)
    generate(a.out, shots=a.shots, rounds_list=a.rounds, bases=a.bases, model_names=a.models, seed=a.seed,
             readout_error=a.readout_error, leakage=a.leakage,
             durations_ns={"1q": a.one_q_ns, "2q": a.two_q_ns, "measure": a.measure_ns, "reset": 1000})


if __name__ == "__main__":
    main()
