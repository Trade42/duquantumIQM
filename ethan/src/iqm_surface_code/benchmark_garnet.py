"""Explicit live Garnet benchmark. Imports/tests never contact hardware.

Credential contents never enter manifests. A prepared run submits only with
--submit; --resume retrieves existing job IDs without submitting again.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time


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


def plot_results(output, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1,2,figsize=(10,4.3),sharey=True,layout="constrained")
    for ax,basis in zip(axes,("X","Z")):
        for dd,color,offset,label in (("off","#bb4f2d",-.045,"DD off"),("xx","#146b9b",.045,"XX DD")):
            cases=sorted([c for c in manifest["cases"] if c["basis"]==basis and c["dd"]==dd
                          and "analysis" in c],key=lambda c:c["rounds"])
            if not cases:
                continue
            x=np.array([c["rounds"] for c in cases],dtype=float)
            y=np.array([c["analysis"]["probability"] for c in cases])*100
            lower=y-np.array([c["analysis"]["ci95"][0] for c in cases])*100
            upper=np.array([c["analysis"]["ci95"][1] for c in cases])*100-y
            ax.errorbar(x+offset,y,yerr=np.array([lower,upper]),marker="o",capsize=4,
                        color=color,label=label,linewidth=1.8)
        ax.set_title(f"{basis}-basis memory",fontweight="bold")
        ax.set_xlabel("Syndrome rounds")
        ax.set_xticks([1,3,5])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y",alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    axes[0].set_ylabel("Decoded logical failure (%)")
    shots=sum(c["analysis"]["shots"] for c in manifest["cases"] if "analysis" in c)
    fig.suptitle(f"Live IQM Garnet · distance 3 · {shots:,} total shots\n"
                 "Error bars: 95% Wilson intervals",fontsize=12)
    fig.savefig(output/"comparison.png",dpi=180)
    plt.close(fig)


def update_report(output, manifest):
    from .decoding import fit_ler
    manifest["fits"] = []
    for basis in ("X", "Z"):
        for dd in ("off", "xx"):
            cases = sorted([c for c in manifest["cases"] if c["basis"] == basis and c["dd"] == dd
                            and "analysis" in c], key=lambda c:c["rounds"])
            if len(cases) >= 3:
                fit = fit_ler(
                    [c["rounds"] for c in cases], [c["analysis"]["errors"] for c in cases],
                    [c["analysis"]["shots"] for c in cases], fit_spam=True)
                # Diagnostic only: constrained amplitude and just three points
                # make the chi-square reference approximate, especially at A=1.
                from scipy.special import xlogy
                deviance = 0.
                for c, p in zip(cases,fit["fitted_probabilities"]):
                    k,n=c["analysis"]["errors"],c["analysis"]["shots"]
                    deviance += 2*(xlogy(k,k/n)+xlogy(n-k,1-k/n)-xlogy(k,p)-xlogy(n-k,1-p))
                fit["binomial_deviance"] = float(deviance)
                manifest["fits"].append(dict(basis=basis, dd=dd, **fit))
    comparisons = []
    for basis in ("X","Z"):
        for rounds in (1,3,5):
            pair = {c["dd"]:c["analysis"] for c in manifest["cases"]
                    if c["basis"]==basis and c["rounds"]==rounds and "analysis" in c}
            if set(pair) != {"off","xx"}:
                continue
            a,b=pair["off"],pair["xx"]
            reduction=a["probability"]-b["probability"]
            # Independent fixed-shot Newcombe interval for p_off - p_DD.
            low = reduction-math.hypot(a["probability"]-a["ci95"][0],b["ci95"][1]-b["probability"])
            high = reduction+math.hypot(a["ci95"][1]-a["probability"],b["probability"]-b["ci95"][0])
            comparisons.append(dict(basis=basis,rounds=rounds,absolute_reduction=reduction,
                ci95_absolute_reduction=[low,high],relative_reduction=reduction/a["probability"] if a["probability"] else None))
    manifest["comparisons"] = comparisons
    write_json(output / "manifest.json", manifest)
    lines = ["# Live IQM Garnet surface-code benchmark", "",
             f"Calibration: `{manifest['calibration_set_id']}`. Distance 3, 17 qubits, native CZ; no routing SWAPs.",
             "Fixed physical layout and calibration-derived decoder weights across DD conditions.",
             "DD uses the IQM server compiler on the nine data qubits. Server accepted XX strategy; inserted pulse count unavailable.",
             "", "| Basis | Rounds | DD | Errors / shots | Logical failure (95% CI) | Decode shots/s |",
             "|---|---:|---|---:|---|---:|"]
    for c in sorted(manifest["cases"], key=lambda c:(c["basis"],c["rounds"],c["dd"])):
        if "analysis" not in c:
            lines.append(f"| {c['basis']} | {c['rounds']} | {c['dd']} | {c['status']} | — | — |")
            continue
        a = c["analysis"]
        lo, hi = a["ci95"]
        lines.append(f"| {c['basis']} | {c['rounds']} | {c['dd']} | {a['errors']} / {a['shots']} | "
                     f"{100*a['probability']:.2f}% ({100*lo:.2f}–{100*hi:.2f}%) | "
                     f"{a['timing']['decode_shots_per_second']:,.0f} |")
    lines += ["", "DD effect (off minus XX; positive means improvement). Pointwise 95% independent-shot Newcombe intervals; "
              "not adjusted for six comparisons or device drift."]
    for c in comparisons:
        low,high=c["ci95_absolute_reduction"]
        relative = "undefined" if c["relative_reduction"] is None else f"{100*c['relative_reduction']:.1f}%"
        lines.append(f"- {c['basis']}, {c['rounds']} rounds: {100*c['absolute_reduction']:.2f} percentage points "
                     f"(95% CI {100*low:.2f}–{100*high:.2f}); relative reduction {relative}.")
    lines += ["", "Per-round rates below use a binomial multi-round fit with a separate preparation/readout amplitude.",
              "Fit assumes stationary independent logical flips; drift, leakage and fit misspecification can violate it."]
    for f in manifest["fits"]:
        lo,hi=f["ci95_profile"]
        lines.append(f"- {f['basis']} DD={f['dd']}: {100*f['epsilon_per_round']:.3f}%/round "
                     f"(95% profile interval {100*lo:.3f}–{100*hi:.3f}%), amplitude {f['amplitude']:.4f}, "
                     f"binomial deviance {f['binomial_deviance']:.2f} on nominal 1 degree of freedom.")
    for f in manifest["fits"]:
        if f["binomial_deviance"]>6.63:
            lines += ["", f"The {f['basis']}/{f['dd']} stationary fit is strained (deviance {f['binomial_deviance']:.2f}); "
                      "prioritize directly measured failure probabilities over its fitted per-round rate. "
                      "Profile intervals exclude model error and drift."]
    lines += ["", "| Circuit | Server compile (s) | Server execution interval (s) | Client wall (s) |",
              "|---|---:|---:|---:|"]
    for c in manifest["cases"]:
        if "analysis" in c:
            t=c.get("server_timing",{})
            lines.append(f"| {c['id']} | {t.get('compilation_s',float('nan')):.3f} | "
                         f"{t.get('execution_s',float('nan')):.3f} | {c['client_wall_s']:.3f} |")
    lines += ["", "Raw counts, immutable Stim/DEM/QPY circuits, calibration observations, job IDs/status/timelines and "
              "measurement arrays are saved alongside this report. A received DD option alone does not prove pulses were inserted.",
              "Decoder is a calibrated Pauli approximation; native R decomposition, pulse timing, leakage, QND failures and "
              "DD-induced noise changes are not fully captured. CPU timing is warmed batch throughput, not streaming latency.",
              "Client wall time includes submission, queue, compilation, execution, polling and result retrieval; "
              "only server timeline intervals may be labeled device processing time."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if manifest["cases"] and all("analysis" in c for c in manifest["cases"]):
        plot_results(output,manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--shots", type=int, default=1000, help="Fixed shots per circuit; default total 12,000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", type=Path, help="Retrieve existing jobs only; never resubmit")
    parser.add_argument("--wait-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    if args.shots < 1 or args.shots > 5000 or args.wait_seconds < 0:
        parser.error("Require 1..5000 shots per circuit and nonnegative wait.")
    secret = read_token(args.token_file)
    output = args.resume or args.output or Path("results/garnet")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from .run_on_hardware import connect_backend, prepare_qiskit
    from .surface_code import make_stim_circuit
    from .noise_models import NoiseModel
    from .dynamical_decoupling import iqm_compilation_options
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    import stim
    import numpy as np
    from qiskit import qpy
    try:
        backend = connect_backend(secret, "garnet")
        if args.resume:
            manifest = json.loads((output/"manifest.json").read_text())
            backend = IQMBackend(backend.client, calibration_set_id=manifest["calibration_set_id"])
        else:
            output.mkdir(parents=True, exist_ok=False)
            architecture = backend.architecture.model_dump(mode="json")
            cal_id = backend.architecture.calibration_set_id
            backend = IQMBackend(backend.client, calibration_set_id=cal_id)
            quality = backend.client.get_quality_metric_set(cal_id).model_dump(mode="json")
            pulse = backend.client.get_calibration_set(cal_id).model_dump(mode="json")
            for name, data in (("architecture",architecture),("quality_metrics",quality),("pulse_calibration",pulse)):
                write_json(output/f"{name}.json", data)
            calibration = calibration_from_observations(architecture, quality, pulse)
            calibration.save(output/"calibration.json")
            couplings = [(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b))
                         for a,b in backend.coupling_map.get_edges()]
            mapping, ranking = select_native_layout(make_stim_circuit(3,3,gate_basis="cz"),
                couplings, backend.physical_qubits, calibration)
            manifest = dict(device="garnet", distance=3, calibration_set_id=str(cal_id),
                created_utc=datetime.now(timezone.utc).isoformat(), placement=mapping, ranking=ranking,
                total_requested_shots=args.shots*12,
                runtime=dict(python=platform.python_version(),os=platform.platform(),
                             processor=platform.processor(),logical_cpus=os.cpu_count()),
                versions={p:importlib.metadata.version(p) for p in ("iqm-client","qiskit","stim","pymatching","numpy")},
                cases=[], assumptions=calibration.metadata)
            model = NoiseModel(name="calibrated_pauli",calibration=calibration,idle=0)
            # Randomize pair order and which DD condition is first in each pair.
            rng = random.Random(1901)
            pairs = [(basis,r) for basis in ("X","Z") for r in (1,3,5)]
            rng.shuffle(pairs)
            for basis, rounds in pairs:
                modes = ["off","xx"]
                rng.shuffle(modes)
                for dd in modes:
                    case_id = f"{basis}_r{rounds}_{dd}"
                    start = time.perf_counter()
                    prepared = prepare_qiskit(backend,3,rounds,basis,calibration=calibration,
                        noise=model,qubit_map=mapping,dd=dd)
                    prepared["circuit"].name = case_id
                    circuit_path = output/f"{case_id}.stim"
                    prepared["stim_circuit"].to_file(circuit_path)
                    prepared["decoder_circuit"].to_file(output/f"{case_id}.decoder.stim")
                    prepared["decoder_circuit"].detector_error_model(decompose_errors=True,
                        approximate_disjoint_errors=True).to_file(output/f"{case_id}.dem")
                    with (output/f"{case_id}.qpy").open("wb") as stream:
                        qpy.dump(prepared["circuit"],stream)
                    manifest["cases"].append(dict(id=case_id,basis=basis,rounds=rounds,dd=dd,
                        shots=args.shots,status="prepared",prepare_s=time.perf_counter()-start,
                        hardware_report=prepared["report"],
                        stim_sha256=hashlib.sha256(circuit_path.read_bytes()).hexdigest()))
            update_report(output, manifest)
            print(f"Prepared {len(manifest['cases'])} circuits, {manifest['total_requested_shots']} shots: {output}",flush=True)
            if not args.submit:
                return
            for case in manifest["cases"]:
                with (output/f"{case['id']}.qpy").open("rb") as stream:
                    circuit = qpy.load(stream)[0]
                options = iqm_compilation_options(case["dd"],target_names=case["hardware_report"]["dd"]["target_qubits"])
                case["submitted_utc"] = datetime.now(timezone.utc).isoformat()
                start = time.perf_counter()
                # Record the job ID before any subsequent query or result wait.
                job = backend.run(circuit,shots=case["shots"],circuit_compilation_options=options)
                case.update(job_id=job.job_id(),status="submitted",submit_s=time.perf_counter()-start)
                update_report(output,manifest)
                print(f"Submitted {case['id']}: {case['job_id']}",flush=True)
        deadline = time.monotonic()+args.wait_seconds
        pending = [c for c in manifest["cases"] if c.get("job_id") and "analysis" not in c
                   and c["status"] not in {"ERROR","CANCELLED"}]
        jobs = {c["id"]:backend.retrieve_job(c["job_id"]) for c in pending}
        while pending:
            for case in pending[:]:
                job = jobs[case["id"]]
                status = job.status().name
                if case["status"] != status:
                    print(f"{case['id']}: {status}",flush=True)
                case["status"] = status
                if status == "DONE":
                    result = job.result(timeout=1)
                    counts = dict(result.get_counts())
                    write_json(output/f"{case['id']}.counts.json",counts)
                    memory = result.get_memory()
                    write_json(output/f"{case['id']}.memory.json",memory)
                    case["server_timeline"] = [t.model_dump(mode="json") for t in result.to_dict().get("timeline",[])]
                    case["server_timing"] = server_timing(case["server_timeline"])
                    case["result_received_utc"] = datetime.now(timezone.utc).isoformat()
                    case["client_wall_s"] = (datetime.fromisoformat(case["result_received_utc"])
                                             - datetime.fromisoformat(case["submitted_utc"])).total_seconds()
                    ideal = stim.Circuit.from_file(output/f"{case['id']}.stim")
                    decoder = stim.Circuit.from_file(output/f"{case['id']}.decoder.stim")
                    analysis, raw, syndromes = analyze_counts(counts,ideal,decoder,expected_shots=case["shots"],memory=memory)
                    case["analysis"] = analysis
                    np.savez_compressed(output/f"{case['id']}.npz",raw_meas=raw,
                        det_events=syndromes["det_events"],obs_flips=syndromes["obs_flips"])
                    print(f"{case['id']}: {analysis['errors']}/{analysis['shots']} logical failures",flush=True)
                    pending.remove(case)
                elif status in {"ERROR","CANCELLED"}:
                    case["error"] = str(job.error_message()).replace(secret,"[REDACTED]")
                    print(f"{case['id']}: {case['error']}",flush=True)
                    pending.remove(case)
            update_report(output,manifest)
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(min(10,max(0,deadline-time.monotonic())))
        print(f"Report: {output/'report.md'}; pending jobs: {len(pending)}",flush=True)
    except Exception as exc:
        message = str(exc).replace(secret,"[REDACTED]")
        if output.exists():
            write_json(output/"error.json",dict(type=type(exc).__name__,message=message))
        raise SystemExit(f"{type(exc).__name__}: {message}") from None


if __name__ == "__main__":
    main()
