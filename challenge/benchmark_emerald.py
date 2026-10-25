"""Vanilla d3 X/Z surface-code memory on Emerald, with an 18,000-shot ceiling.

Preparation fixes the native embedding, literal circuit, DD-off requests and
uniform-prior lookup tables. Submission uses those exact requests once; collect
retrieves existing jobs. Replay uses saved measurements without SDK or credentials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import stim

from iqm_surface_code.benchmark_garnet import read_token, server_timing, write_json
from iqm_surface_code.benchmark_vanilla_lut import (
    build_with_memory_profile, prepare_vanilla, score_hardware, vanilla_circuit)
from iqm_surface_code.decoders.lookup_table import LookupTableDecoder
from iqm_surface_code.hardware_optimization import native_stim

from .benchmark_headline import challenge_fit

ROUNDS = (1, 2, 3)
SHOT_LIMIT = 18000


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def couplings(backend):
    return sorted({tuple(sorted((backend.index_to_qubit_name(a), backend.index_to_qubit_name(b))))
                   for a, b in backend.coupling_map.get_edges()})


def select_baseline_layout(circuit, edges, qubits):
    """First deterministic exact graph embedding; no calibration ranking."""
    import networkx as nx
    from iqm_surface_code.build_emerald_qubit_rotated import interaction_pairs, validate_layout
    key = lambda name: (int(name[2:]), name) if name.startswith("QB") and name[2:].isdigit() else (0, name)
    hardware, required = nx.Graph(), nx.Graph()
    hardware.add_nodes_from(sorted(qubits, key=key))
    hardware.add_edges_from(sorted(tuple(sorted(e, key=key)) for e in edges))
    required.add_nodes_from(sorted(circuit.get_final_qubit_coordinates()))
    required.add_edges_from(sorted(set(tuple(sorted(e)) for e in interaction_pairs(circuit))))
    iterator = nx.algorithms.isomorphism.GraphMatcher(hardware, required).subgraph_monomorphisms_iter()
    try:
        embedding = next(iterator)
    except StopIteration:
        raise ValueError("No native distance-three surface-code embedding on this architecture.") from None
    mapping = {logical: physical for physical, logical in embedding.items()}
    return mapping, dict(**validate_layout(circuit, mapping, edges),
                         selection="first exact graph embedding in deterministic node order",
                         calibration_weighted=False, networkx_version=nx.__version__)


def validate_request(request, case, manifest, edges):
    """Check the actual request, including DD/reset/heralding and every CZ pair."""
    if str(request["calibration_set_id"]) != manifest["calibration_set_id"]:
        raise ValueError("Request calibration differs from the frozen baseline.")
    if request["shots"] != case["shots"] or len(request["circuits"]) != 1:
        raise ValueError("Request shot/circuit inventory differs from the protocol.")
    if request["dd_mode"] != "disabled" or request["heralding_mode"] != "none" or request.get("active_reset_cycles") not in (None, 0):
        raise ValueError("Vanilla requests require DD, heralding and active reset off.")
    from iqm_surface_code.build_emerald_qubit_rotated import interaction_pairs
    mapping = {int(k): v for k, v in manifest["placement"].items()}
    expected = Counter(frozenset((mapping[a], mapping[b]))
                       for a, b in interaction_pairs(vanilla_circuit(case["rounds"], case["basis"])))
    actual = Counter()
    legal = {frozenset(e) for e in edges}
    for instruction in request["circuits"][0]["instructions"]:
        if instruction["name"] not in {"prx", "cz", "measure", "barrier"}:
            raise ValueError("Unexpected instruction in the native vanilla request.")
        if not set(instruction["locus"]) <= set(mapping.values()):
            raise ValueError("Request addresses qubits outside the selected patch.")
        if instruction["name"] == "cz":
            edge = frozenset(instruction["locus"])
            if edge not in legal:
                raise ValueError("Request contains a non-native CZ.")
            actual[edge] += 1
    if actual != expected:
        raise ValueError("Request CZ interactions differ from the canonical circuit.")


def prepare_baseline(output, backend, *, shots=3000, noise_p=.003, fixture=False):
    if not isinstance(shots, int) or not 1 <= shots <= 3000 or not 0 < noise_p < .1:
        raise ValueError("Require 1..3000 shots per point and 0 < noise_p < 0.1.")
    if len(backend.physical_qubits) != 54 and not fixture:
        raise ValueError("Hardware baseline requires the 54-qubit Emerald architecture.")
    from qiskit import qpy
    from iqm_surface_code.dynamical_decoupling import iqm_compilation_options
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "architecture.json", backend.architecture.model_dump(mode="json"))
    protocol = dict(device="emerald", distance=3, bases=["X", "Z"], rounds=list(ROUNDS),
                    shots_per_point=shots, total_physical_shots=6*shots, shot_limit=SHOT_LIMIT,
                    no_reset=True, dd="disabled", heralding="none", uniform_decoder_p=noise_p,
                    decoder="complete logical-class ML lookup", training_shots=0,
                    qiskit_optimization_level=0, custom_optimization=False,
                    layout="fixed deterministic native graph embedding; no calibration ranking",
                    fit="P_L(r) = (1 - (1 - 2*epsilon_L)**r) / 2; amplitude=1; R=1,2,3",
                    acquisition_order_seed=8193, local_validation_shots=20000)
    write_json(output / "protocol.json", protocol)
    edges = couplings(backend)
    mapping, layout = select_baseline_layout(vanilla_circuit(3, "Z"), edges, backend.physical_qubits)
    manifest = dict(schema_version=1, device="emerald", hardware_eligible=not fixture, distance=3,
                    calibration_set_id=str(backend.architecture.calibration_set_id),
                    physical_qubits=len(backend.physical_qubits), placement=mapping, layout=layout, noise_p=noise_p,
                    edges=edges, protocol_sha256=digest(output/"protocol.json"), created_utc=utc(),
                    charged_shots=0, shot_limit=SHOT_LIMIT, cases=[], fits=[],
                    versions={p: importlib.metadata.version(p) for p in
                              ("iqm-client", "qiskit", "stim", "numpy", "scipy", "pymatching", "networkx")})
    specifications = [(b, r) for b in ("X", "Z") for r in ROUNDS]
    random.Random(8193).shuffle(specifications)
    for basis, rounds in specifications:
        name = f"{basis}_r{rounds}"
        prepared = prepare_vanilla(backend, rounds, basis, mapping, noise_p)
        prepared["compiled"].name = "emerald_vanilla_" + name
        native, columns = native_stim(prepared["compiled"], prepared["ideal"], mapping, backend)
        source = prepared["ideal"]
        arbitrary = np.random.default_rng(9000+rounds).integers(2, size=(64, source.num_measurements), dtype=np.uint8).astype(bool)
        a = source.compile_m2d_converter().convert(measurements=arbitrary, separate_observables=True)
        b = native.compile_m2d_converter().convert(measurements=arbitrary[:, columns], separate_observables=True)
        if not all(np.array_equal(x, y) for x, y in zip(a, b)):
            raise ValueError("Source/native measurement frames differ.")
        source.to_file(output/f"{name}.stim")
        native.to_file(output/f"{name}.native.stim")
        model = prepared["decoder_circuit"]
        model.to_file(output/f"{name}.decoder.stim")
        model.detector_error_model(approximate_disjoint_errors=True).to_file(output/f"{name}.dem")
        with (output/f"{name}.qpy").open("wb") as stream:
            qpy.dump(prepared["compiled"], stream)
        print(f"Preparing {name}: {source.num_detectors} detector bits", flush=True)
        decoder = build_with_memory_profile(model)
        decoder.save(output/f"{name}.lut.npz")
        dets, truth = model.compile_detector_sampler(seed=8200+rounds+(basis=="X")).sample(20000, separate_observables=True)
        failed = int(np.any(decoder.decode_batch(dets) != truth, axis=1).sum())
        if abs(failed/20000-decoder.metadata["model_bayes_failure"]) > .02:
            raise ValueError("Matched-noise validation differs from the modeled Bayes failure.")
        case = dict(id=name, basis=basis, rounds=rounds, shots=shots, status="prepared",
                    native_measurement_columns=columns, hardware_report=prepared["report"],
                    lut=decoder.metadata, local_validation=dict(shots=20000, failures=failed,
                    seed=8200+rounds+(basis=="X")))
        request = backend.create_run_request(prepared["compiled"], shots=shots,
                                             circuit_compilation_options=iqm_compilation_options("off"))
        payload = request.model_dump(mode="json")
        validate_request(payload, case, manifest, edges)
        write_json(output/f"{name}.request.json", payload)
        case["artifact_sha256"] = {f"{name}.{suffix}": digest(output/f"{name}.{suffix}") for suffix in
                                   ("stim", "native.stim", "decoder.stim", "dem", "qpy", "lut.npz", "request.json")}
        manifest["cases"].append(case)
        write_json(output/"manifest.json", manifest)
        del decoder, prepared, dets, truth
        gc.collect()
    manifest["prepared_utc"] = utc()
    manifest["source_dependencies"] = {str(Path(__file__).relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/"): digest(__file__)}
    write_json(output/"manifest.json", manifest)
    return manifest


def verify(output, manifest):
    output = Path(output)
    if digest(output/"protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Frozen baseline protocol changed.")
    if {(c["basis"], c["rounds"]) for c in manifest["cases"]} != {(b, r) for b in ("X", "Z") for r in ROUNDS} or len(manifest["cases"]) != 6:
        raise ValueError("Baseline requires exactly X/Z memory at rounds 1, 2 and 3.")
    for case in manifest["cases"]:
        for relative, expected in case["artifact_sha256"].items():
            if digest(output/relative) != expected:
                raise ValueError(f"Frozen baseline artifact changed: {relative}")
    repository = Path(__file__).resolve().parents[1]
    for relative, expected in manifest.get("source_dependencies", {}).items():
        if digest(repository/relative) != expected:
            raise ValueError(f"Frozen baseline source changed: {relative}")


def reserve_submission(output, manifest, case):
    if case.get("submit_intent_utc") or case.get("job_id"):
        raise ValueError("Submission already attempted; collect the existing job instead.")
    if not manifest["hardware_eligible"]:
        raise ValueError("An offline fixture cannot be submitted as Emerald hardware.")
    charged = sum(c["shots"] for c in manifest["cases"] if c.get("submit_intent_utc"))
    if charged != manifest["charged_shots"] or charged+case["shots"] > min(manifest["shot_limit"], SHOT_LIMIT):
        raise ValueError("Physical-shot ledger mismatch or baseline budget exceeded.")
    case.update(submit_intent_utc=utc(), status="submission_intent")
    manifest["charged_shots"] = charged+case["shots"]
    write_json(Path(output)/"manifest.json", manifest)


def submit_cases(output, manifest, backend):
    from iqm.iqm_client import RunRequest
    from iqm.qiskit_iqm.iqm_job import IQMJob
    output = Path(output)
    verify(output, manifest)
    if str(backend.architecture.calibration_set_id) != manifest["calibration_set_id"]:
        raise ValueError("Backend is not pinned to the prepared calibration.")
    if any(c.get("submit_intent_utc") and not c.get("job_id") for c in manifest["cases"]):
        raise ValueError("Ambiguous submission exists; resolve its job ID before any further submission.")
    for case in manifest["cases"]:
        if case.get("job_id"):
            continue
        payload = json.loads((output/f"{case['id']}.request.json").read_text())
        validate_request(payload, case, manifest, couplings(backend))
        reserve_submission(output, manifest, case)
        circuit_job = backend.client.submit_run_request(RunRequest.model_validate(payload))
        job = IQMJob(backend, circuit_job)
        case.update(job_id=job.job_id(), status="submitted", submitted_utc=utc())
        write_json(output/"manifest.json", manifest)
        print(f"Submitted {case['id']}: {case['job_id']}", flush=True)


def save_report(output, manifest):
    output = Path(output)
    fits = []
    for basis in ("X", "Z"):
        cases = sorted((c for c in manifest["cases"] if c["basis"] == basis), key=lambda c:c["rounds"])
        if len(cases) == 3 and all("analysis" in c for c in cases):
            fits.append(dict(basis=basis, **challenge_fit(ROUNDS, [c["analysis"]["errors"] for c in cases], [c["shots"] for c in cases])))
    manifest["fits"] = fits
    write_json(output/"manifest.json", manifest)
    lines = ["# Vanilla surface-code baseline on IQM Emerald", "",
             "Distance 3, nine data qubits, eight reused ancillas, rounds 1–3 in X/Z memory.",
             "Literal H-CZ-H compilation, optimization level 0, fixed native graph embedding, DD off.",
             f"Uniform p={manifest['noise_p']} circuit prior and complete logical-class lookup; all shots retained.",
             f"Calibration: `{manifest['calibration_set_id']}`. Charged shots: {manifest['charged_shots']:,} / {SHOT_LIMIT:,}.", "",
             "| Basis | Rounds | Failures / shots | Failure probability, 95% Wilson interval |",
             "|---|---:|---:|---:|"]
    for c in sorted(manifest["cases"], key=lambda c:(c["basis"], c["rounds"])):
        if "analysis" in c:
            a=c["analysis"]; lo,hi=a["ci95"]
            lines.append(f"| {c['basis']} | {c['rounds']} | {a['errors']} / {c['shots']} | {100*a['probability']:.3f}% ({100*lo:.3f}–{100*hi:.3f}%) |")
        else:
            lines.append(f"| {c['basis']} | {c['rounds']} | {c['status']} | |")
    lines += ["", "Challenge fit: `P_L(r) = [1 - (1 - 2 epsilon_L)^r]/2`, amplitude fixed to one.",
              "Profile intervals describe the fitted binomial equation. Deviance records round-dependent model mismatch."]
    for fit in fits:
        lo,hi=fit["ci95_profile"]
        lines.append(f"- {fit['basis']}: **{100*fit['epsilon_per_round']:.3f}% LER** ({100*lo:.3f}–{100*hi:.3f}%), deviance {fit['deviance']:.2f} / {fit['nominal_df']} df.")
    lines += ["", "The prior models logical H/CZ circuit faults. Native R synthesis, timing, spatial noise variation, leakage and readout disturbance contribute model mismatch.",
              "Saved records preserve original shot order and source/native frames. Decoder states and requests were fixed before acquisition."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def collect(output, manifest, backend, *, wait_seconds=1800):
    output = Path(output)
    verify(output, manifest)
    jobs = {c["id"]: backend.retrieve_job(c["job_id"]) for c in manifest["cases"]
            if c.get("job_id") and "analysis" not in c and c["status"] not in {"ERROR", "CANCELLED"}}
    deadline = time.monotonic()+wait_seconds
    while jobs:
        for c in manifest["cases"]:
            if c["id"] not in jobs:
                continue
            job = jobs[c["id"]]
            status = job.status().name
            if status != c["status"]:
                print(f"{c['id']}: {status}", flush=True)
            c["status"] = status
            if status == "DONE":
                result = job.result(timeout=1)
                counts, memory = dict(result.get_counts()), result.get_memory()
                write_json(output/f"{c['id']}.counts.json", counts)
                write_json(output/f"{c['id']}.memory.json", memory)
                c["analysis"] = score_hardware(output, c, counts, memory)
                path = output/f"{c['id']}.shots.npz"
                with np.load(path, allow_pickle=False) as arrays:
                    saved = {key: arrays[key].copy() for key in arrays.files}
                saved["native_raw_meas"] = saved["raw_meas"][:, c["native_measurement_columns"]]
                np.savez_compressed(path, **saved)
                c["analysis"]["dataset_sha256"] = digest(path)
                c["server_timeline"] = [t.model_dump(mode="json") for t in result.to_dict().get("timeline", [])]
                c["server_timing"] = server_timing(c["server_timeline"])
                c["collected_utc"] = utc()
                print(f"{c['id']}: {c['analysis']['errors']}/{c['shots']} logical failures", flush=True)
                del jobs[c["id"]]
            elif status in {"ERROR", "CANCELLED"}:
                del jobs[c["id"]]
        save_report(output, manifest)
        if not jobs or time.monotonic() >= deadline:
            break
        time.sleep(min(10, max(0, deadline-time.monotonic())))
    return manifest


def replay(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    verify(output, manifest)
    total = 0
    for c in manifest["cases"]:
        path = output/f"{c['id']}.shots.npz"
        if digest(path) != c["analysis"]["dataset_sha256"]:
            raise ValueError("Saved measurement hash changed.")
        with np.load(path, allow_pickle=False) as arrays:
            dets, truth = arrays["det_events"], arrays["obs_flips"]
            if len(dets) != c["shots"]:
                raise ValueError("Returned shot count differs.")
            if not np.array_equal(arrays["raw_meas"][:, c["native_measurement_columns"]], arrays["native_raw_meas"]):
                raise ValueError("Native measurement columns differ.")
            for suffix, raw_key in (("stim", "raw_meas"), ("native.stim", "native_raw_meas")):
                circuit = stim.Circuit.from_file(output/f"{c['id']}.{suffix}")
                converted = circuit.compile_m2d_converter().convert(measurements=arrays[raw_key], separate_observables=True)
                if not all(np.array_equal(a, b) for a, b in zip(converted, (dets, truth))):
                    raise ValueError("Source/native detector or observable frame differs.")
            decoder = LookupTableDecoder.load(output/f"{c['id']}.lut.npz")
            failed = int(np.any(decoder.decode_batch(dets) != truth, axis=1).sum())
            if failed != c["analysis"]["errors"]:
                raise ValueError("Replayed decoder failures differ.")
        total += len(dets)
    for fit in manifest["fits"]:
        points = sorted((c for c in manifest["cases"] if c["basis"] == fit["basis"]), key=lambda c:c["rounds"])
        actual = challenge_fit(ROUNDS, [c["analysis"]["errors"] for c in points], [c["shots"] for c in points])
        if not np.isclose(actual["epsilon_per_round"], fit["epsilon_per_round"], atol=1e-7, rtol=0):
            raise ValueError("Replayed challenge fit differs.")
    if total != manifest["charged_shots"] or total > SHOT_LIMIT:
        raise ValueError("Shot ledger differs from replayed measurements.")
    return dict(device="emerald", cases=len(manifest["cases"]), physical_shots=total, fits=manifest["fits"])


def export_bundle(output, destination):
    output, destination = Path(output), Path(destination)
    replay(output)
    destination.mkdir(parents=True, exist_ok=False)
    for source in output.iterdir():
        if source.is_file() and source.name != "error.json":
            shutil.copyfile(source, destination/source.name)
    (destination/".gitattributes").write_text("* -text whitespace=cr-at-eol\n", encoding="utf-8")
    replay(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "collect", "replay"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=Path("apikey.jom"))
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=1800)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args(argv)
    if args.action == "replay":
        from examples.reproduce import network_disabled
        with network_disabled():
            print(json.dumps(replay(args.run), indent=2))
        return
    if args.action == "run" and not args.submit:
        parser.error("Hardware acquisition requires --submit.")
    if args.wait_seconds < 0:
        parser.error("Wait seconds must be nonnegative.")
    secret = read_token(args.token_file)
    try:
        from .run_on_hardware import connect_backend
        from iqm.qiskit_iqm.iqm_provider import IQMBackend
        initial = connect_backend(secret, "emerald")
        cal_id = initial.architecture.calibration_set_id if args.action == "prepare" else json.loads((args.run/"manifest.json").read_text())["calibration_set_id"]
        backend = IQMBackend(initial.client, calibration_set_id=cal_id)
        if args.action == "prepare":
            manifest = prepare_baseline(args.run, backend, shots=args.shots)
            write_json(args.run/"quality_metrics.json", backend.client.get_quality_metric_set(cal_id).model_dump(mode="json"))
            write_json(args.run/"pulse_calibration.json", backend.client.get_calibration_set(cal_id).model_dump(mode="json"))
            save_report(args.run, manifest)
        else:
            manifest = json.loads((args.run/"manifest.json").read_text())
            if args.action == "run":
                submit_cases(args.run, manifest, backend)
            collect(args.run, manifest, backend, wait_seconds=args.wait_seconds)
        if args.export:
            export_bundle(args.run, args.export)
    except Exception as exc:
        message = str(exc).replace(secret, "[REDACTED]")
        if args.run.exists():
            write_json(args.run/"error.json", dict(type=type(exc).__name__, message=message))
        raise SystemExit(f"{type(exc).__name__}: {message}") from None


if __name__ == "__main__":
    main()
