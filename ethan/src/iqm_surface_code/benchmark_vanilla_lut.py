"""Challenge vanilla distance-3 memory + complete lookup-table Garnet baseline.

Runs 1/2/3 syndrome rounds in X and Z. DD is disabled. No logical operations,
calibration-weighted placement/decoder, custom pulse optimization or LUT training.
Only explicit --submit sends QPU jobs; --resume retrieves existing IDs only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import threading
import time
import numpy as np
import stim

from .benchmark_garnet import read_token, write_json, select_native_layout, server_timing
from .decoders.lookup_table import build_lookup_table, LookupTableDecoder
from .decoding import decode_syndromes, fit_ler
from .extract_syndromes import extract_syndromes
from .internal_helpers import counts_to_measurement_array
from .noise_models import NoiseModel, add_noise
from .surface_code import make_stim_circuit


def vanilla_circuit(rounds, basis):
    """Canonical Stim schedule, with literal H-CZ-H conversion of each CX layer.

    Retains original ticks and all conversion H gates. Does not call the custom
    H-cancellation/ASAP optimizer. Mandatory IQM native gate synthesis still runs.
    """
    original=make_stim_circuit(3,rounds,basis,no_reset=True,gate_basis="cx")
    out=stim.Circuit()
    for inst in original.flattened():
        if inst.name == "CX":
            targets=inst.targets_copy()
            out.append("H",targets[1::2])
            out.append("TICK")
            out.append("CZ",targets)
            out.append("TICK")
            out.append("H",targets[1::2])
        else:
            out.append(inst)
    return out


def prepare_vanilla(backend, rounds, basis, mapping, noise_p):
    from qiskit import transpile
    from .stim_to_qiskit import stim_to_qiskit
    from .build_emerald_qubit_rotated import validate_layout
    ideal=vanilla_circuit(rounds,basis)
    couplings=[(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b))
               for a,b in backend.coupling_map.get_edges()]
    report=validate_layout(ideal,mapping,couplings)
    qc,dense,order=stim_to_qiskit(ideal)
    layout=[backend.qubit_name_to_index(mapping[q]) for q in sorted(dense)]
    compiled=transpile(qc,backend=backend,initial_layout=layout,routing_method="none",
                       optimization_level=0,seed_transpiler=1901)
    serialized=backend.serialize_circuit(compiled)
    native=[i for i in serialized.instructions if i.name == "cz"]
    expected=24*rounds
    edges={frozenset(e) for e in couplings}
    if len(native)!=expected or any(frozenset(i.locus) not in edges for i in native):
        raise ValueError("Vanilla compilation changed CZ count or inserted a non-native interaction.")
    if compiled.count_ops().get("reset",0) or compiled.count_ops().get("measure",0)!=ideal.num_measurements:
        raise ValueError("Vanilla compilation changed reset/measurement convention.")
    if any(i.operation.num_qubits==2 and i.operation.name not in {"cz","barrier"} for i in compiled.data):
        raise ValueError("Unexpected two-qubit gate in vanilla compilation.")
    model=NoiseModel.from_legacy(dict(after_clifford_depolarization=noise_p,
        after_reset_flip_probability=noise_p,before_measure_flip_probability=noise_p,
        before_round_data_depolarization=noise_p))
    decoder_circuit=add_noise(ideal,model)
    report.update(depth=compiled.depth(),native_ops=dict(compiled.count_ops()),dd="disabled",
        qiskit_optimization_level=0,custom_h_cancellation=False,custom_asap=False,
        logical_operations=[],calibration_weighted_placement=False,
        decoder_noise="uniform legacy circuit Pauli",decoder_p=noise_p,
        native_timing_mismatch="Mandatory R synthesis and pulse timing differ from logical H/CZ noise schedule")
    return dict(ideal=ideal,compiled=compiled,decoder_circuit=decoder_circuit,report=report,measurement_order=order)


def build_with_memory_profile(circuit):
    try:
        import psutil
        process=psutil.Process()
    except ImportError:
        process=None
    stop=threading.Event()
    peak=[process.memory_info().rss if process else 0]
    def monitor():
        while not stop.wait(.02):
            peak[0]=max(peak[0],process.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True) if process else None
    if thread: thread.start()
    try:
        decoder=build_lookup_table(circuit)
        if process: peak[0]=max(peak[0],process.memory_info().rss)
    finally:
        stop.set()
        if thread: thread.join()
    decoder.metadata["measured_process_peak_rss_bytes"]=peak[0] if process else None
    decoder.metadata["memory_note"]="Sampled whole-process RSS every 20ms, includes SDK; not isolated allocator peak"
    return decoder


def memory_syndromes(counts, memory, ideal, shots):
    counts_to_measurement_array(counts,ideal.num_measurements,shots)
    records=[record.replace(" ","") for record in memory]
    normalized=Counter()
    for record,count in counts.items(): normalized[record.replace(" ","")]+=count
    if Counter(records)!=normalized:
        raise ValueError("Original memory differs from aggregate counts.")
    raw=np.asarray([[bit=="1" for bit in record[::-1]] for record in records],dtype=bool)
    syndromes=extract_syndromes(raw,ideal,print_summary=False)
    syndromes["stim_circuit"]=ideal
    return raw,syndromes


def score_hardware(output,case,counts,memory):
    name=case["id"]
    ideal=stim.Circuit.from_file(output/f"{name}.stim")
    noisy=stim.Circuit.from_file(output/f"{name}.decoder.stim")
    start=time.perf_counter()
    decoder=LookupTableDecoder.load(output/f"{name}.lut.npz")
    load_s=time.perf_counter()-start
    start=time.perf_counter()
    raw,syndromes=memory_syndromes(counts,memory,ideal,case["shots"])
    convert_s=time.perf_counter()-start
    scored=decode_syndromes(syndromes,noisy,matcher=decoder)
    dets=syndromes["det_events"]
    decoder.decode_batch(dets)
    times=[]
    for _ in range(31):
        start=time.perf_counter()
        decoder.decode_batch(dets)
        times.append(time.perf_counter()-start)
    median=float(np.median(times))
    np.savez_compressed(output/f"{name}.shots.npz",raw_meas=raw,det_events=dets,obs_flips=syndromes["obs_flips"])
    dataset_hash=hashlib.sha256((output/f"{name}.shots.npz").read_bytes()).hexdigest()
    return dict(**scored.to_dict(),detector_event_fraction=float(dets.mean()),
        shot_order="original acquisition order",dataset_sha256=dataset_hash,
        timing=dict(lut_load_s=load_s,conversion_s=convert_s,decode_batch_median_s=median,
            decode_us_per_shot=1e6*median/case["shots"],shots_per_second=case["shots"]/median,repeats=31),
        decoder="complete full-history LUT; independent uniform circuit DEM",training_shots=0,
        evaluated_shots_excluded=0,unseen_history_fallback=None)


def fit_postfirst_ler(rounds, errors, shots):
    """Exploratory first-round-offset model, P(r)=[1-V1(1-2e)^(r-1)]/2.

    V1 is the visibility after the first round. This permits the first round
    to have different exposure from subsequent repeated measurement cycles.
    It is not the challenge's zero-offset equation.
    """
    from functools import lru_cache
    from scipy.optimize import minimize_scalar,brentq
    from scipy.special import xlogy
    from scipy.stats import chi2
    r,k,n=(np.asarray(v,dtype=float) for v in (rounds,errors,shots))
    if r.ndim!=1 or r.shape!=k.shape or r.shape!=n.shape or len(set(r))<3 or np.any(r<1):
        raise ValueError("Post-first fit needs three distinct positive round counts and equal 1D arrays.")
    if (not all(np.all(np.isfinite(v)) for v in (r,k,n)) or np.any(n<1) or np.any(k<0) or np.any(k>n)
            or any(np.any(v!=np.floor(v)) for v in (r,k,n))):
        raise ValueError("Fit inputs must be valid finite binomial counts and integer rounds.")
    def nll(e,v):
        p=.5*(1-v*np.power(1-2*e,r-1))
        return float(-np.sum(xlogy(k,p)+xlogy(n-k,1-p)))
    @lru_cache(maxsize=1024)
    def best(e):
        fit=minimize_scalar(lambda v:nll(e,v),bounds=(0,1),method="bounded",options={"xatol":1e-12})
        v=min((0.,float(fit.x),1.),key=lambda v:nll(e,v))
        return v,nll(e,v)
    outer=minimize_scalar(lambda e:best(float(e))[1],bounds=(0,.5),method="bounded",options={"xatol":1e-12})
    epsilon=min((0.,float(outer.x),.5),key=lambda e:best(e)[1])
    visibility,value=best(epsilon)
    def profile(e): return best(float(e))[1]-value-chi2.ppf(.95,1)/2
    lo=0. if epsilon==0 or profile(0)<=0 else brentq(profile,0,epsilon)
    hi=.5 if epsilon==.5 or profile(.5)<=0 else brentq(profile,epsilon,.5)
    return dict(epsilon_per_round=epsilon,ci95_profile=[lo,hi],amplitude=visibility,
        first_round_visibility=visibility,round_offset=1,fit_spam=True,
        fitted_probabilities=(.5*(1-visibility*(1-2*epsilon)**(r-1))).tolist(),
        negative_log_likelihood=value,boundary_fit=epsilon<1e-8 or epsilon>.5-1e-8,
        interpretation="effective repeated-cycle rate after first round; exploratory model chosen after primary-fit mismatch",
        uncertainty_method="asymptotic profile likelihood, conditional on shifted stationary binomial model")


def fitted_results(cases):
    from scipy.special import xlogy
    fits=[]
    for basis in ("X","Z"):
        points=sorted([c for c in cases if c["basis"]==basis and "analysis" in c],key=lambda c:c["rounds"])
        if len(points)<3: continue
        r,k,n=([c[key] if key!="errors" else c["analysis"][key] for c in points]
               for key in ("rounds","errors","shots"))
        primary=fit_ler(r,k,n,fit_spam=False)
        primary["round_offset"]=0
        diagnostic=fit_postfirst_ler(r,k,n)
        for fit in (primary,diagnostic):
            deviance=0.
            for c,p in zip(points,fit["fitted_probabilities"]):
                k,n=c["analysis"]["errors"],c["shots"]
                deviance+=2*(xlogy(k,k/n)+xlogy(n-k,1-k/n)-xlogy(k,p)-xlogy(n-k,1-p))
            fit.update(basis=basis,deviance=float(deviance),nominal_degrees_of_freedom=len(points)-(2 if fit["fit_spam"] else 1),
                fit_role="exploratory post-first-cycle diagnostic" if fit["fit_spam"] else "primary challenge equation")
            fits.append(fit)
    return fits


def save_report(output,manifest):
    manifest["fits"]=fitted_results(manifest["cases"])
    write_json(output/"manifest.json",manifest)
    lines=["# Vanilla surface-code + lookup-table baseline on IQM Garnet", "",
        "Challenge reference: `reference/README.md`, Qiskit -> hardware -> detection events -> decoder -> multi-round LER.",
        "Distance 3 rotated memory, no ancilla reset, 17 physical qubits. DD disabled. No logical gates, "
        "calibration-tuned layout/weights, custom H cancellation or custom ASAP scheduling. Mandatory native IQM synthesis remains.",
        f"Calibration `{manifest['calibration_set_id']}`. Uniform decoder prior p={manifest['noise_p']}; calibration is archived for context only.",
        "The complete full-history LUT is frozen before submission; no hardware labels train it. Every shot is retained.","",
        "| Basis | Rounds | Failures / shots | Logical failure (95% Wilson CI) | Lookup shots/s |",
        "|---|---:|---:|---|---:|"]
    for c in sorted(manifest["cases"],key=lambda c:(c["basis"],c["rounds"])):
        if "analysis" in c:
            a=c["analysis"];lo,hi=a["ci95"]
            lines.append(f"| {c['basis']} | {c['rounds']} | {a['errors']} / {a['shots']} | "
                         f"{100*a['probability']:.2f}% ({100*lo:.2f}â€“{100*hi:.2f}%) | {a['timing']['shots_per_second']:,.0f} |")
        else: lines.append(f"| {c['basis']} | {c['rounds']} | {c['status']} | â€” | â€” |")
    lines += ["", "Primary LER fit: P(r) = [1 - (1 - 2 epsilon)^r]/2 (challenge equation). "
              "Exploratory post-first-cycle fit: P(r) = [1 - V1(1 - 2 epsilon)^(r-1)]/2, "
              "where V1 is visibility after the first round. This diagnostic was added after observing primary-fit mismatch.",
              "Intervals are asymptotic profile likelihood, conditional on a stationary binomial model. "
              "Device drift, initialization offsets, leakage and fit error are not included."]
    for f in manifest["fits"]:
        lo,hi=f["ci95_profile"]
        lines.append(f"- {f['basis']}, {f['fit_role']}: **{100*f['epsilon_per_round']:.3f}%/round** "
                     f"(95% profile interval {100*lo:.3f}â€“{100*hi:.3f}%), visibility={f['amplitude']:.4f}; "
                     f"deviance {f['deviance']:.2f}, nominal df {f['nominal_degrees_of_freedom']}.")
    lines += ["", "Large deviance or a boundary fit limits interpretation: prefer measured per-round-count probabilities "
              "when the stationary model does not describe the data. This is an encoded baseline, and does not "
              "establish an advantage over an unencoded qubit or below-threshold distance scaling.","",
              "| Circuit | LUT entries | LUT build (s) | Table (MiB) | Sampled process peak (MiB) | Server execution interval (s) |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in manifest["cases"]:
        d=c["lut"]
        lines.append(f"| {c['id']} | {d['entries']:,} | {d['build_s']:.3f} | {d['table_bytes']/2**20:.3f} | "
            f"{(d.get('measured_process_peak_rss_bytes') or 0)/2**20:.1f} | "
            f"{c.get('server_timing',{}).get('execution_s',float('nan')):.3f} |")
    lines += ["", "LUT construction is exponential. Round counts 1/2/3 have 8/16/24 detection bits and "
        "256/65,536/16,777,216 entries. The maximum is 16 MiB of decisions, plus up to about 512 MiB of transform "
        "workspace. Four rounds would require 2**32 decisions and is rejected by this implementation.",
        "LUT probabilities use double precision and a Stim independent-fault DEM with explicit disjoint-channel "
        "approximation. Separated components of one error remain correlated. Exact ML refers to this modeled "
        "distribution within numerical precision; it does not imply that the uniform model matches Garnet.",
        "Decoder mismatch includes native R decomposition, pulse durations, spatially varying noise, leakage and "
        "non-QND readout. Timing is warmed batch lookup throughput; preprocessing, loading and conversion are separate. "
        "Server execution intervals include device setup/overhead and are not pulse durations.",
        "Previous DD measurements used a different schedule/layout, decoder and time window. These runs cannot isolate "
        "a DD-only or decoder-only causal improvement relative to that older dataset.","",
        "Artifacts: raw original-order memory/counts, shot/syndrome arrays, ideal/noisy Stim and DEM definitions, "
        "QPY native circuits, frozen LUTs/hashes, calibration observations, job IDs and server timelines."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if manifest["cases"] and all("analysis" in c for c in manifest["cases"]):
        plot_report(output,manifest)


def plot_report(output,manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout="constrained")
    for ax,basis,color in zip(axes,("X","Z"),("#146b9b","#bb4f2d")):
        points=sorted([c for c in manifest["cases"] if c["basis"]==basis],key=lambda c:c["rounds"])
        x=np.array([c["rounds"] for c in points]);y=np.array([c["analysis"]["probability"] for c in points])
        lo=np.array([c["analysis"]["ci95"][0] for c in points]);hi=np.array([c["analysis"]["ci95"][1] for c in points])
        ax.errorbar(x,y*100,yerr=np.array([y-lo,hi-y])*100,fmt="o",capsize=4,color=color,label="Hardware + LUT")
        for fit in manifest["fits"]:
            if fit["basis"]==basis:
                r=np.linspace(fit.get("round_offset",0),3,150)
                p=.5*(1-fit["amplitude"]*(1-2*fit["epsilon_per_round"])**(r-fit.get("round_offset",0)))
                ax.plot(r,p*100,"--" if fit["fit_spam"] else "-",alpha=.7,
                    color="#777777" if fit["fit_spam"] else color,
                    label="Post-first-cycle diagnostic" if fit["fit_spam"] else "Challenge fit")
        ax.set(title=f"{basis}-basis memory",xlabel="Syndrome rounds",xticks=[1,2,3],ylim=(0,50))
        ax.grid(axis="y",alpha=.2);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel("Decoded logical failure (%)")
    shots=sum(c["shots"] for c in manifest["cases"])
    fig.suptitle(f"Live Garnet vanilla surface-code / LUT baseline Â· {shots:,} shots\n"
                 "Distance 3 Â· DD off Â· 95% Wilson error bars",fontsize=12)
    fig.savefig(output/"comparison.png",dpi=180);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file",required=True)
    parser.add_argument("--shots",type=int,default=3000)
    parser.add_argument("--noise-p",type=float,default=.003)
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--wait-seconds",type=float,default=1800)
    args=parser.parse_args(argv)
    if not 1<=args.shots<=5000 or not 0<args.noise_p<.1 or args.wait_seconds<0:
        parser.error("Require 1..5000 shots, 0<p<0.1 and a nonnegative wait.")
    secret=read_token(args.token_file)
    output=args.resume or args.output or Path("results/decoders/lookup_table")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from .run_on_hardware import connect_backend
    from iqm.iqm_client import CircuitCompilationOptions, DDMode
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    from qiskit import qpy
    try:
        backend=connect_backend(secret,"garnet")
        if args.resume:
            manifest=json.loads((output/"manifest.json").read_text())
            backend=IQMBackend(backend.client,calibration_set_id=manifest["calibration_set_id"])
        else:
            output.mkdir(parents=True,exist_ok=False)
            cal_id=backend.architecture.calibration_set_id
            backend=IQMBackend(backend.client,calibration_set_id=cal_id)
            write_json(output/"architecture.json",backend.architecture.model_dump(mode="json"))
            write_json(output/"quality_metrics.json",backend.client.get_quality_metric_set(cal_id).model_dump(mode="json"))
            write_json(output/"pulse_calibration.json",backend.client.get_calibration_set(cal_id).model_dump(mode="json"))
            edges=[(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b)) for a,b in backend.coupling_map.get_edges()]
            mapping,ranking=select_native_layout(vanilla_circuit(3,"Z"),edges,backend.physical_qubits)
            manifest=dict(device="garnet",distance=3,no_reset=True,dd="disabled",noise_p=args.noise_p,
                calibration_set_id=str(cal_id),placement=mapping,placement_ranking=ranking,
                created_utc=datetime.now(timezone.utc).isoformat(),total_requested_shots=args.shots*6,cases=[],
                versions={p:importlib.metadata.version(p) for p in ("iqm-client","qiskit","stim","numpy","scipy")},
                runtime=dict(python=platform.python_version(),os=platform.platform(),processor=platform.processor(),
                             logical_cpus=os.cpu_count(),thread_environment={k:os.environ.get(k) for k in
                             ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}),
                challenge_reference="challenge/README.md",baseline_definition=__doc__,
                calibration_usage="archived only; not used for layout ranking or decoder probabilities")
            specifications=[(b,r) for b in ("X","Z") for r in (1,2,3)]
            random.Random(1917).shuffle(specifications)
            for basis,rounds in specifications:
                name=f"{basis}_r{rounds}_vanilla_lut"
                prepared=prepare_vanilla(backend,rounds,basis,mapping,args.noise_p)
                prepared["compiled"].name=name
                prepared["ideal"].to_file(output/f"{name}.stim")
                prepared["decoder_circuit"].to_file(output/f"{name}.decoder.stim")
                dem=prepared["decoder_circuit"].detector_error_model(approximate_disjoint_errors=True)
                dem.to_file(output/f"{name}.dem")
                with (output/f"{name}.qpy").open("wb") as stream: qpy.dump(prepared["compiled"],stream)
                print(f"Building {name}: {dem.num_detectors} detector bits, {2**dem.num_detectors:,} LUT entries",flush=True)
                decoder=build_with_memory_profile(prepared["decoder_circuit"])
                decoder.save(output/f"{name}.lut.npz")
                dets,obs=prepared["decoder_circuit"].compile_detector_sampler(seed=300+rounds+(basis=="X")).sample(20000,separate_observables=True)
                validation=decode_syndromes(dict(det_events=dets,obs_flips=obs),prepared["decoder_circuit"],matcher=decoder)
                if abs(validation.probability-decoder.metadata["model_bayes_failure"])>.02:
                    raise ValueError("Matched-noise LUT validation differs from modeled Bayes failure.")
                case=dict(id=name,basis=basis,rounds=rounds,shots=args.shots,status="prepared",
                    hardware_report=prepared["report"],measurement_order=prepared["measurement_order"],
                    lut=decoder.metadata,local_validation=validation.to_dict(),
                    ideal_sha256=hashlib.sha256((output/f"{name}.stim").read_bytes()).hexdigest(),
                    decoder_sha256=hashlib.sha256((output/f"{name}.decoder.stim").read_bytes()).hexdigest())
                manifest["cases"].append(case)
                save_report(output,manifest)
                print(f"Frozen {name}; build {decoder.metadata['build_s']:.2f}s, simulated failure {validation.probability:.4f}",flush=True)
                del decoder,prepared,dets,obs
                gc.collect()
            print(f"Prepared 6 cases, {args.shots*6} shots: {output}",flush=True)
            if not args.submit: return
            for c in manifest["cases"]:
                with (output/f"{c['id']}.qpy").open("rb") as stream: qc=qpy.load(stream)[0]
                c.update(status="submission_started",submitted_utc=datetime.now(timezone.utc).isoformat())
                save_report(output,manifest)
                job=backend.run(qc,shots=c["shots"],circuit_compilation_options=CircuitCompilationOptions(dd_mode=DDMode.DISABLED))
                c.update(job_id=job.job_id(),status="submitted")
                save_report(output,manifest)
                print(f"Submitted {c['id']}: {c['job_id']}",flush=True)
        pending=[c for c in manifest["cases"] if c.get("job_id") and "analysis" not in c and c["status"] not in {"ERROR","CANCELLED"}]
        jobs={c["id"]:backend.retrieve_job(c["job_id"]) for c in pending}
        deadline=time.monotonic()+args.wait_seconds
        while pending:
            for c in pending[:]:
                job=jobs[c["id"]];status=job.status().name
                if c["status"]!=status: print(f"{c['id']}: {status}",flush=True)
                c["status"]=status
                if status=="DONE":
                    result=job.result(timeout=1)
                    counts,memory=dict(result.get_counts()),result.get_memory()
                    write_json(output/f"{c['id']}.counts.json",counts)
                    write_json(output/f"{c['id']}.memory.json",memory)
                    timeline=[t.model_dump(mode="json") for t in result.to_dict().get("timeline",[])]
                    c["server_timeline"],c["server_timing"]=timeline,server_timing(timeline)
                    c["result_received_utc"]=datetime.now(timezone.utc).isoformat()
                    c["analysis"]=score_hardware(output,c,counts,memory)
                    print(f"{c['id']}: {c['analysis']['errors']}/{c['shots']} LUT logical failures",flush=True)
                    pending.remove(c)
                elif status in {"ERROR","CANCELLED"}:
                    c["error"]=str(job.error_message()).replace(secret,"[REDACTED]")
                    print(f"{c['id']}: {c['error']}",flush=True);pending.remove(c)
            save_report(output,manifest)
            if not pending or time.monotonic()>=deadline: break
            time.sleep(min(10,max(0,deadline-time.monotonic())))
        print(f"Report: {output/'report.md'}; pending {len(pending)}",flush=True)
    except Exception as exc:
        message=str(exc).replace(secret,"[REDACTED]")
        if output.exists(): write_json(output/"error.json",dict(type=type(exc).__name__,message=message))
        raise SystemExit(f"{type(exc).__name__}: {message}") from None


if __name__=="__main__":
    main()
