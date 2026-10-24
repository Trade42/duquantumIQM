"""Pilot-select, freeze, then independently confirm the integrated Garnet stack.

The physical configuration, residual noise parameters and decoder family are
selected on pilot shots only. Confirmation uses fresh, interleaved jobs on the
same calibration and physical patch, including vanilla and previous-stack controls.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import math
import random
import time
import numpy as np
import stim

from .benchmark_garnet import read_token,write_json,calibration_from_observations,select_native_layout,server_timing
from .benchmark_vanilla_lut import prepare_vanilla,memory_syndromes,fit_postfirst_ler
from .decoding import make_matcher,wilson_interval,fit_ler
from .decoders.lookup_table import build_lookup_table,LookupTableDecoder
from .hardware_optimization import PulseExecutor,optimize_native,native_stim,native_noise

PROFILES={
    "vanilla":dict(native=False,dd="off",fast=False),
    "current":dict(native=False,dd="xx",fast=False),
    "native_off":dict(native=True,dd="off",fast=False),
    "native_xx":dict(native=True,dd="xx",fast=False),
    "native_xy4":dict(native=True,dd="xy4",fast=False),
    "native_adaptive":dict(native=True,dd="adaptive",fast=False),
    "fast_off":dict(native=True,dd="off",fast=True),
    "fast_adaptive":dict(native=True,dd="adaptive",fast=True),
    "protected_xx":dict(native="retain",dd="xx",fast=False),
    "protected_xy4":dict(native="retain",dd="xy4",fast=False),
    "protected_adaptive":dict(native="retain",dd="adaptive",fast=False),
    "fenced_xx":dict(native="fenced",dd="xx",fast=False),
    "fenced_xy4":dict(native="fenced",dd="xy4",fast=False),
    "fused_xx":dict(native="fused",dd="xx",fast=False),
    "fused_xy4":dict(native="fused",dd="xy4",fast=False),
}
PROFILE_SETS=dict(standard=["vanilla","current","native_off","native_xx","native_xy4","native_adaptive","fast_off","fast_adaptive"],
                 protective=["current","protected_xx","protected_xy4","protected_adaptive","fenced_xx","fenced_xy4"],
                 terminal=["current","fused_xx","fused_xy4"])


def summarize(predictions,truth):
    failed=np.any(np.asarray(predictions)!=truth,axis=1)
    k,n=int(failed.sum()),len(failed)
    return dict(errors=k,shots=n,probability=k/n,ci95=wilson_interval(k,n)),failed


def choose_pilot_candidate(pilots,basis):
    candidates=[c for c in pilots if c["basis"]==basis and c["profile"]!="vanilla"]
    return min(candidates,key=lambda c:(c["pilot_score"]["errors"],c["pulse_audit"]["total_dd_added_iq_pulses"]))


def qualified_release(manifest):
    """Promote on fresh primary-endpoint improvement, otherwise keep current.

    Two memory bases are tested. The endpoint uses a 97.5% Newcombe interval
    per basis (Bonferroni 95% family confidence). This is a promotion decision,
    never a second opportunity to retune the decoder on confirmation labels.
    """
    from scipy.stats import norm
    release={}
    for basis in ("X","Z"):
        cases={c["condition"]:c for c in manifest["cases"] if c["phase"]=="confirmation"
               and c["basis"]==basis and c["rounds"]==3 and "analysis" in c}
        if not {"current","integrated"}<=set(cases):continue
        a,b=(cases[key]["analysis"] for key in ("current","integrated"))
        z=float(norm.ppf(1-.05/4))
        alo,ahi=wilson_interval(a["errors"],a["shots"],z)
        blo,bhi=wilson_interval(b["errors"],b["shots"],z)
        difference=a["probability"]-b["probability"]
        low=difference-math.hypot(a["probability"]-alo,bhi-b["probability"])
        high=difference+math.hypot(ahi-a["probability"],b["probability"]-blo)
        candidate=manifest["selection"][basis]
        promote=candidate["profile"]!="current" and low>0
        release[basis]=(dict(profile=candidate["profile"],decoder=candidate["decoder"]) if promote else
                        dict(profile="current",decoder=dict(family="mwpm",parameters={})))
        release[basis].update(promoted=promote,endpoint_rounds=3,absolute_reduction=difference,
            ci975_absolute_reduction=[low,high],calibration_set_id=manifest["calibration_set_id"],
            reason="Fresh confirmation improved primary endpoint" if promote else "Retain prior stack; no qualified replacement")
    return release


def prepare_case(output,case,backend,executor,calibration,quality,mapping):
    from .run_on_hardware import prepare_qiskit
    from .internal_helpers import get_qubit_lists
    from qiskit import qpy
    profile=PROFILES[case["profile"]]
    if case["profile"]=="vanilla":
        p=prepare_vanilla(backend,case["rounds"],case["basis"],mapping,.003)
        reference,qc,decoder=p["ideal"],p["compiled"],p["decoder_circuit"]
    else:
        p=prepare_qiskit(backend,3,case["rounds"],case["basis"],calibration=calibration,qubit_map=mapping,dd="off")
        reference,qc,decoder=p["stim_circuit"],p["circuit"],p["decoder_circuit"]
    before=dict(qc.count_ops())
    if profile["native"]: qc,optimization=optimize_native(qc,mode=profile["native"] if isinstance(profile["native"],str) else "aggressive")
    else: optimization=dict(before=before,after=before,depth_before=qc.depth(),depth_after=qc.depth())
    native,columns=native_stim(qc,reference,mapping,backend)
    data,anc=get_qubit_lists(reference)
    targets=[mapping[q] for q in data]
    anc_names=[mapping[q] for q in anc]
    qc.name=case["id"]
    run,context,audit=executor.compile(qc,shots=case["shots"],dd=profile["dd"],targets=targets,
                                    ancilla_names=anc_names,fast_ancillas=profile["fast"])
    for suffix,circuit in (("source",reference),("native",native),("reference_decoder",decoder)):
        circuit.to_file(output/f"{case['id']}.{suffix}.stim")
    with (output/f"{case['id']}.qpy").open("wb") as stream:qpy.dump(qc,stream)
    case.update(optimization=optimization,pulse_audit=audit,native_measurement_columns=columns,
                source_sha256=hashlib.sha256(str(reference).encode()).hexdigest(),
                native_sha256=hashlib.sha256(str(native).encode()).hexdigest(),status="prepared")
    return run,context


def collect(output,manifest,cases,executor,secret):
    from .extract_syndromes import extract_syndromes
    pending=[c for c in cases if c.get("job_id") and "data_file" not in c and c.get("status") not in {"failed","cancelled"}]
    start=time.monotonic()
    while pending and time.monotonic()-start<1800:
        for c in pending[:]:
            data=executor.status(c["job_id"]);status=str(data.status)
            if c["status"]!=status:print(c["id"],status,flush=True)
            c["status"]=status
            if status=="completed":
                counts,memory=executor.memory(c["job_id"],c["pulse_audit"],c["shots"])
                write_json(output/f"{c['id']}.counts.json",counts);write_json(output/f"{c['id']}.memory.json",memory)
                reference=stim.Circuit.from_file(output/f"{c['id']}.source.stim")
                native=stim.Circuit.from_file(output/f"{c['id']}.native.stim")
                raw,s=memory_syndromes(counts,memory,reference,c["shots"])
                native_raw=raw[:,c["native_measurement_columns"]]
                ns=extract_syndromes(native_raw,native,print_summary=False)
                if not (np.array_equal(s["det_events"],ns["det_events"]) and np.array_equal(s["obs_flips"],ns["obs_flips"])):
                    raise ValueError("Native/source detector frames differ on the same physical records.")
                filename=f"{c['id']}.shots.npz"
                np.savez_compressed(output/filename,raw_meas=raw,native_raw_meas=native_raw,
                                    det_events=ns["det_events"],obs_flips=ns["obs_flips"])
                c["data_file"]=filename;c["dataset_sha256"]=hashlib.sha256((output/filename).read_bytes()).hexdigest()
                c["server_timeline"]=[t.model_dump(mode="json") for t in data.timeline]
                c["server_timing"]=server_timing(c["server_timeline"])
                write_json(output/f"{c['id']}.job.json",data.model_dump(mode="json"))
                pending.remove(c)
            elif status in {"failed","cancelled"}:
                c["error"]=str(data.errors).replace(secret,"[REDACTED]");pending.remove(c)
        write_json(output/"manifest.json",manifest)
        if pending:time.sleep(2)
    if pending:raise TimeoutError("Pulse jobs still pending; resume this manifest, do not resubmit.")


def load_data(output,case):
    with np.load(output/case["data_file"]) as data:
        return data["det_events"].copy(),data["obs_flips"].copy()


def decoder_for(output,case,backend,calibration,quality,parameters,family,*,persist=True):
    profile=case["profile"]
    if profile in {"vanilla","current"}:
        model=stim.Circuit.from_file(output/f"{case['id']}.reference_decoder.stim")
        family="lut" if profile=="vanilla" else "mwpm"
    else:
        ideal=stim.Circuit.from_file(output/f"{case['id']}.native.stim")
        model=native_noise(ideal,backend,calibration,quality,fast_ancillas=PROFILES[profile]["fast"],
            measurement_operations=case["pulse_audit"].get("measurement_operations"),**parameters)
    start=time.perf_counter()
    decoder=build_lookup_table(model) if family=="lut" else make_matcher(model)
    build_s=time.perf_counter()-start
    if persist:
        model.to_file(output/f"{case['id']}.selected_decoder.stim")
        model.detector_error_model(approximate_disjoint_errors=True).to_file(output/f"{case['id']}.selected.dem")
        if family=="lut":decoder.save(output/f"{case['id']}.selected.lut.npz")
    return decoder,dict(family=family,parameters=parameters,build_s=build_s,
                        model_sha256=hashlib.sha256(str(model).encode()).hexdigest())


def select_pilot(output,case,backend,calibration,quality):
    dets,truth=load_data(output,case)
    candidates=[]
    if case["profile"] in {"vanilla","current"}:
        decoder,config=decoder_for(output,case,backend,calibration,quality,{},"mwpm")
        result,failed=summarize(decoder.decode_batch(dets,enable_correlations=True),truth)
    else:
        ideal=stim.Circuit.from_file(output/f"{case['id']}.native.stim")
        for phase in (0.,.05,.15,.3,.5):
            for backaction in (0.,.5,1.):
                parameters=dict(phase_round=phase,backaction_scale=backaction)
                model=native_noise(ideal,backend,calibration,quality,fast_ancillas=PROFILES[case["profile"]]["fast"],
                    measurement_operations=case["pulse_audit"].get("measurement_operations"),**parameters)
                decoder=make_matcher(model)
                score,_=summarize(decoder.decode_batch(dets,enable_correlations=True),truth)
                candidates.append(dict(parameters=parameters,**score))
        best=min(candidates,key=lambda c:(c["errors"],sum(c["parameters"].values())))
        parameters=best["parameters"]
        decoder,config=decoder_for(output,case,backend,calibration,quality,parameters,"lut")
        score_lut,fail_lut=summarize(decoder.decode_batch(dets),truth)
        if score_lut["errors"]<=best["errors"]:
            result,failed=score_lut,fail_lut
        else:
            decoder,config=decoder_for(output,case,backend,calibration,quality,parameters,"mwpm")
            result,failed=summarize(decoder.decode_batch(dets,enable_correlations=True),truth)
    case.update(pilot_score=result,selected_decoder=config,model_selection_candidates=candidates,
                selection_data_only=True)
    print("PILOT",case["profile"],case["basis"],f"{result['errors']}/{result['shots']}",config,flush=True)


def score_confirmation(output,case,backend,calibration,quality,config):
    dets,truth=load_data(output,case)
    decoder,actual=decoder_for(output,case,backend,calibration,quality,config.get("parameters",{}),config["family"])
    decoder.decode_batch(dets,enable_correlations=True)
    timings=[]
    for _ in range(15):
        start=time.perf_counter();predictions=decoder.decode_batch(dets,enable_correlations=True)
        timings.append(time.perf_counter()-start)
    result,failed=summarize(predictions,truth)
    result.update(decoder=actual,batch_decode_median_s=float(np.median(timings)),
                  decode_shots_per_second=case["shots"]/float(np.median(timings)))
    # Paired decoder-only ablation on identical held-out physical shots.
    reference=stim.Circuit.from_file(output/f"{case['id']}.reference_decoder.stim")
    refdecoder=make_matcher(reference)
    refscore,reffail=summarize(refdecoder.decode_batch(dets,enable_correlations=True),truth)
    rescued=int(np.count_nonzero(reffail & ~failed));harmed=int(np.count_nonzero(~reffail & failed))
    result["paired_decoder_ablation"]=dict(reference=refscore,rescued=rescued,harmed=harmed,
        reduction=(rescued-harmed)/case["shots"],same_physical_shots=True)
    np.savez_compressed(output/f"{case['id']}.predictions.npz",predictions=predictions,failed=failed,reference_failed=reffail)
    case["analysis"]=result
    print("CONFIRM",case["condition"],case["basis"],case["rounds"],f"{result['errors']}/{result['shots']}",flush=True)


def save_report(output,manifest):
    confirm=[c for c in manifest["cases"] if c["phase"]=="confirmation" and "analysis" in c]
    fits=[]
    for condition in ("vanilla","current","integrated"):
        for basis in ("X","Z"):
            points=sorted([c for c in confirm if c["condition"]==condition and c["basis"]==basis],key=lambda c:c["rounds"])
            if len(points)<3:continue
            r=[c["rounds"] for c in points];k=[c["analysis"]["errors"] for c in points];n=[c["shots"] for c in points]
            for kind,f in (("challenge",fit_ler(r,k,n)),("post_first",fit_postfirst_ler(r,k,n))):
                from scipy.special import xlogy
                f["deviance"]=float(2*sum(xlogy(ki,ki/ni)+xlogy(ni-ki,1-ki/ni)-xlogy(ki,p)-xlogy(ni-ki,1-p)
                    for ki,ni,p in zip(k,n,f["fitted_probabilities"])))
                f["nominal_df"]=len(points)-(2 if kind=="post_first" else 1)
                fits.append(dict(condition=condition,basis=basis,kind=kind,**f))
    manifest["fits"]=fits
    comparisons=[]
    for basis in ("X","Z"):
        for rounds in (1,2,3):
            points={c["condition"]:c for c in confirm if c["basis"]==basis and c["rounds"]==rounds}
            if not {"current","integrated"}<=set(points):continue
            a,b=(points[c]["analysis"] for c in ("current","integrated"))
            diff=a["probability"]-b["probability"]
            lo=diff-math.hypot(a["probability"]-a["ci95"][0],b["ci95"][1]-b["probability"])
            hi=diff+math.hypot(a["ci95"][1]-a["probability"],b["probability"]-b["ci95"][0])
            comparisons.append(dict(basis=basis,rounds=rounds,absolute_reduction=diff,ci95=[lo,hi],
                relative_reduction=diff/a["probability"] if a["probability"] else None,
                physical_profile_changed=points["current"]["profile"]!=points["integrated"]["profile"],
                uncertainty_method="pointwise independent-shot Newcombe interval; excludes drift/model-selection uncertainty"))
    manifest["comparisons"]=comparisons
    release=qualified_release(manifest)
    if len(release)==len(manifest.get("bases",["X","Z"])):
        manifest["validated_release"]=release
        write_json(output/"validated_release.json",release)
    write_json(output/"manifest.json",manifest)
    lines=["# Integrated Garnet hardware validation", "",f"Calibration `{manifest['calibration_set_id']}`; fixed 17-qubit patch; no routing SWAPs.",
        "Pilot data select physical profile, residual model parameters and decoder family; confirmation jobs are fresh and interleaved. "
        "All shots retained. All submitted pulse playlists are saved, with actual before/after DD pulse counts.","",
        "| Condition | Basis | Rounds | Errors / shots | Logical failure (95% Wilson CI) | DD IQ pulses | Native R gates |",
        "|---|---|---:|---:|---|---:|---:|"]
    for c in sorted(confirm,key=lambda c:(c["basis"],c["rounds"],c["condition"])):
        a=c["analysis"];lo,hi=a["ci95"]
        lines.append(f"| {c['condition']} | {c['basis']} | {c['rounds']} | {a['errors']}/{c['shots']} | "
          f"{100*a['probability']:.2f}% ({100*lo:.2f}–{100*hi:.2f}%) | {c['pulse_audit']['total_dd_added_iq_pulses']} | "
          f"{c['optimization']['after'].get('r',0)} |")
    lines += ["", "Selected profiles: `"+json.dumps(manifest.get("selection",{}),sort_keys=True)+"`", "",
        "Primary challenge and first-round-offset fits are both retained. Fits assume stationary independent logical flips; "
        "failure probabilities are the direct observations. First-round-offset estimates are effective repeated-cycle rates, not the same estimator as the challenge equation."]
    for f in fits:
        lo,hi=f["ci95_profile"]
        lines.append(f"- {f['condition']} {f['basis']} {f['kind']}: {100*f['epsilon_per_round']:.3f}%/round "
                     f"(95% profile interval {100*lo:.3f}–{100*hi:.3f}%), deviance {f['deviance']:.2f} / nominal df {f['nominal_df']}.")
    lines += ["", "Current minus selected-policy failure probability (positive favors selected policy; pointwise 95% Newcombe intervals):"]
    for c in comparisons:
        lo,hi=c["ci95"]
        lines.append(f"- {c['basis']}, {c['rounds']} rounds: {100*c['absolute_reduction']:.2f} percentage points "
                     f"({100*lo:.2f}–{100*hi:.2f}); physical configuration changed: {c['physical_profile_changed']}.")
    lines += ["When X selects `current`, its confirmation arm is an independent replication of the same configuration; "
              "differences do not measure an X implementation change. Intervals do not cover device drift or all exploratory comparisons."]
    if release:
        lines += ["", "Release gate (fresh 3-round endpoint; Bonferroni correction over X/Z):"]
        for basis,r in release.items():
            lo,hi=r["ci975_absolute_reduction"]
            lines.append(f"- {basis}: `{r['profile']}`, promoted={r['promoted']}; "
                         f"97.5% per-basis difference interval {100*lo:.2f}–{100*hi:.2f} percentage points.")
    lines += ["", "Diagnostics: prior runs compared different circuits/decoders and different fit conventions; "
        "their headline LER values were not interchangeable. The previous static decoder omitted native gate counts, "
        "readout state disturbance/leakage proxies and DD-dependent residual phase noise. "
        "Its simulations were sensitivity studies, not measured hardware suppression factors.",
        "Native Clifford reconstruction remaps every measurement bit and validates the source/native detector frame "
        "on the exact same records. New decoder models include calibrated native gate/readout rates, "
        "a QND/leakage-inspired Pauli proxy and pilot-selected residual phase noise. "
        "The proxy is not a qutrit leakage model and receives no hidden leakage flags.",
        "Aggressively removing every timing barrier regressed on hardware and was rejected. Measurement-fenced "
        "compilation retains readout boundaries while allowing native optimization within unitary regions. "
        "PulLA compiles the exact executable playlist, "
        "applies measured implementation calibration and adaptive DD, and audits the inserted microwave pulses. "
        "Fast readout is selected only if pilot evidence supports it. Transversal algorithm compression has no operations "
        "to remove in an identity memory experiment; it must be evaluated on algorithm workloads separately.",
        "Remaining limitations: pilot selection uncertainty, device drift, non-Pauli leakage/crosstalk, "
        "and residual model mismatch. No unencoded-memory or below-threshold scaling claim is made."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if confirm and len(confirm)==len([c for c in manifest["cases"] if c["phase"]=="confirmation"]):
        plot_validation(output,manifest)


def plot_validation(output,manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bases=manifest.get("bases",["X","Z"])
    fig,axes=plt.subplots(1,len(bases),figsize=(5*len(bases),4.3),sharey=True,layout="constrained",squeeze=False)
    axes=axes[0]
    cases=[c for c in manifest["cases"] if c["phase"]=="confirmation" and "analysis" in c]
    for ax,basis in zip(axes,bases):
        for condition,color,offset,label in (("current","#bb4f2d",-.04,"Previous stack"),
                                            ("integrated","#146b9b",.04,"Selected policy")):
            rows=sorted([c for c in cases if c["basis"]==basis and c["condition"]==condition],key=lambda c:c["rounds"])
            if not rows:continue
            x=np.array([c["rounds"] for c in rows],dtype=float)
            y=np.array([c["analysis"]["probability"] for c in rows])*100
            lo=np.array([c["analysis"]["ci95"][0] for c in rows])*100
            hi=np.array([c["analysis"]["ci95"][1] for c in rows])*100
            ax.errorbar(x+offset,y,yerr=[y-lo,hi-y],marker="o",capsize=4,color=color,label=label)
        ax.set(title=f"{basis}-basis memory",xlabel="Syndrome rounds",xticks=[1,2,3],ylim=(0,35))
        ax.grid(axis="y",alpha=.2);ax.legend(frameon=False)
    axes[0].set_ylabel("Decoded logical failure (%)")
    fig.suptitle("Fresh Garnet validation · profiles frozen on separate pilot shots\n"
                 "95% Wilson error bars · all shots retained",fontsize=12)
    fig.savefig(output/"comparison.png",dpi=180);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file",required=True)
    parser.add_argument("--pilot-shots",type=int,default=500)
    parser.add_argument("--confirmation-shots",type=int,default=2000)
    parser.add_argument("--profile-set",choices=list(PROFILE_SETS),default="protective")
    parser.add_argument("--bases",nargs="+",choices=["X","Z"],default=["X","Z"])
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args(argv)
    if not 100<=args.pilot_shots<=2000 or not 1000<=args.confirmation_shots<=5000:parser.error("Shots outside bounded experiment range.")
    if not args.submit and not args.resume:parser.error("Explicit --submit or --resume required; imports never run hardware.")
    secret=read_token(args.token_file)
    output=args.resume or Path("results/integrated")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from .run_on_hardware import connect_backend
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    from iqm.station_control.client.serializers.run_serializers import serialize_run_definition
    try:
        b=connect_backend(secret,"garnet")
        if args.resume:
            manifest=json.loads((output/"manifest.json").read_text())
            b=IQMBackend(b.client,calibration_set_id=manifest["calibration_set_id"])
            quality=json.loads((output/"quality_metrics.json").read_text())
            from .calibration import CalibrationSnapshot
            cal=CalibrationSnapshot.load(output/"calibration.json")
            mapping={int(q):p for q,p in manifest["placement"].items()}
        else:
            output.mkdir(parents=True,exist_ok=False)
            calid=b.architecture.calibration_set_id;b=IQMBackend(b.client,calibration_set_id=calid)
            architecture=b.architecture.model_dump(mode="json")
            quality=b.client.get_quality_metric_set(calid).model_dump(mode="json")
            pulse=b.client.get_calibration_set(calid).model_dump(mode="json")
            for name,data in (("architecture",architecture),("quality_metrics",quality),("pulse_calibration",pulse)):write_json(output/f"{name}.json",data)
            cal=calibration_from_observations(architecture,quality,pulse);cal.save(output/"calibration.json")
            from .surface_code import make_stim_circuit
            edges=[(b.index_to_qubit_name(a),b.index_to_qubit_name(z)) for a,z in b.coupling_map.get_edges()]
            mapping,_=select_native_layout(make_stim_circuit(3,3,gate_basis="cz"),edges,b.physical_qubits,cal)
            manifest=dict(calibration_set_id=str(calid),placement=mapping,cases=[],created_utc=datetime.now(timezone.utc).isoformat(),
                pilot_shots=args.pilot_shots,confirmation_shots=args.confirmation_shots,
                profile_set=args.profile_set,bases=args.bases,profile_definitions={p:PROFILES[p] for p in PROFILE_SETS[args.profile_set]},
                confirmation_conditions=["vanilla","current","integrated"] if args.profile_set=="standard" else ["current","integrated"],
                max_planned_shots=len(args.bases)*len(PROFILE_SETS[args.profile_set])*args.pilot_shots+
                    len(args.bases)*3*(3 if args.profile_set=="standard" else 2)*args.confirmation_shots,
                selection_rule="Pilot-only minimum failures; include current stack as fallback; ties prefer fewer pulses; freeze before independent confirmation")
            write_json(output/"manifest.json",manifest)
        executor=PulseExecutor(b,secret)
        def submit_cases(specs,phase):
            random.Random(2159 if phase=="pilot" else 2161).shuffle(specs)
            for profile,basis,rounds,condition in specs:
                name=f"{phase}_{condition}_{basis}_r{rounds}"
                if any(c["id"]==name for c in manifest["cases"]):continue
                c=dict(id=name,profile=profile,basis=basis,rounds=rounds,condition=condition,phase=phase,
                       shots=manifest["pilot_shots"] if phase=="pilot" else manifest["confirmation_shots"])
                run,context=prepare_case(output,c,b,executor,cal,quality,mapping)
                c["status"]="submission_started";manifest["cases"].append(c);write_json(output/"manifest.json",manifest)
                job=executor.submit(run,context);c.update(job_id=str(job.job_id),status="submitted")
                write_json(output/"manifest.json",manifest)
                (output/f"{name}.playlist.pb").write_bytes(serialize_run_definition(run).SerializeToString())
                print("SUBMITTED",name,c["job_id"],"DD IQ pulses",c["pulse_audit"]["total_dd_added_iq_pulses"],flush=True)
        if "selection" not in manifest:
            submit_cases([(p,basis,3,p) for p in PROFILE_SETS[manifest.get("profile_set","standard")] for basis in manifest.get("bases",["X","Z"])],"pilot")
            pilots=[c for c in manifest["cases"] if c["phase"]=="pilot"]
            collect(output,manifest,pilots,executor,secret)
            for c in pilots:
                if "pilot_score" not in c:
                    select_pilot(output,c,b,cal,quality);write_json(output/"manifest.json",manifest)
            selection={}
            for basis in manifest.get("bases",["X","Z"]):
                winner=choose_pilot_candidate(pilots,basis)
                selection[basis]=dict(profile=winner["profile"],decoder=winner["selected_decoder"],pilot_case=winner["id"])
            manifest["selection"]=selection;manifest["selection_frozen_utc"]=datetime.now(timezone.utc).isoformat()
            write_json(output/"frozen_selection.json",selection);write_json(output/"manifest.json",manifest)
            print("FROZEN_SELECTION",selection,flush=True)
        specs=[]
        for basis in manifest.get("bases",["X","Z"]):
            for rounds in (1,2,3):
                for condition in manifest.get("confirmation_conditions",["vanilla","current","integrated"]):
                    profile=manifest["selection"][basis]["profile"] if condition=="integrated" else condition
                    specs.append((profile,basis,rounds,condition))
        submit_cases(specs,"confirmation")
        cases=[c for c in manifest["cases"] if c["phase"]=="confirmation"]
        collect(output,manifest,cases,executor,secret)
        for c in cases:
            if "analysis" not in c:
                config=manifest["selection"][c["basis"]]["decoder"] if c["condition"]=="integrated" else dict(family="mwpm",parameters={})
                score_confirmation(output,c,b,cal,quality,config);save_report(output,manifest)
        save_report(output,manifest)
        print("COMPLETE",str(output),flush=True)
    except Exception as exc:
        message=str(exc).replace(secret,"[REDACTED]")
        if output.exists():write_json(output/"error.json",dict(type=type(exc).__name__,message=message))
        raise SystemExit(type(exc).__name__+": "+message) from None


if __name__=="__main__":main()
