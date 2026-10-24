"""Measure gate-count and logical-failure effects of transversal compilation.

python -m challenge.benchmark_logical --shots 100000 --synthetic-calibration
Baseline materializes logical gates; optimized uses Pauli frames and exact
within-block H/Pauli simplification. QEC rounds, patch, basis and noise match.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm,chi2

from .calibration import CalibrationSnapshot, synthetic_snapshot
from .decoding import wilson_interval,fit_ler
from .experiment import simulate_memory
from .logical_gates import make_logical_algorithm, plan_pauli_h
from .noise_models import noise_suite
from .surface_code import make_stim_circuit
from .build_emerald_qubit_rotated import rank_placements

PAPERS = [
    dict(title="Quantum Circuit Optimisation and MBQC Scheduling with a Pauli Tracking Library",url="https://arxiv.org/abs/2405.03970",
         use="Track known Pauli corrections through Clifford gates instead of applying physical pulses."),
    dict(title="Transversal Logical Clifford gates on rotated surface codes with reconfigurable neutral atom arrays",url="https://arxiv.org/abs/2412.01391",
         use="H changes patch orientation; fold-S requires a distinct mid-cycle protocol and connectivity."),
    dict(title="Error correction of transversal CNOT gates for scalable surface code computation",url="https://arxiv.org/abs/2408.01393",
         use="Transversal CNOT induces cross-block error correlations; independent memory decoding is insufficient."),
    dict(title="Decoding across transversal Clifford gates in the surface code",url="https://arxiv.org/abs/2505.13599",
         use="Gate-boundary decoding and fault distance must be checked, not inferred from gate depth."),
    dict(title="Restrictions on Transversal Encoded Quantum Gate Sets",url="https://arxiv.org/abs/0811.4262",
         use="A fixed error-detecting code cannot have a universal transversal unitary gate set."),
]


def workloads():
    return {
        "clifford_block": dict(rounds=5,operations=["H","X","H","Z"]*8+["H"],
            description="Highly compressible synthetic 33-gate Pauli/H block, net H; not a general algorithm speedup."),
        "scheduled_corrections": dict(rounds=14,operations={r:[("X","Y","Z")[(r-1)%3]] for r in range(1,14)},
            description="13 known Pauli corrections separated by mandatory QEC rounds; no QEC rounds removed."),
        "single_h_control": dict(rounds=5,operations=["H"],
            description="Already minimal logical H; identical physical circuits must give zero claimed gain."),
    }


def operation_counts(circuit):
    one = sum(len(i.targets_copy()) for i in circuit if i.name in {"H","X","Y","Z","S","S_DAG"})
    two = sum(len(i.targets_copy())//2 for i in circuit if i.name in {"CX","CZ"})
    return dict(one_qubit=one,two_qubit=two,total_clifford=one+two,
        measurements=circuit.num_measurements,detectors=circuit.num_detectors,ticks=circuit.num_ticks)


def logical_cost(algorithm,distance):
    before = sum(distance**2 if g == "H" else 2*distance-1 if g == "Y" else distance
                 for g in algorithm.plan.source_operations)
    after = distance**2*sum(plan_pauli_h(block).hadamard for block in algorithm.schedule.values())
    return dict(source_logical_gates=len(algorithm.plan.source_operations),
        physical_algorithm_one_qubit_before=before,physical_algorithm_one_qubit_after=after,
        physical_logical_gates_after=sum(plan_pauli_h(block).hadamard for block in algorithm.schedule.values()),
        removed_algorithm_one_qubit=before-after,
        algorithm_one_qubit_reduction_fraction=(before-after)/before if before else 0.)


def compare_results(before,after,*,identical=False):
    pb,pa = before["probability"],after["probability"]
    if identical:
        return dict(absolute_failure_reduction=0.,relative_failure_reduction=0.,
                    absolute_reduction_ci95=[0.,0.],relative_reduction_ci95=[0.,0.],
                    identical_physical_circuit=True,interval_method="same physical circuit; reused samples")
    # Bonferroni: two 97.5% Wilson intervals -> at least nominal 95% joint
    # asymptotic coverage for a difference or ratio of independent estimates.
    z = norm.ppf(1-.05/4)
    lb,hb = wilson_interval(before["errors"],before["shots"],z)
    la,ha = wilson_interval(after["errors"],after["shots"],z)
    return dict(absolute_failure_reduction=pb-pa,
        relative_failure_reduction=(pb-pa)/pb if pb else None,
        absolute_reduction_ci95=[lb-ha,hb-la],
        relative_reduction_ci95=[1-ha/lb,1-la/hb] if lb > 0 else None,
        identical_physical_circuit=False,
        interval_method="Bonferroni combination of independent 97.5% Wilson intervals; asymptotic")


def benchmark_cycle_ler(*,shots=100000,distance=3,p=.001,seed=8001,round_counts=(2,5,8,11,14)):
    """Fit a correction-every-cycle family, not a fixed memory's intrinsic LER.

    r-1 = 3k+1 keeps the same ending position in the periodic X/Y/Z correction
    schedule. Both variants preserve all r QEC rounds; only physical correction
    pulses differ. The exponential fit is an effective workload-cycle model.
    """
    model = noise_suite(p)["uniform"]
    rows,index = [],0
    for basis in ("X","Z"):
        for optimize in (False,True):
            points = []
            for rounds in round_counts:
                operations = {r:[("X","Y","Z")[(r-1)%3]] for r in range(1,rounds)}
                result,_,_ = simulate_memory(distance,rounds,basis=basis,model=model,shots=shots,
                    seed=seed+index,logical_operations=operations,optimize_logical_gates=optimize)
                index += 1
                points.append(result)
            # One initial QEC round is fixed preparation overhead; fit the r-1
            # correction-bearing computational cycles, not an unphysical
            # negative SPAM error forced by counting preparation as a cycle.
            fit = fit_ler([r["rounds"]-1 for r in points],[r["errors"] for r in points],
                          [r["shots"] for r in points],fit_spam=True)
            fit.update(cycle_counts=[r["rounds"]-1 for r in points],fixed_preparation_qec_rounds=1)
            pearson = sum((point["errors"]-point["shots"]*prob)**2/(point["shots"]*prob*(1-prob))
                          for point,prob in zip(points,fit["fitted_probabilities"]))
            fit.update(pearson_chi2=pearson,degrees_of_freedom=len(points)-2,
                       goodness_of_fit_pvalue=float(chi2.sf(pearson,len(points)-2)))
            fit["fit_diagnostic_warning"] = ("Exponential cycle model has p<0.05; interpret effective rate cautiously."
                if fit["goodness_of_fit_pvalue"] < .05 else None)
            row = dict(basis=basis,optimized=optimize,points=points,fit=fit,
                metric="Effective logical error per QEC cycle with scheduled corrections; not intrinsic memory LER",
                independent_round_fit_assumption=True,physical_p=p,seed=seed)
            rows.append(row)
            print(f"Cycle LER {basis} {'after' if optimize else 'before'}: {fit['epsilon_per_round']:.7g} "
                  f"CI={fit['ci95_profile']}",flush=True)
    return rows


def plot_benchmark(comparisons,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    selected = [c for c in comparisons if c["workload"] != "single_h_control"]
    y = np.arange(len(selected))
    labels = [f"{c['model']} / {c['workload']} / {c['basis']}" for c in selected]
    for offset,name,color in [(-.16,"before","#b45b38"),(.16,"after","#26778c")]:
        p = np.array([c[name]["probability"] for c in selected])
        lo = np.array([c[name]["ci95"][0] for c in selected])
        hi = np.array([c[name]["ci95"][1] for c in selected])
        axes[0].errorbar(p,y+offset,xerr=[p-lo,hi-p],fmt="o",label=name,color=color,capsize=2)
    axes[0].set_yticks(y,labels,fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Decoded workload logical-failure probability (95% point CI)")
    axes[0].legend()
    unique = {}
    for c in comparisons:
        unique[c["workload"]] = c
    names = list(unique)
    for offset,name,color in [(-.18,"before","#b45b38"),(.18,"after","#26778c")]:
        counts = [unique[k]["counts_"+name]["total_clifford"] for k in names]
        axes[1].bar(np.arange(len(names))+offset,counts,.36,label=name,color=color)
    axes[1].set_xticks(np.arange(len(names)),names,rotation=25,ha="right",fontsize=8)
    axes[1].set_ylabel("Total circuit Clifford gates, including QEC")
    axes[1].legend()
    fig.suptitle("Simulated transversal compilation — assumed/synthetic noise; fixed QEC rounds",fontsize=12)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def markdown_report(payload):
    lines = ["# Transversal-gate benchmark", "", "Simulated results; not IQM measurements. Baseline materializes corrections; optimized tracks Pauli frames and simplifies each unitary block. Every scheduled QEC round is retained.", "",
        "| Workload | Noise | Basis | Logical failure before | After | Relative reduction |", "|---|---|---|---:|---:|---:|"]
    for c in payload["comparisons"]:
        relative = c["comparison"]["relative_failure_reduction"]
        change = "n/a" if relative is None else f"{100*relative:.1f}%"
        lines.append(f"| {c['workload']} | {c['model']} | {c['basis']} | {c['before']['errors']}/{c['before']['shots']} ({100*c['before']['probability']:.3f}%) | {c['after']['errors']}/{c['after']['shots']} ({100*c['after']['probability']:.3f}%) | {change} |")
    if payload.get("cycle_ler"):
        lines += ["", "## Effective workload-cycle LER", "", "Separate fixed-shot sweeps use 2, 5, 8, 11 and 14 total QEC rounds. One initial QEC round is fixed preparation overhead; the fit uses 1, 4, 7, 10 and 13 correction-bearing computational cycles with a fitted SPAM amplitude. These are effective rates for the computational cycle, not a claim that the untouched memory circuit improved. Goodness-of-fit diagnostics are retained in JSON.", "", "| Basis | Before (95% profile interval) | After (95% profile interval) | Reduction |", "|---|---:|---:|---:|"]
        warnings = []
        for basis in ("X","Z"):
            before = next(r for r in payload["cycle_ler"] if r["basis"] == basis and not r["optimized"])["fit"]
            after = next(r for r in payload["cycle_ler"] if r["basis"] == basis and r["optimized"])["fit"]
            fmt = lambda r:f"{r['epsilon_per_round']:.7f} [{r['ci95_profile'][0]:.7f}, {r['ci95_profile'][1]:.7f}]"
            reduction = 1-after["epsilon_per_round"]/before["epsilon_per_round"]
            lines.append(f"| {basis} | {fmt(before)} | {fmt(after)} | {100*reduction:.1f}% |")
            if before.get("fit_diagnostic_warning") or after.get("fit_diagnostic_warning"):
                warnings.append(f"{basis}: approximate cycle model has a goodness-of-fit warning; see JSON diagnostics.")
        lines += ["",*warnings]
    lines += ["", "Intervals, seeds, counts and hashes are in results.json. Relative-reduction intervals use a conservative combination of independent Wilson intervals. Identical-circuit controls reuse samples and claim zero improvement.", "",
        "These are workload failure probabilities, not improved intrinsic memory LER per QEC round. No quantum speedup, arbitrary-algorithm gain, or IQM pulse-count gain is inferred. A strong existing physical Clifford compiler can already remove some within-block redundancy; the baseline is explicitly unoptimized. The scheduled-correction case separates corrections with QEC and does not rely on removing those rounds.", "", "Sources:", ""]
    lines += [f"- [{p['title']}]({p['url']}): {p['use']}" for p in PAPERS]
    return "\n".join(lines)+"\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shots",type=int,default=100000)
    p.add_argument("--distance",type=int,default=3)
    p.add_argument("--p",type=float,default=.001)
    p.add_argument("--seed",type=int,default=71)
    p.add_argument("--models",nargs="+",default=["uniform","leakage"])
    p.add_argument("--workloads",nargs="+",default=list(workloads()))
    p.add_argument("--bases",nargs="+",choices=["X","Z"],default=["X","Z"])
    p.add_argument("--calibration")
    p.add_argument("--synthetic-calibration",action="store_true")
    p.add_argument("--output",default="results/transversal")
    p.add_argument("--fit-cycle-ler",action="store_true")
    p.add_argument("--cycle-seed",type=int,default=8001)
    a = p.parse_args(argv)
    if a.calibration and a.synthetic_calibration:
        p.error("Choose one calibration source.")
    calibration = CalibrationSnapshot.load(a.calibration) if a.calibration else synthetic_snapshot(seed=a.seed) if a.synthetic_calibration else None
    models,programs = noise_suite(a.p,calibration),workloads()
    if set(a.models)-set(models) or set(a.workloads)-set(programs):
        p.error("Unknown model/workload, or calibrated model requested without snapshot.")
    output = Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    mapping = None
    if calibration:
        calibration.save(output/"calibration.json")
        placement = rank_placements(make_stim_circuit(a.distance,14,gate_basis="cz"),calibration=calibration)[0]
        mapping = placement.stim_to_physical
        (output/"placement.json").write_text(json.dumps(asdict(placement),indent=2)+"\n")
    comparisons,index = [],0
    for workload in a.workloads:
        spec = programs[workload]
        for basis in a.bases:
            base = make_logical_algorithm(a.distance,spec["rounds"],spec["operations"],basis=basis,optimize=False)
            opt = make_logical_algorithm(a.distance,spec["rounds"],spec["operations"],basis=basis,optimize=True)
            counts_before,counts_after = operation_counts(base.circuit),operation_counts(opt.circuit)
            assert counts_before["measurements"] == counts_after["measurements"]
            for name in a.models:
                results,circuits = [],[]
                for optimize in (False,True):
                    seed = a.seed+2*index+int(optimize)
                    if optimize and base.circuit == opt.circuit:
                        result = dict(results[0],logical_algorithm=opt.metadata())
                        circuit = circuits[0]
                    else:
                        result,circuit,decoder = simulate_memory(a.distance,spec["rounds"],
                            basis=basis,model=models[name],shots=a.shots,seed=seed,qubit_map=mapping,
                            logical_operations=spec["operations"],optimize_logical_gates=optimize)
                    results.append(result);circuits.append(circuit)
                    stem = f"{workload}_{basis}_{name}_{'after' if optimize else 'before'}"
                    (output/f"{stem}.stim").write_text(str(circuit)+"\n")
                    (output/f"{stem}.dem").write_text(str(circuit.detector_error_model(
                        decompose_errors=True,approximate_disjoint_errors=True))+"\n")
                index += 1
                comparison = compare_results(*results,identical=base.circuit == opt.circuit)
                row = dict(workload=workload,description=spec["description"],model=name,basis=basis,
                    counts_before=counts_before,counts_after=counts_after,
                    logical_cost=logical_cost(opt,a.distance),before=results[0],after=results[1],comparison=comparison)
                comparisons.append(row)
                print(f"{workload} {basis} {name}: {results[0]['errors']}/{a.shots} -> {results[1]['errors']}/{a.shots}; "
                      f"Clifford gates {counts_before['total_clifford']} -> {counts_after['total_clifford']}",flush=True)
    payload = dict(metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(a),
        noise_source=calibration.metadata if calibration else "Assumed circuit-level noise; not measured hardware",
        packages={k:importlib.metadata.version(k) for k in ("stim","pymatching","numpy","scipy")},
        models={name:{k:v for k,v in asdict(models[name]).items() if k != "calibration"} for name in a.models},
        baseline="Materialized transversal gates with explicit logical-step barriers, before logical optimization",
        metric="Decoded failure probability for the full workload, not intrinsic per-round memory LER",
        mandatory_qec_rounds_preserved=True,
        placement="Same fixed patch for both variants, bases and models",papers=PAPERS),comparisons=comparisons)
    if a.fit_cycle_ler:
        payload["cycle_ler"] = benchmark_cycle_ler(shots=a.shots,distance=a.distance,p=a.p,seed=a.cycle_seed)
    (output/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    (output/"report.md").write_text(markdown_report(payload),encoding="utf-8")
    plot_benchmark(comparisons,output/"comparison.png")
    print(f"Saved {output.resolve()}")
    return payload


if __name__ == "__main__":
    main()
