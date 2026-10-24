"""Fixed-shot DD on/off comparisons on identical QEC schedules.

python -m iqm_surface_code.benchmark_dd --shots 50000 --output results/dd_demo
All default calibration values are explicitly synthetic. Slow-phase noise is
an extra independent stress assumption, not inferred from the calibrated T2.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path

from .benchmark_logical import compare_results
from .calibration import CalibrationSnapshot,synthetic_snapshot
from .build_emerald_qubit_rotated import rank_placements
from .dynamical_decoupling import DDConfig
from .experiment import simulate_memory
from .noise_models import NoiseModel
from .surface_code import make_stim_circuit


def report_markdown(payload):
    lines = ["# Dynamical-decoupling benchmark", "",
        "Local simulations; synthetic/assumed noise unless a measured snapshot was supplied. Every DD variant retains the same QEC rounds, measurements, two-qubit interactions and modeled wall-clock duration. DD adds noisy one-qubit pulses.", "",
        "| Scenario | Basis | DD | Failure off | Failure on | Relative reduction (95% interval) | Added pulses |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in payload["comparisons"]:
        b,a,c = row["before"],row["after"],row["comparison"]
        ratio,ci = c["relative_failure_reduction"],c["relative_reduction_ci95"]
        change = "n/a" if ratio is None else f"{100*ratio:.1f}%"
        if ci is not None:
            change += f" [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]"
        lines.append(f"| {row['scenario']} | {row['basis']} | {row['sequence']} | {100*b['probability']:.3f}% | {100*a['probability']:.3f}% | {change} | {a['dd']['pulse_count']} |")
    lines += ["", "Intervals are per-comparison conservative combinations of independent Wilson intervals. Negative reductions mean DD performed worse. Rates above are full-memory logical-failure probabilities for one fixed round count, not fitted per-round LER.", "",
        "Markov-only T1/T2 is not refocused in this model. Slow-phase noise is Gaussian and constant within each individual idle window, with independent draws between windows; its average Z channel is analytically twirled. Symmetric instantaneous pi pulses refocus that extra component, while background decay and additive pulse errors remain. Finite-pulse control errors beyond the configured pulse-error channel and actual IQM scheduling require hardware characterization.", "",
        "Sources: [IQM DD compilation options](https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html), [IQM DD strategy](https://docs.iqm.tech/iqm-station-control-client/api/iqm.station_control.interface.models.circuit.DDStrategy.html), [superconducting-qubit noise spectroscopy and DD](https://arxiv.org/abs/1101.4707).", ""]
    return "\n".join(lines)


def plot_comparisons(rows,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    selected = [r for r in rows if r["sequence"] == "xx"]
    if not selected:
        selected = rows
    y = np.arange(len(selected))
    fig,ax = plt.subplots(figsize=(10,5),layout="constrained")
    for offset,key,color in [(-.15,"before","#b45b38"),(.15,"after","#26778c")]:
        probabilities = np.array([r[key]["probability"] for r in selected])
        low = np.array([r[key]["ci95"][0] for r in selected])
        high = np.array([r[key]["ci95"][1] for r in selected])
        ax.errorbar(probabilities,y+offset,xerr=[probabilities-low,high-probabilities],fmt="o",capsize=3,
                    label="DD off" if key == "before" else "DD on",color=color)
    ax.set_yticks(y,[f"{r['scenario']} / {r['basis']} / {r['sequence']}" for r in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Decoded logical-failure probability (95% point interval)")
    ax.set_title("DD comparison — assumed/synthetic noise; fixed duration and QEC rounds")
    ax.legend()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distance",type=int,default=3)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--shots",type=int,default=50000)
    p.add_argument("--sequences",nargs="+",choices=["xx","xy4","xy8"],default=["xx","xy4","xy8"])
    p.add_argument("--scenarios",nargs="+",choices=["markov_only","slow_phase","slow_phase_leakage"],
                   default=["markov_only","slow_phase","slow_phase_leakage"])
    p.add_argument("--bases",nargs="+",choices=["X","Z"],default=["X","Z"])
    p.add_argument("--calibration")
    p.add_argument("--detuning-std",type=float,default=.0008)
    p.add_argument("--pulse-error",type=float)
    p.add_argument("--seed",type=int,default=1901)
    p.add_argument("--output",default="results/dd_demo")
    a = p.parse_args(argv)
    snapshot = CalibrationSnapshot.load(a.calibration) if a.calibration else synthetic_snapshot(seed=a.seed)
    base = NoiseModel(name="markov_only",calibration=snapshot,coherence=True,idle=0)
    models = dict(markov_only=base,
        slow_phase=base.with_changes(name="slow_phase",idle_detuning_std_rad_per_ns=a.detuning_std),
        slow_phase_leakage=base.with_changes(name="slow_phase_leakage",idle_detuning_std_rad_per_ns=a.detuning_std,
                                           leakage=.001,seepage=.01,leakage_spread=.02))
    output = Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    snapshot.save(output/"calibration.json")
    template = make_stim_circuit(a.distance,a.rounds,gate_basis="cz")
    placement = rank_placements(template,calibration=snapshot)[0]
    mapping = placement.stim_to_physical
    (output/"placement.json").write_text(json.dumps(asdict(placement),indent=2)+"\n")
    comparisons,index = [],0
    for scenario in a.scenarios:
        for basis in a.bases:
            before,circuit,decoder = simulate_memory(a.distance,a.rounds,basis=basis,model=models[scenario],
                shots=a.shots,seed=a.seed+index,qubit_map=mapping)
            index += 1
            stem = f"{scenario}_{basis}_off"
            (output/f"{stem}.stim").write_text(str(circuit)+"\n")
            (output/f"{stem}.dem").write_text(str(decoder.detector_error_model(decompose_errors=True,
                approximate_disjoint_errors=True))+"\n")
            for sequence in a.sequences:
                config = DDConfig(sequence=sequence,pulse_error_probability=a.pulse_error)
                after,circuit,decoder = simulate_memory(a.distance,a.rounds,basis=basis,model=models[scenario],
                    shots=a.shots,seed=a.seed+index,qubit_map=mapping,dd=config)
                index += 1
                assert before["circuit_info"]["num_measurements"] == after["circuit_info"]["num_measurements"]
                assert before["circuit_info"]["two_qubit_gates"] == after["circuit_info"]["two_qubit_gates"]
                row = dict(scenario=scenario,basis=basis,sequence=sequence,before=before,after=after,
                           comparison=compare_results(before,after))
                comparisons.append(row)
                stem = f"{scenario}_{basis}_{sequence}"
                (output/f"{stem}.stim").write_text(str(circuit)+"\n")
                (output/f"{stem}.dem").write_text(str(decoder.detector_error_model(decompose_errors=True,
                    approximate_disjoint_errors=True))+"\n")
                print(f"{scenario} {basis} {sequence}: {before['errors']}/{a.shots} -> {after['errors']}/{a.shots}; "
                      f"{after['dd']['pulse_count']} pulses",flush=True)
    payload = dict(metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(a),
        calibration_source=snapshot.metadata,
        models={name:{k:v for k,v in asdict(models[name]).items() if k != "calibration"} for name in a.scenarios},
        packages={name:importlib.metadata.version(name) for name in ["stim","pymatching","numpy","scipy"]},
        physical_pulse_model="additive pulse error plus unreduced full-window Markov decay",
        slow_noise_model="independent Gaussian static Z detuning per idle window; instantaneous DD filter",
        actual_server_dd_tested=False),comparisons=comparisons)
    (output/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    (output/"report.md").write_text(report_markdown(payload),encoding="utf-8")
    plot_comparisons(comparisons,output/"comparison.png")
    return payload


if __name__ == "__main__":
    main()
