"""Reproducible memory/noise sweeps, diagnostics, plots and circuit artifacts.

Example: python -m iqm_surface_code.experiment --rounds 1 3 5 --shots 10000
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.special import betaln, xlogy
from scipy.optimize import brentq

from .surface_code import make_stim_circuit, get_circuit_info
from .noise_models import noise_suite, add_noise
from .calibration import CalibrationSnapshot, synthetic_snapshot
from .build_emerald_qubit_rotated import rank_placements
from .decoding import make_matcher, fit_ler, wilson_interval
from .leakage import sample_leakage


def confidence_sequence(errors, shots, alpha=.05):
    """Anytime-valid beta-mixture Bernoulli interval (optional stopping safe)."""
    if shots < 1 or not 0 <= errors <= shots or not 0 < alpha < 1:
        raise ValueError("Invalid binomial counts or confidence level.")
    k, n = errors, shots
    const = betaln(k+.5, n-k+.5)-betaln(.5, .5)-np.log(1/alpha)
    def root(p):
        return const-xlogy(k, p)-xlogy(n-k, 1-p)
    p = k/n
    lo = 0. if k == 0 else brentq(root, 0, p)
    hi = 1. if k == n else brentq(root, p, 1)
    return [lo, hi]


def simulate_memory(distance, rounds, *, basis="Z", no_reset=True, model=None,
                    shots=10000, batch_size=10000, target_errors=None, seed=0,
                    gate_basis="cz", qubit_map=None, decoder_model=None,
                    correlated=True, transversal_gate_set="pauli_h", logical_operations=(),
                    logical_after_round=1, optimize_logical_gates=False,dd=None):
    """Fixed shots by default; optional paper-style early stopping by failures.

    Decoder always uses the static Pauli DEM, including when physical sampling
    includes persistent leakage. This model mismatch is explicitly recorded.
    """
    if shots < 1 or batch_size < 1 or (target_errors is not None and target_errors < 1):
        raise ValueError("Shot, batch and target-error budgets must be positive.")
    model = model or noise_suite()["uniform"]
    from .logical_gates import make_logical_algorithm
    algorithm = make_logical_algorithm(distance,rounds,logical_operations,basis=basis,
        no_reset=no_reset,gate_basis=gate_basis,transversal_gate_set=transversal_gate_set,
        logical_after_round=logical_after_round,optimize=optimize_logical_gates)
    ideal = algorithm.circuit
    from .dynamical_decoupling import plan_dd
    dd_plan = plan_dd(ideal,model,dd,qubit_map=qubit_map)
    circuit = add_noise(ideal, model, qubit_map=qubit_map,dd=dd_plan)
    decoder = circuit if decoder_model is None else add_noise(ideal, decoder_model, qubit_map=qubit_map,dd=dd_plan)
    matcher = make_matcher(decoder, correlated=correlated)
    leakage_enabled = bool(model.leakage or (model.calibration and
        any(model.calibration.qubits[p].leakage_per_gate for p in qubit_map.values())))
    sampler = None if leakage_enabled else circuit.compile_detector_sampler(seed=seed)
    rng = np.random.default_rng(seed)
    actual, errors, raw_errors = 0, 0, 0
    firing = np.zeros(ideal.num_detectors, dtype=np.int64)
    leakage_totals, started = [], time.perf_counter()
    while actual < shots:
        count = min(batch_size, shots-actual)
        if leakage_enabled:
            raw, diagnostic = sample_leakage(circuit, model, count,
                seed=int(rng.integers(0, 2**63)), qubit_map=qubit_map)
            dets, obs = ideal.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
            leakage_totals.append((count, diagnostic))
        else:
            dets, obs = sampler.sample(count, separate_observables=True)
        predictions = matcher.decode_batch(dets, enable_correlations=correlated)
        errors += int(np.any(predictions != obs, axis=1).sum())
        raw_errors += int(np.any(obs, axis=1).sum())
        firing += dets.sum(axis=0, dtype=np.int64)
        actual += count
        if target_errors is not None and errors >= target_errors:
            break
    try:
        graphlike_distance = len(decoder.shortest_graphlike_error())
    except ValueError as error:
        if "Failed to find any graphlike logical errors" not in str(error):
            raise
        graphlike_distance = None  # No witness under this model's supported faults.
    result = dict(distance=distance, rounds=rounds, basis=basis, no_reset=no_reset,
        gate_basis=gate_basis, model=model.name, decoder_model=(decoder_model or model).name,
        decoder="correlated MWPM" if correlated else "MWPM", seed=seed,
        shots=actual, max_shots=shots, errors=errors, probability=errors/actual,
        ci95=list(wilson_interval(errors, actual)) if target_errors is None else confidence_sequence(errors, actual),
        interval_method="Wilson fixed-shot" if target_errors is None else "anytime-valid beta-mixture confidence sequence",
        target_errors=target_errors, stopped_early=actual < shots,
        raw_logical_flip_probability=raw_errors/actual,
        detector_firing_rates=(firing/actual).tolist(),
        mean_syndrome_weight=float(firing.sum()/actual), elapsed_seconds=time.perf_counter()-started,
        circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
        decoder_sha256=hashlib.sha256(str(decoder).encode()).hexdigest(),
        circuit_info=get_circuit_info(circuit),
        dd=dd_plan.metadata() if dd_plan else dict(enabled=False),
        logical_algorithm=algorithm.metadata(),
        graphlike_distance=graphlike_distance,
        leakage_decoder_mismatch=leakage_enabled,
        dem_disjoint_channel_approximation=True)
    if leakage_totals:
        result["leakage"] = dict(model=leakage_totals[0][1]["model"],
            leakage_entries=sum(d["leakage_entries"] for _, d in leakage_totals),
            leaked_measurements=sum(d["leaked_measurements"] for _, d in leakage_totals),
            mean_leaked_fraction=sum(s*d["mean_leaked_fraction"] for s, d in leakage_totals)/actual,
            final_leaked_fraction=sum(s*d["final_leaked_fraction"] for s, d in leakage_totals)/actual,
            hidden_flags_given_to_decoder=False)
    return result, circuit, decoder


def plot_results(records, fits, path, title="Surface-code memory simulation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    groups = defaultdict(list)
    for record in records:
        groups[(record["model"], record["basis"], record["no_reset"])].append(record)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    names = list(dict.fromkeys(r["model"] for r in records))
    colors = {name: plt.get_cmap("tab10")(i % 10) for i,name in enumerate(names)}
    max_probability = max((r["ci95"][1] for r in records), default=.5)
    for (model, basis, no_reset), values in groups.items():
        values = sorted(values, key=lambda r: r["rounds"])
        rounds = np.array([r["rounds"] for r in values])
        probs = np.array([r["probability"] for r in values])
        low = np.array([r["ci95"][0] for r in values])
        high = np.array([r["ci95"][1] for r in values])
        label = f"{model} {basis} {'no reset' if no_reset else 'reset'}"
        axes[0].errorbar(rounds, probs, yerr=[np.maximum(0,probs-low), np.maximum(0,high-probs)],
                         marker="o" if basis == "X" else "^", linestyle="-" if basis == "X" else "--",
                         color=colors[model], capsize=2, label=label, linewidth=1)
    axes[0].set(xlabel="Stabilizer rounds", ylabel="Decoded logical error probability",
                ylim=(0, max(.03, min(.55, max_probability*1.15))))
    axes[0].legend(fontsize=6)
    labels, eps, lows, highs = [], [], [], []
    for fit in fits:
        labels.append(f"{fit['model']} {fit['basis']} {'NR' if fit['no_reset'] else 'R'}")
        eps.append(fit["epsilon_per_round"])
        lows.append(fit["ci95_profile"][0])
        highs.append(fit["ci95_profile"][1])
    if labels:
        values = np.array(eps)
        axes[1].errorbar(np.arange(len(labels)), values,
            yerr=[values-np.array(lows), np.array(highs)-values], fmt="o", capsize=3)
        axes[1].set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=6)
    axes[1].set(ylabel="Fitted logical error rate per round")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--bases", nargs="+", choices=["X", "Z"], default=["X", "Z"])
    p.add_argument("--models", nargs="+", default=["uniform", "biased_z", "readout_heavy", "correlated_zz", "leakage"])
    p.add_argument("--p", type=float, default=.001)
    p.add_argument("--shots", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--target-errors", type=int)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gate-basis", choices=["cx", "cz"], default="cz")
    p.add_argument("--dd",choices=["off","xx","xy4","xy8"],default="off")
    p.add_argument("--dd-target",choices=["data","all"],default="data")
    p.add_argument("--dd-pulse-ns",type=float)
    p.add_argument("--dd-pulse-error",type=float)
    p.add_argument("--idle-detuning-std",type=float,default=0.,help="Extra independent slow Z-detuning stress, rad/ns per idle window")
    p.add_argument("--reset-mode", choices=["no-reset", "reset", "both"], default="no-reset")
    p.add_argument("--calibration")
    p.add_argument("--synthetic-calibration", action="store_true")
    p.add_argument("--grid", help="JSON rectangular array of QB names/nulls; needed for nonstandard topology")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--decoder-model", help="Use same parity circuit with another noise model's decoder weights")
    p.add_argument("--uncorrelated", action="store_true")
    p.add_argument("--leakage-rate", type=float, help="Override per-participant two-qubit leakage in leakage models")
    p.add_argument("--seepage", type=float, help="Override leakage recovery probability per moment")
    p.add_argument("--leakage-spread", type=float)
    p.add_argument("--placement-readout-weight", type=float, default=1.)
    p.add_argument("--placement-coherence-weight", type=float, default=1.)
    p.add_argument("--placement-leakage-weight", type=float, default=0.)
    p.add_argument("--fit-spam", action="store_true")
    p.add_argument("--transversal-gate-set", choices=["none","pauli","pauli_h","css"], default="pauli_h")
    p.add_argument("--logical-operations", nargs="*", default=[])
    p.add_argument("--logical-after-round", type=int, default=1)
    p.add_argument("--optimize-logical-gates", action="store_true")
    p.add_argument("--output", default="results/surface_code")
    args = p.parse_args(argv)
    if args.calibration and args.synthetic_calibration:
        p.error("Choose real or synthetic calibration, not both.")
    calibration = (CalibrationSnapshot.load(args.calibration) if args.calibration else
                   synthetic_snapshot(args.rows, args.cols, args.seed) if args.synthetic_calibration else None)
    if args.dd != "off" and calibration is None:
        p.error("Local DD requires --calibration or --synthetic-calibration for idle-window timing.")
    models = noise_suite(args.p, calibration)
    if args.idle_detuning_std:
        if calibration is None:
            p.error("Slow-detuning simulation requires --calibration or --synthetic-calibration.")
        models = {name:model.with_changes(idle_detuning_std_rad_per_ns=args.idle_detuning_std)
                  if model.calibration else model for name,model in models.items()}
        if any(not models[name].calibration for name in args.models if name in models):
            p.error("--idle-detuning-std requires calibrated noise models.")
    from .dynamical_decoupling import DDConfig
    dd_config = None if args.dd == "off" else DDConfig(sequence=args.dd,target=args.dd_target,
        pulse_duration_ns=args.dd_pulse_ns,pulse_error_probability=args.dd_pulse_error,
        durations_ns=calibration.durations_ns if calibration else None)
    overrides = {k:v for k,v in {"leakage":args.leakage_rate, "seepage":args.seepage,
                                 "leakage_spread":args.leakage_spread}.items() if v is not None}
    if overrides:
        models = {name:model.with_changes(**overrides) if "leakage" in name else model
                  for name,model in models.items()}
    if "all" in args.models:
        args.models = list(models)
    if set(args.models) - set(models) or (args.decoder_model and args.decoder_model not in models):
        p.error(f"Available models: {', '.join(models)}; calibrated models require a snapshot.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if calibration:
        calibration.save(output/"calibration.json")
    grid = json.loads(Path(args.grid).read_text()) if args.grid else None
    template = make_stim_circuit(args.distance, max(args.rounds), gate_basis=args.gate_basis)
    placement = None
    if calibration or grid:
        placement = rank_placements(template, calibration=calibration, grid=grid,
             rows=args.rows, cols=args.cols, readout_weight=args.placement_readout_weight,
             coherence_weight=args.placement_coherence_weight,
             leakage_weight=args.placement_leakage_weight)[0]
        (output/"placement.json").write_text(json.dumps(asdict(placement), indent=2)+"\n")
    reset_modes = [True, False] if args.reset_mode == "both" else [args.reset_mode == "no-reset"]
    records, fits, index = [], [], 0
    for no_reset in reset_modes:
        for name in args.models:
            for basis in args.bases:
                group = []
                for rounds in sorted(set(args.rounds)):
                    result, circuit, decoder = simulate_memory(args.distance, rounds,
                        basis=basis, no_reset=no_reset, model=models[name], shots=args.shots,
                        batch_size=args.batch_size, target_errors=args.target_errors,
                        seed=args.seed+index, gate_basis=args.gate_basis,
                        qubit_map=placement.stim_to_physical if placement else None,
                        decoder_model=models.get(args.decoder_model), correlated=not args.uncorrelated,
                        transversal_gate_set=args.transversal_gate_set,logical_operations=args.logical_operations,
                        logical_after_round=args.logical_after_round,optimize_logical_gates=args.optimize_logical_gates,
                        dd=dd_config)
                    index += 1
                    stem = f"{name}_{basis}_{'nr' if no_reset else 'r'}_{rounds}"
                    (output/f"{stem}.stim").write_text(str(circuit)+"\n")
                    (output/f"{stem}.dem").write_text(str(decoder.detector_error_model(
                        decompose_errors=True, approximate_disjoint_errors=True))+"\n")
                    print(f"{stem}: {result['errors']}/{result['shots']} P={result['probability']:.5g}", flush=True)
                    group.append(result)
                    records.append(result)
                if len(group) >= (3 if args.fit_spam else 2):
                    fit = fit_ler([r["rounds"] for r in group], [r["errors"] for r in group],
                                   [r["shots"] for r in group], fit_spam=args.fit_spam)
                    fit.update(model=name, basis=basis, no_reset=no_reset,
                        optional_stopping=bool(args.target_errors),
                        fit_interval_caveat="Profile fit intervals are asymptotic, not anytime-valid." if args.target_errors else None)
                    fits.append(fit)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ["stim", "pymatching", "numpy", "scipy"]},
        arguments=vars(args), calibration_source=calibration.metadata if calibration else None,
        placement_policy="Fixed patch chosen on Z/no-reset maximum-round schedule; reused across models/bases for comparison.",
        models={name: {k:v for k,v in asdict(models[name]).items() if k != "calibration"} for name in args.models},
        paper="2606.19482v2 (supplied Markdown)",
        limitations=["Leakage uses persistent phenomenological flags, not qutrit dynamics.",
            "T1/T2 and coherent_twirl use Pauli approximations; exact alternative in aer_simulation.py.",
            "No leakage-clearing reset is assumed for Emerald no-reset experiments.",
            "Static MWPM DEM does not represent persistent hidden leakage correlations.",
            "Placement minimizes a proxy over translations/reflections/rotations, not all graph embeddings."])
    payload = dict(metadata=metadata, records=records, fits=fits)
    (output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    source = calibration.metadata.get("source", "Calibration snapshot") if calibration else "Uniform/stress assumptions"
    plot_results(records, fits, output/"logical_errors.png",
                  title=f"Simulated d={args.distance} memory — {source}")
    print(f"Saved {output.resolve()}")
    return payload


if __name__ == "__main__":
    main()
