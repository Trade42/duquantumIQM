"""d=3 no-reset memory on IQM hardware through iyanu.qpu, then every decoder on the same shots.

    python -m iyanu.hw_run prepare --device garnet --rounds 1 2 3 4           # read-only
    DQ_QPU=run DQ_QPU_MAX_SHOTS=1000 python -m iyanu.hw_run run --device garnet --rounds 2 --shots 1000 --task garnet-pilot
    python -m iyanu.hw_run decode archive/iyanu/hw_data/<date>-<task>/<job id> [...]

`prepare` and `run` share one preparation: live calibration, a placement ranked on it over
the device's coupler grid, native compilation with zero SWAPs, and a check of the serialized
run request (every CZ on a coupler and equal to the placement). `decode` scores PyMatching
(uniform and calibrated weights) and the MLP trained only on the calibrated decoder circuit,
per round count with 95% Wilson intervals, and fits eps_L across round counts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def prepare(device, rounds_list, *, bases=("Z",), distance=3, measure_ns, variant="ours"):
    from challenge.run_on_hardware import prepare_qiskit
    from iyanu import qpu
    from iyanu.garnet_sim import NATIVE, grid_from_couplings
    b = qpu.backend(device)
    cal = qpu.calibration(b, measure_ns=measure_ns)
    grid = grid_from_couplings(qpu.couplings(b))
    prepared = []
    for basis in bases:
        for r in rounds_list:
            if variant == "ours":
                p = prepare_qiskit(b, distance, r, basis, calibration=cal, grid=grid, rows=len(grid), cols=len(grid[0]))
            else:
                from iyanu.circuits import prepare as prepare_variant
                p = prepare_variant(b, variant, distance, r, basis, calibration=cal, grid=grid)
            p["variant"] = variant
            extra = set(p["circuit"].count_ops())-NATIVE-{"delay"}
            if extra:
                raise qpu.QPUGuardError(f"{basis} rounds={r}: non-native operations {extra}")
            p["rounds"], p["basis"] = r, basis
            p["circuit"].name = f"d{distance}_r{r}_{basis}_noreset" + ("" if variant == "ours" else f"_{variant}")
            prepared.append(p)
    sent = qpu.check_request(b.create_run_request([p["circuit"] for p in prepared], shots=1), prepared, b)
    return b, cal, prepared, sent


def report(b, cal, prepared, sent):
    from iyanu import qpu
    p0 = prepared[0]
    phys = sorted(set(p0["placement"].stim_to_physical.values()), key=lambda q: int(q[2:]))
    print(f"device {b.name}, calibration set {qpu.calset(b)}")
    print(f"placement: {len(phys)} qubits {phys}; score {p0['placement'].score:.3f}; "
          f"{p0['report']['placement_candidates']} candidates")
    for p, s in zip(prepared, sent):
        print(f"  {s['name']}: CZ {s['cz']} (all on couplers, match placement), routing SWAPs "
              f"{p['report']['routing_swaps']}, depth {p['report']['depth']}")
    used = {frozenset((p0["placement"].stim_to_physical[a], p0["placement"].stim_to_physical[c]))
            for a, c in __import__("challenge.build_emerald_qubit_rotated", fromlist=["x"]).interaction_pairs(p0["stim_circuit"])}
    cz = [cal.edge_error(*sorted(e)) for e in used]
    ro = [cal.qubits[q].readout_error for q in phys]
    t1 = [cal.qubits[q].t1_ns/1e3 for q in phys]
    print(f"  used couplers: CZ infidelity median {np.median(cz):.4f}, max {max(cz):.4f}; "
          f"readout error median {np.median(ro):.4f}, max {max(ro):.4f}; T1 median {np.median(t1):.1f} us")


def decode(dirs, steps, device):
    import stim
    from challenge.decoding import make_matcher, wilson_interval, fit_ler
    from challenge.calibration import CalibrationSnapshot
    from challenge.noise_models import NoiseModel, add_noise
    from challenge.extract_syndromes import extract_syndromes
    from iyanu.neural_gpu import NeuralConfig, train_neural_decoder, paired_comparison
    rows = []
    for d in map(Path, dirs):
        meta = json.loads((d/"meta.json").read_text())
        cal = CalibrationSnapshot.load(d/"calibration.json")
        data = np.load(d/"measurements.npz")
        bases = meta.get("bases", ["Z"]*len(meta["sent"]))
        for i, (s, r, basis) in enumerate(zip(meta["sent"], meta["rounds"], bases)):
            ideal = stim.Circuit((d/f"{s['name']}.stim").read_text())
            qmap = {int(k): v for k, v in meta["stim_to_physical"][i].items()}
            syn = extract_syndromes(data[f"c{i}"].astype(bool), ideal, print_summary=False)
            dets, obs = np.asarray(syn["det_events"], bool), np.asarray(syn["obs_flips"], bool)
            calibrated = add_noise(ideal, NoiseModel(name="calibrated_pauli", calibration=cal, idle=0), qubit_map=qmap)
            uniform = add_noise(ideal, NoiseModel.uniform(0.005), qubit_map=qmap)
            m_cal, m_uni = make_matcher(calibrated), make_matcher(uniform)
            preds = {"pymatching_uniform": m_uni.decode_batch(dets, enable_correlations=True),
                     "pymatching_calibrated": m_cal.decode_batch(dets, enable_correlations=True)}
            if steps:
                nn = train_neural_decoder(calibrated, NeuralConfig(steps=steps, seed=r, device=device))
                preds["mlp"] = nn.decode_batch(dets)
            n = len(obs)
            requested = meta["shots"]
            row = dict(job=meta["job_id"], basis=basis, rounds=r, shots=n, requested_shots=requested,
                       kept_fraction=n/requested, options=meta.get("options"), detection_rate=float(dets.mean()),
                       raw_flip_rate=float(np.any(obs, 1).mean()),
                       errors={k: int(np.any(v != obs, 1).sum()) for k, v in preds.items()})
            if "mlp" in preds:
                row["mlp_vs_calibrated"] = paired_comparison(preds["mlp"], preds["pymatching_calibrated"], obs)
            rows.append(row)
            print(f"{basis} rounds {r} ({n}/{requested} shots kept, detection rate {row['detection_rate']:.3f}): " + "  ".join(
                f"{k} {e/n:.4f} [{wilson_interval(e, n)[0]:.4f},{wilson_interval(e, n)[1]:.4f}]"
                for k, e in row["errors"].items()), flush=True)
    fits = {}
    for basis in sorted({r["basis"] for r in rows}):
        sub = [r for r in rows if r["basis"] == basis]
        if len({r["rounds"] for r in sub}) < 2:
            continue
        fits[basis] = {}
        for k in sub[0]["errors"]:
            f = fits[basis][k] = fit_ler([r["rounds"] for r in sub], [r["errors"][k] for r in sub], [r["shots"] for r in sub])
            print(f"  [{basis}] {k:<22} eps_L = {f['epsilon_per_round']:.4f} [{f['ci95_profile'][0]:.4f}, {f['ci95_profile'][1]:.4f}]")
    out = Path(dirs[0]).parent/"decoded.json"
    out.write_text(json.dumps(dict(rows=rows, fits=fits), indent=2, default=str))
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run"):
        s = sub.add_parser(name)
        s.add_argument("--device", default="garnet")
        s.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4])
        s.add_argument("--bases", nargs="+", default=["Z"], choices=["Z", "X"])
        s.add_argument("--variant", default="ours", choices=["ours", "ours-iqm", "vanilla"],
                       help="circuit construction/compilation (see iyanu.circuits)")
        s.add_argument("--measure-ns", type=float, default=1500., help="readout duration assumed by the noise model")
        if name == "run":
            s.add_argument("--shots", type=int, required=True)
            s.add_argument("--task", required=True)
            s.add_argument("--purpose", default="")
            s.add_argument("--dd", action="store_true", help="IQM dynamical decoupling (standard strategy)")
            s.add_argument("--heralding", action="store_true", help="zeros heralding: post-selects shots, counted")
            s.add_argument("--active-reset", type=int, default=None, help="active reset cycles before each shot")
    s = sub.add_parser("decode")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "decode":
        return decode(a.dirs, a.steps, a.device)
    b, cal, prepared, sent = prepare(a.device, a.rounds, bases=a.bases, measure_ns=a.measure_ns, variant=a.variant)
    report(b, cal, prepared, sent)
    if a.cmd == "run":
        from iyanu import qpu
        opts = qpu.compilation_options(dd=a.dd, heralding=a.heralding, active_reset=a.active_reset)
        out = qpu.submit(b, prepared, shots=a.shots, task=a.task, purpose=a.purpose, calibration_snapshot=cal, options=opts)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
