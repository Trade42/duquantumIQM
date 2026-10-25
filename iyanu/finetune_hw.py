"""Decoders pre-trained on simulation, fine-tuned on hardware shots, scored on held-out hardware shots.

    python -m iyanu.finetune_hw --train iyanu/hw_data/<date>-garnet-train/* --test iyanu/hw_data/<date>-garnet-test/*

For each (basis, rounds) circuit in the jobs: training shots (all --train jobs) are split
90/10 into fit and validation; the test shots (--test jobs) are only ever scored. Decoders:

  pm_uniform        PyMatching, uniform p=0.005 weights
  pm_calibrated     PyMatching, weights from the job's recorded live calibration (Pauli model)
  pm_coherence      PyMatching on the calibration with T1/T2 idle decoherence; readout duration
                    chosen on the validation split. The learned decoders pre-train on this model.
  lookup_sim        full-history lookup built from uniform-noise simulation (the benchmark's kind)
  lookup_hw         full-history lookup built from the hardware training shots
  mlp_sim, mlp_ft   MLP trained on calibrated simulation; then fine-tuned on hardware fit shots
  conv_sim, conv_ft 3D conv decoder, likewise
  pre_sim           conv pre-decoder (simulation labels) + calibrated PyMatching
  uf, bposd, belief_find   ldpc Union-Find, BP+OSD, BP+Union-Find on the calibrated DEM
  ens_ft            mean of mlp_ft and conv_ft flip probabilities (no extra fitting)

Lookup tables answer unseen syndromes with calibrated PyMatching and report how often.
Fine-tuning keeps the checkpoint with the fewest validation failures. Reports P_L with 95%
Wilson intervals, paired comparison with pm_calibrated, and eps_L fits per basis.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(dirs):
    """{(basis, rounds): dict(ideal, qmap, cal, dets, obs, jobs)} merged over jobs."""
    import stim
    from challenge.calibration import CalibrationSnapshot
    from challenge.extract_syndromes import extract_syndromes
    out = {}
    for d in map(Path, dirs):
        if not (d/"meta.json").exists() or not (d/"measurements.npz").exists():
            print(f"skipping {d}: job not finished (no meta.json/measurements.npz)")
            continue
        meta = json.loads((d/"meta.json").read_text())
        if not meta.get("success"):
            print(f"skipping {d}: job did not succeed")
            continue
        data = np.load(d/"measurements.npz")
        cal = CalibrationSnapshot.load(d/"calibration.json")
        for i, s in enumerate(meta["sent"]):
            key = (meta.get("bases", ["Z"]*len(meta["sent"]))[i], meta["rounds"][i])
            ideal = stim.Circuit((d/f"{s['name']}.stim").read_text())
            syn = extract_syndromes(data[f"c{i}"].astype(bool), ideal, print_summary=False)
            dets, obs = np.asarray(syn["det_events"], bool), np.asarray(syn["obs_flips"], bool)
            qmap = {int(k): v for k, v in meta["stim_to_physical"][i].items()}
            if key in out:
                if out[key]["ideal"] != ideal or out[key]["qmap"] != qmap:
                    raise ValueError(f"{key}: jobs differ in circuit or placement; refusing to merge.")
                out[key]["dets"] = np.vstack([out[key]["dets"], dets])
                out[key]["obs"] = np.vstack([out[key]["obs"], obs])
                out[key]["jobs"].append(meta["job_id"])
            else:
                out[key] = dict(ideal=ideal, qmap=qmap, cal=cal, dets=dets, obs=obs, jobs=[meta["job_id"]],
                                calibration_set_id=meta["calibration_set_id"])
    return out


def _keys(dets):
    if dets.shape[1] > 63:
        raise ValueError("lookup supports at most 63 detectors")
    return dets.astype(np.uint64) @ (np.uint64(1) << np.arange(dets.shape[1], dtype=np.uint64))


class Lookup:
    """Majority observable per full syndrome; unseen syndromes go to a fallback decoder."""

    def __init__(self, dets, obs, fallback):
        keys = _keys(dets)
        uniq, inv = np.unique(keys, return_inverse=True)
        ones = np.bincount(inv, weights=obs[:, 0].astype(float), minlength=len(uniq))
        total = np.bincount(inv, minlength=len(uniq))
        self.table = dict(zip(uniq.tolist(), (ones*2 > total).tolist()))
        self.fallback, self.last_unseen = fallback, 0.

    def decode_batch(self, dets):
        keys = _keys(dets).tolist()
        hit = np.array([k in self.table for k in keys])
        out = np.zeros((len(dets), 1), bool)
        out[hit, 0] = [self.table[k] for k, h in zip(keys, hit) if h]
        if (~hit).any():
            out[~hit] = self.fallback(dets[~hit])
        self.last_unseen = float((~hit).mean())
        return out


def finetune(module, forward, dets, obs, val, *, steps, batch, lr, device, eval_every=200):
    """Continue training on hardware shots; keep the state with the fewest validation failures."""
    import torch
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    vd = torch.from_numpy(val[0].astype(np.float32)).to(device)
    vo = val[1]
    rng = np.random.default_rng(0)

    def val_fail():
        with torch.inference_mode():
            p = np.concatenate([(forward(vd[i:i+8192]) > 0).cpu().numpy() for i in range(0, len(vd), 8192)])
        return int(np.any(p != vo, 1).sum())

    best, best_state, history = val_fail(), copy.deepcopy(module.state_dict()), []
    for step in range(steps):
        idx = rng.integers(0, len(dets), batch)
        x = torch.from_numpy(dets[idx].astype(np.float32)).to(device)
        y = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
        loss = loss_fn(forward(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1) % eval_every == 0:
            f = val_fail()
            history.append((step+1, f))
            if f < best:
                best, best_state = f, copy.deepcopy(module.state_dict())
    module.load_state_dict(best_state)
    return dict(val_failures=best, val_shots=len(vo), history=history)


def main(argv=None):
    from challenge.decoding import make_matcher, wilson_interval, fit_ler
    from challenge.noise_models import NoiseModel, add_noise
    from iyanu.neural_gpu import NeuralConfig, train_neural_decoder, paired_comparison, pick_device
    from iyanu.conv_decoders import ConvConfig, ConvDecoder, ConvPredecoder
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--sim-steps", type=int, default=3000)
    ap.add_argument("--ft-steps", type=int, default=4000)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--lookup-sim-shots", type=int, default=2_000_000)
    ap.add_argument("--skip", nargs="*", default=[], help="decoder names to skip")
    ap.add_argument("--pretrain-pauli", action="store_true", help="pre-train on the Pauli calibration model instead of the coherence one")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="iyanu/hw_data/decoder_eval")
    a = ap.parse_args(argv)
    device = pick_device(a.device)
    train, test = load(a.train), load(a.test)
    rows = []
    for key in sorted(test, key=lambda k: (k[0], k[1])):
        basis, rounds = key
        if key not in train:
            print(f"{key}: no training shots, skipped")
            continue
        tr, te = train[key], test[key]
        if tr["ideal"] != te["ideal"] or tr["qmap"] != te["qmap"]:
            raise ValueError(f"{key}: training and test circuits differ")
        started = time.perf_counter()
        n_fit = int(0.9*len(tr["obs"]))
        perm = np.random.default_rng(1).permutation(len(tr["obs"]))
        fit_idx, val_idx = perm[:n_fit], perm[n_fit:]
        fit = (tr["dets"][fit_idx], tr["obs"][fit_idx])
        val = (tr["dets"][val_idx], tr["obs"][val_idx])
        calibrated = add_noise(tr["ideal"], NoiseModel(name="calibrated_pauli", calibration=tr["cal"], idle=0), qubit_map=tr["qmap"])
        uniform = add_noise(tr["ideal"], NoiseModel.uniform(0.005), qubit_map=tr["qmap"])
        m_cal, m_uni = make_matcher(calibrated), make_matcher(uniform)
        pm_cal = lambda d: m_cal.decode_batch(d, enable_correlations=True)
        pm_uni = lambda d: m_uni.decode_batch(d, enable_correlations=True)
        dec, info = {"pm_uniform": pm_uni, "pm_calibrated": pm_cal}, {}
        # Coherence-aware calibration (T1/T2 during idles); the readout duration, which the
        # calibration does not give, is chosen on the validation split, never on test shots.
        best = None
        for m_ns in (300., 800., 1500., 3000.):
            c = dataclasses.replace(tr["cal"], durations_ns={**tr["cal"].durations_ns, "measure": m_ns, "reset": m_ns})
            circ = add_noise(tr["ideal"], NoiseModel(name="calibrated_coherence", calibration=c, idle=0, coherence=True),
                             qubit_map=tr["qmap"])
            mt = make_matcher(circ)
            f = int(np.any(mt.decode_batch(val[0], enable_correlations=True) != val[1], 1).sum())
            if best is None or f < best[0]:
                best = (f, m_ns, circ, mt)
        info["pm_coherence"] = dict(measure_ns=best[1], val_failures=best[0], candidates_ns=[300, 800, 1500, 3000])
        m_coh = best[3]
        dec["pm_coherence"] = lambda d: m_coh.decode_batch(d, enable_correlations=True)
        if not a.pretrain_pauli:
            calibrated = best[2]  # pre-train the learned decoders on the closer model
        if "ldpc" not in a.skip:
            from iyanu.ldpc_decoders import LdpcDecoder
            for kind in ("uf", "bposd", "belief_find"):
                ld = LdpcDecoder(calibrated, kind)
                dec[kind] = ld.decode_batch
                info[kind] = ld.config
        if "lookup_sim" not in a.skip:
            sd, so = uniform.compile_detector_sampler(seed=rounds).sample(a.lookup_sim_shots, separate_observables=True)
            lk = Lookup(sd, so, pm_uni)
            dec["lookup_sim"] = lk.decode_batch
            info["lookup_sim"] = dict(table_size=len(lk.table), sim_shots=a.lookup_sim_shots, fallback="pm_uniform")
        if "lookup_hw" not in a.skip:
            lk_hw = Lookup(*fit, pm_cal)
            dec["lookup_hw"] = lk_hw.decode_batch
            info["lookup_hw"] = dict(table_size=len(lk_hw.table), fit_shots=n_fit, fallback="pm_calibrated")
        if "mlp" not in a.skip:
            mlp = train_neural_decoder(calibrated, NeuralConfig(steps=a.sim_steps, seed=rounds, device=a.device))
            dec["mlp_sim"] = copy.deepcopy(mlp).decode_batch
            info["mlp_ft"] = finetune(mlp.model.train(), mlp.model, *fit, val, steps=a.ft_steps, batch=2048,
                                      lr=a.ft_lr, device=mlp.device)
            mlp.model.eval()
            dec["mlp_ft"] = mlp.decode_batch
        if "conv" not in a.skip:
            conv = ConvDecoder(calibrated, ConvConfig(channels=a.channels, steps=a.sim_steps, batch=2048,
                                                      seed=rounds, device=a.device)).train(calibrated)
            dec["conv_sim"] = copy.deepcopy(conv).decode_batch
            info["conv_ft"] = finetune(conv.model, conv.logits, *fit, val, steps=a.ft_steps, batch=2048,
                                       lr=a.ft_lr, device=conv.device)
            dec["conv_ft"] = conv.decode_batch
        if "mlp_ft" in dec and "conv_ft" in dec:
            import torch

            def ensemble(d, mlp=mlp, conv=conv):
                # Mean of the two fine-tuned decoders' flip probabilities; no extra fitting.
                out = []
                with torch.inference_mode():
                    for i in range(0, len(d), 4096):
                        x = torch.from_numpy(np.asarray(d[i:i+4096], dtype=np.float32)).to(conv.device)
                        p = (torch.sigmoid(mlp.model(x.to(mlp.device))).to(conv.device) + torch.sigmoid(conv.logits(x)))/2
                        out.append((p > 0.5).cpu().numpy())
                return np.concatenate(out)
            dec["ens_ft"] = ensemble
        if "pre" not in a.skip:
            pre = ConvPredecoder(calibrated, ConvConfig(channels=a.channels, steps=a.sim_steps, batch=2048,
                                                        seed=rounds, device=a.device)).train()
            dec["pre_sim"] = pre.decode_batch
        dets, obs = te["dets"], te["obs"]
        preds = {k: f(dets) for k, f in dec.items()}
        n = len(obs)
        row = dict(basis=basis, rounds=rounds, test_shots=n, train_shots=len(tr["obs"]), fit_shots=n_fit,
                   test_jobs=te["jobs"], train_jobs=tr["jobs"], calibration_set_id=te["calibration_set_id"],
                   detection_rate=float(dets.mean()), info=info, seconds=time.perf_counter()-started,
                   errors={k: int(np.any(p != obs, 1).sum()) for k, p in preds.items()},
                   vs_pm_calibrated={k: paired_comparison(p, preds["pm_calibrated"], obs)
                                     for k, p in preds.items() if k != "pm_calibrated"})
        if "lookup_hw" in dec:
            dec["lookup_hw"](dets)
            row["lookup_hw_unseen_fraction"] = lk_hw.last_unseen
        rows.append(row)
        print(f"{basis} r{rounds} ({n} test shots): " + "  ".join(
            f"{k} {e/n:.4f}" for k, e in sorted(row["errors"].items(), key=lambda kv: kv[1])), flush=True)

    fits = defaultdict(dict)
    for basis in sorted({r["basis"] for r in rows}):
        sub = sorted((r for r in rows if r["basis"] == basis), key=lambda r: r["rounds"])
        if len(sub) < 2:
            continue
        for k in sub[0]["errors"]:
            fits[basis][k] = fit_ler([r["rounds"] for r in sub], [r["errors"][k] for r in sub],
                                     [r["test_shots"] for r in sub])
    print("\nP_L on held-out test shots, 95% Wilson; eps_L fit over the rounds:")
    for basis in sorted(fits):
        sub = sorted((r for r in rows if r["basis"] == basis), key=lambda r: r["rounds"])
        print(f"[{basis}]")
        for k in sorted(fits[basis], key=lambda k: fits[basis][k]["epsilon_per_round"]):
            pts = "  ".join(f"r{r['rounds']} {r['errors'][k]/r['test_shots']:.4f}" for r in sub)
            f = fits[basis][k]
            print(f"  {k:<14} {pts}   eps_L {f['epsilon_per_round']:.4f} [{f['ci95_profile'][0]:.4f}, {f['ci95_profile'][1]:.4f}]")
    out = Path(a.output+time.strftime("-%Y%m%dT%H%M%S")+".json")
    out.write_text(json.dumps(dict(args=vars(a), device=str(device), rows=rows, fits=fits), indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
