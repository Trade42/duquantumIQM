"""Real shots: fine-tune the pre-trained recurrent decoders and score them on the held-out batch.

python -m mlDecoder.finetune --backend rnn_lstm --from <pre-training run id> --config mlDecoder/configs/finetune.json
python -m mlDecoder.finetune --backend rnn_lstm --from <run id> --config ... --pilot      # smoke test only
python -m mlDecoder.finetune --backend rnn_lstm --from <run id> --config ... --retune-tau <finetune run id>  # tau only

For each (device, patch, basis) group of the pre-training run:
1. load every InfoRecord E1 run of that patch and basis (hardware.load_run: hashes, circuit, splits),
   keeping the runs whose ``barriers`` matches the batches' setting;
2. wrap each run's splits as Datasets (data.inforecord_dataset) under the decoder prior, the
   calibrated model named in the config (its noisy circuit gives correlated MWPM its weights);
3. score the pre-trained ("synthetic only") model, fine-tune a copy on the ``finetune`` split of
   the fine-tuning batches (early stopping on ``validation``, never on test; weights pulled
   towards the pre-trained ones), and score the fine-tuned model;
4. on the ``test`` split (the whole time-separated test batch): both models against correlated
   MWPM (calibrated prior), MWPM, and correlated MWPM with uniform weights, paired, with the LER
   record labelled HARDWARE (device, calibration set, batch) or SIMULATED for the stand-in.

``--pilot`` decodes the pilot runs (all their shots) with every decoder and records P_L: a loader and
decoder smoke test, not a result and not training data. ``--eval-split validation`` scores on the
validation split instead of test: a code check for data without a test batch, labelled so.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np

from . import cascade, data, evaluate, hardware
from .data import DatasetSpec
from .run import Log, decoder_record, references_for, speed_record, write_json, device_for


def prior_spec(cfg: dict, group: dict, rounds: int) -> DatasetSpec:
    prior = cfg["prior"]
    cal = hardware.device_calibration(group["device"], group["calset"], group["patch"], window_ns=group["window_ns"],
                                      **prior.get("options", {}))
    return DatasetSpec(distance=3, rounds=rounds, basis=group["basis"], noise=prior["noise"], p=float(prior["p"]),
                       calibration=cal)


def group_runs(group: dict, root: Path | None, simulated: bool, log) -> list[hardware.Run]:
    paths = hardware.find_runs(root, device=group["device"], simulated=simulated)
    record_root = hardware.device_root(group["device"], root) / ("sim" if simulated else "")
    runs = []
    for path in paths:
        run = hardware.load_run(path, record_root=record_root)
        if run.spec.get("family") != "surface_memory" or run.spec["patch"] != group["patch"] or run.spec["basis"] != group["basis"]:
            continue
        if float(run.spec.get("delay_ns") or 0) != 0:
            continue                                   # E4's delayed variants are other circuits.
        runs.append(run)
    batch_barriers = {bool(r.spec["barriers"]) for r in runs if int(r.manifest["batch"]) >= 1}
    if len(batch_barriers) > 1:
        raise ValueError(f"{group['name']}: the batches disagree on barriers")
    setting = batch_barriers.pop() if batch_barriers else True
    kept = [r for r in runs if bool(r.spec["barriers"]) == setting]
    log(f"  {group['name']}: {len(runs)} runs, {len(kept)} with barriers={setting} "
        f"({', '.join(r.manifest['run_id'] for r in kept)})")
    return kept


def split_datasets(runs, cfg, group, split: str, batches=None) -> list[data.Dataset]:
    out = []
    for run in runs:
        if batches is not None and int(run.manifest["batch"]) not in batches:
            continue
        if not (run.split == split).any():
            continue
        out.append(data.inforecord_dataset(prior_spec(cfg, group, int(run.spec["rounds"])), run, split))
    return out


def all_shots(run: hardware.Run, cfg, group) -> data.Dataset:
    """Every shot of a run (the pilot smoke test), whatever its split."""
    spec = prior_spec(cfg, group, int(run.spec["rounds"]))
    ideal, _ = data.make_circuits(spec)
    hardware.check_run_against(run, ideal, spec.calibration)
    meta = dict(run.describe(), label=run.label, dataset_id=f"ir-{run.manifest['run_id']}-all", decoder_prior=spec.slug,
                note="all shots of the run: pilot smoke test, not a result")
    return data.external_dataset(spec, run.dets, run.obs, meta, split="test", raw_meas=run.raw_meas, source=run.label)


def calibrated_prior(cfg: dict, group: dict) -> dict:
    """The config's decoder prior on the group's device patch: the weights of correlated MWPM ("mwpm_corr")."""
    prior = cfg["prior"]
    return dict(noise=prior["noise"], p=float(prior["p"]), calibration=prior_spec(cfg, group, 1).calibration)


def cascade_for(layer1, cfg: dict, group: dict, prior: dict | None = None) -> cascade.Cascade:
    """The two-layer decoder around a model: Relay-BP (then correlated MWPM) on the DEM of ``prior`` (default calibrated)."""
    return cascade.Cascade(layer1, prior or calibrated_prior(cfg, group), tau=0.0,
                           relay_config=cfg.get("cascade", {}).get("relay", {}))


def choose_prior(val, cfg: dict, group: dict, log) -> dict:
    """Layers 2-3's prior from the validation split only: calibrated, or uniform weights if they beat it beyond 1 SE.

    The uniform prior is the team's uniform model at evaluate.UNIFORM_P, the weights of "mwpm_corr_uniform".
    """
    cal = np.concatenate([evaluate.reference(ds, True)["fail"] for ds in val])
    uni = np.concatenate([evaluate.reference(ds, True, uniform=True)["fail"] for ds in val])
    choice = cascade.choose_prior(cal, uni)
    choice["prior"] = (dict(noise="uniform", p=evaluate.UNIFORM_P, calibration=None) if choice["chosen"] == "uniform"
                       else calibrated_prior(cfg, group))
    log(f"  decoder prior: {choice['chosen']} (validation corr MWPM: calibrated {choice['calibrated_failures']}, "
        f"uniform {choice['uniform_failures']} of {choice['shots']}; paired b {choice['only_uniform_fails']}, "
        f"c {choice['only_calibrated_fails']}, 1 SE {choice['se']:.1f})")
    return choice


def tune_cascade(casc: cascade.Cascade, val, cfg: dict, log) -> dict:
    """τ from the validation split only (one-standard-error rule); the validation trade-off curve and layer 1's calibration."""
    parts = cascade.components(casc.layer1, casc.relay, val)
    relay_us = float(np.mean([c["relay_us_per_shot"] for c in parts]))
    tuned = cascade.tune(parts)
    casc.tau = tuned["tau"]
    v, a, se = tuned["validation"], tuned["argmin"], tuned["paired_vs_argmin"]
    log(f"  cascade: tau {casc.tau:.2f} from validation ({v['failures']}/{v['shots']} failures, "
        f"{v['deferred_fraction']:.3f} deferred); argmin tau {a['tau']:.2f} ({a['validation']['failures']} failures, "
        f"{a['validation']['deferred_fraction']:.3f} deferred); paired b {se['only_tau_fails']}, c {se['only_argmin_fails']}, "
        f"1 SE {se['se']:.1f}")
    return dict(tau=casc.tau, rule=tuned["rule"], validation_shots=v["shots"], validation=v,
                argmin_tau=a["tau"], argmin_validation=a["validation"], paired_vs_argmin=se,
                validation_curve=cascade.curve(parts, relay_us=relay_us, layer1_us=0.0),
                relay_us_per_validation_shot=relay_us, calibration=cascade.calibration(parts),
                layer1_alone_validation_failures=cascade.failures_at(parts, 0.0)[0],
                relay_alone_validation_failures=cascade.failures_at(parts, 1.0 + 1e-9)[0])


def retune_tau(args) -> int:
    """Re-tune an earlier fine-tuning run's cascade (layers 2-3's prior, then τ) on its validation split, in place.

    No training and no test split: the run's own config, fine-tuned models and validation shots, under the
    current rules. The replaced entry is kept under ``cascade_history``. For a rule change made before the test
    batch is opened.
    """
    run_dir = data.results_root() / args.backend / args.retune_tau
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("mode") != "finetune" or result.get("pretrained_run") != args.source:
        raise ValueError(f"{args.retune_tau} is not a fine-tuning run of {args.source}")
    cfg = result["config"]
    log = Log(run_dir / "log.txt")
    module = importlib.import_module(f".models.{args.backend}", __package__)
    decoder_cls = type(module.build(data.make_circuits(DatasetSpec())[0], {}))
    device = device_for(cfg)
    git = evaluate.git_state()
    log(f"retune-tau at {datetime.now(timezone.utc).isoformat()}: pid {os.getpid()}, commit {git['commit'][:7]}"
        f"{' (dirty tree)' if git['dirty'] else ''}; validation split only")
    for name, e in result["groups"].items():
        if args.groups and name not in args.groups or not e.get("model_file") or not e.get("cascade"):
            continue
        group = e["group"]
        log(f"== {name} (retune tau)")
        runs = group_runs(group, args.root, args.simulated, log)
        ft_batches = set(cfg.get("finetune_batches", [1, 2])) | ({0} if cfg.get("finetune_on_pilot_phase") else set())
        val = split_datasets(runs, cfg, group, "validation", ft_batches)
        if sum(d.shots for d in val) != e["cascade"]["validation_shots"]:
            raise ValueError(f"{name}: {sum(d.shots for d in val)} validation shots, the run tuned on "
                             f"{e['cascade']['validation_shots']}")
        choice = choose_prior(val, cfg, group, log)
        casc = cascade_for(decoder_cls.load(run_dir / e["model_file"], None, device), cfg, group, choice["prior"])
        new = tune_cascade(casc, val, cfg, log)
        old = e["cascade"]
        log(f"  replaces tau {old['tau']:.2f} ({old['rule']}; prior "
            f"{old.get('prior_choice', {}).get('chosen', 'calibrated')}); argmin tau now {new['argmin_tau']:.2f}")
        now = datetime.now(timezone.utc).isoformat()
        e.setdefault("cascade_history", []).append(dict(old, replaced_utc=now))
        e["cascade"] = dict(new, prior=choice.pop("prior"), prior_choice=choice, retuned_utc=now,
                            retuned_commit=git["commit"], retuned_dirty=git["dirty"])
        write_json(result_path, result)
    return 0


def score(decoders: dict, test: data.Dataset, cfg: dict, group: dict) -> dict:
    refs, records = references_for(test, cfg)
    point = dict(spec=asdict(test.spec), slug=test.spec.slug, group=group["name"], references=records,
                 dataset=dict(test.manifest, shot_indices=None), always_zero=evaluate.point(np.any(test.obs != 0, axis=1)),
                 paired_mwpm_vs_mwpm_corr=evaluate.paired(refs["mwpm"]["fail"], refs["mwpm_corr"]["fail"]),
                 paired_refs_vs_mwpm_corr={n: evaluate.paired(r["fail"], refs["mwpm_corr"]["fail"])
                                           for n, r in refs.items() if n not in ("mwpm_corr", "mwpm")},
                 decoders=[])
    for name, decoder in decoders.items():
        record = decoder_record(decoder, test, refs, decoder.device)
        record["name"] = name
        point["decoders"].append(record)
    names = list(decoders)
    if len(names) >= 2 and names[0].endswith("_pretrained") and names[1].endswith("_finetuned"):
        fa = evaluate.failure_mask(evaluate.decode_dataset(decoders[names[1]], test)[0], test.obs)
        fb = evaluate.failure_mask(evaluate.decode_dataset(decoders[names[0]], test)[0], test.obs)
        point["paired_finetuned_vs_pretrained"] = evaluate.paired(fa, fb)
    return point


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--from", dest="source", required=True, help="Pre-training run id (results/<backend>/<id>)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", type=Path, help="InfoRecord root (default: hardware.inforecord_root())")
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--pilot", action="store_true", help="Smoke test: decode the pilot runs with every decoder")
    parser.add_argument("--simulated", action="store_true", help="Use InfoRecord's SIMULATED stand-in runs (code check)")
    parser.add_argument("--eval-split", default="test", choices=["test", "validation"])
    parser.add_argument("--use-finetuned", metavar="RUN_ID",
                        help="Score the fine-tuned models and validation-tuned tau of an earlier finetune run "
                             "(no training or tuning here): for a test batch that lands after the fine-tuning batches")
    parser.add_argument("--provisional", action="store_true",
                        help="Label HARDWARE numbers PROVISIONAL (the mini PC's batch review has not passed yet)")
    parser.add_argument("--retune-tau", metavar="RUN_ID",
                        help="Re-tune the cascade (layers 2-3's prior, then tau) of an earlier fine-tuning run on its "
                             "validation split under the current rules, in place (the old entry is kept); no training, "
                             "no test split")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_dir = data.results_root() / args.backend / args.source
    if args.retune_tau:
        return retune_tau(args)
    pre = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
    kind = "pilot" if args.pilot else "finetune"
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{kind}-{args.source[:15]}"
    run_dir = data.results_root() / args.backend / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    log = Log(run_dir / "log.txt")
    result_path = run_dir / "result.json"
    module = importlib.import_module(f".models.{args.backend}", __package__)
    decoder_cls = type(module.build(data.make_circuits(DatasetSpec())[0], {}))
    device = device_for(cfg)
    result = dict(run_id=run_id, backend=args.backend, config_path=Path(args.config).as_posix(), config=cfg,
                  mode=kind, pretrained_run=args.source, eval_split=args.eval_split, simulated_standin=args.simulated,
                  started_utc=datetime.now(timezone.utc).isoformat(), status="running", pid=os.getpid(),
                  environment=evaluate.environment(torch_needed=True), git=evaluate.git_state(), groups={}, points=[],
                  label="SIMULATED" if args.simulated else "HARDWARE",
                  step03="InfoRecord's det_events.npy / obs_flips.npy (the m2d conversion of the raw record, checked); "
                         "decoded per shot; a shot fails if ANY observable is wrong.")
    if args.provisional and not args.simulated:
        result["label"] = "HARDWARE (PROVISIONAL: batch review pending)"
    if args.eval_split != "test" and not args.pilot:
        result["label"] += " (validation split: code check, not a test)"
    log(f"run {run_id}: {kind}, backend {args.backend}, from {args.source}, pid {os.getpid()}")
    for name, g in pre["groups"].items():
        if args.groups and name not in args.groups or g.get("status") != "complete":
            continue                              # Includes groups dropped from the pre-training run.
        group = g["group"]
        log(f"== {name}")
        runs = group_runs(group, args.root, args.simulated, log)
        pretrained = decoder_cls.load(source_dir / g["model_file"], None, device)
        entry = dict(group=group, runs=[r.describe() for r in runs])
        result["groups"][name] = entry
        if args.pilot:
            for run in runs:
                if int(run.manifest["batch"]) != 0:
                    continue
                point = score({f"{args.backend}_pretrained": pretrained}, all_shots(run, cfg, group), cfg, group)
                point["pilot"] = True
                result["points"].append(point)
                d = point["decoders"][0]["point"]
                log(f"  pilot {run.manifest['run_id']}: {args.backend} (synthetic only) {d['p_l']:.4f} "
                    f"{d['ci95_wilson']}, corr MWPM {point['references']['mwpm_corr']['point']['p_l']:.4f}, "
                    f"uniform {point['references']['mwpm_corr_uniform']['point']['p_l']:.4f}, "
                    f"always-0 {point['always_zero']['p_l']:.4f}  (n={d['shots']})")
            write_json(result_path, result)
            continue
        ft_batches = set(cfg.get("finetune_batches", [1, 2])) | ({0} if cfg.get("finetune_on_pilot_phase") else set())
        train = split_datasets(runs, cfg, group, "finetune", ft_batches)
        val = split_datasets(runs, cfg, group, "validation", ft_batches)
        tests = split_datasets(runs, cfg, group, args.eval_split, None if args.eval_split == "test" else ft_batches)
        entry.update(finetune_shots=sum(d.shots for d in train), validation_shots=sum(d.shots for d in val),
                     test_shots=sum(d.shots for d in tests))
        log(f"  finetune {entry['finetune_shots']:,} shots in {len(train)} sets, validation {entry['validation_shots']:,}, "
            f"{args.eval_split} {entry['test_shots']:,}")
        decoders = {f"{args.backend}_pretrained": pretrained}
        if args.use_finetuned:
            earlier = json.loads((data.results_root() / args.backend / args.use_finetuned / "result.json").read_text(encoding="utf-8"))
            e = earlier["groups"].get(name, {})
            if e.get("model_file"):
                tuned = decoder_cls.load(data.results_root() / args.backend / args.use_finetuned / e["model_file"], None, device)
                decoders[f"{args.backend}_finetuned"] = tuned
                entry.update(train=e.get("train"), model_file=e["model_file"], metadata=e.get("metadata"),
                             finetuned_in=args.use_finetuned, finetune_shots=e.get("finetune_shots"),
                             validation_shots=e.get("validation_shots"))
                log(f"  using the fine-tuned model of {args.use_finetuned} (trained on {e.get('finetune_shots')} shots)")
            if e.get("cascade"):
                casc = cascade_for(decoders.get(f"{args.backend}_finetuned", pretrained), cfg, group,
                                   e["cascade"].get("prior"))
                casc.tau = float(e["cascade"]["tau"])
                entry["cascade"] = dict(e["cascade"], tuned_in=args.use_finetuned)
                decoders[f"{args.backend}_cascade"] = casc
                log(f"  cascade tau {casc.tau:.2f}, tuned on validation in {args.use_finetuned}")
        elif train and val:
            tuned = decoder_cls.load(source_dir / g["model_file"], None, device)
            tuned.config["train"] = {**tuned.config.get("train", {}), **cfg["train"]}
            tuned.checkpoint_path = str(run_dir / "models" / f"{name}.ckpt")
            info = tuned.fit(train, val, device=device, seed=int(cfg.get("seed", 0)), log=log)
            tuned.save(run_dir / "models" / f"{name}.pt")
            entry.update(train=info, model_file=f"models/{name}.pt", metadata=tuned.metadata())
            decoders[f"{args.backend}_finetuned"] = tuned
            log(f"  fine-tuned {info['wall_seconds']:.0f} s ({info['stop_reason']}), val P_L {info['final_val_p_l']:.4f}")
        if cfg.get("cascade", {}).get("enabled") and val and not args.use_finetuned:
            best = decoders.get(f"{args.backend}_finetuned", pretrained)
            choice = choose_prior(val, cfg, group, log)
            casc = cascade_for(best, cfg, group, choice["prior"])
            entry["cascade"] = dict(tune_cascade(casc, val, cfg, log), prior=choice.pop("prior"), prior_choice=choice)
            decoders[f"{args.backend}_cascade"] = casc
        for test in tests:
            point_decoders = dict(decoders)
            if f"{args.backend}_cascade" in decoders:
                casc = decoders[f"{args.backend}_cascade"]
                point_decoders["relay_bp"] = casc.relay(test.spec.rounds)
            point = score(point_decoders, test, cfg, group)
            if f"{args.backend}_cascade" in decoders:
                detail = casc.decode_detailed(test.dets, raw_meas=test.raw if casc.layer1.uses_raw else None)
                point["cascade"] = dict(tau=casc.tau, shots=test.shots, deferred=int((detail["layer"] >= 1).sum()),
                                        mwpm_fallback=int((detail["layer"] == 2).sum()),
                                        deferred_fraction=float((detail["layer"] >= 1).mean()))
            result["points"].append(point)
            c = point["references"]["mwpm_corr"]["point"]
            log(f"  {args.eval_split} R={test.spec.rounds}: " + ", ".join(
                f"{d['name']} {d['point']['p_l']:.4f}" for d in point["decoders"]) + f", corr MWPM {c['p_l']:.4f}, "
                f"uniform {point['references']['mwpm_corr_uniform']['point']['p_l']:.4f} (n={c['shots']})")
        if tests and cfg.get("eval", {}).get("benchmark", True):
            best = decoders.get(f"{args.backend}_finetuned", pretrained)
            entry["speed"] = [speed_record(best, t, cfg, best.device) for t in tests
                              if t.spec.rounds in cfg.get("eval", {}).get("speed_rounds", [])]
            if f"{args.backend}_cascade" in decoders:
                casc = decoders[f"{args.backend}_cascade"]
                entry["cascade_speed"] = [dict(speed_record(casc, t, cfg, casc.device), tau=casc.tau) for t in tests
                                          if t.spec.rounds in cfg.get("eval", {}).get("speed_rounds", [])]
        entry["status"] = "complete"
        write_json(result_path, result)
    if result["points"] and not args.pilot:
        record = evaluate.write_ler(run_dir, result)
        result["ler"] = dict(directory="ler", fits=len(record["fits"]))
    result["status"] = "complete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(result_path, result)
    log(f"done: {len(result['points'])} points")
    if not args.no_report:
        from . import report
        report.main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
