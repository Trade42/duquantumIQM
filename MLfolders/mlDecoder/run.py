"""CLI: python -m mlDecoder.run --backend <name> --config mlDecoder/configs/<name>_<size>.json

Grid mode (stage 0): for every grid point, size and freeze the test set (correlated-MWPM
pilot), score correlated and uncorrelated MWPM on it, then (for a learned backend) train one
model, freeze it, and score it on the same shots with paired comparisons and speed benchmarks.

Mixture mode (``"mode": "mixture"``, stage 1): one model per (device, patch, basis) group,
pre-trained on a mixture of noise components and rounds (sequence.Mixture), then scored on the
frozen development sets of the components marked ``eval`` at every round count, against
correlated MWPM, MWPM and correlated MWPM with uniform weights. All of it is SIMULATED.

One result JSON per run under results/<backend>/<run-id>/, rewritten after every point; then
RESULTS.md is regenerated.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from . import data, evaluate
from .data import DatasetSpec

GRID_KEYS = ("p", "noise", "distance", "basis", "rounds")


def expand_grid(grid: dict) -> list[DatasetSpec]:
    """Cartesian product of the grid's lists; ordered by p, distance, noise, basis, rounds."""
    fixed = {k: v for k, v in grid.items() if k not in GRID_KEYS}
    lists = {k: grid[k] if isinstance(grid[k], list) else [grid[k]] for k in GRID_KEYS}
    specs = [DatasetSpec(**fixed, **dict(zip(GRID_KEYS, values))) for values in itertools.product(*lists.values())]
    return sorted(specs, key=lambda s: (s.p, s.distance, s.noise, s.basis, s.rounds))


def resolve(section: dict, spec: DatasetSpec) -> dict:
    """A config section with its per-distance overrides ("by_distance": {"5": {...}}) applied."""
    section = dict(section or {})
    overrides = section.pop("by_distance", {}).get(str(spec.distance), {})
    return {**section, **overrides}


def matches(spec: DatasetSpec, filters: list[str]) -> bool:
    values = asdict(spec)
    for item in filters:
        key, _, value = item.partition("=")
        if str(values[key]) not in value.split(","):
            return False
    return True


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str = ""):
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def test_set(spec: DatasetSpec, cfg: dict, log) -> tuple[data.Dataset, dict]:
    t = cfg["test"]
    leakage = data.uses_leakage_sampler(spec)
    pilot = data.pilot(spec, max_shots=t["pilot_max_shots_leakage" if leakage else "pilot_max_shots"],
                       target_failures=t["pilot_target_failures"], workers=cfg.get("workers", 1), log=log)
    sizing = data.choose_test_shots(pilot, target_failures=t["target_failures"], safety=t["safety"],
                                    min_shots=t["min_shots_leakage" if leakage else "min_shots"],
                                    max_shots=t["max_shots"])
    dataset = data.frozen_test_set(spec, sizing["shots"], workers=cfg.get("workers", 1), log=log)
    return dataset, dict(pilot=pilot, sizing=sizing)


def dataset_record(dataset: data.Dataset, sizing: dict) -> dict:
    m = dataset.manifest
    record = {k: m[k] for k in ("dataset_id", "shots", "dets_sha256", "obs_sha256", "circuit_sha256",
                                "ideal_sha256", "num_detectors", "num_observables", "sampler", "generation_seconds")}
    record.update(sizing=sizing["sizing"], pilot={k: sizing["pilot"][k] for k in ("shots", "failures", "p_hat")},
                  leakage=m.get("leakage"))
    return record


def speed_sample(dataset: data.Dataset, n: int = 65536) -> np.ndarray:
    return data.unpack(dataset.dets_packed[:n], dataset.num_detectors)


def score_references(dataset, cfg, *, benchmark: bool, log) -> tuple[dict, dict]:
    refs, out = {}, {}
    for correlated in (True, False):
        ref = evaluate.reference(dataset, correlated)
        refs[ref["name"]] = ref
        out[ref["name"]] = dict(point={k: ref[k] for k in ("failures", "shots", "p_l", "ci95_wilson")},
                                build_seconds=ref["build_seconds"], decode_seconds=ref["decode_seconds"],
                                decoder=ref["decoder"], threads=1)
    if benchmark:
        sample = speed_sample(dataset)
        for name, correlated in (("mwpm_corr", True), ("mwpm", False)):
            matcher = evaluate.Matcher(dataset.circuit, correlated)
            out[name]["speed"] = dict(cpu_threads_1=evaluate.benchmark(matcher.decode_batch, sample,
                                      batch_sizes=cfg["eval"]["batch_sizes"], min_seconds=cfg["eval"]["min_seconds"]),
                                      cuda="n/a: PyMatching runs on the CPU only")
    return refs, out


def device_for(cfg: dict) -> str:
    wanted = cfg.get("device", "auto")
    import torch
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config asks for CUDA but torch.cuda.is_available() is False")
    return wanted


def train_and_score(backend: str, spec, test, refs, cfg, run_dir: Path, seed: int, log) -> dict:
    import torch
    module = importlib.import_module(f".models.{backend}", __package__)
    model_cfg, train_cfg = resolve(cfg.get("model"), spec), resolve(cfg.get("train"), spec)
    workers = cfg.get("workers", 1)
    device = device_for(cfg)
    sizes = {k: train_cfg.pop(k, None) for k in ("pool_shots", "leakage_pool_shots", "val_shots", "leakage_val_shots")}
    stream_leakage = train_cfg.pop("leakage_stream", False)
    if data.uses_leakage_sampler(spec):
        pool_shots, val_shots = sizes["leakage_pool_shots"], sizes["leakage_val_shots"] or sizes["val_shots"]
        if stream_leakage:
            pool_shots = None  # Fresh shots from worker processes, each used once (train.LeakageStreamSource).
        elif not pool_shots:
            raise ValueError("Leakage specs need train.leakage_pool_shots, or train.leakage_stream")
    else:
        pool_shots, val_shots = sizes["pool_shots"], sizes["val_shots"]
    val_shots = val_shots or 100_000
    started = time.perf_counter()
    train = (data.cached_dataset(spec, "train", int(pool_shots), seed=seed, workers=workers, log=log)
             if pool_shots else spec)
    val = data.cached_dataset(spec, "val", int(val_shots), seed=seed, workers=workers, log=log)
    data_seconds = time.perf_counter() - started
    decoder = module.build(test.circuit, dict(model=model_cfg, train=train_cfg))
    (run_dir / "models").mkdir(exist_ok=True)
    decoder.checkpoint_path = str(run_dir / "models" / f"{spec.slug}.ckpt")
    info = decoder.fit(train, val, device=device, seed=seed, log=log)
    decoder.save(run_dir / "models" / f"{spec.slug}.pt")
    pred, decode_seconds = evaluate.decode_dataset(decoder, test)
    fail = evaluate.failure_mask(pred, test.obs)
    always_zero = np.any(test.obs != 0, axis=1)
    record = dict(name=decoder.name, device=device, point=evaluate.point(fail),
                  train=info, data_seconds=data_seconds, decode_seconds_full_test=decode_seconds,
                  paired_vs_mwpm_corr=evaluate.paired(fail, refs["mwpm_corr"]["fail"]),
                  paired_vs_mwpm=evaluate.paired(fail, refs["mwpm"]["fail"]),
                  paired_vs_always_zero=evaluate.paired(fail, always_zero))
    e = cfg["eval"]
    record["metadata"] = decoder.metadata()
    if not e.get("benchmark", True):
        record["speed"] = "not measured in this run (eval.benchmark is false)"
        return record
    sample = speed_sample(test)
    speed = {}
    if device.startswith("cuda"):
        speed["cuda"] = evaluate.benchmark(decoder.decode_batch, sample, batch_sizes=e["batch_sizes"],
                                           min_seconds=e["min_seconds"], sync=torch.cuda.synchronize)
        speed["cuda_parts"] = [decoder.profile(sample[:b]) for b in e["batch_sizes"] if b <= len(sample)]
        speed["cuda_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    decoder.to("cpu")
    threads = torch.get_num_threads()
    for n_threads in (1, threads):
        torch.set_num_threads(n_threads)
        speed[f"cpu_threads_{n_threads}"] = evaluate.benchmark(decoder.decode_batch, sample,
            batch_sizes=e["batch_sizes"], min_seconds=e["min_seconds"], max_call_seconds=e.get("max_cpu_call_seconds", 4.))
    torch.set_num_threads(threads)
    decoder.to(device)
    record["speed"] = speed
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    return record


def checks_for(point: dict, checks: dict) -> dict:
    if not checks or "decoder" not in point:
        return {}
    dec = point["decoder"]
    out = {}
    if "max_train_shots" in checks:
        distinct = dec["train"]["pool_shots"] or dec["train"]["shots_seen"]
        out["distinct_train_shots"] = dict(value=distinct, limit=checks["max_train_shots"],
                                           passed=distinct <= checks["max_train_shots"])
    if "max_train_seconds" in checks:
        out["train_seconds"] = dict(value=dec["train"]["wall_seconds"], limit=checks["max_train_seconds"],
                                    passed=dec["train"]["wall_seconds"] <= checks["max_train_seconds"])
    if checks.get("beat_always_zero"):
        hi = dec["paired_vs_always_zero"]["ci95_paired_bootstrap"][1]
        out["beats_always_zero"] = dict(value=hi, passed=hi < 0,
                                        rule="upper end of paired 95% interval for P_L(model) - P_L(always 0) < 0")
    return out


def group_fits(points: list[dict]) -> list[dict]:
    groups = {}
    for pt in points:
        s = pt["spec"]
        key = (s["p"], s["noise"], s["distance"], s["basis"], s["gate_basis"], s["no_reset"], s["calibration"])
        entries = {name: ref["point"] for name, ref in pt["references"].items()}
        if "decoder" in pt:
            entries[pt["decoder"]["name"]] = pt["decoder"]["point"]
        for name, value in entries.items():
            groups.setdefault((key, name), []).append((s["rounds"], value["failures"], value["shots"]))
    out = []
    for (key, name), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.sort()
        if len({r for r, _, _ in rows}) < 2:
            continue
        fit = evaluate.fits(*zip(*rows))
        out.append(dict(p=key[0], noise=key[1], distance=key[2], basis=key[3], gate_basis=key[4],
                        no_reset=key[5], calibration=key[6], decoder=name, **fit))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", nargs="*", default=[], metavar="KEY=V[,V]", help="Filter grid points, e.g. distance=3")
    parser.add_argument("--resume", metavar="RUN_ID", help="Continue a run, skipping its finished points")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--groups", nargs="*", default=[], help="Mixture mode: only these groups, e.g. garnet_pA_Z")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("backend", args.backend) != args.backend:
        sys.exit(f"Config is for backend {cfg['backend']!r}, not {args.backend!r}")
    mixture = cfg.get("mode") == "mixture"
    specs = [] if mixture else [s for s in expand_grid(cfg["grid"]) if matches(s, args.only)]
    run_id = args.resume or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{Path(args.config).stem}"
    run_dir = data.results_root() / args.backend / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Log(run_dir / "log.txt")
    result_path = run_dir / "result.json"
    learned = args.backend != "mwpm"
    if args.resume:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = dict(run_id=run_id, backend=args.backend, config_path=Path(args.config).as_posix(), config=cfg,
                      only=args.only, started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                      environment=evaluate.environment(torch_needed=learned), git=evaluate.git_state(),
                      pid=os.getpid(), points=[], fits=[])
    if mixture:
        return run_mixture(args, cfg, run_dir, result, result_path, log, learned)
    done = {json.dumps(p["spec"], sort_keys=True) for p in result["points"]}
    log(f"run {run_id}: backend {args.backend}, {len(specs)} grid points, pid {os.getpid()}, results {run_dir}")
    seed = int(cfg.get("seed", 0))
    budget = float(cfg.get("max_total_train_seconds", float("inf")))
    spent = sum(p.get("decoder", {}).get("train", {}).get("wall_seconds", 0) for p in result["points"])
    for index, spec in enumerate(specs, 1):
        if json.dumps(asdict(spec), sort_keys=True) in done:
            continue
        log(f"[{index}/{len(specs)}] {spec.slug}")
        started = time.perf_counter()
        test, sizing = test_set(spec, cfg, log)
        refs, ref_records = score_references(test, cfg, benchmark=not learned or cfg["eval"].get("bench_references", False), log=log)
        point = dict(spec=asdict(spec), slug=spec.slug, dataset=dataset_record(test, sizing), references=ref_records,
                     always_zero=evaluate.point(np.any(test.obs != 0, axis=1)),
                     paired_mwpm_vs_mwpm_corr=evaluate.paired(refs["mwpm"]["fail"], refs["mwpm_corr"]["fail"]))
        c = ref_records["mwpm_corr"]["point"]
        u = ref_records["mwpm"]["point"]
        log(f"  N={test.shots:,}  corr MWPM {c['failures']} ({c['p_l']:.3g})  MWPM {u['failures']} ({u['p_l']:.3g})")
        if learned:
            if spent >= budget:
                point["decoder_skipped"] = f"total training budget {budget:.0f} s exhausted"
                log("  " + point["decoder_skipped"])
            else:
                point["decoder"] = train_and_score(args.backend, spec, test, refs, cfg, run_dir, seed, log)
                spent += point["decoder"]["train"]["wall_seconds"]
                d, pv = point["decoder"]["point"], point["decoder"]["paired_vs_mwpm_corr"]
                log(f"  {args.backend} {d['failures']} ({d['p_l']:.3g})  vs corr MWPM: {pv['a_wrong_b_right']} vs "
                    f"{pv['b_wrong_a_right']} discordant, dP_L {pv['delta_p_l']:+.2e} "
                    f"[{pv['ci95_paired_bootstrap'][0]:+.2e}, {pv['ci95_paired_bootstrap'][1]:+.2e}] -> {pv['verdict']}"
                    f"  (train {point['decoder']['train']['wall_seconds']:.0f} s, spent {spent:.0f} s)")
        point["checks"] = checks_for(point, cfg.get("checks"))
        for name, check in point["checks"].items():
            log(f"  check {name}: {'PASS' if check['passed'] else 'FAIL'} ({check['value']})")
        point["seconds"] = time.perf_counter() - started
        result["points"].append(point)
        write_json(result_path, result)
    result["fits"] = group_fits(result["points"])
    record = evaluate.write_ler(run_dir, result)
    result["ler"] = dict(directory="ler", files=["ler_points.csv", "ler_fits.json", "ler_<d>_<basis>_<noise>_p<p>.png"],
                         fits=len(record["fits"]), lambdas=len(record["lambdas"]))
    result["status"] = "complete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["total_train_seconds"] = spent
    failed = [f"{p['slug']}:{n}" for p in result["points"] for n, c in p.get("checks", {}).items() if not c["passed"]]
    result["checks_failed"] = failed
    write_json(result_path, result)
    log(f"done: {len(result['points'])} points, training {spent:.0f} s; checks failed: {failed or 'none'}")
    if not args.no_report:
        from . import report
        report.main([])
    return 1 if failed else 0


# ---------------------------------------------------------------- mixture mode (stage 1)

def mixture_groups(cfg: dict) -> list[dict]:
    """(device, patch, basis) groups of a mixture config, in config order."""
    out = []
    for dev in cfg["devices"]:
        for patch in dev["patches"]:
            for basis in cfg["bases"]:
                out.append(dict(name=f"{dev['name']}_{patch}_{basis}", device=dev["name"], calset=dev["calset"],
                                window_ns=dev["window_ns"], patch=patch, basis=basis))
    return out


def group_components(cfg: dict, group: dict):
    """The group's sequence.Component list (calibrated components on the group's patch) and eval flags."""
    from . import hardware
    from .sequence import Component
    comps, evals = [], []
    for c in cfg["mixture"]:
        cal = None
        if c["noise"].startswith("calibrated_"):
            cal = hardware.device_calibration(group["device"], group["calset"], group["patch"],
                                              window_ns=group["window_ns"], **c.get("options", {}))
        comps.append(Component(c["noise"], float(c["p"]), cal, float(c["weight"])))
        evals.append(bool(c.get("eval")))
    return comps, evals


def score_point(spec, test, sizing, refs, ref_records, cfg) -> dict:
    """A run point: the frozen set, its references and their paired comparisons with correlated MWPM."""
    point = dict(spec=asdict(spec), slug=spec.slug, dataset=dataset_record(test, sizing), references=ref_records,
                 always_zero=evaluate.point(np.any(test.obs != 0, axis=1)),
                 paired_mwpm_vs_mwpm_corr=evaluate.paired(refs["mwpm"]["fail"], refs["mwpm_corr"]["fail"]))
    point["paired_refs_vs_mwpm_corr"] = {name: evaluate.paired(ref["fail"], refs["mwpm_corr"]["fail"])
                                         for name, ref in refs.items() if name not in ("mwpm_corr", "mwpm")}
    return point


def references_for(test, cfg) -> tuple[dict, dict]:
    refs, records = score_references(test, cfg, benchmark=False, log=None)
    if cfg.get("uniform_reference", True):
        ref = evaluate.reference(test, True, uniform=True)
        refs[ref["name"]] = ref
        records[ref["name"]] = dict(point={k: ref[k] for k in ("failures", "shots", "p_l", "ci95_wilson")},
                                    build_seconds=ref["build_seconds"], decode_seconds=ref["decode_seconds"],
                                    decoder=ref["decoder"], threads=1)
    return refs, records


def decoder_record(decoder, test, refs, device) -> dict:
    pred, seconds = evaluate.decode_dataset(decoder, test)
    fail = evaluate.failure_mask(pred, test.obs)
    record = dict(name=decoder.name, device=device, point=evaluate.point(fail), decode_seconds_full_test=seconds,
                  paired_vs_mwpm_corr=evaluate.paired(fail, refs["mwpm_corr"]["fail"]),
                  paired_vs_mwpm=evaluate.paired(fail, refs["mwpm"]["fail"]),
                  paired_vs_always_zero=evaluate.paired(fail, np.any(test.obs != 0, axis=1)))
    if "mwpm_corr_uniform" in refs:
        record["paired_vs_mwpm_corr_uniform"] = evaluate.paired(fail, refs["mwpm_corr_uniform"]["fail"])
    return record


def speed_record(decoder, test, cfg, device) -> dict:
    """Warmed throughput and latency of the decoder (CUDA, CPU 1 and all threads) and of correlated MWPM."""
    import torch
    e = cfg["eval"]
    sample = speed_sample(test)
    out = dict(rounds=test.spec.rounds, spec=test.spec.slug)
    if device.startswith("cuda"):
        out["cuda"] = evaluate.benchmark(decoder.decode_batch, sample, batch_sizes=e["batch_sizes"],
                                         min_seconds=e["min_seconds"], sync=torch.cuda.synchronize)
        out["cuda_parts"] = [decoder.profile(sample[:b]) for b in e["batch_sizes"] if b <= len(sample)]
    decoder.to("cpu")
    threads = torch.get_num_threads()
    for n_threads in (1, threads):
        torch.set_num_threads(n_threads)
        out[f"cpu_threads_{n_threads}"] = evaluate.benchmark(decoder.decode_batch, sample, batch_sizes=e["batch_sizes"],
                                                             min_seconds=e["min_seconds"],
                                                             max_call_seconds=e.get("max_cpu_call_seconds", 4.))
    torch.set_num_threads(threads)
    decoder.to(device)
    matcher = evaluate.Matcher(test.circuit, True)
    out["mwpm_corr_cpu_threads_1"] = evaluate.benchmark(matcher.decode_batch, sample, batch_sizes=e["batch_sizes"],
                                                        min_seconds=e["min_seconds"])
    return out


def run_mixture(args, cfg, run_dir, result, result_path, log, learned) -> int:
    """Stage 1: per (device, patch, basis), pre-train one model on the mixture; score it on every eval set."""
    from .sequence import Mixture
    t = cfg["test"]
    groups = [g for g in mixture_groups(cfg) if not args.groups or g["name"] in args.groups]
    seed = int(cfg.get("seed", 0))
    budget = float(cfg.get("max_total_train_seconds", float("inf")))
    result.setdefault("groups", {})
    spent = sum(g.get("train", {}).get("wall_seconds", 0) for g in result["groups"].values())
    keep_raw = bool(t.get("keep_raw", True))
    log(f"run {result['run_id']}: backend {args.backend}, mixture mode, groups {[g['name'] for g in groups]}, "
        f"pid {os.getpid()}, results {run_dir}")
    for group in groups:
        if result["groups"].get(group["name"], {}).get("status") in ("complete", "dropped"):
            continue
        comps, evals = group_components(cfg, group)
        rounds = tuple(int(r) for r in cfg["rounds"])
        d, basis = int(cfg["distance"]), group["basis"]
        eval_specs = [c.spec(d, r, basis) for c, e in zip(comps, evals) if e for r in cfg["eval"].get("rounds", rounds)]
        log(f"== {group['name']}: {len(comps)} components, {len(eval_specs)} eval sets")
        tests = []
        for spec in eval_specs:
            test, sizing = test_set_raw(spec, cfg, log, keep_raw)
            refs, records = references_for(test, cfg)
            tests.append((spec, test, sizing, refs, records))
            c = records["mwpm_corr"]["point"]
            log(f"  {spec.slug}: N={test.shots:,} corr MWPM {c['failures']} ({c['p_l']:.4f}), uniform-weight "
                f"{records.get('mwpm_corr_uniform', {}).get('point', {}).get('p_l', float('nan')):.4f}")
        entry = dict(status="running", group=group, mixture=None)
        result["groups"][group["name"]] = entry
        decoder = None
        if learned:
            if spent >= budget:
                entry.update(status="skipped", reason=f"training budget {budget:.0f} s exhausted")
                log("  " + entry["reason"])
                write_json(result_path, result)
                continue
            import importlib
            module = importlib.import_module(f".models.{args.backend}", __package__)
            mix = Mixture(d, basis, tuple(comps), rounds, float(cfg.get("curriculum", {}).get("fraction", 0.15)),
                          int(cfg.get("curriculum", {}).get("max_rounds", 5)))
            entry["mixture"] = mix.describe()
            v = cfg["val"]
            model_cfg = dict(model=cfg.get("model", {}), train={**cfg.get("train", {}), "workers": cfg.get("workers", 6)},
                             family=f"{group['device']}:{group['patch']}")
            raw_model = bool(model_cfg["model"].get("raw_record"))
            vals = [data.cached_dataset(c.spec(d, r, basis), "val", int(v["shots"]), seed=seed, workers=cfg.get("workers", 6),
                                        log=log, keep_raw=raw_model)
                    for c, e in zip(comps, evals) if e for r in v["rounds"]]
            device = device_for(cfg)
            decoder = module.build(tests[0][1].circuit, model_cfg)
            (run_dir / "models").mkdir(exist_ok=True)
            decoder.checkpoint_path = str(run_dir / "models" / f"{group['name']}.ckpt")
            info = decoder.fit(mix, vals, device=device, seed=seed, log=log)
            decoder.save(run_dir / "models" / f"{group['name']}.pt")
            spent += info["wall_seconds"]
            entry.update(train=info, metadata=decoder.metadata(), model_file=f"models/{group['name']}.pt",
                         val_sets=[x.manifest["dataset_id"] for x in vals])
            log(f"  trained {info['wall_seconds']:.0f} s ({info['stop_reason']}), {info['shots_seen']:,} shots, "
                f"val P_L {info['final_val_p_l']:.4f}; spent {spent:.0f} s")
        for spec, test, sizing, refs, records in tests:
            point = score_point(spec, test, sizing, refs, records, cfg)
            point["group"] = group["name"]
            if decoder is not None:
                point["decoder"] = decoder_record(decoder, test, refs, decoder.device)
                point["decoder"]["train"] = {k: entry["train"][k] for k in ("wall_seconds", "shots_seen", "stop_reason")}
                point["decoder"]["metadata"] = dict(parameters=entry["metadata"]["parameters"],
                                                    architecture=entry["metadata"]["architecture"])
                pd, pv = point["decoder"]["point"], point["decoder"]["paired_vs_mwpm_corr"]
                log(f"  {spec.slug}: {args.backend} {pd['p_l']:.4f} vs corr MWPM {records['mwpm_corr']['point']['p_l']:.4f}"
                    f"  dP_L {pv['delta_p_l']:+.2e} [{pv['ci95_paired_bootstrap'][0]:+.2e}, {pv['ci95_paired_bootstrap'][1]:+.2e}]"
                    f" -> {pv['verdict']}")
            result["points"] = [q for q in result["points"] if not (q.get("group") == group["name"] and q["spec"] == point["spec"])]
            result["points"].append(point)
        if decoder is not None and cfg["eval"].get("benchmark", True):
            entry["speed"] = [speed_record(decoder, test, cfg, decoder.device)
                              for spec, test, *_ in tests[:len(cfg["eval"].get("rounds", rounds))]
                              if spec.rounds in cfg["eval"].get("speed_rounds", [])]
        entry["status"] = "complete"
        write_json(result_path, result)
    result["fits"] = group_fits(result["points"])
    record = evaluate.write_ler(run_dir, result)
    result["ler"] = dict(directory="ler", fits=len(record["fits"]), lambdas=len(record["lambdas"]))
    result["status"] = ("complete" if all(g.get("status") in ("complete", "dropped") for g in result["groups"].values())
                        else "partial")
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["total_train_seconds"] = spent
    write_json(result_path, result)
    log(f"done: {len(result['points'])} points, training {spent:.0f} s")
    if not args.no_report:
        from . import report
        report.main([])
    return 0


def test_set_raw(spec: DatasetSpec, cfg: dict, log, keep_raw: bool):
    t = cfg["test"]
    leakage = data.uses_leakage_sampler(spec)
    pilot = data.pilot(spec, max_shots=t["pilot_max_shots_leakage" if leakage else "pilot_max_shots"],
                       target_failures=t["pilot_target_failures"], workers=cfg.get("workers", 1), log=log)
    sizing = data.choose_test_shots(pilot, target_failures=t["target_failures"], safety=t["safety"],
                                    min_shots=t["min_shots_leakage" if leakage else "min_shots"], max_shots=t["max_shots"])
    dataset = data.frozen_test_set(spec, sizing["shots"], workers=cfg.get("workers", 1), log=log, keep_raw=keep_raw)
    return dataset, dict(pilot=pilot, sizing=sizing)


if __name__ == "__main__":
    sys.exit(main())
