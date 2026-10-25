"""Why does the MLP beat correlated MWPM? Shot-level analysis of where the two disagree.

python -m mlDecoder.analysis why --run <mlp run-id> [--noise leakage] [--distance 3]

Analysis only, no timing: frozen test shots, the run's saved MLPs (CPU, one thread) and
the cached correlated-MWPM predictions. Writes results/analysis/why_<noise>/ with
summary.json (per config) and summary.md (included in RESULTS.md by the report).
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import sys
import numpy as np

from . import data, evaluate
from .data import DatasetSpec, results_root

GROUPS = ("both_right", "both_wrong", "mlp_only_right", "mwpm_only_right")
STREAK_BINS = ((0, 1, "0–1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 10**9, "≥5"))
EVENT_BINS = ((0, 0, "0"), (1, 2, "1–2"), (3, 4, "3–4"), (5, 8, "5–8"), (9, 16, "9–16"), (17, 10**9, "≥17"))
LOGIT_BINS = ((0, .5, "<0.5"), (.5, 1, "0.5–1"), (1, 2, "1–2"), (2, 4, "2–4"), (4, np.inf, "≥4"))


def stabilizer_weights(distance: int) -> dict:
    """Ancilla qubit -> number of data qubits it checks (2 on the boundary, 4 in the bulk)."""
    from challenge.surface_code import make_stim_circuit
    template = make_stim_circuit(distance, 1, "Z", True, gate_basis="cx")
    _, anc = data.get_qubit_lists(template)
    weights = dict.fromkeys(anc, 0)
    for inst in template.flattened():
        if inst.name == "CX":
            for q in (t.value for t in inst.targets_copy()):
                if q in weights:
                    weights[q] += 1
    return weights


def shot_features(dets: np.ndarray, layout: dict) -> dict:
    """Per shot: total events, longest firing streak on one check, streaks >= 3, most firings on one check."""
    grid = layout["grid"]
    valid = grid >= 0
    series = np.zeros((len(dets), *grid.shape), dtype=np.uint8)
    series[:, valid] = dets[:, grid[valid]]
    run = np.zeros((len(dets), grid.shape[1]), dtype=np.int32)
    longest = np.zeros(len(dets), dtype=np.int32)
    streaks3 = np.zeros(len(dets), dtype=np.int32)
    for t in range(grid.shape[0]):
        run = (run + 1) * series[:, t]
        longest = np.maximum(longest, run.max(axis=1))
        streaks3 += (run == 3).sum(axis=1)  # A streak is counted once, when it reaches 3.
    return dict(total=dets.sum(axis=1, dtype=np.int32), longest=longest, streaks3=streaks3,
                max_on_one_check=series.sum(axis=1).max(axis=1))


def histogram(values: np.ndarray, bins) -> dict:
    return {label: int(((values >= lo) & (values <= hi)).sum()) if hi != np.inf else int((values >= lo).sum())
            for lo, hi, label in bins}


def firing_classes(dets: np.ndarray, layout: dict, weights: dict) -> dict:
    """Mean firings per shot by check type, time class and spatial class."""
    if not len(dets):
        return {}
    rounds = layout["rounds"]
    stab_weight = np.array([weights[s["qubit"]] for s in layout["stabilizers"]])[layout["stabilizer"]]
    time_class = np.where(layout["is_final"], "final (data readout)",
                          np.where(layout["round"] == 0, "round 0", np.where(layout["round"] == 1, "round 1", "bulk rounds")))
    mean = dets.mean(axis=0)
    out = {f"type {k}": float(mean[layout["type"] == k].sum()) for k in ("X", "Z")}
    out.update({k: float(mean[time_class == k].sum()) for k in ("round 0", "round 1", "bulk rounds", "final (data readout)")})
    out.update({"weight-2 (boundary) checks": float(mean[stab_weight == 2].sum()),
                "weight-4 (bulk) checks": float(mean[stab_weight == 4].sum())})
    out["per round"] = [float(mean[layout["round"] == t].sum()) for t in range(rounds + 1)]
    return out


def conditional(rows: list[dict], key: str, bins) -> list[dict]:
    """P_L(MLP) and P_L(MWPM) per feature bin, with each bin's share of the total failure difference."""
    feature = np.concatenate([r[key] for r in rows])
    mlp_fail = np.concatenate([r["mlp_fail"] for r in rows])
    ref_fail = np.concatenate([r["ref_fail"] for r in rows])
    advantage = int(ref_fail.sum() - mlp_fail.sum())
    out = []
    for lo, hi, label in bins:
        sel = (feature >= lo) & (feature <= hi)
        n, km, kr = int(sel.sum()), int(mlp_fail[sel].sum()), int(ref_fail[sel].sum())
        out.append(dict(bin=label, shots=n, shot_fraction=n / len(feature), mlp_failures=km, mwpm_failures=kr,
                        p_l_mlp=km / n if n else None, p_l_mwpm=kr / n if n else None,
                        share_of_advantage=(kr - km) / advantage if advantage else None))
    return out


def why(run_id: str, noise: str = "leakage", distance: int = 3) -> dict:
    import torch
    from .models.mlp import MLPDecoder
    torch.set_num_threads(1)
    started = datetime.now(timezone.utc).isoformat()
    run_dir = results_root() / "mlp" / run_id
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    weights = stabilizer_weights(distance)
    configs, pooled = [], {"X": [], "Z": []}
    for pt in sorted(result["points"], key=lambda p: (p["spec"]["basis"], p["spec"]["rounds"])):
        s = pt["spec"]
        if s["noise"] != noise or s["distance"] != distance or "decoder" not in pt:
            continue
        spec = DatasetSpec.from_dict(s)
        test = data.load_dataset(results_root() / "datasets" / pt["dataset"]["dataset_id"], spec)
        dets, obs = test.dets, test.obs.astype(bool)
        model = MLPDecoder.load(run_dir / "models" / f"{spec.slug}.pt", test.circuit, "cpu")
        mlp_pred = model.decode_batch(dets)
        logits = model.logits(dets)[:, 0]
        ref = evaluate.reference(test, True)
        mlp_fail, ref_fail = evaluate.failure_mask(mlp_pred, obs), ref["fail"]
        recorded = pt["decoder"]["point"]["failures"]
        # The run scored on CUDA with float16 autocast; here float32 on the CPU: borderline logits may flip.
        if abs(int(mlp_fail.sum()) - recorded) > max(5, .005 * recorded):
            raise RuntimeError(f"{spec.slug}: reloaded MLP gives {int(mlp_fail.sum())} failures, the run recorded {recorded}")
        layout = data.detector_layout(test.circuit)
        feats = shot_features(dets, layout)
        masks = dict(both_right=~mlp_fail & ~ref_fail, both_wrong=mlp_fail & ref_fail,
                     mlp_only_right=~mlp_fail & ref_fail, mwpm_only_right=mlp_fail & ~ref_fail)
        entry = dict(slug=spec.slug, basis=s["basis"], rounds=s["rounds"], shots=len(obs),
                     counts={g: int(m.sum()) for g, m in masks.items()},
                     mlp_failures=int(mlp_fail.sum()), mlp_failures_recorded_cuda_fp16=recorded,
                     mwpm_failures=int(ref_fail.sum()), groups={})
        for g in GROUPS:
            m = masks[g]
            entry["groups"][g] = dict(
                mean_total_events=float(feats["total"][m].mean()) if m.any() else None,
                mean_longest_streak=float(feats["longest"][m].mean()) if m.any() else None,
                mean_streaks_ge3=float(feats["streaks3"][m].mean()) if m.any() else None,
                mean_max_on_one_check=float(feats["max_on_one_check"][m].mean()) if m.any() else None,
                total_events=histogram(feats["total"][m], EVENT_BINS),
                longest_streak=histogram(feats["longest"][m], STREAK_BINS),
                streaks_ge3={k: int(v) for k, v in zip(("0", "1", "2", "≥3"),
                             [(feats["streaks3"][m] == i).sum() for i in range(3)] + [(feats["streaks3"][m] >= 3).sum()])},
                fires=firing_classes(dets[m], layout, weights),
                abs_logit=(dict(median=float(np.nanmedian(np.abs(logits[m]))),
                                histogram=histogram(np.abs(logits[m][~np.isnan(logits[m])]), LOGIT_BINS))
                           if m.any() and not np.isnan(logits[m]).all() else None))
        configs.append(entry)
        pooled[s["basis"]].append(dict(mlp_fail=mlp_fail, ref_fail=ref_fail, longest=feats["longest"], total=feats["total"]))
    conditional_tables = {}
    for basis, rows in list(pooled.items()) + [("X+Z", pooled["X"] + pooled["Z"])]:
        if rows:
            conditional_tables[basis] = dict(longest_streak=conditional(rows, "longest", STREAK_BINS),
                                             total_events=conditional(rows, "total", EVENT_BINS))
    summary = dict(run_id=run_id, noise=noise, distance=distance, p=result["points"][0]["spec"]["p"],
                   started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                   note=("Analysis only (no timing), on frozen test shots, the run's saved MLPs on one CPU thread "
                         "(float32; the run scored them on CUDA with float16 autocast, which differs on a few borderline "
                         "shots per 100k), and cached correlated-MWPM predictions. Streak: consecutive rounds in which "
                         "one check's detector fires; streaks >= 3 are counted once each. Simulated data."),
                   configs=configs, conditional=conditional_tables)
    out = results_root() / "analysis" / f"why_{noise}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8", newline="\n")
    return summary


def markdown(summary: dict) -> str:
    s = summary
    lines = [f"### Why the MLP wins: d={s['distance']} `{s['noise']}` p={s['p']:g} (run `{s['run_id']}`)", "",
             s["note"], "",
             "**Who gets which shots right**, per configuration (MLP-only right = correlated MWPM wrong, MLP right):", "",
             "| basis | R | N | both right | both wrong | MLP-only right | MWPM-only right | MLP fails | MWPM fails |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in s["configs"]:
        k = c["counts"]
        lines.append(f"| {c['basis']} | {c['rounds']} | {c['shots']:,} | {k['both_right']:,} | {k['both_wrong']:,} | "
                     f"{k['mlp_only_right']:,} | {k['mwpm_only_right']:,} | {c['mlp_failures']:,} | {c['mwpm_failures']:,} |")
    lines += ["", "**The two discordant groups**, means per shot pooled over configurations (weighted by shots in the group):", "",
              "| group | shots | events | longest streak | streaks ≥ 3 | most firings on one check | median abs(logit) | "
              "type X | type Z | round 0 | round 1 | bulk rounds | final | weight-2 checks | weight-4 checks |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ("mlp_only_right", "mwpm_only_right", "both_wrong"):
        w = np.array([c["counts"][g] for c in s["configs"]], dtype=float)
        if w.sum() == 0:
            continue
        def avg(get):
            vals = [get(c["groups"][g]) for c in s["configs"]]
            return float(sum(v * x for v, x in zip(vals, w) if v is not None) / w[[v is not None for v in vals]].sum())
        cls = ["type X", "type Z", "round 0", "round 1", "bulk rounds", "final (data readout)",
               "weight-2 (boundary) checks", "weight-4 (bulk) checks"]
        lines.append(f"| {g.replace('_', ' ')} | {int(w.sum()):,} | {avg(lambda x: x['mean_total_events']):.2f} | "
                     f"{avg(lambda x: x['mean_longest_streak']):.2f} | {avg(lambda x: x['mean_streaks_ge3']):.2f} | "
                     f"{avg(lambda x: x['mean_max_on_one_check']):.2f} | "
                     f"{avg(lambda x: x['abs_logit']['median'] if x['abs_logit'] else None):.2f} | "
                     + " | ".join(f"{avg(lambda x, k=k: x['fires'].get(k) if x['fires'] else None):.2f}" for k in cls) + " |")
    for basis, tables in s["conditional"].items():
        for key, title in (("longest_streak", "longest streak"), ("total_events", "total detection events")):
            lines += ["", f"**P_L conditional on the {title}**, basis {basis}, pooled over R "
                          "(share = the bin's part of MWPM failures minus MLP failures):", "",
                      f"| {title} | shots | fraction | P_L MLP | P_L MWPM | share of advantage |", "|---|---:|---:|---:|---:|---:|"]
            for row in tables[key]:
                fmt = lambda v: f"{v:.2e}" if v is not None else "–"
                lines.append(f"| {row['bin']} | {row['shots']:,} | {row['shot_fraction']:.3f} | {fmt(row['p_l_mlp'])} | "
                             f"{fmt(row['p_l_mwpm'])} | {row['share_of_advantage']:.2f} |" if row["share_of_advantage"] is not None
                             else f"| {row['bin']} | {row['shots']:,} | {row['shot_fraction']:.3f} | {fmt(row['p_l_mlp'])} | "
                                  f"{fmt(row['p_l_mwpm'])} | – |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- overfitting audit

def _z(k1, n1, k2, n2) -> float:
    """(p2 - p1) in units of its binomial standard error."""
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1*(1 - p1)/n1 + p2*(1 - p2)/n2)
    return float((p2 - p1) / se) if se > 0 else 0.


def held_in_training_sample(spec, train: dict, seed: int, shots: int = 200_000):
    """Shots the model actually trained on: the start of its pool, or its stream's first batches regenerated."""
    if train["pool_shots"]:
        pool = data.cached_dataset(spec, "train", train["pool_shots"], seed=seed)
        n = min(shots, pool.shots)
        return pool.dets_packed[:n], pool.obs[:n], f"first {n:,} shots of the {pool.shots:,}-shot pool"
    if data.uses_leakage_sampler(spec):  # Fresh leakage stream: regenerate its first chunks.
        parts = [data.sample_shots(spec, data.LEAKAGE_CHUNK, data.chunk_seed(spec, "train", seed, data.LEAKAGE_STREAM_BASE + k))
                 for k in range(shots // data.LEAKAGE_CHUNK)]
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                f"the stream's first {shots:,} fresh shots, regenerated")
    stream, batch, dets, obs = data.Stream(spec, seed), train["batch_size"], [], []
    while sum(len(o) for o in obs) < shots:
        d, o = stream.next(batch)
        dets.append(d)
        obs.append(o)
    return np.concatenate(dets), np.concatenate(obs), f"the stream's first {sum(len(o) for o in obs):,} shots, regenerated"


def audit(run_id: str, device: str = "cuda") -> dict:
    from .models.mlp import MLPDecoder
    run_dir = results_root() / "mlp" / run_id
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    seed = int(result["config"].get("seed", 0))
    rows = []
    for pt in sorted(result["points"], key=lambda p: (p["spec"]["p"], p["spec"]["noise"], p["spec"]["distance"],
                                                         p["spec"]["basis"], p["spec"]["rounds"])):
        if "decoder" not in pt:
            continue
        spec = DatasetSpec.from_dict(pt["spec"])
        tr = pt["decoder"]["train"]
        test = data.load_dataset(results_root() / "datasets" / pt["dataset"]["dataset_id"], spec)
        model = MLPDecoder.load(run_dir / "models" / f"{spec.slug}.pt", test.circuit, device)
        val = data.cached_dataset(spec, "val", tr["val_shots"], seed=seed)
        held_dets, held_obs, held_note = held_in_training_sample(spec, tr, seed)
        scores = {}
        for name, packed, obs in (("train", held_dets, held_obs), ("val", val.dets_packed, val.obs),
                                  ("test", test.dets_packed, test.obs)):
            fail = evaluate.failure_mask(model.decode_packed(packed), obs)
            scores[name] = evaluate.point(fail)
        z_train_test = _z(scores["train"]["failures"], scores["train"]["shots"], scores["test"]["failures"], scores["test"]["shots"])
        z_val_test = _z(scores["val"]["failures"], scores["val"]["shots"], scores["test"]["failures"], scores["test"]["shots"])
        rows.append(dict(slug=spec.slug, spec=pt["spec"], parameters=pt["decoder"]["metadata"]["parameters"],
                         data=tr["data"], pool_shots=tr["pool_shots"], shots_seen=tr["shots_seen"], epochs=tr["epochs"],
                         train_sample=held_note, scores=scores, z_train_beats_test=z_train_test, z_val_beats_test=z_val_test,
                         flag=z_train_test > 2, stop_reason=tr["stop_reason"], best_step=tr["best_step"], steps=tr["steps"],
                         seeds=dict(train_stream=data.chunk_seed(spec, "train", seed, data.STREAM_CHUNK),
                                    train_pool_chunk0=data.chunk_seed(spec, "train", seed, 0),
                                    val_chunk0=data.chunk_seed(spec, "val", seed, 0),
                                    test_chunk0=data.chunk_seed(spec, "test", 0, 0))))
    summary = dict(run_id=run_id, kind="audit", rows=rows, device=device,
                   separation=dict(
                       seed_sequences=(f"numpy SeedSequence(entropy={data.DATA_SEED}, spawn_key=(8 uint32 words of "
                                       f"sha256(spec), DATA_VERSION={data.DATA_VERSION}, split id {data.SPLITS}, run seed, chunk))"),
                       run_seed=seed, test_seed="0 (canonical: test and pilot shots depend on the spec alone)",
                       stream_chunks=(f"Pauli stream: chunk {data.STREAM_CHUNK}; leakage pool: chunks 0..; fresh leakage "
                                      f"stream: chunks {data.LEAKAGE_STREAM_BASE}.."),
                       test_used_for=("final scoring only: run.train_and_score decodes the frozen test set after fit() "
                                      "has returned and frozen the weights"),
                       val_used_for=("early stopping and checkpoint choice only: train._train keeps the state with the "
                                     "fewest val failures (ties: lower val loss) and stops after patience_evals evaluations "
                                     "without improvement"),
                       hyperparameters=("fixed in configs/mlp_*.json before any test scoring; no search was run on val or test"),
                       test_size=("chosen from a correlated-MWPM pilot on the separate 'pilot' split, never from test"),
                       features=("detection events only: MLPDecoder._check / decode_packed take det_events; obs_flips "
                                 "enter only as BCE labels and in failure counting")))
    out = results_root() / "analysis" / f"audit_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [f"### Overfitting audit: run `{run_id}`", "",
             "P_L of each frozen model on shots it trained on (held-in), on its validation set and on the frozen test set "
             "(Wilson 95%). z = (P_test − P_train)/σ; **flag** if the model does better on its training shots than on test "
             "by more than 2σ. Epochs = training shots seen / pool size (pool runs only).", "",
             "Separation: " + "; ".join(f"**{k.replace('_', ' ')}**: {v}" for k, v in summary["separation"].items()) + ".", "",
             "| noise | p | d | basis | R | params | data | pool | epochs | P_L train (held-in) | P_L val | P_L test | z train<test | z val<test | flag |",
             "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    cell = lambda x: f"{x['p_l']:.2e} [{x['ci95_wilson'][0]:.2e}, {x['ci95_wilson'][1]:.2e}]"
    for r in rows:
        s, sc = r["spec"], r["scores"]
        pool = r["pool_shots"]
        cells = [s["noise"], f"{s['p']:g}", s["distance"], s["basis"], s["rounds"], f"{r['parameters']:,}",
                 "pool" if pool else "fresh", f"{pool:,}" if pool else "–", f"{r['epochs']:.0f}" if r["epochs"] else "–",
                 cell(sc["train"]), cell(sc["val"]), cell(sc["test"]), f"{r['z_train_beats_test']:+.1f}",
                 f"{r['z_val_beats_test']:+.1f}", "**FLAG**" if r["flag"] else ""]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    flagged = [r["slug"] for r in rows if r["flag"]]
    lines += ["", f"Flagged (train beats test by > 2σ): {', '.join(flagged) if flagged else 'none'}."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


# ---------------------------------------------------------------- fresh-test replication

def replicate(run_id: str, noise: str = "leakage", distance: int = 3, device: str = "cuda", workers: int = 4) -> dict:
    """Score the run's MLPs and correlated MWPM on new, independent 'test2' sets; refit eps (A=1) on each."""
    from challenge.decoding import fit_ler
    from .models.mlp import MLPDecoder
    run_dir = results_root() / "mlp" / run_id
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    groups = {}
    for pt in sorted(result["points"], key=lambda p: (p["spec"]["basis"], p["spec"]["rounds"])):
        s = pt["spec"]
        if s["noise"] != noise or s["distance"] != distance or "decoder" not in pt:
            continue
        spec = DatasetSpec.from_dict(s)
        test = data.load_dataset(results_root() / "datasets" / pt["dataset"]["dataset_id"], spec)
        test2 = data.cached_dataset(spec, "test2", test.shots, seed=0, workers=workers)
        model = MLPDecoder.load(run_dir / "models" / f"{spec.slug}.pt", test.circuit, device)
        row = dict(rounds=s["rounds"], shots=test.shots, shots2=test2.shots, dataset2=test2.manifest["dataset_id"])
        for label, ds in (("test", test), ("test2", test2)):
            mlp_fail = evaluate.failure_mask(model.decode_packed(ds.dets_packed), ds.obs)
            ref_fail = evaluate.reference(ds, True)["fail"]
            row[label] = dict(mlp=int(mlp_fail.sum()), mwpm_corr=int(ref_fail.sum()),
                              paired=evaluate.paired(mlp_fail, ref_fail))
        groups.setdefault(s["basis"], []).append(row)
    fits_out = []
    for basis, rows in sorted(groups.items()):
        rounds = [r["rounds"] for r in rows]
        entry = dict(basis=basis, rounds=rounds, min_mwpm_failures_test2=min(r["test2"]["mwpm_corr"] for r in rows))
        for dec in ("mlp", "mwpm_corr"):
            for label, n_key in (("test", "shots"), ("test2", "shots2")):
                f = fit_ler(rounds, [r[label][dec] for r in rows], [r[n_key] for r in rows])
                entry[f"{dec}_{label}"] = dict(epsilon=f["epsilon_per_round"], ci95_profile=f["ci95_profile"])
            a, b = entry[f"{dec}_test"], entry[f"{dec}_test2"]
            se = lambda x: (x["ci95_profile"][1] - x["ci95_profile"][0]) / (2 * 1.96)
            z = (b["epsilon"] - a["epsilon"]) / np.hypot(se(a), se(b))
            entry[f"{dec}_agreement"] = dict(z=float(z), agree=bool(abs(z) < 2),
                                             overlap=bool(a["ci95_profile"][1] >= b["ci95_profile"][0] and
                                                          b["ci95_profile"][1] >= a["ci95_profile"][0]))
        entry["ratio_test"] = entry["mlp_test"]["epsilon"] / entry["mwpm_corr_test"]["epsilon"]
        entry["ratio_test2"] = entry["mlp_test2"]["epsilon"] / entry["mwpm_corr_test2"]["epsilon"]
        fits_out.append(entry)
    summary = dict(run_id=run_id, kind="replicate", noise=noise, distance=distance, p=result["points"][0]["spec"]["p"],
                   rows=groups, fits=fits_out,
                   note=("test2: a new canonical split stream (split id 5, seed 0), the same spec and the same N as the "
                         "frozen test set, generated after the models were frozen. A = 1 fits (fit_ler), profile 95% "
                         "intervals; agreement z = difference / combined standard error. Simulated data."))
    out = results_root() / "analysis" / f"replicate_{noise}_d{distance}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    cell = lambda x: f"{x['epsilon']:.3e} [{x['ci95_profile'][0]:.3e}, {x['ci95_profile'][1]:.3e}]"
    lines = [f"### Fresh-test replication: d={distance} `{noise}` p={summary['p']:g} (run `{run_id}`)", "", summary["note"], "",
             "| basis | decoder | ε_L frozen test [95%] | ε_L test2 [95%] | z | agree |", "|---|---|---:|---:|---:|---|"]
    for f in fits_out:
        for dec, label in (("mlp", "MLP"), ("mwpm_corr", "correlated MWPM")):
            ag = f[f"{dec}_agreement"]
            lines.append(f"| {f['basis']} | {label} | {cell(f[dec + '_test'])} | {cell(f[dec + '_test2'])} | "
                         f"{ag['z']:+.2f} | {'yes' if ag['agree'] else '**NO**'} |")
        lines.append(f"| {f['basis']} | ratio MLP/MWPM | {f['ratio_test']:.3f} | {f['ratio_test2']:.3f} | | |")
    lines += ["", "Per round count, failures on test2 (MLP / correlated MWPM) and the paired ΔP_L:", "",
              "| basis | R | N | test: MLP / MWPM | test2: MLP / MWPM | test2 ΔP_L [95%] |", "|---|---:|---:|---:|---:|---:|"]
    for basis, rows in sorted(groups.items()):
        for r in rows:
            pr = r["test2"]["paired"]
            lines.append(f"| {basis} | {r['rounds']} | {r['shots2']:,} | {r['test']['mlp']} / {r['test']['mwpm_corr']} | "
                         f"{r['test2']['mlp']} / {r['test2']['mwpm_corr']} | {pr['delta_p_l']:+.2e} "
                         f"[{pr['ci95_paired_bootstrap'][0]:+.2e}, {pr['ci95_paired_bootstrap'][1]:+.2e}] |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["why", "audit", "replicate"])
    parser.add_argument("--run", required=True, help="An mlp run id under results/mlp/")
    parser.add_argument("--noise", default="leakage")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "why":
        why(args.run, args.noise, args.distance)
    elif args.command == "audit":
        audit(args.run, args.device)
    else:
        replicate(args.run, args.noise, args.distance, args.device)
    print(f"wrote results under {results_root() / 'analysis'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
