"""Regenerate mlDecoder/RESULTS.md from every result JSON under results/ (idempotent).

python -m mlDecoder.report

Per spec, the latest run of each (backend, config) wins. The MWPM baseline comes
from the latest point carrying it; fits are recomputed from the selected points.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from .data import results_root

OUT = Path(__file__).resolve().parent / "RESULTS.md"
FIGURES = Path(__file__).resolve().parent / "figures"


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*/*/result.json")):
        if path.parent.parent.name == "datasets":
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        run["_config"] = Path(run["config_path"]).stem
        runs.append(run)
    return sorted(runs, key=lambda r: r["run_id"])


def spec_key(s: dict) -> tuple:
    return (s["p"], s["noise"], s["distance"], s["basis"], s["rounds"], s["gate_basis"], s["no_reset"], str(s["calibration"]))


def e(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.2e}" if abs(x) < 1e-2 else f"{x:.4f}"


def ci(pair) -> str:
    return f"[{e(pair[0])}, {e(pair[1])}]"


def signed(x: float) -> str:
    return ("+" if x > 0 else "") + e(x)


def speed(rows, batch, field) -> str:
    for row in rows or []:
        if row.get("batch") == batch:
            if "skipped" in row:
                return "skip"
            value = row[field]
            return f"{value/1e3:,.0f}k" if field == "shots_per_second" else f"{value:,.1f}"
    return "–"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if h not in ("basis", "decoder", "verdict", "noise") else "---"
                                                            for h in header) + "|"]
    return out + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def verdict(paired: dict) -> str:
    lo, hi = paired["ci95_paired_bootstrap"]
    return "**win**" if hi < 0 else "loss" if lo > 0 else "tie"


def gof(fit: dict) -> str:
    g = fit["goodness_of_fit"]
    return f"{g['pearson_chi2']:.1f}/{g['dof']} (p={g['pearson_p']:.2f})" if g["dof"] > 0 else "–"


def ler_section(run: dict) -> list[str]:
    import csv
    run_dir = results_root() / run["backend"] / run["run_id"]
    path = run_dir / "ler" / "ler_fits.json"
    if not path.exists():
        return []
    record = json.loads(path.read_text(encoding="utf-8"))
    with (run_dir / "ler" / "ler_points.csv").open(encoding="utf-8") as stream:
        points = [{k: (v if k in ("decoder", "basis", "noise", "calibration", "group") else float(v)) for k, v in row.items()}
                  for row in csv.DictReader(stream)]
    for point in points:
        point["rounds"], point["d"] = int(point["rounds"]), int(point["d"])
        point.setdefault("calibration", "")
        point.setdefault("group", "")
    for f in record["fits"]:
        f.setdefault("calibration_label", "")
        f.setdefault("group", "")
    lines = [f"### {run['backend']} ({run['_config']}), run `{run['run_id']}`", ""]
    rows = []
    mixture = run.get("config", {}).get("mode") == "mixture" or run.get("mode") in ("finetune", "pilot")
    order = lambda f: (f["group"], f["p"], f["noise"], f["calibration_label"], f["d"], f["basis"],
                       f["decoder"] != "mwpm_corr", f["decoder"])
    lead = (lambda f: [f["group"], f["noise"], options(f["calibration_label"]), f"{f['p']:g}", f["basis"]]) if mixture else \
           (lambda f: [f["noise"], f"{f['p']:g}", f["d"], f["basis"]])
    lead_head = ["group", "noise", "options", "p", "basis"] if mixture else ["noise", "p", "d", "basis"]
    for f in sorted(record["fits"], key=order):
        if "no_spam" not in f:
            continue
        ns, sp, ratio = f["no_spam"], f.get("spam"), f.get("ratio_to_mwpm_corr")
        short = f["reference_shortfall"]
        rows.append([*lead(f), f["decoder"], len(f["rounds"]),
                     f"{e(ns['epsilon'])} {ci(ns['ci95_profile'])}", gof(ns),
                     f"{e(sp['epsilon'])} {ci(sp['ci95_profile'])}" if sp else "–", f"{sp['amplitude']:.3f}" if sp else "–",
                     gof(sp) if sp else "–",
                     (f"{ratio['no_spam']:.3f} " + (f"[{ratio['no_spam_ci95_paired_bootstrap'][0]:.3f}, "
                      f"{ratio['no_spam_ci95_paired_bootstrap'][1]:.3f}]" if ratio["no_spam_ci95_paired_bootstrap"] else ""))
                     if ratio and ratio["no_spam"] is not None else "–",
                     (f"{ratio['spam']:.3f} " + (f"[{ratio['spam_ci95_paired_bootstrap'][0]:.3f}, "
                      f"{ratio['spam_ci95_paired_bootstrap'][1]:.3f}]" if ratio.get("spam_ci95_paired_bootstrap") else ""))
                     if ratio and ratio.get("spam") is not None else "–",
                     f["min_reference_failures"] if not short else f"{f['min_reference_failures']} (short at R={','.join(short)})"])
    if rows:
        lines += table([*lead_head, "decoder", "#R", "ε_L A=1 [95%]", "χ²/dof (p)", "ε_L SPAM [95%]",
                        "A", "χ²/dof (p) ", "ε/ε_corr A=1 [95%]", "ε/ε_corr SPAM [95%, reduced]", "min corr k"], rows)
        lines.append("")
    sens_rows = []
    for f in sorted(record["fits"], key=order):
        sens = f.get("sensitivity", {})
        if "no_spam" not in f or not sens:
            continue
        def cell(fit):
            if not fit:
                return "–"
            g = fit["goodness_of_fit"]
            return f"{e(fit['epsilon'])} {ci(fit['ci95_profile'])} (p={g['pearson_p']:.2g})"
        free = sens.get("free_amplitude")
        sens_rows.append([*lead(f), f["decoder"], cell(f["no_spam"]),
                          cell(sens.get("a1_r_min_3")), cell(sens.get("a1_r_min_5")), cell(free),
                          f"{free['amplitude']:.5f}" if free else "–",
                          f"{free['round_offset']:.2f}" if free and free["round_offset"] is not None else "–",
                          f"{free['lr_test_a_above_1']['p_value']:.2g}" if free else "–", f.get("trusted_fit") or "none"])
    if sens_rows:
        lines += ["**Sensitivity fits** (mlDecoder's, not the team's `fit_ler`): the A = 1 form refit on R ≥ 3 and R ≥ 5, "
                  "and a free-amplitude fit with A in (0, 2], written P_L = [1 − (1 − 2ε)^(R − R₀)]/2 with "
                  "R₀ = ln A / −ln(1 − 2ε). The **headline is the A = 1 all-R fit** (the challenge's formula); "
                  "\"trusted\" names the first fit, in the order all-R, free A, R ≥ 3, R ≥ 5, whose Pearson p ≥ 0.05. "
                  "A > 1 (significant by the likelihood-ratio p) is the boundary-round effect of the no-reset circuit, "
                  "not SPAM: round 1 has only basis-type detectors and round 2's detector is m₂ alone, so the curve "
                  "starts R₀ rounds late. With A forced to 1, dropping early rounds shrinks that bias but does not remove it.", ""]
        lines += table([*lead_head, "decoder", "A=1 all R", "A=1 R≥3", "A=1 R≥5", "free A",
                        "A", "R₀", "LR p (A>1)", "trusted"], sens_rows)
        lines.append("")
    lam = [[l["noise"], f"{l['p']:g}", l["basis"], l["decoder"], f"{l['d_from']}→{l['d_to']}",
            (f"{l['lambda_no_spam']:.2f}" + (f" [{l['ci95_bootstrap'][0]:.2f}, {l['ci95_bootstrap'][1]:.2f}]"
                                            if l["ci95_bootstrap"] else "")) if l["lambda_no_spam"] else "–"]
           for l in sorted(record["lambdas"], key=lambda l: (l["p"], l["noise"], l["basis"], l["decoder"], l["d_from"]))]
    if lam:
        lines += ["Λ = ε_L(d)/ε_L(d+2), no-SPAM fits, independent-bootstrap 95% interval:", ""]
        lines += table(["noise", "p", "basis", "decoder", "d", "Λ [95%]"], lam)
        lines.append("")
    if mixture:
        lines += speed_section(run)
    keys = sorted({(q["noise"], q["p"], q["calibration"], patch_of(q)) for q in points}, key=lambda x: (x[1], x[0], x[3], x[2]))
    for noise, p, cal, patch in keys:
        tag = (f"_{patch}" + (f"_{options(cal)}" if cal else "")) if mixture else ""
        name = f"ler_{run['backend']}_{run['_config']}_{noise}_p{p:g}{tag}.png"
        draw_figure(record, points, noise, p, FIGURES / name, f"{run['backend']} ({run['_config']})", cal, patch,
                    kind=record.get("label", "SIMULATED"))
        lines += [f"![P_L against R, {noise} p={p:g} {patch} {options(cal)}](figures/{name})", ""]
    return lines


def options(calibration_label: str) -> str:
    """A device slug without its device, calibration set and patch (e.g. ramsey-w1096-mf-noleak-asym-qndx)."""
    parts = calibration_label.split("-")
    return "-".join(parts[3:]) if len(parts) > 3 else calibration_label


def patch_of(point: dict) -> str:
    """The patch of a mixture point's group (garnet_pA_Z -> garnet_pA), or ''."""
    return point["group"].rsplit("_", 1)[0] if point.get("group") else ""


def speed_section(run: dict) -> list[str]:
    """Mixture runs: per group, the decoder's and correlated MWPM's warmed speed at the speed rounds."""
    rows = []
    for name, g in run.get("groups", {}).items():
        for sp in g.get("speed", []):
            rows.append([name, sp["rounds"], speed(sp.get("cuda"), 65536, "shots_per_second"),
                         speed(sp.get("cuda"), 1, "latency_per_shot_us"),
                         speed(sp.get("cpu_threads_1"), 1, "latency_per_shot_us"),
                         speed(sp.get("cpu_threads_1"), 4096, "shots_per_second"),
                         speed(sp.get("mwpm_corr_cpu_threads_1"), 65536, "shots_per_second"),
                         speed(sp.get("mwpm_corr_cpu_threads_1"), 1, "latency_per_shot_us")])
    if not rows:
        return []
    return ["Speed on this laptop (warmed; end to end, numpy in and out). Decoder: CUDA throughput at batch 65,536 and "
            "latency per shot at batch 1, CPU (1 thread) latency at batch 1 and throughput at batch 4,096; correlated "
            "MWPM on one CPU thread.", ""] + table(
        ["group", "R", "CUDA shots/s (b=65536)", "CUDA µs (b=1)", "CPU-1 µs (b=1)", "CPU-1 shots/s (b=4096)",
         "corr MWPM shots/s", "corr MWPM µs (b=1)"], rows) + [""]


def draw_figure(record, points, noise, p, path: Path, label: str, cal: str = "", patch: str = "", *,
                kind: str = "SIMULATED") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import plot_ler_panel
    sel = [q for q in points if q["noise"] == noise and q["p"] == p and q["calibration"] == cal and patch_of(q) == patch]
    ds, bases = sorted({q["d"] for q in sel}), sorted({q["basis"] for q in sel})
    fig, axes = plt.subplots(len(ds), len(bases), figsize=(4.2*len(bases), 3.0*len(ds)), squeeze=False,
                             constrained_layout=True)
    for i, d in enumerate(ds):
        for j, basis in enumerate(bases):
            fits_by = {f["decoder"]: f for f in record["fits"]
                       if (f["d"], f["basis"], f["noise"], f["p"], f["calibration_label"]) == (d, basis, noise, p, cal)
                       and patch_of(f) == patch}
            pts = [q for q in sel if q["d"] == d and q["basis"] == basis]
            plot_ler_panel(axes[i, j], fits_by, pts, f"d={d}, {basis} memory")
    fig.suptitle(f"{kind}, noise model '{noise}', p={p:g}{', ' + patch + ' ' + options(cal) if patch else ''}, "
                 f"no reset, CZ: {label}", fontsize=8)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=80, metadata={"Software": None})
    plt.close(fig)


def mixture_section(run: dict) -> list[str]:
    """A mixture run (stage 1): per group, the model against the three MWPM baselines on every eval set."""
    lines = [f"## {run['backend']} ({run['_config']}), run `{run['run_id']}`: SIMULATED development", "",
             "One model per (device patch, basis), pre-trained on the run's noise mixture (all R together), scored on the "
             "frozen development sets of the components marked eval. **These are simulated shots, not results.** "
             "ΔP_L = P_L(model) − P_L(correlated MWPM, the set's own calibrated prior), paired-bootstrap 95%. "
             "\"uniform\" is correlated MWPM with weights from the uniform model.", ""]
    for name, g in run.get("groups", {}).items():
        tr = g.get("train", {})
        if tr:
            lines += [f"**{name}**: {g['metadata']['architecture']}, {g['metadata']['parameters']:,} parameters; "
                      f"trained {tr['wall_seconds']:.0f} s on {tr['shots_seen']/1e6:,.1f}M shots "
                      f"({tr.get('reused_shots', 0)/1e6:,.2f}M of them reused leakage chunks), {tr['stop_reason']}; "
                      f"val P_L {tr['final_val_p_l']:.4f}.", ""]
        body = []
        for pt in sorted((q for q in run["points"] if q.get("group") == name),
                         key=lambda q: (q["spec"]["noise"], str(q["spec"]["calibration"]), q["spec"]["rounds"])):
            s_ = pt["spec"]
            c = pt["references"]["mwpm_corr"]["point"]
            u = pt["references"].get("mwpm_corr_uniform", {}).get("point")
            row = [s_["noise"], options(cal_slug(s_["calibration"])), f"{s_['p']:g}", s_["rounds"], f"{c['shots']:,}"]
            if "decoder" in pt:
                d, pc = pt["decoder"], pt["decoder"]["paired_vs_mwpm_corr"]
                row += [f"{e(d['point']['p_l'])} {ci(d['point']['ci95_wilson'])}"]
            else:
                pc = None
                row += ["–"]
            row += [f"{e(c['p_l'])}", e(u["p_l"]) if u else "–",
                    f"{signed(pc['delta_p_l'])} {ci(pc['ci95_paired_bootstrap'])}" if pc else "–",
                    verdict(pc) if pc else "–"]
            body.append(row)
        if body:
            lines += table(["noise", "options", "p", "R", "N", f"{run['backend']} P_L [95%]", "corr MWPM",
                            "uniform", "ΔP_L vs corr [95%]", "verdict"], body) + [""]
    return lines


def cal_slug(calibration) -> str:
    from .evaluate import cal_label
    return cal_label(calibration)


def pooled_rows(points: list[dict]) -> tuple[int, list[list[str]]]:
    """One group's points pooled over R: failures per decoder, paired ΔP_L against calibrated and uniform-weight
    correlated MWPM from the summed discordant counts."""
    from .evaluate import paired_counts
    n = sum(q["references"]["mwpm_corr"]["point"]["shots"] for q in points)

    def delta(entries: list[dict]) -> str:
        pc = paired_counts(sum(x["a_wrong_b_right"] for x in entries), sum(x["b_wrong_a_right"] for x in entries), n)
        return f"{signed(pc['delta_p_l'])} {ci(pc['ci95_paired_bootstrap'])} {verdict(pc)}"

    rows = []
    for d0 in points[0]["decoders"]:
        mine = [next(d for d in q["decoders"] if d["name"] == d0["name"]) for q in points]
        k = sum(d["point"]["failures"] for d in mine)
        rows.append([d0["name"], f"{k:,} ({e(k / n)})", delta([d["paired_vs_mwpm_corr"] for d in mine]),
                     delta([d["paired_vs_mwpm_corr_uniform"] for d in mine]) if "paired_vs_mwpm_corr_uniform" in d0 else "–"])
    for ref in ("mwpm_corr", "mwpm_corr_uniform", "mwpm"):
        if ref in points[0]["references"]:
            k = sum(q["references"][ref]["point"]["failures"] for q in points)
            vs = [q["paired_refs_vs_mwpm_corr"][ref] for q in points if ref in q.get("paired_refs_vs_mwpm_corr", {})]
            rows.append([ref, f"{k:,} ({e(k / n)})", delta(vs) if len(vs) == len(points) else "–", "–"])
    return n, rows


def hardware_section(run: dict) -> list[str]:
    """A fine-tune or pilot run on real shots (finetune.py): per group, every decoder against the MWPM baselines."""
    pilot = run.get("mode") == "pilot"
    label = run.get("label", "HARDWARE")
    title = ("pilot smoke test (loader and decoders; not a result)" if pilot else
             f"{run['eval_split']} split" + ("" if run["eval_split"] == "test" else ": a code check, not a test"))
    lines = [f"## {label}: {run['backend']}, {title}, run `{run['run_id']}`", "",
             f"Pre-trained models from `{run['pretrained_run']}`. Correlated MWPM's weights come from the decoder prior "
             f"(`{run['config']['prior']['noise']}` with {run['config']['prior'].get('options')} on the run's device patch); "
             "\"uniform\" is correlated MWPM with uniform-model weights. ΔP_L is paired against correlated MWPM on the same "
             "shots (bootstrap 95%).", ""]
    if not pilot:
        lines += ["The validation split (the fine-tuning batches' held-out 20%) served early stopping and the cascade's τ, "
                  "so numbers on it lean optimistic. The test batch, never trained or tuned on, is the honest number.", ""]
    batches = run.get("config", {}).get("finetune_batches", [1, 2])
    for name, g in run.get("groups", {}).items():
        runs = g.get("runs", [])
        cal = sorted({r["calibration_set_id"][:8] for r in runs})
        lines.append(f"**{name}**: {len(runs)} runs ({', '.join(sorted({r['label'] for r in runs}))}, calibration sets "
                     f"{', '.join(cal) or '–'}, batches {sorted({r['batch'] for r in runs})})"
                     + (f"; fine-tuned on {g['finetune_shots']:,} shots of batches {batches}, early-stopped on "
                        f"{g['validation_shots']:,}, {g['train']['wall_seconds']:.0f} s ({g['train']['stop_reason']})"
                        if g.get("train") else "") + ".")
        casc = g.get("cascade")
        if casc and not pilot:
            argmin, choice = casc.get("argmin_tau"), casc.get("prior_choice")
            lines.append(f"Cascade τ = {casc['tau']:.2f}, tuned on {casc['validation_shots']:,} validation shots ("
                         + (f"one-standard-error rule; the argmin τ was {argmin:.2f}" if argmin is not None else casc["rule"])
                         + ")" + (f", re-tuned on validation at {casc['retuned_utc'][:16]} UTC (the replaced entry is kept)"
                                  if casc.get("retuned_utc") else "")
                         + (f"; τ from `{casc['tuned_in']}`" if casc.get("tuned_in") else "") + "."
                         + (f" Relay-BP and its MWPM fallback use the **{choice['chosen']}** prior, chosen on validation "
                            f"by the same test: correlated MWPM fails {choice['calibrated_failures']:,} shots with "
                            f"calibrated weights and {choice['uniform_failures']:,} with uniform weights, of "
                            f"{choice['shots']:,} (1 SE {choice['se']:.1f})." if choice else ""))
        lines.append("")
        mine = [q for q in run["points"] if q.get("group") == name]
        if mine and not pilot:
            n, rows = pooled_rows(mine)
            lines += [f"Pooled over its {len(mine)} points ({n:,} shots, the same number at each R); ΔP_L is paired, "
                      "from the summed discordant shots (bootstrap 95%):", ""]
            lines += table(["Decoder", "Failures (P_L)", "ΔP_L vs corr MWPM [95%]", "ΔP_L vs uniform-weight MWPM [95%]"],
                           rows) + [""]
        body = []
        for pt in sorted((q for q in run["points"] if q.get("group") == name), key=lambda q: q["spec"]["rounds"]):
            c = pt["references"]["mwpm_corr"]["point"]
            u = pt["references"].get("mwpm_corr_uniform", {}).get("point")
            row = [pt["spec"]["rounds"], f"{c['shots']:,}"]
            for d in pt["decoders"]:
                pc = d["paired_vs_mwpm_corr"]
                row += [f"{e(d['point']['p_l'])} {ci(d['point']['ci95_wilson'])}",
                        f"{signed(pc['delta_p_l'])} {ci(pc['ci95_paired_bootstrap'])} {verdict(pc)}"]
            row += [f"{e(c['p_l'])} {ci(c['ci95_wilson'])}", e(u["p_l"]) if u else "–", e(pt["always_zero"]["p_l"])]
            body.append(row)
        if body:
            head = ["R", "N"]
            for d in next(q for q in run["points"] if q.get("group") == name)["decoders"]:
                head += [f"{d['name']} P_L [95%]", "ΔP_L vs corr [95%]"]
            lines += table(head + ["corr MWPM [95%]", "uniform", "always 0"], body) + [""]
    if not pilot:
        lines += speed_section(run)
    return lines


def build(runs: list[dict]) -> str:
    baseline, speeds, learned = {}, {}, {}
    real = [run for run in runs if run.get("mode") in ("finetune", "pilot")]
    runs = [run for run in runs if run.get("mode") not in ("finetune", "pilot")]
    for run in runs:
        for pt in run["points"]:
            key = spec_key(pt["spec"])
            baseline[key] = (run, pt)
            if "speed" in pt["references"].get("mwpm_corr", {}):
                speeds[key] = pt
            if "decoder" in pt and run.get("config", {}).get("mode") != "mixture":
                learned.setdefault((run["backend"], run["_config"]), {})[key] = (run, pt)
    lines = ["# ML decoders: results", "",
             "> **Real-data sections are headed HARDWARE (device, calibration set, batch); everything else is "
             "SIMULATED (development), not the evaluation.** Sal's direction is to train and evaluate the decoders on "
             "Garnet and Emerald data (InfoRecord); simulated sets serve pre-training and development only.", "",
             "Generated by `python -m mlDecoder.report` from the result JSONs under `mlDecoder/results/` "
             "(not committed). Do not edit by hand. The runs behind this file:", ""]
    for run in runs:
        lines.append(f"- `{run['backend']}/{run['run_id']}` ({run['_config']}, {run['status']}, "
                     f"{len(run['points'])} points, commit `{run['git']['commit'][:7]}`{', dirty tree' if run['git']['dirty'] else ''})")
    if runs:
        env = runs[-1]["environment"]
        gpu = env.get("gpu", {})
        lines += ["", f"Machine: {env['cpu']} ({env['logical_cpus']} logical CPUs)"
                  + (f", {gpu['name']} ({gpu['vram_mib']} MiB)" if gpu else "")
                  + f"; Python {env['python']}; " + ", ".join(f"{k} {v}" for k, v in sorted(env["packages"].items())) + "."]
    lines += ["", "**Definitions.** P_L is the fraction of test shots whose predicted observable flip is wrong. "
              "Intervals are Wilson 95% for one decoder, and paired-bootstrap 95% (10,000 resamples of the shots) "
              "for a difference on the same shots. ε_L is the per-round rate from `challenge.decoding.fit_ler`: "
              "**no SPAM**, P(r) = [1 − (1 − 2ε)^r]/2, is the challenge's own form; **SPAM** fits an amplitude A as well. "
              "Speeds are measured on this laptop, after a warm-up, at a given batch size; PyMatching runs on one CPU thread.", ""]
    # ------------------------------------------------ baseline
    groups = {}
    for key, (run, pt) in sorted(baseline.items()):
        groups.setdefault((key[0], key[1], key[7]), []).append((key, pt))
    lines += ["## The bar: MWPM on the frozen test sets", "",
              "Correlated MWPM (`challenge.decoding.make_matcher(noisy, correlated=True)`) and uncorrelated MWPM on the "
              "same frozen shots. N is sized from a correlated-MWPM pilot so that MWPM expects at least 100 failures "
              "(×1.25 margin, at least 100,000 shots, capped at 10⁷). Δ is P_L(MWPM) − P_L(correlated MWPM). "
              "Speed columns: throughput at batch 65,536 and latency per shot at batch 1, one thread. "
              "Leakage sets come from `challenge.leakage.sample_leakage`, decoded with the static Pauli DEM "
              "(a model mismatch the team's pipeline records).", ""]
    for (p, noise, cal), rows in groups.items():
        lines += [f"### {noise}, p = {p:g}" + (f", {cal_slug(None if cal == 'None' else cal)}" if cal != "None" else ""), ""]
        body = []
        for key, pt in rows:
            c, u = pt["references"]["mwpm_corr"]["point"], pt["references"]["mwpm"]["point"]
            pr = pt["paired_mwpm_vs_mwpm_corr"]
            sp = speeds.get(key, {}).get("references", {})
            cs, us = sp.get("mwpm_corr", {}).get("speed", {}), sp.get("mwpm", {}).get("speed", {})
            cap = " (cap)" if pt["dataset"]["sizing"]["cap_binds"] else ""
            body.append([key[2], key[3], key[4], f"{c['shots']:,}{cap}", c["failures"], f"{e(c['p_l'])} {ci(c['ci95_wilson'])}",
                         u["failures"], f"{e(u['p_l'])} {ci(u['ci95_wilson'])}",
                         f"{signed(pr['delta_p_l'])} {ci(pr['ci95_paired_bootstrap'])}",
                         speed(cs.get("cpu_threads_1"), 65536, "shots_per_second"), speed(cs.get("cpu_threads_1"), 1, "latency_per_shot_us"),
                         speed(us.get("cpu_threads_1"), 65536, "shots_per_second"), speed(us.get("cpu_threads_1"), 1, "latency_per_shot_us"),
                         f"`{pt['dataset']['dataset_id']}`"])
        lines += table(["d", "basis", "R", "N", "corr k", "corr P_L [95%]", "MWPM k", "MWPM P_L [95%]",
                        "Δ MWPM − corr [95%]", "corr shots/s", "corr µs (b=1)", "MWPM shots/s", "MWPM µs (b=1)", "dataset"], body)
        lines.append("")
    # ------------------------------------------------ the LER record, per run
    latest = {}
    for run in runs:
        if run["status"] == "complete":
            latest[(run["backend"], run["_config"])] = run
    hardware_lines = []
    for run in real:
        if run["status"] == "complete":
            hardware_lines += hardware_section(run) + ler_section(run)
    if hardware_lines:
        lines[lines.index("## The bar: MWPM on the frozen test sets"):0] = hardware_lines
    lines += ["## The LER record (steps 03 and 04)", "",
              "For the latest complete run of each config: P_L at every R, both fits of "
              "2·P_L = 1 − A(1 − 2ε)^R (`fit_ler`, A = 1 is the challenge's formula; A fitted is the SPAM form) "
              "with profile-likelihood 95% intervals and goodness of fit (Pearson χ²/dof, p-value). "
              "ε/ε_corr is the ratio to correlated MWPM on the same shots, with a paired-bootstrap 95% interval: "
              "2,000 resamples in the A = 1 form, and a reduced 200-resample bootstrap in the SPAM form "
              "(`fit_ler(fit_spam=True)` refit per resample). Λ = ε_L(d)/ε_L(d+2). Everything here is **simulated**, under the named noise model. "
              "Per-run files: `results/<backend>/<run-id>/ler/` (`ler_points.csv`, `ler_fits.json`, one PNG per panel).", ""]
    for key in sorted(latest):
        lines += ler_section(latest[key])
    # ------------------------------------------------ analyses
    analyses = sorted((results_root() / "analysis").glob("*/summary.md"))
    if analyses:
        lines += ["## Analyses", "", "Shot-level studies of where the decoders disagree "
                  "(`python -m mlDecoder.analysis why --run <mlp run-id> --noise <noise>`); simulated data.", ""]
        for path in analyses:
            lines += path.read_text(encoding="utf-8").replace("### ", "### ", 1).splitlines() + [""]
    # ------------------------------------------------ stage 1: mixture runs (latest complete per config)
    for key in sorted(latest):
        if latest[key].get("config", {}).get("mode") == "mixture":
            lines += mixture_section(latest[key])
    # ------------------------------------------------ learned decoders
    for (backend, config), points in sorted(learned.items()):
        wins = sum(verdict(pt["decoder"]["paired_vs_mwpm_corr"]) == "**win**" for _, pt in points.values())
        losses = sum(verdict(pt["decoder"]["paired_vs_mwpm_corr"]) == "loss" for _, pt in points.values())
        lines += [f"## {backend} ({config})", "",
                  f"{len(points)} points against correlated MWPM on the same shots: **{wins} wins, "
                  f"{len(points) - wins - losses} ties, {losses} losses** (paired 95% interval for ΔP_L = "
                  f"P_L({backend}) − P_L(corr MWPM) below, containing, or above zero). "
                  "\"Disc.\" is the discordant pair: shots only this decoder gets wrong / shots only correlated MWPM gets wrong. "
                  "Speed: CUDA throughput at batch 65,536, CUDA and 1-thread CPU latency per shot at batch 1, all end to end "
                  "(numpy in, numpy out).", ""]
        groups = {}
        for key, (run, pt) in sorted(points.items()):
            groups.setdefault((key[0], key[1]), []).append((key, pt))
        for (p, noise), rows in groups.items():
            lines += [f"### {noise}, p = {p:g}", ""]
            body = []
            for key, pt in rows:
                d = pt["decoder"]
                pc, pz = d["paired_vs_mwpm_corr"], d["paired_vs_always_zero"]
                c = pt["references"]["mwpm_corr"]["point"]
                tr, md = d["train"], d["metadata"]
                sp = d["speed"] if isinstance(d["speed"], dict) else {}  # Some runs skip timing.
                body.append([key[2], key[3], key[4], f"{d['point']['shots']:,}", d["point"]["failures"],
                             f"{e(d['point']['p_l'])} {ci(d['point']['ci95_wilson'])}", f"{e(c['p_l'])}",
                             f"{signed(pc['delta_p_l'])} {ci(pc['ci95_paired_bootstrap'])}",
                             f"{pc['a_wrong_b_right']}/{pc['b_wrong_a_right']}", verdict(pc),
                             "yes" if pz["ci95_paired_bootstrap"][1] < 0 else "no",
                             f"{tr['wall_seconds']:.0f}", f"{tr['shots_seen']/1e6:,.1f}M", f"{md['parameters']:,}",
                             speed(sp.get("cuda"), 65536, "shots_per_second"),
                             speed(sp.get("cuda"), 1, "latency_per_shot_us"),
                             speed(sp.get("cpu_threads_1"), 1, "latency_per_shot_us")])
            lines += table(["d", "basis", "R", "N", "k", "P_L [95%]", "corr P_L", "ΔP_L vs corr [95%]", "disc.",
                            "verdict", "beats 0", "train s", "shots seen", "params", "CUDA shots/s", "CUDA µs (b=1)",
                            "CPU-1 µs (b=1)"], body)
            lines.append("")
        failed = [c for run in runs if run["backend"] == backend and run["_config"] == config for c in run.get("checks_failed", [])]
        checks = [(pt["slug"], name, chk) for _, pt in points.values() for name, chk in pt.get("checks", {}).items()]
        if checks:
            lines += ["Checks: " + "; ".join(f"{name} {'PASS' if chk['passed'] else 'FAIL'} ({chk['value']:.4g})"
                                             for _, name, chk in checks) + ".", ""]
        elif failed:
            lines += [f"Failed checks: {', '.join(failed)}.", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    text = build(load_runs(results_root()))
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
