"""Markdown tables for the README's B3 section, straight from the B3 runs' result.json and ler_fits.json.

It wrote the three tables in README.md's HARDWARE B3 section (epsilon_L, pooled P_L, cascade speed). From the repo
root, against this committed record (on Linux the interpreter is MLfolders/mlDecoder/.venv/bin/python):

    export MLDECODER_RESULTS=MLfolders/mlDecoder/results_record PYTHONPATH=MLfolders PYTHONIOENCODING=utf-8
    MLfolders/mlDecoder/.venv/Scripts/python.exe MLfolders/mlDecoder/results_record/readme_b3_tables.py RUN_GARNET RUN_EMERALD

with RUN_GARNET = 20261024-215152-finetune-20261024-191223 and RUN_EMERALD = 20261024-220320-finetune-20261024-195414.
"""
import json
import sys
from pathlib import Path

from mlDecoder import data
from mlDecoder.evaluate import paired_counts

NAMES = {"rnn_lstm_finetuned": "LSTM, fine-tuned", "rnn_lstm_cascade": "cascade", "relay_bp": "Relay-BP",
         "mwpm_corr": "correlated MWPM", "mwpm_corr_uniform": "uniform-weight MWPM", "rnn_lstm_pretrained": "LSTM, pre-trained"}
ORDER = list(NAMES)
LABEL = {"garnet_pA_Z": "Garnet pA, Z", "garnet_pA_X": "Garnet pA, X", "emerald_eA_Z": "Emerald eA, Z", "emerald_eA_X": "Emerald eA, X"}


def f3(x):
    return f"{x:.3f}"


eps_rows, pooled_rows, speed_rows = [], [], []
for run_id in sys.argv[1:]:
    run_dir = data.results_root() / "rnn_lstm" / run_id
    r = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    fits = json.loads((run_dir / "ler" / "ler_fits.json").read_text(encoding="utf-8"))["fits"]
    for g, e in r["groups"].items():
        casc = e["cascade"]
        pts = [p for p in r["points"] if p["group"] == g]
        n = sum(p["references"]["mwpm_corr"]["point"]["shots"] for p in pts)
        deferred = sum(p["cascade"]["deferred"] for p in pts) / n
        by = {f["decoder"]: f for f in fits if f["group"] == g}
        cells = []
        for name in ORDER:
            f = by[name]
            ns, ratio = f["no_spam"], f.get("ratio_to_mwpm_corr")
            cell = f"{f3(ns['epsilon'])} [{f3(ns['ci95_profile'][0])}, {f3(ns['ci95_profile'][1])}]"
            if ratio:
                lo, hi = ratio["no_spam_ci95_paired_bootstrap"]
                cell += f"<br>×{f3(ratio['no_spam'])} [{f3(lo)}, {f3(hi)}]"
            cells.append(cell)
        prior = (casc.get("prior_choice") or {}).get("chosen", "calibrated")
        eps_rows.append([LABEL[g], f"{casc['tau']:.2f} ({prior}; {deferred:.1%} deferred)"] + cells)

        def pooled(name):
            mine = [next(d for d in p["decoders"] if d["name"] == name) for p in pts]
            k = sum(d["point"]["failures"] for d in mine)
            out = [f"{k / n:.4f}"]
            for field in ("paired_vs_mwpm_corr", "paired_vs_mwpm_corr_uniform"):
                pc = paired_counts(sum(d[field]["a_wrong_b_right"] for d in mine), sum(d[field]["b_wrong_a_right"] for d in mine), n)
                lo, hi = pc["ci95_paired_bootstrap"]
                out.append(f"{pc['delta_p_l']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            return out
        ref = {k: sum(p["references"][k]["point"]["failures"] for p in pts) / n for k in ("mwpm_corr", "mwpm_corr_uniform")}
        pooled_rows.append([LABEL[g], f"{n:,}"] + pooled("rnn_lstm_finetuned") + pooled("rnn_lstm_cascade")[:2]
                           + [f"{ref['mwpm_corr']:.4f}", f"{ref['mwpm_corr_uniform']:.4f}"])
        for s in e.get("cascade_speed", []):
            row1 = lambda rows, b: next(x for x in rows if x.get("batch") == b and "skipped" not in x)
            big = max((x for x in s["cuda"] if "skipped" not in x), key=lambda x: x["batch"])
            cpu, m = row1(s["cpu_threads_1"], 1), row1(s["mwpm_corr_cpu_threads_1"], 1)
            speed_rows.append([LABEL[g], s["rounds"], f"{casc['tau']:.2f}", f"{cpu['latency_per_shot_us'] / 1000:.2f} ms "
                               f"(p99 {cpu['latency_per_call_p99_us'] / 1000:.2f})", f"{big['shots_per_second']:,.0f} at {big['batch']:,}",
                               f"{m['latency_per_shot_us']:.0f} µs"])


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


print("\n".join(table(["Device, basis", "τ (prior; deferred)"] + [NAMES[k] for k in ORDER], eps_rows)))
print()
print("\n".join(table(["Device, basis", "Shots", "LSTM P_L", "ΔP_L vs corr MWPM", "ΔP_L vs uniform MWPM", "Cascade P_L",
                       "Cascade ΔP_L vs corr MWPM", "corr MWPM P_L", "uniform MWPM P_L"], pooled_rows)))
print()
print("\n".join(table(["Device, basis", "R", "τ", "Cascade, 1 CPU thread, batch 1", "Cascade, CUDA, shots/s at batch",
                       "corr MWPM, 1 thread, batch 1"], speed_rows)))
