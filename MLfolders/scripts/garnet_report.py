"""The dry run's results as markdown, written into docs/garnet-design.md (between results:begin and results:end).

    uv run python scripts/garnet_report.py [--no-write]

Reads only ``InfoRecord/sim/`` (``checks/*.json``, ``ler/*/*/ler_fits.json``, ``derived/*/*/summary.json``):
no simulation, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "InfoRecord" / "sim"
DOC = REPO / "docs" / "garnet-design.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def load(name: str):
    return json.loads((SIM / "checks" / name).read_text())


def ci(lo, hi, f="{:.4f}") -> str:
    return f"[{f.format(lo)}, {f.format(hi)}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    summary = load("summary.json")
    st = load("structure.json")
    nl = load("noiseless.json")
    mv = load("mps_vs_statevector.json")
    dc = load("decode.json")
    ft = load("features.json")
    sz = load("sizes.json")
    out = [BEGIN, ""]
    out.append(f"All numbers are **SIMULATED**: {summary['stand_in']}, calibration set `{summary['calibration_set_id']}`, "
               f"Aer MPS, seeded. Shots per circuit: {json.dumps(summary['dry_shots'])}. {summary['jobs']} jobs, "
               f"{summary['runs']} dataset runs, {summary['ler_records']} LER records; {summary['wall_s'] / 60:.1f} min wall "
               "on 4 cores under the sim lock.")
    out.append("")
    # structure
    fam = defaultdict(list)
    for r in st:
        fam[r["experiment"]].append(r)
    out += ["**Structure** (`checks/structure.json`; distance = `len(shortest_graphlike_error())` of the noisy copy):", "",
            "| experiment | circuits | detectors | measurements | observables | SWAPs | DEM errors | graphlike distance |",
            "|---|---|---|---|---|---|---|---|"]
    for e, rs in sorted(fam.items()):
        dist = sorted({r.get("graphlike_distance") for r in rs if r.get("graphlike_distance") is not None})
        dem = [r["dem_errors"] for r in rs if "dem_errors" in r]
        out.append(f"| {e} | {len(rs)} | {min(r['detectors'] for r in rs)}–{max(r['detectors'] for r in rs)} | "
                   f"{min(r['measurements'] for r in rs)}–{max(r['measurements'] for r in rs)} | "
                   f"{sorted({r['observables'] for r in rs})} | {max(r['swaps'] for r in rs)} | "
                   f"{(str(min(dem)) + '–' + str(max(dem))) if dem else 'n/a'} | {dist if dist else 'n/a'} |")
    e5 = sorted((r["rounds"], r.get("graphlike_distance")) for r in fam.get("E5", []) if "Z" in r["circuit"])
    out += ["", f"E5's graphlike distance under the calibrated noise, (R, distance), Z dual (the X dual is the same): {e5}, that is "
            "⌈R/2⌉. With no record flips in the model (uniform circuit noise) it is R (tests/test_garnet_experiments.py::test_distances). "
            "Without reset a misread bit m_k fires D_k and D_(k+2): one record flip spans two rounds of the time direction, so the "
            "stability experiment's time-like distance halves; E1's distance stays 3. Every detector count is the designed one "
            "(tests/test_garnet_experiments.py::test_counts).", ""]
    # noiseless
    out.append(f"**Noiseless, through the hardware path** (`checks/noiseless.json`): all {len(nl)} hardware circuits, "
               f"{nl[0]['shots']} shots each, Aer without noise → `hw.save_raw` → `decode.load_memory` → `detection_events`: "
               f"**{sum(r['detection_events'] for r in nl)} detection events, {sum(r['observable_flips'] for r in nl)} observable flips**. "
               f"The records are not trivially quiet: up to {max(r['random_columns'] for r in nl)} random columns per circuit.")
    out.append("")
    out.append(f"**MPS against statevector** (`checks/mps_vs_statevector.json`, {mv['circuit']}, {mv['shots']} noisy shots, seed {mv['seed']}): "
               f"identical records: **{mv['identical_records']}**; detection fraction {mv['matrix_product_state']['detection_fraction']:.4f} "
               f"(MPS, {mv['matrix_product_state']['seconds']} s) and {mv['statevector']['detection_fraction']:.4f} "
               f"(statevector, {mv['statevector']['seconds']} s).")
    out.append("")
    # decoding
    p = dc["pooled"]
    out += [f"**E1 decoded against 'always 0'** (`checks/decode.json`): PyMatching on the noisy copy of the circuit that ran "
            f"(`calibrated_noise(..., idle_ns=0)`, the stand-in's own noise). Pooled over every E1 run: "
            f"P_L = {p['p_pm']:.4f} {ci(*p['p_pm_ci'])} decoded, against {p['p_raw']:.4f} {ci(*p['p_raw_ci'])} undecoded "
            f"({p['shots']:,} shots): **better: {p['better_than_always_0']}**.", "",
            "| basis | R | shots | P_L PyMatching [Wilson 95%] | P_L always 0 [Wilson 95%] |", "|---|---|---|---|---|"]
    for r in dc["rows"]:
        out.append(f"| {r['basis']} | {r['rounds']} | {r['shots']} | {r['p_pm']:.4f} {ci(*r['p_pm_ci'])} | {r['p_raw']:.4f} {ci(*r['p_raw_ci'])} |")
    out.append("")
    # LER
    rows = []
    for f in sorted((SIM / "ler").glob("*/*/ler_fits.json")):
        d = json.loads(f.read_text())
        if "free_A" not in d:
            continue
        a, b = d["fixed_A"], d["free_A"]
        rows.append(f"| {d['experiment']} | {d['run_id']} | {a['eps']:.4f} {ci(*a['eps_ci_bootstrap'])} | {a['chi2_per_dof']:.2f} | "
                    f"{b['eps']:.4f} {ci(*b['eps_ci_bootstrap'])} | {b['amplitude']:.3f} | {b['chi2_per_dof']:.2f} | {sum(a['n']):,} |")
    out += ["**LER records** (`ler/<experiment>/<run id>/`): the fit 2P = 1 − A(1 − 2ε)^R by maximum likelihood "
            "(`dqiqm.stats.fit_eps`), 95% bootstrap intervals (1,000 resamples); each record also has `ler_points.csv` "
            "(P_L per R with Wilson intervals) and `ler.png`. At these shot counts the intervals are wide: the point is the "
            "pipeline, not the numbers.", "",
            "| exp | run | ε_L (A = 1) [95%] | χ²/dof | ε_L (free A) [95%] | A | χ²/dof | shots |", "|---|---|---|---|---|---|---|---|"]
    out += rows
    out.append("")
    # features
    out += ["**Derived features against known injected errors** (`checks/features.json`): Stim runs of E1 Z R=10 and E3's "
            "repeated readout with errors of known rate, written and analysed by the record's own writer:", "",
            "| run | same-stabilizer p_ij, lag 1 | lag 2 | lag 3 | record flips | state flips |", "|---|---|---|---|---|---|"]
    for k, v in ft.items():
        if "same_stab_lag1" in v:
            out.append(f"| {k} | {v['same_stab_lag1']:.4f} | {v['same_stab_lag2']:.4f} | {v['same_stab_lag3']:.4f} | | |")
        else:
            out.append(f"| {k} | | | | {v['record_flip_mean']:.4f} | {v['state_flip_mean']:.4f} |")
    both = next(v for k, v in ft.items() if "record 0.02, state 0.01" in k)
    out += ["", "Record flips show at k→k+2 (lag 2) only, state flips at k→k+1 only, and the E3 estimator recovers both "
            f"rates; with both injected (0.02 and 0.01) it gives {both['record_flip_mean']:.4f} and {both['state_flip_mean']:.4f} "
            "(the estimator is first order in the rates)."]
    out.append("")
    dm = summary["derived_means"]
    out += ["On the stand-in itself (means over runs, `derived/*/*/summary.json`):", "",
            "| experiment | detection rate | same-stab lag 1 | same-stab lag 2 | cross-stab lag 0 |", "|---|---|---|---|---|"]
    for e, v in dm.items():
        out.append(f"| {e} | {v['det_rate_mean']:.4f} | {v['same_stab_lag1']:.4f} | {v['same_stab_lag2']:.4f} | {v['cross_stab_lag0']:.4f} |")
    rate = lambda name: json.loads((SIM / "derived" / name.split("_")[1] / name / "summary.json").read_text())["det_rate_mean"]  # noqa: E731
    out += ["", "E4 on the stand-in, against E1 at the same R in the same batch (detection rate): the stand-in puts no noise on "
            "`delay`, so the idles change nothing beyond shot noise. On hardware this is where T1 and T2 show.", "",
            "| basis | R | E1 | E4 1 µs | E4 2 µs | E4 5 µs |", "|---|---|---|---|---|---|"]
    for basis in "ZX":
        for r in (2, 3, 5, 8):
            try:
                e1 = rate(f"b2_E1_d3_{basis}_r{r:02d}_pA")
                e4 = [rate(f"b2_E4_d3_{basis}_r{r:02d}_pA_t{t:05d}ns") for t in (1000, 2000, 5000)]
            except FileNotFoundError:
                continue  # only the representative runs are in git
            out.append(f"| {basis} | {r} | {e1:.4f} | " + " | ".join(f"{x:.4f}" for x in e4) + " |")
    e3 = [json.loads(f.read_text()) for f in sorted((SIM / "derived" / "E3").glob("*repeat*/summary.json"))]
    if e3:
        import csv

        with open(REPO / "InfoRecord" / "calibration" / "garnet" / "qubits.csv") as f:
            ro = np.mean([float(r["readout_error"]) for r in csv.DictReader(f)])
        out += ["", f"E3's repeated readouts on the stand-in: record flips {np.mean([s['record_flip_mean'] for s in e3]):.4f} per readout "
                f"(the calibration's mean assignment error over the 20 qubits is {ro:.4f}), state flips {np.mean([s['state_flip_mean'] for s in e3]):+.4f}: the stand-in's "
                "readout error is a pure record flip, as Aer models it, and E1's same-stabilizer lag-2 p_ij is its no-reset "
                "signature."]
    out.append("")
    # sizes
    out += ["**Sizes** (`checks/sizes.json`), per million shots:", "", "| circuit | measurements | detectors | uint8 .npy | bit-packed .npy | shots.csv | shots.csv.gz |",
            "|---|---|---|---|---|---|---|"]
    for name, v in sz.items():
        m = v["per_million_shots_MB"]
        out.append(f"| {name} | {v['measurements']} | {v['detectors']} | {m['uint8 npy (raw + dets + obs)']:.0f} MB | "
                   f"{m['bit-packed npy (raw + dets + obs)']:.0f} MB | {m['shots.csv']:.0f} MB | {m['shots.csv.gz']:.0f} MB |")
    out += ["", "Parquet: not measured; `pyarrow` is not in `uv.lock`, and the brief forbids adding it. Arrays above "
            "2^24 entries (for E1 R=20, about 50,000 shots) are bit-packed automatically.", "", END]
    md = "\n".join(out)
    print(md)
    if args.no_write:
        return 0
    text = DOC.read_text()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(head + md + tail)
    print(f"wrote {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
