"""Print the closure test's results as markdown for docs/garnet-calib.md (read-only, no simulation).

    uv run python scripts/garnet_calib_report.py

Reads InfoRecord/characterization/sim/{closure.json, compensate.json, derived/<world>/r<NN>/table.csv}.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dqiqm.characterization import table as T  # noqa: E402
from dqiqm.stats import wilson  # noqa: E402

SIM = REPO / "InfoRecord" / "characterization" / "sim"
KINDS = ("eps", "eta", "delta", "phi")


def kind(p: str) -> str:
    return "phi" if p.startswith("phi_") else p


def tables(world: str) -> list[list[dict]]:
    return [T.read(d / "table.csv") for d in sorted((SIM / "derived" / world).glob("r*"))]


def main() -> int:
    c = json.loads((SIM / "closure.json").read_text())
    print(f"All SIMULATED: stand-in `{c['stand_in']}`, calibration set `{c['calibration_set_id']}`, Aer MPS, seeded "
          f"({c['seeds']['aer']}; errors {c['seeds']['errors']}). Jobs: "
          + ", ".join(f"{j['label']} {j['circuits']} x {j['shots']}" for j in c["jobs"])
          + f". Worlds: noiseless 1, calibrated noise with nothing injected {c['worlds']['none']}, injected {c['worlds']['injected']} "
          f"(per qubit eps, eta uniform in +-{c['injected_ranges_rad']['eps']} rad; per coupler delta, phi_a, phi_b in "
          f"+-{c['injected_ranges_rad']['delta']} rad, a fresh draw per replica).\n")
    print("**Coverage of the injected values** (95% intervals; if they are right, the covered count falls in the "
          "binomial range about 95% of the time):\n")
    print("| parameter | intervals | hold the injected value | fraction | binomial 95% range of the count | median se (rad) | rms of (estimate - injected)/se |")
    print("|---|---|---|---|---|---|---|")
    inj = tables("injected")
    for k in KINDS:
        v = c["coverage_injected"][k]
        rows = [r for t in inj for r in t if kind(r["parameter"]) == k]
        se = np.array([float(r["se"]) for r in rows])
        z = np.array([(float(r["fit_value"]) - float(r["injected"])) / float(r["se"]) for r in rows])
        lo, hi = v["expected_range"]
        print(f"| {k} | {v['n']} | {v['covered']} | {v['fraction']:.3f} | {lo:.0f}–{hi:.0f} | {np.median(se):.2g} | "
              f"{math.sqrt(np.mean(z ** 2)):.2f} |")
    print("\n**Nothing injected** (every estimate's truth is 0):\n")
    print("| world | parameter | estimates | intervals holding 0 | largest \\|estimate\\| (rad) | largest \\|estimate\\|/se |")
    print("|---|---|---|---|---|---|")
    for w, o in c["nothing_injected"].items():
        for k in KINDS:
            v = o[k]
            print(f"| {w} | {k} | {v['n']} | {v['covered']} | {v['max_abs']:.2g} | {v['max_abs_z']:.2f} |")
    # a bias per locus would show as a mean z far from 0 over the replicas
    none = tables("none")
    z = defaultdict(list)
    for t in none:
        for r in t:
            z[(r["locus"], r["parameter"])].append(float(r["fit_value"]) / float(r["se"]))
    n = len(none)
    worst = sorted(((abs(np.mean(v)) * math.sqrt(len(v)), key) for key, v in z.items()), reverse=True)[:3]
    print(f"\nPer-locus bias check over the {n} nothing-injected noisy replicas: for each (locus, parameter), "
          f"sqrt(n) x mean(estimate/se) is N(0, 1) if unbiased. Over {len(z)} (locus, parameter) pairs the largest are "
          + ", ".join(f"{key[0]} {key[1]} {s:.2f}" for s, key in worst) + ".")
    nl = c.get("noiseless_ideal_vs_counts")
    if nl:
        print(f"\nNoiseless world: all {nl['deterministic_columns']} deterministic columns (ideal P1 = 0 or 1) are exact, "
              f"and the {nl['random_columns']} random ones scatter as binomial: z = (P_hat - p)/sqrt(p(1 - p)/shots) has "
              f"mean {nl['mean_z']:+.3f}, standard deviation {nl['std_z']:.3f} and max |z| {nl['max_abs_z']:.2f}.")
    p = c["pilot"]
    print("\n**The pilot, dry** (its 13 circuits on the stand-in, 1,000 shots):\n")
    print("| record | D (C1 no-merge) | R (C2 no-merge) | readout | verdict |")
    print("|---|---|---|---|---|")
    for label, key in (("stand-in", "stand_in"), ("mock merged", "mock_merged")):
        s = p[key]
        ro = "ok" if s["checks"]["readout_n00"] and s["checks"]["readout_n02"] else "fails"
        print(f"| {label} | {s['D']:.4f} | {s['R']:.4f} | {ro} | **{s['verdict']}** |")
    comp = SIM / "compensate.json"
    if comp.exists():
        k = json.loads(comp.read_text())
        print(f"\n**The compensated E1 twin** (noise-free Aer statevector, {k['shots']} shots, the injected world r01's local "
              f"phases after every CZ; the C2-measured phases differ from the injected by at most "
              f"{k['max_abs_phase_error_rad']:.4f} rad):\n")
        print("Cells: shots with any detection event, with its Wilson 95% interval as a fraction, and observable flips.\n")
        print("| circuit | ideal | with the phases | compensated (true phases) | compensated (measured phases) |")
        print("|---|---|---|---|---|")
        for name, r in k["circuits"].items():
            cells = []
            for x in ("ideal", "plain", "compensated_true", "compensated_measured"):
                k, n = r[x]["shots_with_events"], r[x]["shots"]
                lo, hi = wilson(k, n)
                cells.append(f"{k}/{n} shots with events [{float(lo):.4f}, {float(hi):.4f}], {r[x]['observable_flips']} flips")
            print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
