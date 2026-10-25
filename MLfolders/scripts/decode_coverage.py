"""Do the 95% intervals of dqiqm.stats.fit_eps cover the true ε_L 95% of the time?

Draws many synthetic experiments with a known truth, k ~ Binomial(n, P(ε, A, rounds)) at each
round count, fits each one, and counts how often each interval contains the true ε_L. The truth
defaults to the baseline's Z-basis free-A fit (``data/sim/decode-fit/baseline_fit.json``): ε =
0.002, A = 0.9987, rounds 1-10, 100,000 shots per round count.

Cases:
* ``A = 1 form, A = 1 truth``: the model is right; curvature interval;
* ``free A form, A < 1 truth``: the model is right; curvature and bootstrap intervals;
* ``A = 1 form, A < 1 truth``: the model is wrong by a little (the stand-in's case), to show
  what it does to the coverage.

Writes ``data/sim/decode-fit/coverage.json``.

Run:
    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
        uv run python scripts/decode_coverage.py
"""

from __future__ import annotations

import os

# The fits do tiny linear algebra: threaded BLAS only spins, on a shared machine. Set before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dqiqm import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "sim" / "decode-fit"
SEED = 2026


def coverage(hits: int, trials: int) -> dict:
    lo, hi = stats.wilson(hits, trials)
    return {"hits": hits, "trials": trials, "coverage": hits / trials, "wilson_lo": lo, "wilson_hi": hi}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--amplitude", type=float, default=0.9987)
    ap.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=1000, help="experiments per case, curvature interval")
    ap.add_argument("--boot-trials", type=int, default=200, help="experiments for the bootstrap interval")
    ap.add_argument("--bootstrap", type=int, default=300, help="replicates per bootstrap")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    r = np.array(args.rounds)
    n = np.full(len(r), args.shots)
    rng = np.random.default_rng(SEED)

    cases = [
        ("A = 1 form, A = 1 truth", False, 1.0),
        ("free A form, A < 1 truth", True, args.amplitude),
        ("A = 1 form, A < 1 truth", False, args.amplitude),
    ]
    results = {}
    for name, free, amp in cases:
        p_true = stats.p_model(args.eps, r, amp)
        hits, est = 0, []
        for _ in range(args.trials):
            k = rng.binomial(n, p_true)
            f = stats.fit_eps(r, k, n, free_amplitude=free, bootstrap=0)
            hits += f.eps_ci_curvature[0] <= args.eps <= f.eps_ci_curvature[1]
            est.append(f.eps)
        est = np.array(est)
        results[name] = {
            "free_amplitude": free,
            "true_amplitude": amp,
            "curvature": coverage(int(hits), args.trials),
            "mean_eps": float(est.mean()),
            "bias": float(est.mean() - args.eps),
            "sd_eps": float(est.std(ddof=1)),
        }

    name = "free A form, A < 1 truth"
    p_true = stats.p_model(args.eps, r, args.amplitude)
    hits = 0
    for t in range(args.boot_trials):
        k = rng.binomial(n, p_true)
        f = stats.fit_eps(r, k, n, free_amplitude=True, bootstrap=args.bootstrap, seed=SEED + t)
        hits += f.eps_ci_bootstrap[0] <= args.eps <= f.eps_ci_bootstrap[1]
    results[name]["bootstrap"] = coverage(int(hits), args.boot_trials) | {"replicates": args.bootstrap}

    summary = {
        "what": "coverage of fit_eps's 95% intervals on synthetic binomial data with a known truth",
        "truth": {"eps": args.eps, "amplitude": args.amplitude, "rounds": args.rounds, "shots_per_point": args.shots},
        "seed": SEED,
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"truth eps = {args.eps}, A = {args.amplitude}, rounds {r.min()}-{r.max()}, {args.shots:,} shots per point")
    print("| case | interval | covered | coverage [Wilson 95%] | mean eps - truth | sd eps |")
    print("|---|---|---|---|---|---|")
    for name, res in results.items():
        for kind in ("curvature", "bootstrap"):
            if kind in res:
                c = res[kind]
                print(
                    f"| {name} | {kind} | {c['hits']}/{c['trials']} | {c['coverage']:.3f} "
                    f"[{c['wilson_lo']:.3f}, {c['wilson_hi']:.3f}] | {res['bias']:+.2e} | {res['sd_eps']:.2e} |"
                )
    print(f"wrote {args.out}/coverage.json; elapsed {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
