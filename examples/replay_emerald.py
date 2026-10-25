"""Reproduce the vanilla Emerald memory baseline from saved hardware records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from challenge.benchmark_emerald import replay
from examples.reproduce import network_disabled

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "benchmarks/emerald/vanilla/20261025T022556Z"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    with network_disabled():
        result = replay(args.bundle)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"IQM Emerald vanilla surface code: verified {result['cases']} cases, {result['physical_shots']:,} hardware shots")
        for fit in result["fits"]:
            lo, hi = fit["ci95_profile"]
            print(f"{fit['basis']}: {100*fit['epsilon_per_round']:.3f}% LER (95% profile CI {100*lo:.3f}-{100*hi:.3f}%), deviance {fit['deviance']:.2f}/{fit['nominal_df']}")
    return result


if __name__ == "__main__":
    main()
