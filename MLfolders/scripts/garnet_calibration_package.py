"""Write the Garnet calibration pre-training package from a committed snapshot, offline.

    uv run python scripts/garnet_calibration_package.py [--snapshot PATH] [--out DIR]

Reads the snapshot (default: the InfoRecord's copy of the 2026-10-24 19:13 UTC snapshot,
calibration set a18a947f), chooses the placements from it as the suite does (seeded Stim
ranking), and writes qubits.csv, couplers.csv, durations.json, placement.json and README.md to
``InfoRecord/calibration/garnet/`` (or ``--out``). No network, no token.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dqiqm import inforecord as IR
from dqiqm.experiments import device as dev
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", type=Path, default=dev.DEFAULT_SNAPSHOT)
    ap.add_argument("--out", type=Path, default=REPO / "InfoRecord" / "calibration" / "garnet")
    args = ap.parse_args()
    cal = dev.live_calibration(args.snapshot)
    suite = SU.build(cal, experiments=("E1", "E2", "E5"), e1_rounds=(1,), e2_rounds=(1,), e5_rounds=(2,))
    for f in IR.write_calibration_package(args.out, cal, suite):
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
