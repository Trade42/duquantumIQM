"""Reproduce the Garnet submission and vanilla surface-code comparison offline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import socket

import numpy as np

from challenge.submission import BUNDLE, evaluate


@contextmanager
def network_disabled():
    def deny(*args, **kwargs):
        raise RuntimeError("Network access is disabled during reproduction.")
    original = socket.socket.connect, socket.create_connection
    socket.socket.connect = socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    with network_disabled():
        result = evaluate()
    expected = json.loads((BUNDLE / "evaluation.json").read_text())
    for arm, rows in result["measurements"].items():
        for basis, fit in rows.items():
            reference = expected["measurements"][arm][basis]
            if fit["errors"] != reference["errors"] or fit["shots"] != reference["shots"] or not np.isclose(
                    fit["epsilon_per_round"], reference["epsilon_per_round"], atol=1e-7, rtol=0):
                raise ValueError("Reproduction differs from the published comparison.")
    for basis, comparison in result["comparisons"].items():
        if not np.allclose(comparison["ci975_absolute"], expected["comparisons"][basis]["ci975_absolute"], atol=1e-7, rtol=0):
            raise ValueError("Comparison interval differs from the published calculation.")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("IQM Garnet | distance-3 surface-code memory | rounds 1, 2, 3")
        for basis in ("X", "Z"):
            baseline = result["measurements"]["vanilla"][basis]
            final = result["measurements"]["optimized"][basis]
            reduction = result["comparisons"][basis]["relative_ler_reduction"]
            print(f"{basis}: vanilla {100*baseline['epsilon_per_round']:.3f}% -> optimized {100*final['epsilon_per_round']:.3f}% LER ({100*reduction:.1f}% reduction)")
        print(f"Verified {result['physical_shots']:,} recorded shots, circuit frames, decoder states and comparison intervals.")
    return result


if __name__ == "__main__":
    main()
