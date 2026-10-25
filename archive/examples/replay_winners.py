"""Reproduce the retained Garnet policies from saved data, with no QPU access.

Run from the repository root: python -m examples.replay_winners [--json].
Only base project dependencies and the local iqm_surface_code package are needed.
The example neither retrains decoders nor writes benchmark artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket

import numpy as np

from challenge.benchmark_rounds import PRIMARY, arrays, challenge_fit, load_adopted

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = Path("results/decoders/history/20261025T004600Z")


@contextmanager
def network_disabled():
    def deny(*args, **kwargs):
        raise RuntimeError("Network access is disabled during winner replay.")

    original_connect, original_create = socket.socket.connect, socket.create_connection
    socket.socket.connect, socket.create_connection = deny, deny
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = original_connect, original_create


def replay_winners():
    if Path.cwd().resolve() != REPOSITORY:
        raise ValueError("Run winner replay from the repository root to resolve sealed artifact paths.")
    manifest = json.loads((STUDY/"manifest.json").read_text())
    report = json.loads((STUDY/"report.json").read_text())
    output = dict(study=STUDY.as_posix(), device="IQM Garnet", distance=3,
                  rounds=list(PRIMARY), hardware_access=False, policies={})
    for basis in ("X", "Z"):
        failures, shots, families = [], [], set()
        for rounds in PRIMARY:
            policy, decoder = load_adopted(STUDY, basis, rounds)
            cases = [c for c in manifest["cases"] if c["phase"].startswith("confirm") and
                     c["basis"] == basis and c["rounds"] == rounds and c["profile"] == policy["profile"]]
            if {c["phase"] for c in cases} != {"confirm0", "confirm1"} or len(cases) != 2:
                raise ValueError("Both frozen confirmation blocks are required.")
            failed = total = 0
            for case in cases:
                dets, truth = arrays(STUDY, case)
                prediction = decoder.decode_batch(dets)  # Truth is not an inference input.
                if prediction.shape != truth.shape:
                    raise ValueError("Prediction and observable dimensions differ.")
                failed += int(np.any(prediction != truth, axis=1).sum())
                total += len(dets)
            failures.append(failed)
            shots.append(total)
            families.add(decoder.metadata["family"])
        if len(families) != 1:
            raise ValueError("A qualified policy must not switch decoder family across the fit.")
        expected = next(row["fit"] for row in report["results"] if row["basis"] == basis and
                        row["profile"] == policy["profile"] and row["family"] == policy["family"] and row["window"] == "headline")
        fitted = challenge_fit(PRIMARY, failures, shots)
        if failures != expected["errors"] or shots != expected["shots"] or not np.isclose(
                fitted["epsilon_per_round"], expected["epsilon_per_round"], atol=1e-7, rtol=0):
            raise ValueError("Winner replay differs from the published confirmation.")
        output["policies"][basis] = dict(family=families.pop(), failures=failures, shots=shots, fit=fitted)
    output["replayed_shots"] = sum(sum(p["shots"]) for p in output["policies"].values())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args(argv)
    with network_disabled():
        result = replay_winners()
    if args.json:
        print(json.dumps(result, indent=2))
        return result
    print("Retained Garnet policies - offline replay, no credentials or QPU access")
    for basis, policy in result["policies"].items():
        fit = policy["fit"]
        low, high = fit["ci95_profile"]
        print(f"{basis}: {policy['family']}; failures R1/R2/R3={policy['failures']}; shots={policy['shots']}")
        print(f"   Challenge R1-3 LER: {100*fit['epsilon_per_round']:.3f}% (95% profile CI {100*low:.3f}-{100*high:.3f}%)")
    print(f"Verified {result['replayed_shots']:,} saved shots. Stationary fits rejected; LERs are descriptive.")
    return result


if __name__ == "__main__":
    main()
