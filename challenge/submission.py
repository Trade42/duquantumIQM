"""The Garnet surface-code submission: inference and reproducible evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from .benchmark_headline import challenge_fit, fit_many
from .decoders.hybrid import load_decoder

REPOSITORY = Path(__file__).resolve().parents[1]
BUNDLE = REPOSITORY / "benchmarks/garnet"
ROUNDS = (1, 2, 3)


def verify_bundle(bundle=BUNDLE):
    """Verify every measurement, inference state and scientific dependency."""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError("Artifact path escapes the comparison bundle.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    for relative, expected in manifest["source_dependencies"].items():
        path = (REPOSITORY / relative).resolve()
        if not path.is_relative_to(REPOSITORY):
            raise ValueError("Source path escapes the repository.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Scientific source hash mismatch: {relative}")
    return manifest


def _load_policy(manifest, bundle, basis, rounds, arm):
    if basis not in {"X", "Z"} or rounds not in ROUNDS:
        raise ValueError("The submission supports X/Z memory at rounds 1, 2 and 3.")
    if arm == "optimized":
        return load_decoder(bundle / manifest["policies"][f"{basis}{rounds}"]["path"])
    if arm == "vanilla":
        from iqm_surface_code.decoders.lookup_table import LookupTableDecoder
        case = next(c for c in manifest["cases"] if
                    c["arm"] == arm and c["basis"] == basis and c["rounds"] == rounds)
        return LookupTableDecoder.load(bundle / case["files"]["lut"])
    raise ValueError("Arm must be optimized or vanilla.")


def load_policy(basis, rounds, *, arm="optimized", bundle=BUNDLE):
    """Return a decoder accepting only binary detection-event arrays.

    Input: (shots, 8 * rounds). Output: (shots, 1), reference-relative flips.
    The detector definitions and logical frame must match the bundled circuit.
    """
    bundle = Path(bundle).resolve()
    return _load_policy(verify_bundle(bundle), bundle, basis, rounds, arm)


def _records(bundle, case):
    files = {key: bundle / value for key, value in case["files"].items()}
    with np.load(files["shots"], allow_pickle=False) as data:
        dets, truth = data["det_events"], data["obs_flips"]
        if len(dets) != case["shots"] or len(truth) != case["shots"]:
            raise ValueError("Measurement shot count differs from the manifest.")
        source = stim.Circuit.from_file(files["source"])
        converted = source.compile_m2d_converter().convert(
            measurements=data["raw_meas"], separate_observables=True)
        if not all(np.array_equal(a, b) for a, b in zip(converted, (dets, truth))):
            raise ValueError("Source detector/observable frame differs.")
        if "native" in files:
            native = stim.Circuit.from_file(files["native"])
            columns = case["native_measurement_columns"]
            if sorted(columns) != list(range(source.num_measurements)) or not np.array_equal(
                    data["raw_meas"][:, columns], data["native_raw_meas"]):
                raise ValueError("Native measurement order differs.")
            converted = native.compile_m2d_converter().convert(
                measurements=data["native_raw_meas"], separate_observables=True)
            if not all(np.array_equal(a, b) for a, b in zip(converted, (dets, truth))):
                raise ValueError("Native detector/observable frame differs.")
        return dets.copy(), truth.copy()


def evaluate(*, bundle=BUNDLE, draws=20000, seed=20261025):
    """Recompute all failures, challenge fits and independent-arm comparisons."""
    bundle = Path(bundle).resolve()
    manifest = verify_bundle(bundle)
    groups = {}
    for arm in ("vanilla", "optimized"):
        groups[arm] = {}
        for basis in ("X", "Z"):
            failures, shots, strata = [], [], []
            for rounds in ROUNDS:
                decoder = _load_policy(manifest, bundle, basis, rounds, arm)
                cases = [c for c in manifest["cases"] if
                         (c["arm"], c["basis"], c["rounds"]) == (arm, basis, rounds)]
                if len(cases) != (2 if arm == "optimized" else 1):
                    raise ValueError("Comparison acquisition inventory is incomplete.")
                failed = total = 0
                round_strata = []
                for case in cases:
                    dets, truth = _records(bundle, case)
                    prediction = decoder.decode_batch(dets)
                    if prediction.shape != truth.shape:
                        raise ValueError("Prediction and observable dimensions differ.")
                    errors = int(np.any(prediction != truth, axis=1).sum())
                    if errors != case["errors"]:
                        raise ValueError(f"Score mismatch: {case['id']}")
                    failed += errors
                    total += len(dets)
                    round_strata.append((errors, len(dets)))
                failures.append(failed)
                shots.append(total)
                strata.append(round_strata)
            groups[arm][basis] = dict(fit=challenge_fit(ROUNDS, failures, shots), strata=strata)
    comparisons = {}
    rng = np.random.default_rng(seed)
    for basis in ("X", "Z"):
        samples = {}
        for arm in ("vanilla", "optimized"):
            simulated = np.zeros((draws, len(ROUNDS)), dtype=int)
            for column, strata in enumerate(groups[arm][basis]["strata"]):
                for errors, shots in strata:
                    simulated[:, column] += rng.binomial(shots, errors / shots, size=draws)
            samples[arm] = fit_many(simulated, np.array(groups[arm][basis]["fit"]["shots"]))
        vanilla = groups["vanilla"][basis]["fit"]["epsilon_per_round"]
        optimized = groups["optimized"][basis]["fit"]["epsilon_per_round"]
        comparisons[basis] = dict(
            absolute_ler_reduction=vanilla - optimized,
            relative_ler_reduction=1 - optimized / vanilla,
            ci975_absolute=np.quantile(samples["vanilla"] - samples["optimized"], [.0125, .9875]).tolist(),
            draws=draws, seed=seed, paired=False,
            method="independent binomial bootstrap stratified by rounds and acquisition block; two-basis Bonferroni intervals")
    total = sum(sum(groups[arm][basis]["fit"]["shots"]) for arm in groups for basis in groups[arm])
    if total != manifest["total_shots"]:
        raise ValueError("Comparison total differs from the manifest.")
    return dict(device=manifest["device"], distance=manifest["distance"], rounds=list(ROUNDS),
                calibration_set_id=manifest["calibration_set_id"], physical_shots=total,
                measurements={arm: {basis: row["fit"] for basis, row in values.items()}
                              for arm, values in groups.items()}, comparisons=comparisons)
