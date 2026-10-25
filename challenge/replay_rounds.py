"""Verify the bounded Garnet evidence without credentials or network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import stim

from .benchmark_rounds import (DIAGNOSTIC, OLD, PRIMARY, challenge_fit, compare, decoder, digest,
                               planned_cases, select_release, utc, verify, write_json)
from .decoders.hybrid import binary_rows


def verify_case(output, case):
    output = Path(output)
    source = stim.Circuit.from_file(output/f"{case['id']}.source.stim")
    native = stim.Circuit.from_file(output/f"{case['id']}.native.stim")
    for field, circuit in (("source_sha256", source), ("native_sha256", native)):
        if hashlib.sha256(str(circuit).encode()).hexdigest() != case[field]:
            raise ValueError("Circuit definition hash changed.")
    if digest(output/f"{case['id']}.decoder.stim") != case["model_sha256"]:
        raise ValueError("Physical decoder prior hash changed.")
    path = output/case["data_file"]
    if digest(path) != case["dataset_sha256"]:
        raise ValueError("Hardware dataset hash changed.")
    with np.load(path, allow_pickle=False) as arrays:
        raw = binary_rows(arrays["raw_meas"], source.num_measurements)
        nraw = binary_rows(arrays["native_raw_meas"], native.num_measurements)
        dets = binary_rows(arrays["det_events"], source.num_detectors)
        obs = binary_rows(arrays["obs_flips"], source.num_observables)
        if any(len(a) != case["shots"] for a in (raw, nraw, dets, obs)):
            raise ValueError("Returned shot count changed.")
        columns = case["native_measurement_columns"]
        if sorted(columns) != list(range(source.num_measurements)) or not np.array_equal(raw[:, columns], nraw):
            raise ValueError("Native measurement order differs.")
        for circuit, measurements in ((source, raw), (native, nraw)):
            d, o = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
            if not np.array_equal(d, dets) or not np.array_equal(o, obs):
                raise ValueError("Source/native detector or observable frame differs.")
        return dets.copy(), obs.copy()


def replay(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    verify(output, manifest)
    frozen = json.loads((output/"frozen_selection.json").read_text())
    report = json.loads((output/"report.json").read_text())
    expected = {c["id"]: c for phase in ("pilot", "confirm0", "confirm1")
                for c in planned_cases(manifest, phase, frozen)}
    if set(expected) != {c["id"] for c in manifest["cases"]}:
        raise ValueError("Hardware case inventory differs from the fixed protocol.")
    charged = sum(c["shots"] for c in manifest["cases"] if c.get("submit_intent_utc"))
    if charged != manifest["charged_shots"] or charged >= 100000:
        raise ValueError("Physical shot ledger mismatch or limit exceeded.")
    job_ids = [c["job_id"] for c in manifest["cases"]]
    archived_ids = {c["job_id"] for c in json.loads((OLD/"manifest.json").read_text())["cases"]}
    archived_ids |= {c["job_id"] for c in json.loads((output/"archived_long.json").read_text())}
    if len(set(job_ids)) != len(job_ids) or set(job_ids) & archived_ids:
        raise ValueError("Fresh job IDs are duplicated or overlap archived training.")
    for c in manifest["cases"]:
        if c["phase"].startswith("confirm") and c["submit_intent_utc"] <= manifest["selection_frozen_utc"]:
            raise ValueError("Confirmation preceded the final freeze.")
    zero = [c for c in manifest["cases"] if c["phase"] == "confirm0"]
    one = [c for c in manifest["cases"] if c["phase"] == "confirm1"]
    if max(c["collected_utc"] for c in zero) >= min(c["submit_intent_utc"] for c in one):
        raise ValueError("The acquisition windows overlapped.")
    failures, prediction_groups, shots = {}, 0, 0
    for c in manifest["cases"]:
        if c["shots"] != expected[c["id"]]["shots"] or c["returned_shots"] != c["shots"]:
            raise ValueError("Requested/returned shots differ from the protocol.")
        dets, obs = verify_case(output, c)
        shots += len(dets)
        if c["phase"] == "pilot":
            continue
        with np.load(output/f"{c['id']}.predictions.npz", allow_pickle=False) as saved:
            for family in saved.files:
                pred = decoder(output, manifest, c["profile"], c["basis"], c["rounds"], family).decode_batch(dets)
                if not np.array_equal(pred, saved[family]):
                    raise ValueError("Frozen inference predictions changed.")
                key = (c["basis"], c["profile"], family, c["rounds"], int(c["phase"][-1]))
                failures[key] = np.any(pred != obs, axis=1)
                prediction_groups += 1
    for row in report["results"]:
        rounds = PRIMARY if row["window"] == "headline" else DIAGNOSTIC
        values = [[failures[row["basis"], row["profile"], row["family"], r, b] for b in (0, 1)] for r in rounds]
        errors = [int(sum(x.sum() for x in blocks)) for blocks in values]
        counts = [sum(len(x) for x in blocks) for blocks in values]
        fitted = challenge_fit(rounds, errors, counts)
        if errors != row["fit"]["errors"] or counts != row["fit"]["shots"] or not np.isclose(
                fitted["epsilon_per_round"], row["fit"]["epsilon_per_round"], atol=1e-10, rtol=0):
            raise ValueError("Reported failures or challenge fit changed.")
    for basis, arms in frozen["release_arms"].items():
        for value in report["comparisons"][basis].values():
            a, b = arms[value["first"]], arms[value["second"]]
            first = {r: {block: failures[basis, a["profile"], a["family"], r, block] for block in (0, 1)} for r in PRIMARY}
            second = {r: {block: failures[basis, b["profile"], b["family"], r, block] for block in (0, 1)} for r in PRIMARY}
            fitted = compare(first, second, paired=a["profile"] == b["profile"], draws=value["draws"], seed=value["seed"],
                             alpha=.05/max(1, frozen["comparison_count"]))
            for field in ("absolute_ler_reduction", "ci", "confidence"):
                if not np.allclose(fitted[field], value[field], atol=1e-12, rtol=0):
                    raise ValueError("Adjusted confirmation comparison changed.")
        rates = {name: next(row["fit"]["epsilon_per_round"] for row in report["results"] if row["basis"] == basis and
                           row["profile"] == arm["profile"] and row["family"] == arm["family"] and row["window"] == "headline")
                 for name, arm in arms.items()}
        winner = select_release(arms, report["comparisons"][basis], rates)
        if any(report["adopted"][basis][field] != arms[winner][field] for field in ("profile", "family")) or report["adopted"][basis]["arm"] != winner:
            raise ValueError("Released policy violates the frozen component rule.")
    if shots != charged or report["charged_shots"] != charged or report["returned_shots"] != shots:
        raise ValueError("Report total differs from all-shot evidence.")
    return dict(verified_utc=utc(), cases=len(manifest["cases"]), physical_shots=shots,
                prediction_groups=prediction_groups, fits=len(report["results"]),
                adjusted_comparisons=frozen["comparison_count"], release_rule="verified",
                network="disabled by replay CLI", credentials="not accessed",
                source_native_frames="identical on every raw shot", archived_training_job_overlap=False,
                inference_seal_sha256=manifest["frozen_artifacts_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--write", action="store_true", help="Write a new verification.json; default is read-only")
    args = parser.parse_args(argv)
    if args.runtime:
        import sys
        sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    torch.set_num_threads(2)

    def deny(*args, **kwargs):
        raise RuntimeError("Network disabled during evidence replay.")

    previous_connect, previous_create = socket.socket.connect, socket.create_connection
    socket.socket.connect, socket.create_connection = deny, deny
    try:
        result = replay(args.run)
    finally:
        socket.socket.connect, socket.create_connection = previous_connect, previous_create
    if args.write:
        if (args.run/"verification.json").exists():
            raise ValueError("Verification already published; omit --write for read-only replay.")
        write_json(args.run/"verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
