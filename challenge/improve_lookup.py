"""Build, evaluate and use hardware-calibrated logical lookup tables offline."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pymatching
import stim

from .benchmark_headline import challenge_fit, ler_difference
from .decoders.calibrated_lookup import fit, load
from .decoders.hybrid import model_posterior
from .submission import BUNDLE, REPOSITORY, _load_policy, _records, verify_bundle
from iqm_surface_code.decoders.lookup_table import LookupTableDecoder

ARCHIVE_COMMIT = "be265ab341fac550468b8125a1127469b518ece8"
PILOT = "archive/results/decoders/hybrid/20261024T234900Z/"
LATER = "archive/results/decoders/history/20261025T004600Z/"
ORDERS, PENALTIES, ALPHAS = (0, 1, 2), (.1, .01, .001), (0., 10.)


def _git(path):
    return subprocess.check_output(["git", "show", ARCHIVE_COMMIT+":"+path], cwd=REPOSITORY)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def pilot_records(basis, rounds, manifest):
    """Recover only earlier, compatible pilots; retain their recorded split."""
    old = json.loads(_git(PILOT+"manifest.json"))
    later = json.loads(_git(LATER+"manifest.json"))
    if old["calibration_set_id"] != manifest["calibration_set_id"] or later["calibration_set_id"] != manifest["calibration_set_id"]:
        raise ValueError("Training calibration differs from the frozen physical experiment.")
    target = next(c for c in manifest["cases"] if
                  (c["arm"], c["basis"], c["rounds"]) == ("optimized", basis, rounds))
    source = stim.Circuit.from_file(BUNDLE / target["files"]["source"])
    native = stim.Circuit.from_file(BUNDLE / target["files"]["native"])
    provenance, arrays = [], []
    for prefix, inventory, name in ((PILOT, old, f"pilot_ethan_{basis}_r{rounds}"),
                                     (PILOT, old, f"confirmation_b0_ethan_{basis}_r{rounds}"),
                                     (PILOT, old, f"confirmation_b1_ethan_{basis}_r{rounds}"),
                                     (LATER, later, f"pilot_baseline_{basis}_r{rounds}")):
        case = next(c for c in inventory["cases"] if c["id"] == name)
        if case["job_id"] in {c["job_id"] for c in manifest["cases"] if c["arm"] == "optimized"}:
            raise ValueError("Development records overlap the evaluation acquisition.")
        blob = _git(prefix+case["data_file"])
        if hashlib.sha256(blob).hexdigest() != case["dataset_sha256"]:
            raise ValueError("Archived pilot measurement hash differs.")
        old_source = stim.Circuit(_git(prefix+name+".source.stim").decode())
        old_native = stim.Circuit(_git(prefix+name+".native.stim").decode())
        if old_source != source or old_native != native:
            raise ValueError("Archived pilot circuit/frame differs from the physical experiment.")
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            d, o = data["det_events"].copy(), data["obs_flips"].copy()
            converted = source.compile_m2d_converter().convert(
                measurements=data["raw_meas"], separate_observables=True)
            if not all(np.array_equal(a, b) for a, b in zip(converted, (d, o))):
                raise ValueError("Pilot labels differ from the measured logical frame.")
        provenance.append(dict(path=prefix+case["data_file"], sha256=case["dataset_sha256"],
                               job_id=case["job_id"], shots=len(d),
                               source_sha256=hashlib.sha256(str(source).encode()).hexdigest(),
                               native_sha256=hashlib.sha256(str(native).encode()).hexdigest()))
        arrays.append((d, o))
    if [len(a[0]) for a in arrays] != [8000, 3000, 3000, 800]:
        raise ValueError("Pilot sizes differ from the recorded training/validation protocol.")
    td, to = (np.concatenate((arrays[0][i][:6000], arrays[1][i])) for i in (0, 1))
    vd, vo = (np.concatenate((arrays[0][i][6000:], arrays[2][i], arrays[3][i])) for i in (0, 1))
    return td, to, vd, vo, provenance


def build(run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    manifest = verify_bundle()
    protocol = dict(algorithm="regularized conditional logical-posterior calibration compiled to a complete LUT",
        archive_commit=ARCHIVE_COMMIT, training="first 6000 pilot_ethan shots plus earlier confirmation_b0_ethan (3000)",
        validation="last 2000 pilot_ethan shots plus earlier confirmation_b1_ethan (3000) and 800 pilot_baseline shots",
        evaluation="published confirm0/confirm1 records; opened for scoring after all policies are frozen",
        orders=ORDERS, penalties=PENALTIES, alphas=ALPHAS, matching_feature=True, seed=20261025,
        selection="fewest validation failures; ties prefer fewer parameters, no count update and stronger regularization",
        circuit_changes=False, physical_oracle_inputs=False,
        versions=dict(python=platform.python_version(), numpy=np.__version__, stim=stim.__version__,
                      pymatching=pymatching.__version__))
    _write(run/"protocol.json", protocol)
    frozen = {}
    for basis in ("X", "Z"):
        for rounds in (1, 2, 3):
            td, to, vd, vo, provenance = pilot_records(basis, rounds, manifest)
            path = BUNDLE / manifest["policies"][f"{basis}{rounds}"]["path"] / "model.dem"
            dem = stim.DetectorErrorModel.from_file(path)
            decoder = fit(dem, td, to, vd, vo, orders=ORDERS, penalties=PENALTIES, alphas=ALPHAS,
                          progress=lambda message: print(f"{basis} R{rounds}: {message}", flush=True))
            decoder.metadata.update(archive_commit=ARCHIVE_COMMIT, training_records=provenance)
            target = run/f"{basis}_r{rounds}.npz"
            decoder.save(target)
            frozen[f"{basis}{rounds}"] = dict(file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                metadata=decoder.metadata)
            choice = decoder.metadata["selection"]
            original = decoder.metadata["candidates"][0]["validation_errors"]
            print(f"{basis} R{rounds}: validation {original} -> {choice['validation_errors']} / {len(vd)}; "
                  f"order {choice['order']}, penalty {choice['penalty']}, alpha {choice['alpha']}", flush=True)
    _write(run/"frozen_selection.json", frozen)
    report = evaluate(run)
    _write(run/"report.json", report)
    return report


def evaluate(run):
    run = Path(run)
    frozen = json.loads((run/"frozen_selection.json").read_text())
    manifest = verify_bundle()
    scores, failures = [], {}
    for basis in ("X", "Z"):
        failures[basis] = {name: {} for name in ("calibrated", "lut", "mwpm", "published")}
        for rounds in (1, 2, 3):
            saved = frozen[f"{basis}{rounds}"]
            path = run / saved["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
                raise ValueError("Selected policy differs from its frozen hash.")
            decoder = load(path)
            dem_path = BUNDLE / manifest["policies"][f"{basis}{rounds}"]["path"] / "model.dem"
            dem = stim.DetectorErrorModel.from_file(dem_path)
            matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            plain = LookupTableDecoder((model_posterior(str(dem)) > .5).astype(np.uint8), dem.num_detectors, {})
            published = _load_policy(manifest, BUNDLE, basis, rounds, "optimized")
            cases = [c for c in manifest["cases"] if
                     (c["arm"], c["basis"], c["rounds"]) == ("optimized", basis, rounds)]
            for name in failures[basis]: failures[basis][name][rounds] = {}
            total = {name: 0 for name in failures[basis]}
            for block, case in enumerate(cases):
                dets, truth = _records(BUNDLE, case)
                predictions = dict(calibrated=decoder.decode_batch(dets), lut=plain.decode_batch(dets),
                                   mwpm=matcher.decode_batch(dets, enable_correlations=True),
                                   published=published.decode_batch(dets))
                for name, prediction in predictions.items():
                    bad = np.any(prediction != truth, axis=1)
                    failures[basis][name][rounds][block] = bad
                    total[name] += int(bad.sum())
            dets = np.concatenate([_records(BUNDLE, c)[0] for c in cases])
            decoder.decode_batch(dets)
            times = []
            for _ in range(31):
                start = time.perf_counter(); decoder.decode_batch(dets); times.append(time.perf_counter()-start)
            row = dict(basis=basis, rounds=rounds, shots=len(dets), failures=total,
                       median_batch_decode_seconds=float(np.median(times)),
                       mean_us_per_shot_per_round=float(np.mean(times)*1e6/len(dets)/rounds),
                       table_bytes=decoder.table.nbytes, build_seconds=decoder.metadata["build_seconds"])
            scores.append(row)
            print(f"{basis} R{rounds}: calibrated {total['calibrated']}; LUT {total['lut']}; "
                  f"MWPM {total['mwpm']}; published {total['published']}", flush=True)
    fits, comparisons = {}, {}
    for basis, families in failures.items():
        fits[basis], comparisons[basis] = {}, {}
        for name, by_round in families.items():
            errors = [sum(a.sum() for a in by_round[r].values()) for r in (1, 2, 3)]
            shots = [sum(len(a) for a in by_round[r].values()) for r in (1, 2, 3)]
            fits[basis][name] = challenge_fit((1, 2, 3), errors, shots)
        for reference in ("lut", "mwpm", "published"):
            comparison = ler_difference(families[reference], families["calibrated"], paired=True, seed=20261025)
            comparison["relative_ler_reduction"] = 1-fits[basis]["calibrated"]["epsilon_per_round"]/fits[basis][reference]["epsilon_per_round"]
            comparisons[basis][reference] = comparison
    return dict(protocol=json.loads((run/"protocol.json").read_text()),
                frozen_selection_sha256=hashlib.sha256((run/"frozen_selection.json").read_bytes()).hexdigest(),
                scores=scores, fits=fits, comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "evaluate"):
        item = sub.add_parser(command); item.add_argument("--run", type=Path, required=True)
    decode = sub.add_parser("decode")
    decode.add_argument("--policy", type=Path, required=True)
    decode.add_argument("--dets", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from MLfolders.mlDecoder.evidence import offline
    with offline():
        if args.command == "build": return build(args.run)
        if args.command == "evaluate": return evaluate(args.run)
        prediction = load(args.policy).decode_batch(np.load(args.dets, allow_pickle=False))
        with args.output.open("xb") as stream: np.save(stream, prediction)
        print(f"Decoded {len(prediction):,} shots into {args.output}")


if __name__ == "__main__":
    main()
