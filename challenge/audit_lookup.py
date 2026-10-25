"""Check LUT correctness and compare LUT/MWPM on fixed circuits and records.

python -m challenge.audit_lookup --output results/decoders/lookup_audit/NEW/report.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pymatching
from scipy.stats import binomtest
import stim

from iqm_surface_code.decoders.lookup_table import LookupTableDecoder, build_lookup_table

ROOT = Path(__file__).resolve().parents[1]


def _xor_fault_inplace(probability, mask, p, *, block_size=1 << 18):
    """Update disjoint XOR pairs in bounded scratch space."""
    if not mask or not p:
        return
    pivot = mask.bit_length() - 1
    lower = (1 << pivot) - 1
    for start in range(0, len(probability)//2, block_size):
        keys = np.arange(start, min(start+block_size, len(probability)//2), dtype=np.uint32)
        left = (keys & lower) | ((keys >> pivot) << (pivot+1))
        right = left ^ mask
        a, b = probability[left], probability[right]
        probability[left] = (1-p)*a + p*b
        probability[right] = p*a + (1-p)*b


def direct_joint(dem, *, max_detectors=16, progress=None):
    """Independent probability-space dynamic program, without a Fourier transform."""
    n = dem.num_detectors
    if n > max_detectors or dem.num_observables != 1:
        raise ValueError("Direct enumeration requires one observable and a small DEM.")
    size = 1 << (n + 1)
    keys = np.arange(size, dtype=np.uint32) if n <= 16 else None
    probability = np.zeros(size, np.float64)
    probability[0] = 1
    terms = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                mask ^= 1 << (n + target.val)
            elif not target.is_separator():
                raise ValueError("Unsupported DEM target.")
        p = instruction.args_copy()[0]
        if keys is None:
            _xor_fault_inplace(probability, mask, p)
        else:
            probability = (1 - p) * probability + p * probability[keys ^ mask]
        terms += 1
        if progress and terms % 32 == 0:
            progress(f"Direct probability convolution: {terms} fault terms")
    return probability.reshape(2, -1)


def paired_scores(lut, mwpm, truth, *, seed=20261025, draws=10000):
    left, right, truth = (np.asarray(a, bool).reshape(-1) for a in (lut, mwpm, truth))
    a, b = left != truth, right != truth
    counts = np.bincount(a.astype(int) + 2*b.astype(int), minlength=4)
    samples = np.random.default_rng(seed).multinomial(len(a), counts / len(a), size=draws)
    difference = (samples[:, 1] - samples[:, 2]) / len(a)
    discordant = int(counts[1] + counts[2])
    return dict(shots=len(a), lut_failures=int(a.sum()), mwpm_failures=int(b.sum()),
                lut_only_wrong=int(counts[1]), mwpm_only_wrong=int(counts[2]),
                lut_minus_mwpm_probability=float(a.mean() - b.mean()),
                paired_ci95=np.quantile(difference, [.025, .975]).tolist(),
                mcnemar_exact_p=float(binomtest(int(counts[1]), discordant).pvalue) if discordant else 1.,
                bootstrap_draws=draws, bootstrap_seed=seed)


def _decode_timed(decoder, dets):
    decoder(dets)
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        prediction = decoder(dets)
        elapsed.append(time.perf_counter() - start)
    return prediction, dict(batch=len(dets), repeats=5,
                            median_batch_seconds=float(np.median(elapsed)))


def _records(source, shots):
    with np.load(shots, allow_pickle=False) as data:
        dets, truth = data["det_events"].copy(), data["obs_flips"].copy()
        converted = source.compile_m2d_converter().convert(
            measurements=data["raw_meas"], separate_observables=True)
        if not all(np.array_equal(a, b) for a, b in zip(converted, (dets, truth))):
            raise ValueError("Stored detection/observable frame differs from raw records.")
    return dets, truth


def _baseline_cases():
    garnet = ROOT / "benchmarks/garnet/vanilla"
    emerald = ROOT / "benchmarks/emerald/vanilla/20261025T022556Z"
    for device, folder in (("garnet", garnet), ("emerald", emerald)):
        for path in sorted(folder.glob("*.lut.npz")):
            prefix = str(path)[:-len(".lut.npz")]
            yield device, path.stem, Path(prefix), path


def audit(*, simulated_shots=100000, seed=20261025, progress=print):
    from .submission import BUNDLE, _records as garnet_records, verify_bundle
    manifest = verify_bundle()
    report = dict(seed=seed, simulated_shots=simulated_shots,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                stim=stim.__version__, pymatching=pymatching.__version__),
                  host=platform.platform(), timing="CPU, five warmed whole-batch calls; setup separate",
                  model="Independent joint DEM faults; disjoint-channel approximation retained",
                  models=[], vanilla=[], optimized=[])
    models, aggregate = {}, {device: [] for device in ("garnet", "emerald")}

    def prepare(circuit, *, matching_dem=None):
        dem = circuit if isinstance(circuit, stim.DetectorErrorModel) else circuit.detector_error_model(
            approximate_disjoint_errors=True)
        digest = hashlib.sha256(str(dem).encode()).hexdigest()
        if digest in models:
            return models[digest]
        start = time.perf_counter()
        lut = build_lookup_table(dem)
        lut_setup = time.perf_counter() - start
        graph = matching_dem if matching_dem is not None else (
            dem if isinstance(circuit, stim.DetectorErrorModel) else circuit.detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True))
        start = time.perf_counter()
        matcher = pymatching.Matching.from_detector_error_model(graph, enable_correlations=True)
        mwpm_setup = time.perf_counter() - start
        decode = lambda dets: matcher.decode_batch(dets, enable_correlations=True)
        row = dict(dem_sha256=digest, matching_dem_sha256=hashlib.sha256(str(graph).encode()).hexdigest(),
                   detectors=dem.num_detectors, lut_setup_seconds=lut_setup,
                   mwpm_setup_seconds=mwpm_setup, lut_metadata=lut.metadata)
        if dem.num_detectors <= 16:
            joint = direct_joint(dem)
            histories = ((np.arange(1 << dem.num_detectors, dtype=np.uint32)[:, None]
                          >> np.arange(dem.num_detectors, dtype=np.uint32)) & 1).astype(bool)
            l = lut.decode_batch(histories).ravel().astype(int)
            m = decode(histories).ravel().astype(int)
            optimum = float(np.minimum(joint[0], joint[1]).sum())
            lut_risk = float(joint[1-l, np.arange(len(l))].sum())
            mwpm_risk = float(joint[1-m, np.arange(len(m))].sum())
            row["independent_check"] = dict(probability_mass=float(joint.sum()),
                optimal_failure=optimum, lut_expected_failure=lut_risk,
                mwpm_expected_failure=mwpm_risk, lut_excess_risk=lut_risk-optimum,
                non_tie_decision_errors=int(np.count_nonzero(
                    (l != (joint[1] > joint[0])) & (np.abs(joint[1]-joint[0]) > 1e-12))))
            if abs(lut_risk-optimum) > 1e-10:
                raise AssertionError("LUT disagrees with independently computed logical ML.")
            del joint, histories
        dets, truth, _ = dem.compile_sampler(seed=seed).sample(simulated_shots)
        row["matched_model_simulation"] = paired_scores(lut.decode_batch(dets), decode(dets), truth, seed=seed)
        report["models"].append(row)
        models[digest] = lut, decode, row
        gc.collect()
        return models[digest]

    for device, name, prefix, path in _baseline_cases():
        source = stim.Circuit.from_file(str(prefix)+".stim")
        noisy = stim.Circuit.from_file(str(prefix)+".decoder.stim")
        dets, truth = _records(source, str(prefix)+".shots.npz")
        saved = LookupTableDecoder.load(path)
        rebuilt, matching, model = prepare(noisy)
        lp, lt = _decode_timed(saved.decode_batch, dets)
        mp, mt = _decode_timed(matching, dets)
        differences = int(np.count_nonzero(saved.table != rebuilt.table))
        row = dict(device=device, case=name, rounds=source.num_detectors//8,
                   basis="X" if name.startswith("X") else "Z", dem_sha256=model["dem_sha256"],
                   saved_table_differences=differences,
                   saved_vs_rebuilt_shot_differences=int(np.count_nonzero(lp != rebuilt.decode_batch(dets))),
                   circuit_sha256=hashlib.sha256(str(noisy).encode()).hexdigest(),
                   shots_sha256=hashlib.sha256(Path(str(prefix)+".shots.npz").read_bytes()).hexdigest(),
                   scores=paired_scores(lp, mp, truth, seed=seed), timing=dict(lut=lt, mwpm=mt))
        report["vanilla"].append(row)
        aggregate[device].append((lp, mp, truth))
        if progress:
            progress(f"{device} {name}: LUT {row['scores']['lut_failures']}; "
                     f"same-prior MWPM {row['scores']['mwpm_failures']}; rebuilt differences {differences}")

    for basis in ("X", "Z"):
        for rounds in (1, 2, 3):
            policy = BUNDLE / manifest["policies"][f"{basis}{rounds}"]["path"]
            dem = stim.DetectorErrorModel.from_file(policy / "model.dem")
            lut, matching, model = prepare(dem)
            cases = [c for c in manifest["cases"] if
                     (c["arm"], c["basis"], c["rounds"]) == ("optimized", basis, rounds)]
            arrays = [garnet_records(BUNDLE, case) for case in cases]
            dets, truth = (np.concatenate([a[i] for a in arrays]) for i in (0, 1))
            lp, lt = _decode_timed(lut.decode_batch, dets)
            mp, mt = _decode_timed(matching, dets)
            row = dict(basis=basis, rounds=rounds, dem_sha256=model["dem_sha256"],
                       cases=[c["id"] for c in cases], scores=paired_scores(lp, mp, truth, seed=seed),
                       timing=dict(lut=lt, mwpm=mt))
            report["optimized"].append(row)
            if progress:
                progress(f"optimized {basis} R{rounds}: LUT {row['scores']['lut_failures']}; "
                         f"same-prior MWPM {row['scores']['mwpm_failures']}")
    report["vanilla_totals"] = {device: paired_scores(
        *(np.concatenate([row[i] for row in rows]) for i in (0, 1, 2)), seed=seed)
        for device, rows in aggregate.items()}
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return report


def audit_x_r3(*, progress=print):
    """Check the complete 25-bit X/R3 joint distribution without an FWHT."""
    from .submission import BUNDLE, _records as garnet_records, verify_bundle
    manifest = verify_bundle()
    policy = BUNDLE / manifest["policies"]["X3"]["path"]
    dem = stim.DetectorErrorModel.from_file(policy / "model.dem")
    lut = build_lookup_table(dem)
    start = time.perf_counter()
    joint = direct_joint(dem, max_detectors=24, progress=progress)
    seconds = time.perf_counter() - start
    exact = (joint[1] > joint[0]).astype(np.uint8)
    cases = [c for c in manifest["cases"] if
             (c["arm"], c["basis"], c["rounds"]) == ("optimized", "X", 3)]
    arrays = [garnet_records(BUNDLE, case) for case in cases]
    dets, truth = (np.concatenate([a[i] for a in arrays]) for i in (0, 1))
    keys = dets.astype(np.uint32) @ (np.uint32(1) << np.arange(dem.num_detectors, dtype=np.uint32))
    lp = lut.decode_batch(dets).ravel()
    matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    mp = matcher.decode_batch(dets, enable_correlations=True).ravel()
    disputed = lp != mp
    p1 = joint[1, keys] / joint[:, keys].sum(axis=0)
    expected_error = np.where(lp, 1-p1, p1)
    select = lut.table.astype(bool)
    lut_risk = float(joint[0, select].sum() + joint[1, ~select].sum())
    result = dict(dem_sha256=hashlib.sha256(str(dem).encode()).hexdigest(),
        direct_convolution_seconds=seconds, probability_mass=float(joint.sum()),
        optimal_failure=float(np.minimum(joint[0], joint[1]).sum()),
        fwht_lut_expected_failure=lut_risk,
        all_table_decision_differences=int(np.count_nonzero(exact != lut.table)),
        hardware_prediction_differences=int(np.count_nonzero(exact[keys] != lp)),
        hardware_lut_failures=int(np.count_nonzero(lp != truth.ravel())),
        hardware_direct_failures=int(np.count_nonzero(exact[keys] != truth.ravel())),
        hardware_mwpm_failures=int(np.count_nonzero(mp != truth.ravel())),
        disputed_shots=int(disputed.sum()),
        disputed_lut_failures=int(np.count_nonzero((lp != truth.ravel()) & disputed)),
        disputed_model_mean_lut_error=float(expected_error[disputed].mean()),
        minimum_model_probability_of_observed_history=float(joint[:, keys].sum(axis=0).min()))
    if progress:
        progress(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-shots", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--direct-x-r3", action="store_true", help="check every X/R3 outcome by direct convolution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from MLfolders.mlDecoder.evidence import offline
    with offline():
        result = audit_x_r3() if args.direct_x_r3 else audit(simulated_shots=args.sim_shots, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Saved {args.output}")
    return result


if __name__ == "__main__":
    main()
