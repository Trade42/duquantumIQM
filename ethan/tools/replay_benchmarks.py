"""Verify bundled hardware evidence offline; never retrieve credentials or jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from iqm_surface_code.decoders.lookup_table import LookupTableDecoder
from iqm_surface_code.decoding import make_matcher


def replay(bundle: Path) -> tuple[int, int]:
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    total = 0
    for case in manifest["cases"]:
        files = {}
        for key, relative in case["files"].items():
            path = (bundle / relative).resolve()
            if not path.is_relative_to(bundle):
                raise ValueError("Artifact path escapes bundle.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"][key]:
                raise ValueError(f"Artifact hash mismatch: {relative}")
            files[key] = path

        with np.load(files["shots"], allow_pickle=False) as data:
            dets, obs = data["det_events"], data["obs_flips"]
            if len(dets) != case["shots"] or len(obs) != case["shots"]:
                raise ValueError("Shot count mismatch.")
            reference = stim.Circuit.from_file(files["reference"])
            converted = reference.compile_m2d_converter().convert(
                measurements=data["raw_meas"], separate_observables=True
            )
            if not (np.array_equal(converted[0], dets) and
                    np.array_equal(converted[1], obs)):
                raise ValueError("Source frame mismatch.")
            if "native" in files:
                native = stim.Circuit.from_file(files["native"])
                converted = native.compile_m2d_converter().convert(
                    measurements=data["native_raw_meas"], separate_observables=True
                )
                if not (np.array_equal(converted[0], dets) and
                        np.array_equal(converted[1], obs)):
                    raise ValueError("Native frame mismatch.")
            if case["decoder_family"] == "lut":
                decoder = LookupTableDecoder.load(files["lut"])
            elif case["decoder_family"] == "mwpm":
                decoder = make_matcher(stim.Circuit.from_file(files["decoder"]))
            else:
                raise ValueError(f"Unknown decoder family: {case['decoder_family']}")
            predictions = decoder.decode_batch(dets, enable_correlations=True)
            failed = np.any(predictions != obs, axis=1)
            if int(failed.sum()) != case["errors"]:
                raise ValueError(f"Score mismatch: {case['id']}")
            if "predictions" in files:
                with np.load(files["predictions"], allow_pickle=False) as saved:
                    if not (np.array_equal(predictions, saved["predictions"]) and
                            np.array_equal(failed, saved["failed"])):
                        raise ValueError("Frozen predictions differ.")
        total += case["shots"]
        print(f"{case['study']}/{case['id']}: {case['errors']}/{case['shots']} failures")

    if total != manifest["total_shots"]:
        raise ValueError("Bundle total does not match manifest.")
    print(f"Verified {len(manifest['cases'])} cases, {total:,} hardware shots; no QPU calls.")
    return len(manifest["cases"]), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=Path(__file__).resolve().parents[2] / "archive/ethan/benchmarks",
    )
    args = parser.parse_args()
    replay(args.bundle)


if __name__ == "__main__":
    main()
