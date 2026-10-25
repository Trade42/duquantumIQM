"""Audit imported ML evidence offline, optionally replaying the frozen LSTM.

Default: verify published artifact hashes, raw-record detector conversion, test
splits, pooled failures and LER fits. --decode also recomputes predictions for
all 154,000 B3 shots with the committed weights. No training or QPU access.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
from datetime import datetime
import platform

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RECORD = HERE / "results_record/rnn_lstm"
RUNS = (
    "20261024-215152-finetune-20261024-191223",
    "20261024-220320-finetune-20261024-195414",
)


@contextmanager
def offline():
    def deny(*args, **kwargs):
        raise RuntimeError("Network access is disabled during the ML evidence audit.")
    previous = socket.socket.connect, socket.create_connection
    socket.socket.connect = socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = previous


def verify_artifacts():
    provenance = json.loads((HERE / "upstream_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for relative, expected in provenance["files"].items():
        # Source hashes identify upstream provenance; adapted runtime sources
        # are documented in UPSTREAM.md. Evidence and weights remain exact.
        frozen_roots = (
            "MLfolders/InfoRecord/datasets/", "MLfolders/InfoRecord/emerald/datasets/",
            "MLfolders/InfoRecord/circuits/", "MLfolders/InfoRecord/emerald/circuits/",
            "MLfolders/mlDecoder/results_record/", "MLfolders/mlDecoder/calibration/",
            "MLfolders/mlDecoder/weights/", "MLfolders/mlDecoder/configs/",
        )
        if not relative.startswith(frozen_roots) or relative.endswith(".py"):
            continue
        path = (REPOSITORY / relative).resolve()
        if not path.is_relative_to(REPOSITORY):
            raise ValueError("Evidence path escapes the repository.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Imported evidence hash mismatch: " + relative)
        checked += 1
    weights = json.loads((HERE / "weights/manifest.json").read_text(encoding="utf-8"))
    for model in weights["models"]:
        path = HERE / "weights" / model["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("Frozen model hash mismatch: " + model["file"])
    return provenance, checked


def _decoder(point, name):
    return next(row for row in point["decoders"] if row["name"] == name)


def _timings(group):
    result = {}
    for row in group.get("cascade_speed", []):
        cpu = next(x for x in row["cpu_threads_1"] if x.get("batch") == 1 and "skipped" not in x)
        mwpm = next(x for x in row["mwpm_corr_cpu_threads_1"] if x.get("batch") == 1 and "skipped" not in x)
        cuda = max((x for x in row["cuda"] if "skipped" not in x), key=lambda x:x["batch"])
        result[str(row["rounds"])] = dict(
            cpu_median_us=cpu["latency_per_shot_us"],
            cpu_p99_us=cpu["latency_per_call_p99_us"],
            mwpm_cpu_median_us=mwpm["latency_per_shot_us"],
            cuda_shots_per_second=cuda["shots_per_second"], cuda_batch=cuda["batch"],
        )
    return result


def audit(*, decode=False, progress=None):
    from challenge.decoding import fit_ler
    from .hardware import load_run

    provenance, artifact_count = verify_artifacts()
    models = json.loads((HERE / "weights/manifest.json").read_text(encoding="utf-8"))["models"]
    summaries, cases, differences = [], 0, []
    torch = None
    if decode:
        import torch
        torch.set_num_threads(1)
        from .final import load

    for run_id in RUNS:
        result = json.loads((RECORD / run_id / "result.json").read_text(encoding="utf-8"))
        fits = json.loads((RECORD / run_id / "ler/ler_fits.json").read_text(encoding="utf-8"))["fits"]
        if result["eval_split"] != "test" or result.get("simulated_standin"):
            raise ValueError("Expected a held-out hardware test record.")
        for group_name, group in result["groups"].items():
            info = group["group"]
            selected = next(m for m in models if (m["device"], m["patch"], m["basis"]) ==
                            (info["device"], info["patch"], info["basis"]))
            if selected["cascade"]["tau"] != group["cascade"]["tau"]:
                raise ValueError("Frozen cascade threshold differs from the evaluated threshold.")
            training = json.loads((HERE / "results_record" / selected["source_run"] / "result.json").read_text(encoding="utf-8"))
            frozen = training["groups"][group_name]["cascade"]
            if frozen["tau"] != group["cascade"]["tau"] or datetime.fromisoformat(frozen["retuned_utc"]) >= datetime.fromisoformat(result["started_utc"]):
                raise ValueError("Threshold selection did not precede held-out evaluation.")
            points = sorted((p for p in result["points"] if p["group"] == group_name),
                            key=lambda p:p["spec"]["rounds"])
            row_fits = {f["decoder"]: f for f in fits if f["group"] == group_name}
            totals = dict(lstm=0, cascade=0, matching=0, uniform_matching=0, shots=0, deferred=0)
            neural = load(info["device"], info["patch"], info["basis"]) if decode else None
            replay_failures = 0
            for point in points:
                rounds = point["spec"]["rounds"]
                record_root = REPOSITORY / "MLfolders/InfoRecord"
                if info["device"] == "emerald":
                    record_root = record_root / "emerald"
                case_id = f"b3_E1_d3_{info['basis']}_r{rounds:02d}_{info['patch']}"
                raw = load_run(record_root / "datasets/E1" / case_id, record_root=record_root)
                if raw.manifest["batch"] != 3 or set(raw.split) != {"test"}:
                    raise ValueError("Only B3 held-out test shots belong in this replay.")
                if (raw.device, raw.spec["patch"], raw.spec["basis"], raw.spec["rounds"]) != (
                        info["device"], info["patch"], info["basis"], rounds):
                    raise ValueError("Test case device/patch/basis/rounds mismatch.")
                shots = len(raw.dets)
                lstm = _decoder(point, "rnn_lstm_finetuned")["point"]
                cascade = _decoder(point, "rnn_lstm_cascade")["point"]
                matching = point["references"]["mwpm_corr"]["point"]
                uniform = point["references"]["mwpm_corr_uniform"]["point"]
                if any(v["shots"] != shots for v in [lstm, cascade, matching, uniform]):
                    raise ValueError("Saved decoder scores use different shots.")
                if decode:
                    prediction = neural.decode_batch(raw.dets)
                    failures = int(np.any(prediction != raw.obs.astype(bool), axis=1).sum())
                    replay_failures += failures
                    if failures != lstm["failures"]:
                        differences.append(dict(group=group_name, rounds=rounds,
                                                published_failures=lstm["failures"],
                                                promoted_fp16_cpu_failures=failures))
                for key, value in [("lstm", lstm), ("cascade", cascade), ("matching", matching), ("uniform_matching", uniform)]:
                    totals[key] += value["failures"]
                totals["shots"] += shots
                totals["deferred"] += point["cascade"]["deferred"]
                cases += 1
                if progress:
                    progress(f"{group_name} R{rounds}: verified {shots:,} test shots" + (" and LSTM predictions" if decode else ""))
            rounds = [p["spec"]["rounds"] for p in points]
            shots = [_decoder(p, "rnn_lstm_finetuned")["point"]["shots"] for p in points]
            for name in ["rnn_lstm_finetuned", "rnn_lstm_cascade", "mwpm_corr"]:
                counts = ([p["references"][name]["point"]["failures"] for p in points] if name == "mwpm_corr" else
                          [_decoder(p, name)["point"]["failures"] for p in points])
                actual = fit_ler(rounds, counts, shots)["epsilon_per_round"]
                if not np.isclose(actual, row_fits[name]["no_spam"]["epsilon"], atol=1e-7):
                    raise ValueError("Recomputed challenge LER differs from the imported fit.")
            summaries.append(dict(
                group=group_name, **totals, probabilities={k:totals[k]/totals["shots"] for k in
                    ["lstm", "cascade", "matching", "uniform_matching"]},
                lstm_fit=row_fits["rnn_lstm_finetuned"]["no_spam"],
                cascade_fit=row_fits["rnn_lstm_cascade"]["no_spam"],
                matching_fit=row_fits["mwpm_corr"]["no_spam"],
                lstm_to_matching=row_fits["rnn_lstm_finetuned"]["ratio_to_mwpm_corr"],
                tau=group["cascade"]["tau"], timings=_timings(group),
                promoted_fp16_cpu_failures=replay_failures if decode else None,
            ))
    return dict(upstream_commit=provenance["commit"], verified_artifacts=artifact_count,
                cases=cases, shots=sum(r["shots"] for r in summaries),
                lstm_predictions_recomputed=decode, groups=summaries,
                promoted_replay_differences=differences,
                torch_version=str(torch.__version__) if decode else None,
                numpy_version=np.__version__,
                python_version=platform.python_version(),
                replay_device="CPU, one Torch thread" if decode else None,
                replay_note=("Published scores precede promotion to float16. CPU/runtime/precision differences "
                             "are reported separately; published scores and artifacts are never rewritten."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", action="store_true", help="replay all frozen LSTM predictions on the CPU")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="save a new audit/replay record; existing files are refused")
    args = parser.parse_args(argv)
    with offline():
        result = audit(decode=args.decode, progress=None if args.json or not args.decode else print)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result["groups"]:
            print(f"{row['group']}: LSTM {row['lstm']:,}/{row['shots']:,}; "
                  f"telescope {row['cascade']:,}; tau={row['tau']}")
        print(f"Verified {result['shots']:,} held-out hardware shots in {result['cases']} cases; "
              + ("promoted float16 LSTM predictions recomputed on CPU." if args.decode else "saved ML scores and fits audited."))
        if args.decode:
            for row in result["groups"]:
                print(f"  {row['group']}: published {row['lstm']}; promoted float16 CPU replay "
                      f"{row['promoted_fp16_cpu_failures']} failures")
            print(result["replay_note"])
    return result


if __name__ == "__main__":
    main()
