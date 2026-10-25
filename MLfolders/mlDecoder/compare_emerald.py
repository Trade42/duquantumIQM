"""Compare the Emerald ML telescope with vanilla memory at rounds 1, 2 and 3.

python -m MLfolders.mlDecoder.compare_emerald
python -m MLfolders.mlDecoder.compare_emerald --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from challenge.decoding import fit_ler

ROOT = Path(__file__).resolve().parents[2]
VANILLA = Path("benchmarks/emerald/vanilla/20261025T022556Z/manifest.json")
ML = Path("MLfolders/mlDecoder/results_record/rnn_lstm/"
          "20261024-220320-finetune-20261024-195414/result.json")
ROUNDS = [1, 2, 3]


def _fit(rounds, failures, shots):
    fitted = fit_ler(rounds, failures, shots)
    return dict(rounds=rounds, failures=failures, shots=shots,
                epsilon_per_round=fitted["epsilon_per_round"],
                ci95_profile=fitted["ci95_profile"])


def _score(point, decoder):
    return next(row["point"] for row in point["decoders"] if row["name"] == decoder)


def comparison():
    vanilla = json.loads((ROOT / VANILLA).read_text(encoding="utf-8"))
    ml = json.loads((ROOT / ML).read_text(encoding="utf-8"))
    results, timings = [], []
    for basis in ("X", "Z"):
        base = next(row for row in vanilla["fits"] if row["basis"] == basis)
        base_fit = _fit(base["rounds"], base["errors"], base["shots"])
        points = sorted((p for p in ml["points"] if p["spec"]["basis"] == basis
                         and p["spec"]["rounds"] in ROUNDS),
                        key=lambda p: p["spec"]["rounds"])
        row = dict(basis=basis, vanilla=base_fit)
        for label, decoder in (("lstm", "rnn_lstm_finetuned"),
                               ("telescope", "rnn_lstm_cascade")):
            scores = [_score(point, decoder) for point in points]
            fitted = _fit([p["spec"]["rounds"] for p in points],
                          [p["failures"] for p in scores], [p["shots"] for p in scores])
            row[label] = {**fitted, "relative_ler_reduction":
                          1 - fitted["epsilon_per_round"] / base_fit["epsilon_per_round"]}
        results.append(row)
        group = ml["groups"][f"emerald_eA_{basis}"]
        for label, key in (("lstm", "speed"), ("telescope", "cascade_speed")):
            for case in group[key]:
                cpu = next(r for r in case["cpu_threads_1"] if r["batch"] == 1)
                timings.append(dict(basis=basis, decoder=label, rounds=case["rounds"],
                                    mean_decode_us_per_round=cpu["latency_per_shot_us"] / case["rounds"],
                                    mean_history_us=cpu["latency_per_shot_us"],
                                    median_history_us=cpu["latency_per_call_us"],
                                    p99_history_us=cpu["latency_per_call_p99_us"]))
    return dict(schema="emerald-vanilla-ml/1", rounds=ROUNDS,
                sources=[dict(path=path.as_posix(), sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
                         for path in (VANILLA, ML)],
                fit="P_L(R) = (1 - (1 - 2*epsilon)**R) / 2; binomial maximum likelihood",
                results=results, timing=dict(
                    host="Ryzen 9 8945HS", cpu_threads=1, batch=1,
                    metric="Mean complete-history decode time divided by syndrome round count",
                    records=timings))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="save a new comparison record")
    args = parser.parse_args(argv)
    result = comparison()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result["results"]:
            print(f"Emerald {row['basis']}: vanilla LER {row['vanilla']['epsilon_per_round']:.3%}; "
                  f"ML telescope {row['telescope']['epsilon_per_round']:.3%}; "
                  f"reduction {row['telescope']['relative_ler_reduction']:.1%}")
        for row in result["timing"]["records"]:
            print(f"  {row['basis']} R{row['rounds']:2} {row['decoder']:9}: "
                  f"{row['mean_decode_us_per_round']:.1f} us/round")
    return result


if __name__ == "__main__":
    main()
