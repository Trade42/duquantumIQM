"""Compare decoders on the same simulated shots: PyMatching, MLP, conv full decoder, conv pre-decoder.

For each distance d and noise strength p: a no-reset Z memory with `rounds` (default d) rounds
and uniform circuit noise. Every learned decoder trains on its own fresh shots from that
circuit; all are scored on one independent held-out set. Reports P_L with 95% Wilson
intervals, the paired comparison with PyMatching (only-X-fails / only-PM-fails, exact McNemar),
training time, single-shot latency and batched throughput.

    python -m iyanu.compare_decoders --distances 3 5 --p 0.003 0.005
    python -m iyanu.compare_decoders --distances 3 5 7 --channels 32 --steps 4000   # CUDA
"""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np

DECODERS = ("pymatching", "mlp", "conv_full", "conv_pre")


def build(distance, rounds, p):
    from challenge.surface_code import make_stim_circuit
    from challenge.noise_models import noise_suite, add_noise
    return add_noise(make_stim_circuit(distance, rounds, "Z", no_reset=True, gate_basis="cz"), noise_suite(p)["uniform"])


def train_all(circuit, a, seed, names):
    from challenge.decoding import make_matcher
    from iyanu.neural_gpu import NeuralConfig, train_neural_decoder
    from iyanu.conv_decoders import ConvConfig, ConvDecoder, ConvPredecoder
    conv = ConvConfig(channels=a.channels, layers=a.layers, steps=a.steps, batch=a.batch, seed=seed,
                      device=a.device, threshold=a.threshold)
    out = {}
    for name in names:
        t = time.perf_counter()
        if name == "pymatching":
            m = make_matcher(circuit, correlated=True)
            out[name] = (lambda d, m=m: m.decode_batch(d, enable_correlations=True),
                         dict(backend="pymatching correlated", build_seconds=time.perf_counter()-t))
        elif name == "mlp":
            dec = train_neural_decoder(circuit, NeuralConfig(steps=a.steps, batch=a.batch, seed=seed, device=a.device))
            out[name] = (dec.decode_batch, dec.metadata())
        elif name == "conv_full":
            dec = ConvDecoder(circuit, conv).train(circuit)
            out[name] = (dec.decode_batch, dec.metadata())
        elif name == "conv_pre":
            dec = ConvPredecoder(circuit, conv).train()
            out[name] = (dec.decode_batch, dec.metadata())
            out[name][1]["_decoder"] = dec
        print(f"    {name:<10} ready in {time.perf_counter()-t:6.1f} s", flush=True)
    return out


def main(argv=None):
    from challenge.decoding import wilson_interval
    from iyanu.neural_gpu import latency_ns, paired_comparison, pick_device, throughput
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--p", type=float, nargs="+", default=[0.003])
    ap.add_argument("--rounds", type=int, default=None, help="default: rounds = distance")
    ap.add_argument("--decoders", nargs="+", default=list(DECODERS), choices=DECODERS)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test-shots", type=int, default=100_000)
    ap.add_argument("--latency-repeats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="results/iyanu/compare")
    a = ap.parse_args(argv)
    names = ["pymatching"]+[n for n in a.decoders if n != "pymatching"]
    device = pick_device(a.device)
    print(f"device {device}; decoders {names}", flush=True)

    rows = []
    for p in a.p:
        for d in a.distances:
            rounds = a.rounds or d
            circuit = build(d, rounds, p)
            print(f"d={d} rounds={rounds} p={p}: {circuit.num_detectors} detectors", flush=True)
            seed = a.seed+1000*d+int(round(p*1e5))
            decoders = train_all(circuit, a, seed, names)
            dets, obs = circuit.compile_detector_sampler(seed=seed+7_919_993).sample(a.test_shots, separate_observables=True)
            preds = {n: f(dets) for n, (f, _) in decoders.items()}
            row = dict(distance=d, rounds=rounds, p=p, detectors=circuit.num_detectors, shots=a.test_shots,
                       test_seed=seed+7_919_993, raw_flip_rate=float(np.any(obs, axis=1).mean()), decoders={})
            for n, (f, meta) in decoders.items():
                errors = int(np.any(preds[n] != obs, axis=1).sum())
                meta = {k: v for k, v in meta.items() if not k.startswith("_")}
                entry = dict(errors=errors, p_l=errors/a.test_shots, ci95=wilson_interval(errors, a.test_shots),
                             meta=meta, latency=latency_ns(f, dets, repeats=a.latency_repeats, warmup=50),
                             throughput=throughput(f, dets, min(10_000, a.test_shots)))
                if n != "pymatching":
                    entry["vs_pymatching"] = paired_comparison(preds[n], preds["pymatching"], obs)
                if n == "conv_pre":
                    entry["predecoder_stats"] = decoders[n][1]["_decoder"].decode_batch(dets[:20_000], return_stats=True)[1]
                row["decoders"][n] = entry
            rows.append(row)
            report(row)

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out = Path(a.output)/run_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"results.json").write_text(json.dumps(dict(run_id=run_id, args=vars(a), device=str(device),
        environment=dict(platform=platform.platform()), rows=rows), indent=2, default=str))
    plot(rows, out/"pl_vs_distance.png", a)
    print(f"\nwrote {out}")
    return rows


def report(row):
    print(f"  {'decoder':<11} {'P_L':>8}  {'95% Wilson':<19} {'only-it/only-PM':>15} {'McNemar p':>9}"
          f" {'lat p50 us':>10} {'lat p99 us':>10} {'shots/s':>10} {'train s':>8}")
    for n, e in row["decoders"].items():
        vs = e.get("vs_pymatching")
        pair = f"{vs['only_a_fails']}/{vs['only_b_fails']}" if vs else "-"
        mc = f"{vs['mcnemar_exact_p']:.3g}" if vs else "-"
        train = e["meta"].get("train_seconds", e["meta"].get("build_seconds", 0))
        print(f"  {n:<11} {e['p_l']:8.5f}  [{e['ci95'][0]:.5f}, {e['ci95'][1]:.5f}] {pair:>15} {mc:>9}"
              f" {e['latency']['p50_us']:10.1f} {e['latency']['p99_us']:10.1f}"
              f" {e['throughput']['shots_per_second']:10,.0f} {train:8.1f}")
    if "conv_pre" in row["decoders"]:
        s = row["decoders"]["conv_pre"]["predecoder_stats"]
        print(f"  pre-decoder: {s['events_in']:.2f} -> {s['events_left']:.2f} events/shot, "
              f"{100*s['cleared_fraction']:.1f}% of shots need no matching", flush=True)


def plot(rows, path, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = sorted({r["p"] for r in rows})
    fig, axes = plt.subplots(1, len(ps), figsize=(5*len(ps), 4), squeeze=False)
    for ax, p in zip(axes[0], ps):
        sub = sorted((r for r in rows if r["p"] == p), key=lambda r: r["distance"])
        ds = np.array([r["distance"] for r in sub])
        for i, n in enumerate(sub[0]["decoders"]):
            pl = np.array([r["decoders"][n]["p_l"] for r in sub])
            ci = np.array([r["decoders"][n]["ci95"] for r in sub])
            ax.errorbar(ds+(i-1.5)*0.08, pl, yerr=[pl-ci[:, 0], ci[:, 1]-pl], fmt="o-", capsize=3, label=n)
        ax.set(yscale="log", xticks=ds, xlabel="code distance d (rounds = d)",
               ylabel="logical error probability $P_L$", title=f"uniform p = {p}")
        ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"No-reset Z memory, {a.test_shots} held-out shots per point, 95% Wilson")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
