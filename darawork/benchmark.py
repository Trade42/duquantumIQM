"""Decoder benchmark: identical held-out test shots for every decoder.

Per (noise, basis, distance, rounds): logical error probability with a Wilson
interval, decode time per shot, and a paired comparison with correlated MWPM on
the same shots (exact two-sided binomial test on discordant shots). Per-round
LER per decoder comes from decoding.fit_ler over the round counts.

Example: python -m darawork.benchmark --distances 3 --rounds 1 2 3 4 5 --shots 4000
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import zlib
import numpy as np
from scipy.stats import binomtest

from challenge.surface_code import make_stim_circuit
from challenge.noise_models import NoiseModel, add_noise
from challenge.build_emerald_qubit_rotated import interaction_pairs
from challenge.decoding import fit_ler, wilson_interval
from challenge.leakage import sample_leakage
from .garnet_dataset import GARNET_MEDIANS, garnet_snapshot
from .decoders.ensemble_decoder import EnsembleDecoder, MatchingDecoder
from .decoders.tn_decoder import TNDecoder
from .decoders.lut_decoder import CachedLUT, LUTDecoder
from .decoders.dem_fit import choose_dem
import stim

UNIFORM_P = 0.003
DURATIONS = {"1q": 20, "2q": 80, "measure": 700, "reset": 1000}  # ASSUMED, as in garnet_dataset.


@lru_cache(maxsize=None)
def noisy(noise, distance, rounds, basis):
    """(ideal, static-noise circuit, model, qubit map). Garnet noise = Garnet-median qubits."""
    ideal = make_stim_circuit(distance, rounds, basis, True, gate_basis="cz")
    if noise == "uniform":
        model, qmap = NoiseModel.uniform(UNIFORM_P), None
    else:
        qmap = {q: f"Q{q}" for q in ideal.get_final_qubit_coordinates()}
        couplers = sorted({tuple(sorted((qmap[a], qmap[b]))) for a, b in interaction_pairs(ideal)})
        t2 = GARNET_MEDIANS["t2_echo_us" if noise == "garnet_echo" else "t2_ramsey_us"]
        cal = garnet_snapshot(readout_error=0.02, t2_us=t2, durations_ns=DURATIONS,
                              names=list(qmap.values()), couplers=couplers)
        extra = dict(leakage=1e-3, seepage=0.01, leakage_spread=0.02) if noise == "garnet_leakage" else {}
        if noise not in {"garnet", "garnet_echo", "garnet_leakage"}:
            raise ValueError(f"Unknown noise {noise}")
        model = NoiseModel(name=noise, calibration=cal, coherence=True, idle=0, **extra)
    return ideal, add_noise(ideal, model, qubit_map=qmap), model, qmap


def sample(noise, distance, rounds, basis, shots, seed):
    """(ideal, dets, obs). Leakage is sampled statefully; decoders still see the static DEM."""
    ideal, circuit, model, qmap = noisy(noise, distance, rounds, basis)
    if model.leakage:
        raw, _ = sample_leakage(circuit, model, shots, seed=seed, qubit_map=qmap)
        dets, obs = ideal.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    else:
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return ideal, dets, obs


def nn_path(noise, root="results/nn"):
    return Path(root) / f"{noise}.pt"


def decoders(nn, distance, circuit, ideal, *, chi, members, sigma, train=None):
    """name -> zero-arg factory. A factory raising ValueError marks that decoder infeasible here."""
    out = {"MWPM": lambda: MatchingDecoder(circuit, correlated=False),
           "corr. MWPM": lambda: MatchingDecoder(circuit),
           f"ensemble x{members}": lambda: EnsembleDecoder(circuit, members=members, sigma=sigma),
           "TN": lambda: tn(circuit, distance, chi)}
    if nn is not None and Path(nn).exists():
        from .decoders.nn_decoder import NNDecoder
        out["CNN-GRU"] = lambda: NNDecoder(NNDecoder.load(nn), ideal)
    if train is not None:
        out["LUT"] = lambda: LUTDecoder(*train, MatchingDecoder(circuit))
        if distance == 3:  # Exact TN posteriors, overruled by enough real counts; misses decoded online.
            out["TN cache + data"] = lambda: CachedLUT(train[0], tn(circuit, distance, chi), MatchingDecoder(circuit),
                                                       train_obs=train[1], online=True)
    return out


def tn(circuit_or_dem, distance, chi):
    kw = dict(chi=None if distance == 3 else chi)
    if isinstance(circuit_or_dem, stim.DetectorErrorModel):
        return TNDecoder(circuit_or_dem, **kw)
    return TNDecoder.from_circuit(circuit_or_dem, **kw)


def evaluate(cases, *, chi, members, sigma, only=None):
    """cases: (meta, ideal, decoder circuit, dets, obs, nn checkpoint, (train dets, train obs) or None).

    Every decoder sees the same test shots; training shots never overlap them.
    """
    records = []
    for meta, ideal, circuit, dets, obs, nn, train in cases:
        fails, shots, d = {}, len(dets), meta["distance"]
        tag = " ".join(f"{k}={v}" for k, v in meta.items() if k != "seed")
        for name, make in decoders(nn, d, circuit, ideal, chi=chi, members=members, sigma=sigma, train=train).items():
            if only and name not in only:
                continue
            rec = dict(**meta, decoder=name, shots=shots)
            try:
                t0 = time.perf_counter(); dec = make(); build = time.perf_counter() - t0
                t0 = time.perf_counter(); pred = dec.decode(dets); dt = time.perf_counter() - t0
            except (ValueError, MemoryError) as e:
                records.append({**rec, "infeasible": str(e)})
                print(f"{tag} {name}: infeasible ({e})", flush=True)
                continue
            fail = np.any(pred != obs, axis=1)
            fails[name] = fail
            k = int(fail.sum())
            rec.update(errors=k, probability=k / shots, ci95=list(wilson_interval(k, shots)),
                       us_per_shot=1e6 * dt / shots, build_seconds=build)
            if getattr(dec, "coverage", None) is not None:
                rec.update(lut_coverage=dec.coverage, train_shots=len(train[0]))
            if name != "corr. MWPM" and "corr. MWPM" in fails:
                base = fails["corr. MWPM"]
                only_dec, only_pm = int((fail & ~base).sum()), int((base & ~fail).sum())
                rec.update(only_this_wrong=only_dec, only_corr_mwpm_wrong=only_pm,
                           paired_p=binomtest(only_dec, only_dec + only_pm).pvalue if only_dec + only_pm else 1.0)
            records.append(rec)
            print(f"{tag} {name}: P_L={k / shots:.4f} {rec['us_per_shot']:.1f} us/shot", flush=True)
    return records


def finish(records, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fits = []
    keys = {(r["noise"], r["basis"], r["distance"], r["decoder"]) for r in records if "errors" in r}
    for noise, basis, d, name in sorted(keys):
        pts = sorted((r for r in records if "errors" in r and (r["noise"], r["basis"], r["distance"], r["decoder"]) == (noise, basis, d, name)),
                     key=lambda r: r["rounds"])
        if len(pts) < 2:
            continue  # e.g. TN at d=7 is feasible only at one round count.
        f = fit_ler([p["rounds"] for p in pts], [p["errors"] for p in pts], [p["shots"] for p in pts])
        fits.append(dict(noise=noise, basis=basis, distance=d, decoder=name, rounds=[p["rounds"] for p in pts],
                         epsilon=f["epsilon_per_round"], ci95=f["ci95_profile"],
                         us_per_shot=float(np.mean([p["us_per_shot"] for p in pts]))))
    (out / "results.json").write_text(json.dumps(dict(records=records, fits=fits), indent=2) + "\n")
    plots(records, fits, out)
    (out / "summary.md").write_text(summary(fits))
    print(summary(fits))
    return records, fits


def run(*, noises, bases, distances, rounds_list, shots, chi, members, sigma, out, lut_train=100000, only=None):
    def cases():
        for noise in noises:
            for basis in bases:
                for d in distances:
                    for r in rounds_list:
                        ideal, circuit, _, _ = noisy(noise, d, r, basis)
                        seed = zlib.crc32(f"test/{noise}/{basis}/{d}/{r}".encode())  # < 2**32; training uses >= 2**40.
                        _, dets, obs = sample(noise, d, r, basis, shots, seed)
                        train = sample(noise, d, r, basis, lut_train, 2**41 + seed)[1:] if lut_train else None
                        yield (dict(noise=noise, basis=basis, distance=d, rounds=r, seed=seed),
                               ideal, circuit, dets, obs, nn_path(noise), train)
    return evaluate(cases(), chi=chi, members=members, sigma=sigma, only=only)


def hardware_data(job_dir):
    """{(basis, rounds): (ideal, dets, obs)} from a saved Garnet job (raw records, never edited)."""
    from .garnet_dataset import suite_and_layout
    job_dir = Path(job_dir)
    manifest = json.loads((job_dir / "manifest.json").read_text())
    layout = {int(k): v for k, v in manifest["layout"].items()}
    model = suite_and_layout(0.02, DURATIONS, 1e-3)[0]["garnet_coherence"]
    data = {}
    for f in sorted(job_dir.glob("raw_*_r*.npy")):
        _, basis, r = f.stem.split("_")
        ideal = make_stim_circuit(3, int(r[1:]), basis, True, gate_basis="cz")
        dets, obs = ideal.compile_m2d_converter().convert(measurements=np.load(f), separate_observables=True)
        data[basis, int(r[1:])] = (ideal, add_noise(ideal, model, qubit_map=layout), dets, obs)
    return data, manifest


def hardware(job_dir, *, test_shots, rounds_list=None, nn=None, chi=256, members=21, sigma=0.5, only=None, fit=False):
    """Decode real Garnet shots: the last test_shots of each circuit are the held-out test set.

    DEM decoders use the simulated Garnet-median model (no DD). With fit=True each
    circuit is decoded a second time ("garnet_hw+fit") with that model's DEM fitted
    to the circuit's training split (dem_fit); test shots never enter the fit.
    """
    data, manifest = hardware_data(job_dir)
    def cases():
        for (basis, r), (ideal, circuit, dets, obs) in sorted(data.items()):
            if rounds_list and r not in rounds_list:
                continue
            train = (dets[:-test_shots], obs[:-test_shots])  # LUT/NN/fit all use this split only.
            meta = dict(basis=basis, distance=3, rounds=r, job_id=manifest["job_id"])
            yield (dict(noise="garnet_hw", **meta), ideal, circuit, dets[-test_shots:], obs[-test_shots:], nn, train)
            if fit:
                dem, chosen, cv = choose_dem(circuit, *train)
                print(f"{basis} r={r}: decoder DEM = {chosen} (held-back training errors {cv})", flush=True)
                yield (dict(noise="garnet_hw+fit", dem=chosen, **meta), ideal, dem,
                       dets[-test_shots:], obs[-test_shots:], nn, train)
    return evaluate(cases(), chi=chi, members=members, sigma=sigma, only=only)


class _Decisions:
    """Adapter: a decoder's hard decisions as one-hot 'probabilities' (cache values at d=7)."""

    def __init__(self, decoder):
        self.decoder, self.name = decoder, decoder.name

    def probabilities(self, dets):
        flip = self.decoder.decode(dets)[:, 0]
        return np.stack([~flip, flip], 1).astype(float)


def cache_report(*, noise="uniform", basis="Z", rounds=3, capacities=(256, 4096, 65536), build_shots=100000,
                 test_shots=5000, chi=256, out="results/cache_report"):
    """Weighted cached LUT per distance: LER, decode time, hit rate and table memory per capacity."""
    rows = []
    for d in (3, 5, 7):
        ideal, circuit, _, _ = noisy(noise, d, rounds, basis)
        seed = zlib.crc32(f"test/{noise}/{basis}/{d}/{rounds}".encode())
        _, build, _ = sample(noise, d, rounds, basis, build_shots, 2**42 + seed)
        _, dets, obs = sample(noise, d, rounds, basis, test_shots, seed)
        pm = MatchingDecoder(circuit)
        try:
            model, value = tn(circuit, d, chi), "TN posterior"
        except ValueError:
            model, value = _Decisions(pm), "corr. MWPM decision (TN infeasible)"
        t0 = time.perf_counter(); direct = model.decoder.decode(dets) if isinstance(model, _Decisions) else model.decode(dets)
        direct_us = 1e6 * (time.perf_counter() - t0) / test_shots
        base = dict(distance=d, rounds=rounds, noise=noise, basis=basis, value=value,
                    model_P_L=float((direct != obs).any(1).mean()), model_us_per_shot=direct_us,
                    corr_mwpm_P_L=float((pm.decode(dets) != obs).any(1).mean()))
        for cap in capacities:
            if d == 5 and cap > 4096:
                continue  # ponytail: TN beam ~13 ms/syndrome at d=5; larger tables take >15 min to build.
            t0 = time.perf_counter(); cache = CachedLUT(build, model, pm, capacity=cap); build_s = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = cache.decode(dets); us = 1e6 * (time.perf_counter() - t0) / test_shots
            rows.append(dict(base, capacity=cap, entries=len(cache.table), memory_MB=cache.memory_bytes / 2**20,
                             build_seconds=build_s, hit_rate=cache.hit_rate, P_L=float((pred != obs).any(1).mean()),
                             us_per_shot=us))
            print(rows[-1], flush=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "cache_report.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| d | value | capacity | memory MB | hit rate | P_L cache | P_L model | P_L corr. MWPM | µs/shot cache | µs/shot model |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['distance']} | {r['value']} | {r['capacity']} | {r['memory_MB']:.2f} | {r['hit_rate']:.3f} | {r['P_L']:.4f} | "
              f"{r['model_P_L']:.4f} | {r['corr_mwpm_P_L']:.4f} | {r['us_per_shot']:.1f} | {r['model_us_per_shot']:.1f} |" for r in rows]
    (out / "cache_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def summary(fits):
    lines = ["| noise | basis | d | decoder | ε_L per round [95% CI] | vs corr. MWPM | µs/shot |",
             "|---|---|---|---|---|---|---|"]
    base = {(f["noise"], f["basis"], f["distance"]): f["epsilon"] for f in fits if f["decoder"] == "corr. MWPM"}
    for f in fits:
        b = base.get((f["noise"], f["basis"], f["distance"]))
        rel = f"{100 * (f['epsilon'] / b - 1):+.1f}%" if b else ""
        lines.append(f"| {f['noise']} | {f['basis']} | {f['distance']} | {f['decoder']} | "
                     f"{f['epsilon']:.5f} [{f['ci95'][0]:.5f}, {f['ci95'][1]:.5f}] | {rel} | {f['us_per_shot']:.1f} |")
    return "\n".join(lines) + "\n"


def plots(records, fits, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in records if "errors" in r]
    groups = sorted({(r["noise"], r["basis"], r["distance"]) for r in ok})
    names = sorted({r["decoder"] for r in ok})
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), squeeze=False)
    for ax, g in zip(axes[0], groups):
        for n in names:
            pts = sorted((r for r in ok if (r["noise"], r["basis"], r["distance"]) == g and r["decoder"] == n), key=lambda r: r["rounds"])
            if pts:
                x = [p["rounds"] for p in pts]; y = [p["probability"] for p in pts]
                err = np.clip(np.array([[p["probability"] - p["ci95"][0], p["ci95"][1] - p["probability"]] for p in pts]).T, 0, None)
                ax.errorbar(x, y, yerr=err, marker="o", ms=3, capsize=2, label=n)
        ax.set(title=f"{g[0]} {g[1]} d={g[2]}", xlabel="rounds", ylabel="logical error probability P_L")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "pl_vs_rounds.png", dpi=150); plt.close(fig)
    for fname, xkey, xlabel in (("ler_vs_distance.png", "distance", "code distance d"),
                                ("ler_vs_time.png", "us_per_shot", "decode time per shot (µs, batched)")):
        series = sorted({(f["noise"], f["basis"]) for f in fits})
        fig, axes = plt.subplots(1, len(series), figsize=(4.2 * len(series), 3.6), squeeze=False)
        for ax, s in zip(axes[0], series):
            for n in names:
                pts = sorted((f for f in fits if (f["noise"], f["basis"]) == s and f["decoder"] == n), key=lambda f: f[xkey])
                if pts:
                    x = [p[xkey] for p in pts]; y = [p["epsilon"] for p in pts]
                    err = np.clip(np.array([[p["epsilon"] - p["ci95"][0], p["ci95"][1] - p["epsilon"]] for p in pts]).T, 0, None)
                    ax.errorbar(x, y, yerr=err, marker="o", ms=4, capsize=2, label=n, ls="-" if xkey == "distance" else "none")
            ax.set(title=f"{s[0]} {s[1]}", xlabel=xlabel, ylabel="ε_L per round", yscale="log")
            if xkey == "us_per_shot":
                ax.set_xscale("log")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--noises", nargs="+", default=["uniform", "garnet_echo", "garnet_leakage"])
    p.add_argument("--bases", nargs="+", default=["Z", "X"])
    p.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    p.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--chi", type=int, default=128, help="beam width for TN at d>=5")
    p.add_argument("--members", type=int, default=21)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--hardware", metavar="JOB_DIR", help="decode a saved real-Garnet job instead")
    p.add_argument("--test-shots", type=int, default=5000, help="with --hardware: held-out shots per circuit")
    p.add_argument("--nn", help="with --hardware: NN checkpoint")
    p.add_argument("--fit", action="store_true", help="with --hardware: also decode with the DEM fitted to training shots")
    p.add_argument("--lut-train", type=int, default=100000, help="simulated LUT training shots per circuit")
    p.add_argument("--add-lut", action="store_true",
                   help="retroactive: decode the same test shots with only LUT (+ its fallback) and merge into OUT/results.json")
    p.add_argument("--cache-report", action="store_true", help="weighted cached LUT report for d=3/5/7")
    a = p.parse_args(argv)
    if a.cache_report:
        return cache_report(chi=a.chi)
    out = Path(a.out if not (a.hardware and a.out == "results/benchmark") else "results/benchmark_hw")
    only = {"LUT", "corr. MWPM"} if a.add_lut else None
    if a.hardware:
        records = hardware(a.hardware, test_shots=a.test_shots, rounds_list=a.rounds, nn=a.nn, chi=a.chi,
                           members=a.members, sigma=a.sigma, only=only, fit=a.fit)
    else:
        records = run(noises=a.noises, bases=a.bases, distances=a.distances, rounds_list=a.rounds, shots=a.shots,
                      chi=a.chi, members=a.members, sigma=a.sigma, out=out, lut_train=a.lut_train, only=only)
    if a.add_lut:  # Keep earlier records; LUT records are new, its fallback baseline is already there.
        old = json.loads((out / "results.json").read_text())["records"]
        records = [r for r in old if r["decoder"] != "LUT"] + [r for r in records if r["decoder"] == "LUT"]
    finish(records, out)


if __name__ == "__main__":
    main()
