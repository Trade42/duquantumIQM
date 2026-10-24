"""Selection protocol for real-Garnet LERs, X and Z selected independently.

Per circuit: train (first n-10000 shots) / validation (next 5000) / test (last 5000).
Stage 1, on validation: circuit configuration, by correlated MWPM on the DEM chosen
from training shots, pooled over every session of that configuration.
Stage 2, on validation: decoder for the chosen configuration.
Test, once: the chosen decoder on the chosen configuration, pooled over sessions
(calibration drift is averaged, not picked), with each session shown.
Everything a decoder learns (DEM fit, LUT counts and min_count, TN cache counts)
comes from the training slice only.

Example: python -m darawork.protocol
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np

from .benchmark import hardware_data, tn
from challenge.decoding import fit_ler, wilson_interval
from .decoders.dem_fit import choose_dem
from .decoders.ensemble_decoder import EnsembleDecoder, MatchingDecoder
from .decoders.lut_decoder import LUTDecoder

# 20k-shot jobs only (large enough for three slices), by circuit configuration.
CONFIGS = {
    "original": ["01a16f9c", "01a16fee"],
    "DD": ["01a16fdd"],
    "sqrtY + measure barriers + DD": ["01a16fe0", "01a16feb"],
    "sqrtY + all barriers + DD": ["01a16fe2"],
    "sqrtY + no barriers + DD": ["01a16fd0"],
}
ROUNDS = [1, 2, 3, 4, 5, 6]
VAL = TEST = 5000


def job_dir(prefix, root="darawork/data/garnet_hw"):
    (d,) = [p for p in Path(root).glob(f"*_{prefix}*") if p.is_dir()]
    return d


def slices(n, val=VAL):
    return slice(0, n - val - TEST), slice(n - val - TEST, n - TEST), slice(n - TEST, n)


def circuits(prefixes, basis):
    """(session, rounds, model circuit, dets, obs) for every listed job."""
    for prefix in prefixes:
        data, _ = hardware_data(job_dir(prefix))
        for r in ROUNDS:
            ideal, circuit, dets, obs = data[basis, r]
            yield prefix, r, circuit, dets, obs


def decode(config, basis, split, names, *, jobs=None, val=VAL):
    """errors[name][session][rounds] on the requested slice; decoders trained on the train slice.

    jobs/val: evaluate other jobs (e.g. fresh confirmation runs with val=0).
    """
    errors, timing = {n: {} for n in names}, {n: 0.0 for n in names}
    for session, r, circuit, dets, obs in circuits(jobs or CONFIGS[config], basis):
        tr, va, te = slices(len(dets), val)
        part = {"validation": va, "test": te}[split]
        td, to, d, o = dets[tr], obs[tr], dets[part], obs[part]
        model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        fitted, _, _ = choose_dem(circuit, td, to)
        preds = {}

        def run(name, fn):
            if name in names:
                t0 = time.perf_counter(); preds[name] = fn(); timing[name] += time.perf_counter() - t0
        run("corr. MWPM (model)", lambda: MatchingDecoder(model).decode(d))
        run("corr. MWPM (fit)", lambda: MatchingDecoder(fitted).decode(d))
        run("ensemble x21 (model)", lambda: EnsembleDecoder(model).decode(d))
        run("ensemble x21 (fit)", lambda: EnsembleDecoder(fitted).decode(d))
        run("LUT + corr. MWPM (fit)", lambda: LUTDecoder(td, to, MatchingDecoder(fitted)).decode(d))
        if {"TN (fit)", "TN + data counts (fit)"} & set(names):
            t0 = time.perf_counter()
            P = tn(fitted, 3, None).probabilities(d)
            p = P[:, 1] / np.maximum(P.sum(1), 1e-300)
            dt = time.perf_counter() - t0
            if "TN (fit)" in names:
                preds["TN (fit)"] = (p > 0.5)[:, None]; timing["TN (fit)"] += dt
            if "TN + data counts (fit)" in names:  # Same posterior as a prior, overruled by training counts.
                counts = {}
                for k, f in zip(map(bytes, np.packbits(td, axis=1)), to[:, 0]):
                    n0, f0 = counts.get(k, (0, 0)); counts[k] = (n0 + 1, f0 + int(f))
                nf = np.array([counts.get(bytes(k), (0, 0)) for k in np.packbits(d, axis=1)])
                preds["TN + data counts (fit)"] = ((nf[:, 1] + 10 * p) / (nf[:, 0] + 10) > 0.5)[:, None]
                timing["TN + data counts (fit)"] += dt
        for name, pred in preds.items():
            errors[name].setdefault(session, {})[r] = int((pred != o).any(1).sum())
    return errors, timing


def ler(errs_by_session, shots=TEST):
    """Fit over rounds 1-6 (and 1-3) with errors and shots pooled across sessions."""
    k = [sum(s[r] for s in errs_by_session.values()) for r in ROUNDS]
    n = [shots * len(errs_by_session)] * len(ROUNDS)
    f6, f3 = fit_ler(ROUNDS, k, n), fit_ler(ROUNDS[:3], k[:3], n[:3])
    return dict(eps=f6["epsilon_per_round"], ci95=f6["ci95_profile"], eps_r1_3=f3["epsilon_per_round"],
                P_L=[round(a / b, 4) for a, b in zip(k, n)],
                P_L_ci95=[[round(x, 4) for x in wilson_interval(a, b)] for a, b in zip(k, n)])


def run(bases=("X", "Z"), out="results/protocol"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = dict(written_utc=datetime.now(timezone.utc).isoformat(), protocol=__doc__, configs=CONFIGS)
    decoders = ["corr. MWPM (model)", "corr. MWPM (fit)", "ensemble x21 (model)", "ensemble x21 (fit)",
                "LUT + corr. MWPM (fit)", "TN (fit)", "TN + data counts (fit)"]
    for basis in bases:
        stage1 = {c: ler(decode(c, basis, "validation", ["corr. MWPM (fit)"])[0]["corr. MWPM (fit)"]) for c in CONFIGS}
        config = min(stage1, key=lambda c: stage1[c]["eps"])
        print(f"{basis} stage 1 (validation): " + ", ".join(f"{c}={v['eps']:.4f}" for c, v in stage1.items()) + f" -> {config}", flush=True)
        val, val_t = decode(config, basis, "validation", decoders)
        stage2 = {n: ler(val[n]) for n in decoders}
        decoder = min(stage2, key=lambda n: stage2[n]["eps"])
        print(f"{basis} stage 2 (validation): " + ", ".join(f"{n}={v['eps']:.4f}" for n, v in stage2.items()) + f" -> {decoder}", flush=True)
        test, test_t = decode(config, basis, "test", decoders)  # Test slice read once, after both choices.
        headline = ler(test[decoder])
        sessions = {s: ler({s: e}) for s, e in test[decoder].items()}
        report[basis] = dict(stage1_validation=stage1, chosen_config=config, stage2_validation=stage2,
                             chosen_decoder=decoder, test=headline, test_per_session=sessions,
                             test_all_decoders_for_transparency={n: ler(test[n]) for n in decoders},
                             us_per_shot={n: 1e6 * test_t[n] / (TEST * len(CONFIGS[config]) * len(ROUNDS)) for n in decoders})
        print(f"{basis} TEST: {config} / {decoder}: eps={headline['eps']:.4f} [{headline['ci95'][0]:.4f},{headline['ci95'][1]:.4f}] "
              f"(r1-3 {headline['eps_r1_3']:.4f}); per session " +
              ", ".join(f"{s}={v['eps']:.4f}" for s, v in sessions.items()), flush=True)
        (out / "protocol.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bases", nargs="+", default=["X", "Z"])
    p.add_argument("--out", default="results/protocol")
    a = p.parse_args(argv)
    run(a.bases, a.out)


if __name__ == "__main__":
    main()
