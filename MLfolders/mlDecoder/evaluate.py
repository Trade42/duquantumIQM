"""Scoring on frozen test shots: failures, Wilson and paired-bootstrap intervals, fits, speed."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

from challenge.decoding import fit_ler, make_matcher, wilson_interval
from .data import Dataset, results_root, sha256_bytes, unpack

BATCH_SIZES = (1, 64, 4096, 65536)
DECODE_CHUNK = 65536


# ---------------------------------------------------------------- failures and intervals

def failure_mask(predictions: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """A shot fails if any observable is mispredicted."""
    predictions, obs = np.asarray(predictions), np.asarray(obs)
    if predictions.shape != obs.shape:
        raise ValueError(f"Prediction shape {predictions.shape} != observable shape {obs.shape}")
    return np.any(predictions.astype(bool) != obs.astype(bool), axis=1)


def point(fail: np.ndarray) -> dict:
    k, n = int(fail.sum()), int(len(fail))
    return dict(failures=k, shots=n, p_l=k/n, ci95_wilson=list(wilson_interval(k, n)))


def paired(fail_a: np.ndarray, fail_b: np.ndarray, *, bootstrap: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of decoder A against B on the same shots: dP_L = P_A - P_B.

    Resampling shots with replacement only moves the two discordant counts, so the
    paired bootstrap is an exact multinomial draw over (A-only, B-only, other).
    """
    if fail_a.shape != fail_b.shape:
        raise ValueError("Paired comparison needs the same shots")
    return paired_counts(int((fail_a & ~fail_b).sum()), int((fail_b & ~fail_a).sum()), len(fail_a),
                         bootstrap=bootstrap, seed=seed)


def paired_counts(a_only: int, b_only: int, n: int, *, bootstrap: int = 10000, seed: int = 0) -> dict:
    """``paired`` from the discordant counts alone, e.g. summed over several sets decoded by both decoders."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [a_only/n, b_only/n, 1 - (a_only + b_only)/n], size=bootstrap)
    delta = (draws[:, 0] - draws[:, 1]) / n
    lo, hi = np.percentile(delta, [2.5, 97.5])
    discordant = a_only + b_only
    return dict(shots=n, a_wrong_b_right=a_only, b_wrong_a_right=b_only, delta_p_l=(a_only - b_only)/n,
                ci95_paired_bootstrap=[float(lo), float(hi)], bootstrap_samples=bootstrap,
                mcnemar_exact_p=float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0,
                verdict=("A better" if hi < 0 else "B better" if lo > 0 else "tie (interval contains 0)"))


def fits(rounds, failures, shots) -> dict:
    """Per-round LER both ways. A=1 (no SPAM) is the challenge's own form."""
    out = dict(rounds=list(map(int, rounds)), failures=list(map(int, failures)), shots=list(map(int, shots)))
    if len(set(rounds)) >= 2:
        f = fit_ler(rounds, failures, shots, fit_spam=False)
        out["no_spam"] = dict(epsilon=f["epsilon_per_round"], ci95_profile=f["ci95_profile"],
                              boundary_fit=f["boundary_fit"], form="challenge: P(r)=[1-(1-2e)^r]/2")
    if len(set(rounds)) >= 3:
        f = fit_ler(rounds, failures, shots, fit_spam=True)
        out["spam"] = dict(epsilon=f["epsilon_per_round"], ci95_profile=f["ci95_profile"],
                           amplitude=f["amplitude"], boundary_fit=f["boundary_fit"],
                           form="P(r)=[1-A(1-2e)^r]/2, A fitted")
    return out


def goodness_of_fit(rounds, failures, shots, fitted, parameters: int) -> dict:
    """Pearson chi-square and deviance of a binomial fit, with p-values on #points - #parameters dof."""
    from scipy.stats import chi2
    from scipy.special import xlogy
    k, n, p = (np.asarray(v, dtype=float) for v in (failures, shots, fitted))
    expected = n * p
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = float(np.sum(np.where(expected > 0, (k - expected)**2 / (expected * (1 - p)), 0.)))
    deviance = float(2 * np.sum(xlogy(k, k / np.maximum(expected, 1e-300)) +
                                xlogy(n - k, (n - k) / np.maximum(n - expected, 1e-300))))
    dof = len(k) - parameters
    return dict(pearson_chi2=pearson, deviance=deviance, dof=dof,
                pearson_p=float(chi2.sf(pearson, dof)) if dof > 0 else None,
                deviance_p=float(chi2.sf(deviance, dof)) if dof > 0 else None)


def eps_mle(rounds, failures, shots) -> np.ndarray:
    """Vectorised no-SPAM MLE of epsilon for many count vectors (rows of ``failures``).

    For bootstraps only: a log grid then golden-section refinement, checked against
    ``fit_ler`` in tests. Rows with no failures give 0.
    """
    from scipy.special import xlogy
    r = np.asarray(rounds, dtype=float)
    k = np.atleast_2d(np.asarray(failures, dtype=float))
    n = np.broadcast_to(np.asarray(shots, dtype=float), k.shape)

    def loglik(eps):  # eps (B, G) -> (B, G)
        p = .5 * (1 - np.power(1 - 2*eps[..., None], r))
        p = np.clip(p, 1e-300, 1 - 1e-16)
        return np.sum(xlogy(k[:, None, :], p) + xlogy(n[:, None, :] - k[:, None, :], 1 - p), axis=-1)

    grid = np.concatenate([[0.], np.geomspace(1e-8, .4999, 400)])
    ll = loglik(np.broadcast_to(grid, (len(k), len(grid))))
    best = np.argmax(ll, axis=1)
    lo = grid[np.maximum(best - 1, 0)]
    hi = grid[np.minimum(best + 1, len(grid) - 1)]
    golden = (np.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - golden*(hi - lo), lo + golden*(hi - lo)
        better = loglik(np.stack([a, b], 1))
        left = better[:, 0] > better[:, 1]
        hi, lo = np.where(left, b, hi), np.where(left, lo, a)
    eps = (lo + hi) / 2
    return np.where(k.sum(axis=1) == 0, 0., eps)


def fit_free_amplitude(rounds, failures, shots, *, a_max: float = 2.0) -> dict:
    """Sensitivity fit: 2 P_L = 1 - A (1 - 2 eps)^R with A free in (0, a_max], so A > 1 is allowed.

    Not the team's fit_ler (which bounds A <= 1). Profile likelihood over eps, as fit_ler does,
    and a likelihood-ratio test of A > 1 against the A <= 1 optimum.
    """
    from scipy.optimize import brentq, minimize_scalar
    from scipy.special import xlogy
    from scipy.stats import chi2
    r, k, n = (np.asarray(v, dtype=float) for v in (rounds, failures, shots))

    def nll(eps, a):
        p = .5 * (1 - a * np.power(1 - 2*eps, r))
        if np.any(p < 0) or np.any((p == 0) & (k > 0)) or np.any(p >= 1):
            return np.inf
        return float(-np.sum(xlogy(k, p) + xlogy(n - k, 1 - p)))

    def best_a(eps, upper=a_max):
        top = min(upper, 1 / np.power(1 - 2*eps, r.min()))  # Keeps P_L >= 0 at the smallest R.
        fit = minimize_scalar(lambda a: nll(eps, a), bounds=(1e-9, top), method="bounded", options={"xatol": 1e-10})
        return min(((float(fit.x), fit.fun), (top, nll(eps, top))), key=lambda t: t[1])

    def profile(eps, upper=a_max):
        return best_a(eps, upper)[1]

    def optimum(upper):
        grid = np.concatenate([[0.], np.geomspace(1e-7, .49, 240)])
        values = [profile(e, upper) for e in grid]
        i = int(np.argmin(values))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        fit = minimize_scalar(lambda e: profile(e, upper), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        return min(((float(fit.x), fit.fun), (float(grid[i]), values[i])), key=lambda t: t[1])

    eps, best = optimum(a_max)
    amp = best_a(eps)[0]
    _, bounded = optimum(1.0)
    threshold = chi2.ppf(.95, 1) / 2
    gap = lambda e: profile(e) - best - threshold
    lo = 0. if eps == 0 or gap(0.) <= 0 else brentq(gap, 0., eps)
    hi = .49 if gap(.49) <= 0 else brentq(gap, eps, .49)
    fitted = .5 * (1 - amp * np.power(1 - 2*eps, r))
    lr = max(0., 2 * (bounded - best))
    # A (1-2e)^R = (1-2e)^(R-R0): A > 1 is the curve starting R0 rounds late (boundary rounds), not SPAM.
    offset = float(np.log(amp) / -np.log(1 - 2*eps)) if 0 < eps < .5 else None
    return dict(epsilon=eps, ci95_profile=[lo, hi], amplitude=amp, amplitude_bound=a_max, round_offset=offset,
                fitted_P_L=fitted.tolist(), goodness_of_fit=goodness_of_fit(r, k, n, fitted, 2),
                lr_test_a_above_1=dict(statistic=lr, p_value=float(chi2.sf(lr, 1)) if amp > 1 else 1.0,
                                       note="likelihood ratio, free A in (0, 2] against A <= 1; chi2(1) reference"),
                form="SENSITIVITY: 2 P_L = 1 - A (1 - 2 eps)^R, A in (0, 2] (mlDecoder, not fit_ler)")


def sensitivity_fits(rows) -> dict:
    """The A=1 form on R >= 3 and R >= 5 (first rounds dropped), and the free-amplitude fit."""
    out = {}
    for r_min in (3, 5):
        sub = [row for row in rows if row[0] >= r_min]
        if len({row[0] for row in sub}) >= 2:
            rounds, shots, failures = [s[0] for s in sub], [s[1] for s in sub], [s[2] for s in sub]
            f = fit_ler(rounds, failures, shots, fit_spam=False)
            out[f"a1_r_min_{r_min}"] = dict(rounds=rounds, epsilon=f["epsilon_per_round"], ci95_profile=f["ci95_profile"],
                goodness_of_fit=goodness_of_fit(rounds, failures, shots, f["fitted_probabilities"], 1),
                form=f"SENSITIVITY: challenge form (A = 1) on R >= {r_min} only")
    if len({row[0] for row in rows}) >= 3:
        out["free_amplitude"] = fit_free_amplitude([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows])
    return out


def trusted_fit(entry: dict, alpha: float = .05) -> str | None:
    """The first fit, in order of preference, whose Pearson p-value is acceptable (>= alpha)."""
    sens = entry.get("sensitivity", {})
    for name, fit in (("A=1, all R", entry.get("no_spam")), ("free A, all R", sens.get("free_amplitude")),
                      ("A=1, R>=3", sens.get("a1_r_min_3")), ("A=1, R>=5", sens.get("a1_r_min_5"))):
        if fit and fit["goodness_of_fit"]["pearson_p"] is not None and fit["goodness_of_fit"]["pearson_p"] >= alpha:
            return name
    return None


# ---------------------------------------------------------------- the LER record (steps 03 and 04)

LER_COLUMNS = ("decoder", "d", "basis", "noise", "p", "rounds", "shots", "failures", "P_L", "wilson_lo", "wilson_hi",
               "calibration", "group")
REFERENCE = "mwpm_corr"


def cal_label(calibration) -> str:
    """Short label of a spec's calibration: '' (none), the synthetic seed, or a device patch's slug."""
    from .hardware import DeviceNoise, is_device
    if not calibration:
        return ""
    return DeviceNoise.parse(calibration).slug() if is_device(calibration) else str(calibration).replace(":", "")


def point_decoders(pt: dict) -> list[dict]:
    """The learned decoders scored on a point: 'decoder' (one backend) and/or 'decoders' (several)."""
    return ([pt["decoder"]] if "decoder" in pt else []) + list(pt.get("decoders", []))


def _ler_groups(result: dict) -> dict:
    """{(p, noise, d, basis, calibration, group): {decoder: [(R, shots, failures, both, only_this, only_ref)]}}.

    ``group`` (mixture runs: the device patch and basis a model was trained for) keeps two models'
    scores on a shared set, e.g. the plain leakage model's, apart.
    """
    groups = {}
    for pt in result["points"]:
        s = pt["spec"]
        key = (s["p"], s["noise"], s["distance"], s["basis"], s.get("calibration") or "", pt.get("group") or "")
        decs = groups.setdefault(key, {})
        ref = pt["references"][REFERENCE]["point"]
        decs.setdefault(REFERENCE, []).append((s["rounds"], ref["shots"], ref["failures"], ref["failures"], 0, 0))
        pairs = [("mwpm", pt["references"]["mwpm"]["point"], pt["paired_mwpm_vs_mwpm_corr"])]
        for name, paired in pt.get("paired_refs_vs_mwpm_corr", {}).items():
            pairs.append((name, pt["references"][name]["point"], paired))
        for dec in point_decoders(pt):
            pairs.append((dec["name"], dec["point"], dec["paired_vs_mwpm_corr"]))
        for name, value, pr in pairs:
            both = value["failures"] - pr["a_wrong_b_right"]
            decs.setdefault(name, []).append((s["rounds"], value["shots"], value["failures"], both,
                                              pr["a_wrong_b_right"], pr["b_wrong_a_right"]))
    for decs in groups.values():
        for rows in decs.values():
            rows.sort()
    return groups


def _fit_both(rows) -> dict:
    rounds, shots, failures = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    out = {}
    for name, spam, params in (("no_spam", False, 1), ("spam", True, 2)):
        if len(set(rounds)) < (3 if spam else 2):
            continue
        f = fit_ler(rounds, failures, shots, fit_spam=spam)
        out[name] = dict(epsilon=f["epsilon_per_round"], ci95_profile=f["ci95_profile"], amplitude=f["amplitude"],
                         boundary_fit=f["boundary_fit"], fitted_P_L=f["fitted_probabilities"],
                         goodness_of_fit=goodness_of_fit(rounds, failures, shots, f["fitted_probabilities"], params),
                         form=("challenge: 2 P_L = 1 - (1 - 2 eps)^R (A = 1)" if not spam
                               else "2 P_L = 1 - A (1 - 2 eps)^R, A fitted"))
    return out


def ler_record(result: dict, *, bootstrap: int = 2000, spam_bootstrap: int = 200, seed: int = 0) -> dict:
    """Points, both fits with goodness of fit, ratio to correlated MWPM, and Lambda across d."""
    rng = np.random.default_rng(seed)
    groups = _ler_groups(result)
    points, fits_out = [], []
    eps_boot = {}  # (p, noise, basis, calibration, decoder, d) -> no-SPAM bootstrap draws (independent resampling)
    for (p, noise, d, basis, cal, grp), decs in sorted(groups.items()):
        ref_rows = decs[REFERENCE]
        ref_fit = _fit_both(ref_rows)
        for name, rows in sorted(decs.items()):
            for rounds, n, k, *_ in rows:
                lo, hi = wilson_interval(k, n)
                points.append(dict(decoder=name, d=d, basis=basis, noise=noise, p=p, rounds=rounds, shots=n,
                                   failures=k, P_L=k/n, wilson_lo=lo, wilson_hi=hi, calibration=cal_label(cal),
                                   group=grp))
            rounds = [r[0] for r in rows]
            ref_k = {r[0]: r[2] for r in ref_rows}
            entry = dict(decoder=name, d=d, basis=basis, noise=noise, p=p, calibration=cal or None,
                         calibration_label=cal_label(cal), group=grp, rounds=rounds,
                         distinct_rounds=len(set(rounds)), enough_rounds=len(set(rounds)) >= 5,
                         min_reference_failures=min(ref_k.values()),
                         reference_shortfall={str(r): 100 - k for r, k in ref_k.items() if k < 100},
                         **_fit_both(rows), sensitivity=sensitivity_fits(rows))
            entry["trusted_fit"] = trusted_fit(entry)
            if "no_spam" in entry:
                n_arr = np.array([r[1] for r in rows])
                draws = rng.binomial(n_arr, np.array([r[2] for r in rows]) / n_arr, size=(bootstrap, len(rows)))
                eps_boot[(p, noise, basis, cal, grp, name, d)] = (entry["no_spam"]["epsilon"], eps_mle(rounds, draws, n_arr))
            if name != REFERENCE and "no_spam" in entry:
                cats = np.array([[r[3], r[4], r[5], r[1] - r[3] - r[4] - r[5]] for r in rows], dtype=float)
                n_arr = np.array([r[1] for r in rows])
                draws = np.stack([rng.multinomial(n_arr[i], cats[i] / n_arr[i], size=bootstrap) for i in range(len(rows))], 1)
                k_a, k_b = draws[..., 0] + draws[..., 1], draws[..., 0] + draws[..., 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = eps_mle(rounds, k_a, n_arr) / eps_mle(rounds, k_b, n_arr)
                finite = ratios[np.isfinite(ratios)]
                e_ref = ref_fit["no_spam"]["epsilon"]
                entry["ratio_to_mwpm_corr"] = dict(
                    no_spam=entry["no_spam"]["epsilon"] / e_ref if e_ref > 0 else None,
                    no_spam_ci95_paired_bootstrap=(np.percentile(finite, [2.5, 97.5]).tolist()
                                                   if len(finite) > .9*bootstrap else None),
                    spam=(entry["spam"]["epsilon"] / ref_fit["spam"]["epsilon"]
                          if "spam" in entry and ref_fit.get("spam", {}).get("epsilon") else None),
                    bootstrap_samples=bootstrap,
                    method="paired: shots resampled jointly for both decoders at each R; no-SPAM refit per sample")
                if "spam" in entry and spam_bootstrap:
                    spam_ratios = []
                    for a_row, b_row in zip(k_a[:spam_bootstrap], k_b[:spam_bootstrap]):
                        e_a = fit_ler(rounds, a_row, n_arr, fit_spam=True)["epsilon_per_round"]
                        e_b = fit_ler(rounds, b_row, n_arr, fit_spam=True)["epsilon_per_round"]
                        if e_b > 0:
                            spam_ratios.append(e_a / e_b)
                    entry["ratio_to_mwpm_corr"].update(
                        spam_ci95_paired_bootstrap=(np.percentile(spam_ratios, [2.5, 97.5]).tolist()
                                                    if len(spam_ratios) > .9*spam_bootstrap else None),
                        spam_bootstrap_samples=spam_bootstrap,
                        spam_method=("REDUCED paired bootstrap: the first resamples of the same paired draws, "
                                     "refit with fit_ler(fit_spam=True), A free"))
            fits_out.append(entry)
    lambdas = []
    for (p, noise, basis, cal, grp, name, d), (eps, draws) in sorted(eps_boot.items()):
        nxt = eps_boot.get((p, noise, basis, cal, grp, name, d + 2))
        if nxt is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = draws / nxt[1]
        finite = boot[np.isfinite(boot)]
        lambdas.append(dict(decoder=name, basis=basis, noise=noise, p=p, calibration=cal or None, d_from=d, d_to=d + 2,
                            lambda_no_spam=eps / nxt[0] if nxt[0] > 0 else None,
                            ci95_bootstrap=np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) > .9*bootstrap else None,
                            method="independent binomial resampling of each point; no-SPAM fits"))
    label = result.get("label", "SIMULATED")
    return dict(points=points, fits=fits_out, lambdas=lambdas, reference=REFERENCE, bootstrap=bootstrap,
                spam_bootstrap=spam_bootstrap, label=label, simulated=label == "SIMULATED",
                step03=result.get("step03") or ("Pauli models: Stim's detector sampler on the noisy circuit; leakage "
                "models: challenge.leakage.sample_leakage raw measurements, then the ideal circuit's m2d converter "
                "(readout=asym: readout flips by the value read, then m2d). "
                "Decoded per shot; a shot fails if ANY observable is wrong."),
                rules="Fixed shots (no optional stopping). Points with fewer than 100 reference failures are listed.")


def plot_ler_panel(ax, fits_by_decoder: dict, points: list[dict], title: str) -> None:
    """P_L against R with Wilson bars, the no-SPAM (solid) and SPAM (dashed) fits; MWPM in grey."""
    colors = {REFERENCE: "0.35", "mwpm": "0.65", "mwpm_corr_uniform": "0.8"}
    palette = iter(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    rmax = max(p["rounds"] for p in points)
    grid = np.linspace(.5, rmax + .5, 200)
    for name in sorted(fits_by_decoder, key=lambda x: (x == REFERENCE, x.startswith("mwpm"), x)):
        fit = fits_by_decoder[name]
        color = colors.get(name) or next(palette)
        pts = sorted((p for p in points if p["decoder"] == name), key=lambda p: p["rounds"])
        r = np.array([p["rounds"] for p in pts])
        y = np.array([p["P_L"] for p in pts])
        err = [y - np.array([p["wilson_lo"] for p in pts]), np.array([p["wilson_hi"] for p in pts]) - y]
        label = {REFERENCE: "correlated MWPM", "mwpm": "MWPM", "mwpm_corr_uniform": "corr. MWPM, uniform weights"}.get(name, name)
        ax.errorbar(r, y, yerr=err, fmt="o", ms=3, capsize=2, color=color, label=label)
        for form, style in (("no_spam", "-"), ("spam", "--")):
            if form in fit:
                eps, amp = fit[form]["epsilon"], fit[form]["amplitude"]
                ax.plot(grid, .5*(1 - amp*(1 - 2*eps)**grid), style, lw=1, color=color,
                        label=f"  {'A=1' if form == 'no_spam' else f'A={amp:.3f}'}: ε={eps:.2e}")
    ax.set_xlabel("rounds R")
    ax.set_ylabel("logical error probability P_L")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=5.5, loc="upper left")


def write_ler(run_dir: Path, result: dict, **options) -> dict:
    """results/<backend>/<run-id>/ler/: ler_points.csv, ler_fits.json and one PNG per (d, basis, noise, p)."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    record = ler_record(result, **options)
    out = Path(run_dir) / "ler"
    out.mkdir(exist_ok=True)
    with (out / "ler_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LER_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in record["points"]:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    (out / "ler_fits.json").write_text(json.dumps({k: v for k, v in record.items() if k != "points"}, indent=2) + "\n",
                                       encoding="utf-8")
    keys = sorted({(f["d"], f["basis"], f["noise"], f["p"], f["calibration_label"], f["group"]) for f in record["fits"]})
    for d, basis, noise, p, cal, grp in keys:
        sel = (d, basis, noise, p, cal, grp)
        fits_by = {f["decoder"]: f for f in record["fits"]
                   if (f["d"], f["basis"], f["noise"], f["p"], f["calibration_label"], f["group"]) == sel}
        pts = [q for q in record["points"] if (q["d"], q["basis"], q["noise"], q["p"], q["calibration"], q["group"]) == sel]
        fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
        what = (f"{record['label']} d={d} {basis} memory, " + (f"{cal}" if record["label"].startswith("HARDWARE")
                else f"noise model '{noise}' p={p:g}" + (f", {cal}" if cal else "")) + " (no reset, CZ)")
        plot_ler_panel(ax, fits_by, pts, what)
        fig.savefig(out / f"ler_{d}_{basis}_{noise}_p{p:g}{'_' + cal if cal else ''}{'_' + grp if grp else ''}.png", dpi=120)
        plt.close(fig)
    return record


# ---------------------------------------------------------------- decoding a dataset

def decode_dataset(decoder, dataset: Dataset, *, chunk: int = DECODE_CHUNK, **kwargs) -> tuple[np.ndarray, float]:
    """Predictions (N, num_obs) bool, in shot order, plus decode seconds (chunks unpacked on demand)."""
    whole = getattr(decoder, "decode_dataset", None)
    if whole is not None:                       # Recurrent decoders: they read the Dataset (and its raw record).
        started = time.perf_counter()
        out = whole(dataset)
        return out, time.perf_counter() - started
    packed_path = getattr(decoder, "decode_packed", None)
    out = np.empty(dataset.obs.shape, dtype=bool)
    seconds = 0.
    for start, stop, packed in dataset.batches(chunk):
        if packed_path is None:
            dets = unpack(packed, dataset.num_detectors)
        started = time.perf_counter()
        out[start:stop] = (packed_path(packed, **kwargs) if packed_path is not None
                           else decoder.decode_batch(dets, **kwargs))
        seconds += time.perf_counter() - started
    return out, seconds


class Matcher:
    """The team's PyMatching baseline in the backend shape (correlated or not)."""

    def __init__(self, circuit, correlated: bool):
        started = time.perf_counter()
        self.matcher = make_matcher(circuit, correlated=correlated)
        self.build_seconds = time.perf_counter() - started
        self.correlated = correlated
        self.name = "mwpm_corr" if correlated else "mwpm"
        self.num_detectors = circuit.num_detectors

    def decode_batch(self, det_events, *, enable_correlations=None):
        det_events = np.asarray(det_events, dtype=np.uint8)
        if det_events.ndim != 2 or det_events.shape[1] != self.num_detectors:
            raise ValueError(f"Expected (shots, {self.num_detectors}) detection events, got {det_events.shape}")
        return self.matcher.decode_batch(det_events, enable_correlations=self.correlated).astype(bool)

    def decode_packed(self, packed):
        return self.matcher.decode_batch(packed, bit_packed_shots=True, enable_correlations=self.correlated).astype(bool)


UNIFORM_P = 0.001   # "MWPM with uniform weights": correlated MWPM on the team's uniform model at this p.


def uniform_prior(dataset: Dataset):
    """The dataset's ideal circuit under the team's uniform model (the uniform-weights decoder prior)."""
    from challenge.noise_models import NoiseModel, add_noise
    return add_noise(dataset.circuit.without_noise(), NoiseModel.uniform(UNIFORM_P))


def reference(dataset: Dataset, correlated: bool, *, uniform: bool = False) -> dict:
    """MWPM predictions on a frozen set, cached beside it (keyed by its hashes).

    ``uniform``: correlated MWPM whose weights come from the uniform model (same detectors),
    not from the dataset's own noisy circuit.
    """
    name = "mwpm_corr_uniform" if uniform else "mwpm_corr" if correlated else "mwpm"
    directory = results_root() / "datasets" / dataset.manifest["dataset_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ref_{name}.npz"
    meta_path = directory / f"ref_{name}.json"
    stamp = dict(dets_sha256=dataset.manifest["dets_sha256"], circuit_sha256=dataset.manifest["circuit_sha256"],
                 pymatching=importlib.metadata.version("pymatching"), stim=importlib.metadata.version("stim"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["stamp"] == stamp:
            with np.load(path) as data:
                pred = unpack(data["pred_packed"], dataset.obs.shape[1]).astype(bool)
            if sha256_bytes(pred) == meta["pred_sha256"]:
                return dict(meta, predictions=pred, fail=failure_mask(pred, dataset.obs))
    prior = uniform_prior(dataset) if uniform else dataset.circuit
    matcher = Matcher(prior, correlated or uniform)
    pred, seconds = decode_dataset(matcher, dataset)
    meta = dict(name=name, decoder=("correlated MWPM" if correlated or uniform else "MWPM")
                + " (challenge.decoding.make_matcher)" + (f", weights from uniform p={UNIFORM_P}" if uniform else ""),
                stamp=stamp, build_seconds=matcher.build_seconds, decode_seconds=seconds, threads=1,
                pred_sha256=sha256_bytes(pred), **point(failure_mask(pred, dataset.obs)))
    np.savez_compressed(path, pred_packed=np.packbits(pred.astype(np.uint8), axis=1, bitorder="little"))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dict(meta, predictions=pred, fail=failure_mask(pred, dataset.obs))


# ---------------------------------------------------------------- speed

def benchmark(decode, dets: np.ndarray, *, batch_sizes=BATCH_SIZES, min_seconds: float = .2,
              min_calls_b1: int = 300, max_call_seconds: float = 4., sync=None) -> list[dict]:
    """Warmed throughput and latency per batch size; batch 1 cycles over distinct shots.

    A batch size whose call would take longer than ``max_call_seconds`` (extrapolated
    from the previous size's per-shot time) is skipped and marked as such.
    """
    rows, n, per_shot = [], len(dets), None
    for b in batch_sizes:
        if b > n:
            rows.append(dict(batch=b, skipped=f"batch larger than the {n} available shots"))
            continue
        if per_shot is not None and per_shot * b > max_call_seconds:
            rows.append(dict(batch=b, skipped=f"estimated {per_shot*b:.1f} s per call > {max_call_seconds} s"))
            continue
        batches = [np.ascontiguousarray(dets[i:i + b]) for i in range(0, n - b + 1, b)][:max(64, 4096 // b)]
        decode(batches[0])  # Warm-up.
        if sync:
            sync()
        times, calls = [], 0
        total_started = time.perf_counter()
        while (time.perf_counter() - total_started < min_seconds or calls < 2
               or (b == 1 and calls < min_calls_b1)):
            started = time.perf_counter()
            decode(batches[calls % len(batches)])
            if sync:
                sync()
            times.append(time.perf_counter() - started)
            calls += 1
        times = np.array(times)
        per_shot = float(times.sum() / (b * calls))
        rows.append(dict(batch=b, calls=calls, seconds=float(times.sum()),
                         shots_per_second=float(b*calls/times.sum()),
                         latency_per_shot_us=float(1e6*times.sum()/(b*calls)),
                         latency_per_call_us=float(1e6*np.median(times)),
                         latency_per_call_p99_us=float(1e6*np.percentile(times, 99))))
    return rows


# ---------------------------------------------------------------- provenance

def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def git_state() -> dict:
    """The commit, branch, and whether this package or the team's challenge/ has uncommitted changes."""
    package = Path(__file__).resolve().parent
    def run(*args):
        return subprocess.run(["git", "-C", str(package), *args], capture_output=True, text=True).stdout.strip()
    top = run("rev-parse", "--show-toplevel")
    paths = [str(package)] + ([str(Path(top) / "challenge")] if top else [])
    return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                package=Path(package).relative_to(top).as_posix() if top else package.name,
                dirty=bool(run("status", "--porcelain", "--", *paths)))


def environment(torch_needed: bool = False) -> dict:
    env = dict(python=platform.python_version(), platform=platform.platform(), cpu=cpu_name(),
               logical_cpus=os.cpu_count(),
               packages={k: importlib.metadata.version(k) for k in ("stim", "pymatching", "numpy", "scipy")})
    if torch_needed:
        import torch
        env["packages"]["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = dict(name=props.name, vram_mib=props.total_memory // 2**20,
                              capability=f"{props.major}.{props.minor}")
    return env
