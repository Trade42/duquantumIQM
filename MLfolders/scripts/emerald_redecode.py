"""emerald-redecode: Emerald's batch (E1, E4) re-decoded with its compiled 1,036 ns readout window.

    uv run python scripts/emerald_redecode.py select    # B1+B2 only: the exact rebuild, both decodes, the choice
    uv run python scripts/emerald_redecode.py report    # needs the committed selection: B3, records, fits, X/Z, ratios
    uv run python scripts/emerald_redecode.py doc       # docs/emerald-redecode.md from the two JSON files

The batch decoded every Emerald run with ``device.hardware_noise(cal, spec)`` while Emerald had no compile
file, so its readout window fell back on ``measure_ns``, 360 ns (``fallback``). emerald-compile (main
``53b2a19``) added ``data/calibration/emerald/pulse/compile.23e12f6b-….json``, and the same call now gives the
compiled window, 1,036 ns on every eA and eB ancilla (``compiled``). ``fallback`` is that call with
``windows=None``, which emerald-compile pins byte for byte to the batch's prior.

Order (the brief's Must 1 and 2): ``select`` opens only B1 and B2 (it refuses a ``b3_`` path). Every run is
rebuilt from its committed raw memory and checked before anything is trusted: the native circuit equals
``placement.json``'s, the memory equals the dataset's ``raw_meas.npy``, the detection events and observable
equal the dataset's, and the fallback prior gives back the dataset's decoder note, its DEM size and its
per-shot ``pm_fail`` exactly. Then the rule below chooses the prior per patch and basis on the validation
shots of B1+B2, and the fit model on B1+B2; both are written to ``ler/summary/redecode_selection.json``,
which is committed before ``report`` runs. ``report`` refuses unless that file is committed and unchanged,
and its rules equal this script's.

Records: one new LER record per (experiment, basis, patch, [delay,] batch) under
``ler/<exp>/<batch run id>_c1036/``, written by ``inforecord.write_ler`` and tagged as ``batch_run`` tags
its own, with ``priors: hardware_noise compiled-1036ns (53b2a19)``. The suffix keeps them out of
``batch_run.read_points``'s ``*_b[123]`` glob. The batch's records, datasets and derived files are read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import batch_run as B  # noqa: E402

from dqiqm import decode as D  # noqa: E402
from dqiqm import inforecord as IR  # noqa: E402
from dqiqm import stats  # noqa: E402
from dqiqm.experiments import circuits as C  # noqa: E402
from dqiqm.experiments import device as dev  # noqa: E402
from dqiqm.experiments import emerald as EM  # noqa: E402
from dqiqm.experiments import suite as SU  # noqa: E402

ROOT = EM.RECORD
SUMMARY = ROOT / "ler" / "summary"
SELECTION = SUMMARY / "redecode_selection.json"
REPORT = SUMMARY / "redecode_report.json"
POINTS = SUMMARY / "redecode_E1_points.csv"
DOC = REPO / "docs" / "emerald-redecode.md"
MAIN_SHA = "53b2a19"
PRIORS = {"fallback": B.PRIORS, "compiled": f"hardware_noise compiled-1036ns ({MAIN_SHA})"}
WINDOWS = {"fallback": None, "compiled": "auto"}
WINDOW_NS = {"fallback": 360.0, "compiled": 1036.0}
SUFFIX = "_c1036"
SEED, MODEL_SHOTS, BOOT = 2026, 20_000, 1000
QB10 = "QB10"

RULE = ("Per patch and basis: pool the validation shots of B1 and B2 (shots.csv split = validation, 20% per job, "
        "checked against suite.split) over every E1 round count; decode them under both priors; the prior with fewer "
        "failures is chosen; an exact tie chooses compiled. McNemar (exact binomial on the discordant shots) is "
        "reported beside it, for information, not as the rule. B3 is never opened.")
FIT_RULE = ("Per patch and basis, on B1+B2 (all shots, finetune and validation) decoded with the chosen prior: fit "
            "A = 1 on R >= 2, and the first-round model 2 P_L = 1 - (1 - 2 eps_1)(1 - 2 eps)^(R - 1) on every R; "
            "the one with the lower chi2/dof wins that pool; the model that wins more of the four pools is used for "
            "all (a 2-2 tie: the lower summed chi2/dof).")
E4_RULE = ("E4 has no test batch and runs on eA only: it takes eA's per-basis E1 choice, and both priors are "
           "reported for it.")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def need(ok: bool, msg: str) -> None:
    if not ok:
        raise SystemExit(f"STOP: {msg}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def note(prior: str, cal: Any, info: dict[str, Any]) -> str:
    """``batch_run.decoder_priors``'s note, with the prior's own label."""
    return (f"priors: {PRIORS[prior]}. dqiqm.experiments.device.hardware_noise(calibration set {cal.calibration_set_id}, "
            f"spec): calibrated_noise with the {info['idle_channel']} idle over {info['idle_ns']:g} ns per mid-circuit "
            f"readout layer (readout window {info['readout_window_ns']:g} ns from {info['readout_window_source']}, plus a "
            f"{info['delay_ns']:g} ns delay), T2 {info['t2']}, final readout {info['final_readout']}")


# ---------------------------------------------------------------------------
# The runs, rebuilt and decoded.
# ---------------------------------------------------------------------------


def run_dirs(batches: tuple[int, ...]) -> list[Path]:
    """The committed E1 and E4 datasets of ``batches`` (E4 is in B2 only), chosen by name before any is opened."""
    out = [ds for exp in ("E1", "E4") for b in batches for ds in sorted((ROOT / "datasets" / exp).glob(f"b{b}_*"))]
    allowed = {f"b{b}" for b in batches}
    need(all(p.name.split("_", 1)[0] in allowed for p in out), f"a dataset outside batches {batches}")
    return out


def group_id(spec: Any, batch: int) -> tuple[str, str]:
    exp, run_id, _ = B.ler_group(spec, batch)
    return exp, run_id


def committed_failures(exp: str, run_id: str, name: str) -> int:
    with open(ROOT / "ler" / exp / run_id / "ler_points.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["circuit"] == name]
    need(len(rows) == 1, f"{exp}/{run_id}: {len(rows)} rows for {name}")
    return int(rows[0]["failures"])


def decode_run(ds: Path, fz: B.Frozen, cals: dict[Path, Any]) -> dict[str, Any]:
    """One run, checked against everything committed, and decoded under both priors."""
    m = json.loads((ds / "manifest.json").read_text())
    name, batch = m["circuit"], int(m["batch"])
    spec = fz.suite[name]
    need(C.native_sha256(spec.hardware) == fz.placement[name]["native_sha256"] == m["native_sha256"], f"{ds.name}: native_sha256")
    job_dir = (ROOT / m["raw"]["path"]).parent
    raw, _ = D.load_memory(job_dir, int(m["raw"]["key"][1:]), circuit=spec.stim)
    need(np.array_equal(raw, IR.load_array(ds, "raw_meas")), f"{ds.name}: memory.npz != raw_meas.npy")
    dets, obs = D.detection_events(spec.stim, raw)
    need(np.array_equal(dets, IR.load_array(ds, "dets")) and np.array_equal(obs, IR.load_array(ds, "obs")),
         f"{ds.name}: detection events or observable differ from the dataset's")
    snap = (ROOT / m["calibration"]["job"]).resolve()
    if snap not in cals:
        cals[snap] = B.live_calibration("emerald", snap)
    cal = cals[snap]
    with open(ds / "shots.csv") as f:
        rows = list(csv.DictReader(f))
    split = np.array([r["split"] for r in rows], dtype=object)
    need((split == SU.split(m["split"]["job"], batch, len(raw))).all(), f"{ds.name}: split labels differ from suite.split")
    pm_fail = np.array([r["pm_fail"] == "1" for r in rows])
    exp, gid = group_id(spec, batch)
    out: dict[str, Any] = {"dataset": ds, "run_id": ds.name, "circuit": name, "spec": spec, "batch": batch, "exp": exp,
                           "group": gid, "split": split, "dets": dets, "obs": obs[:, 0], "raw": raw, "cal": cal,
                           "snapshot": snap, "manifest": m}
    for prior in ("fallback", "compiled"):
        nm, info = dev.hardware_noise(cal, spec, windows=WINDOWS[prior])
        noisy = spec.noisy(nm)
        n_err = noisy.detector_error_model(decompose_errors=True).num_errors
        fail = D.decode(noisy, dets)[:, 0] != obs[:, 0]
        if spec.rounds > 1:
            need(info["readout_window_ns"] == WINDOW_NS[prior], f"{ds.name}: {prior} window {info['readout_window_ns']}")
        out[prior] = {"fail": fail, "info": info, "noisy": noisy, "dem_errors": int(n_err), "note": note(prior, cal, info)}
    f = out["fallback"]
    need(f["note"] == m["decoder"]["noise"], f"{ds.name}: the fallback note differs from the dataset's decoder note")
    need(f["dem_errors"] == m["decoder"]["num_dem_errors"], f"{ds.name}: DEM {f['dem_errors']} != {m['decoder']['num_dem_errors']}")
    need(np.array_equal(f["fail"], pm_fail), f"{ds.name}: the fallback decode differs from shots.csv pm_fail")
    need(int(f["fail"].sum()) == committed_failures(exp, gid, name), f"{ds.name}: failures differ from ler_points.csv")
    if "compile." not in out["compiled"]["info"]["readout_window_source"] and spec.rounds > 1:
        raise SystemExit(f"STOP: {ds.name}: the compiled prior did not read the compile file")
    return out


def decode_all(batches: tuple[int, ...]) -> list[dict[str, Any]]:
    fz = B.Frozen("emerald")
    cals: dict[Path, Any] = {}
    runs = []
    for ds in run_dirs(batches):
        runs.append(decode_run(ds, fz, cals))
    log(f"REBUILT {len(runs)} runs of batches {batches}: native, memory, detection events, split, fallback note, DEM size "
        f"and per-shot pm_fail all equal the committed records ({sum(len(r['obs']) for r in runs):,} shots)")
    return runs


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------


def mcnemar(fail_a: np.ndarray, fail_b: np.ndarray) -> dict[str, Any]:
    """Shots that fail under a only and under b only, with the exact two-sided binomial p."""
    a_only, b_only = int((fail_a & ~fail_b).sum()), int((fail_b & ~fail_a).sum())
    p = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": float(p)}


def choose(k_fallback: int, k_compiled: int) -> str:
    """The rule: fewer validation failures wins; a tie chooses compiled."""
    return "compiled" if k_compiled <= k_fallback else "fallback"


def wilson(k: int, n: int) -> list[float]:
    lo, hi = stats.wilson(k, n)
    return [float(lo), float(hi)]


def _nll(theta: np.ndarray, r: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    eps, eps1 = theta
    p = np.clip(0.5 * (1 - (1 - 2 * eps1) * (1 - 2 * eps) ** (r - 1)), 1e-15, 1 - 1e-15)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def _mle_first(r, k, n, x0) -> np.ndarray:
    res = minimize(_nll, np.asarray(x0, dtype=float), args=(r, k, n), method="L-BFGS-B",
                   bounds=[(0.0, 0.5 - 1e-9), (0.0, 0.5 - 1e-9)], options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000})
    return res.x


def fit_first_round(rounds, k, n, *, bootstrap: int = BOOT, seed: int = SEED) -> dict[str, Any]:
    """2 P_L = 1 - (1 - 2 eps_1)(1 - 2 eps)^(R - 1): a first round of its own (eps_1), then eps per round.

    The same as A free with A = (1 - 2 eps_1) / (1 - 2 eps), unbounded above 1, which ``stats.fit_eps`` caps.
    Binomial maximum likelihood; the bootstrap redraws every point as Binomial(n, k/n), seeded.
    """
    r, k, n = (np.asarray(x, dtype=float) for x in (rounds, k, n))
    singles = stats.eps_from_p(k / n, r)
    e0 = float(np.clip(np.nanmedian(singles), 1e-4, 0.45))
    best = min((_mle_first(r, k, n, x0) for x0 in ([e0, e0], [e0, max(k[0] / n[0], 1e-4)])), key=lambda x: _nll(x, r, k, n))
    eps, eps1 = map(float, best)
    p = 0.5 * (1 - (1 - 2 * eps1) * (1 - 2 * eps) ** (r - 1))
    chi2 = float(np.sum((k - n * p) ** 2 / (n * np.clip(p * (1 - p), 1e-15, None))))
    out = {"form": "F = (1-2eps_1)(1-2eps)^(n-1)", "eps": eps, "eps_1": eps1, "amplitude": (1 - 2 * eps1) / (1 - 2 * eps),
           "chi2": chi2, "dof": len(r) - 2, "chi2_per_dof": chi2 / (len(r) - 2), "bootstrap": bootstrap, "seed": seed}
    if bootstrap:
        rng = np.random.default_rng(seed)
        draws = np.array([_mle_first(r, rng.binomial(n.astype(np.int64), k / n).astype(float), n, best)
                          for _ in range(bootstrap)])
        out["eps_ci_bootstrap"] = [float(x) for x in np.percentile(draws[:, 0], [2.5, 97.5])]
        out["eps_1_ci_bootstrap"] = [float(x) for x in np.percentile(draws[:, 1], [2.5, 97.5])]
    return out


def fit_r2(r, k, n, *, bootstrap: int = BOOT) -> dict[str, Any]:
    """A = 1 on R >= 2 (``stats.fit_eps``)."""
    keep = np.asarray(r) >= 2
    f = stats.fit_eps(np.asarray(r)[keep], np.asarray(k)[keep], np.asarray(n)[keep], free_amplitude=False,
                      bootstrap=bootstrap, seed=SEED)
    return {"form": "F = (1-2eps)^n on R >= 2", "eps": f.eps, "eps_ci_bootstrap": list(f.eps_ci_bootstrap), "chi2": f.chi2,
            "dof": f.dof, "chi2_per_dof": f.chi2_per_dof, "eps_se_curvature": f.eps_se}


def pool(runs: list[dict[str, Any]], prior: str, mask=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for x in runs:
        sel = np.ones(len(x["obs"]), bool) if mask is None else mask(x)
        by[x["spec"].rounds][0] += int(x[prior]["fail"][sel].sum())
        by[x["spec"].rounds][1] += int(sel.sum())
    r = np.array(sorted(by))
    return r, np.array([by[i][0] for i in r]), np.array([by[i][1] for i in r])


def e1(runs, basis=None, patch=None, batches=None):
    return [x for x in runs if x["exp"] == "E1" and (basis is None or x["spec"].params["basis"] == basis)
            and (patch is None or x["spec"].params["patch"] == patch) and (batches is None or x["batch"] in batches)]


CASES = [(p, b) for p in ("eA", "eB") for b in ("Z", "X")]


# ---------------------------------------------------------------------------
# select: B1 and B2 only.
# ---------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> int:
    runs = decode_all((1, 2))
    table = {}
    for patch, basis in CASES:
        rs = e1(runs, basis, patch)
        val = lambda x: x["split"] == "validation"  # noqa: E731
        ff = np.concatenate([x["fallback"]["fail"][val(x)] for x in rs])
        fc = np.concatenate([x["compiled"]["fail"][val(x)] for x in rs])
        n, kf, kc = len(ff), int(ff.sum()), int(fc.sum())
        mc = mcnemar(fc, ff)
        table[f"{patch}_{basis}"] = {"patch": patch, "basis": basis, "runs": len(rs), "validation_shots": n,
                                     "fallback": {"failures": kf, "p_l": kf / n, "wilson95": wilson(kf, n)},
                                     "compiled": {"failures": kc, "p_l": kc / n, "wilson95": wilson(kc, n)},
                                     "mcnemar": {"compiled_only": mc["a_only"], "fallback_only": mc["b_only"], "p": mc["p"]},
                                     "choice": choose(kf, kc)}
        log(f"VALIDATION {patch} {basis}: n {n}; fallback {kf} ({kf / n:.4f}), compiled {kc} ({kc / n:.4f}); "
            f"compiled-only {mc['a_only']}, fallback-only {mc['b_only']}, McNemar p {mc['p']:.3f} -> {table[f'{patch}_{basis}']['choice']}")
    fits, wins = {}, defaultdict(int)
    for patch, basis in CASES:
        prior = table[f"{patch}_{basis}"]["choice"]
        r, k, n = pool(e1(runs, basis, patch), prior)
        a, b = fit_r2(r, k, n, bootstrap=0), fit_first_round(r, k, n, bootstrap=0)
        win = "A1_R>=2" if a["chi2_per_dof"] < b["chi2_per_dof"] else "first_round"
        wins[win] += 1
        fits[f"{patch}_{basis}"] = {"prior": prior, "A1_R>=2": a, "first_round": b, "winner": win}
        log(f"FIT MODEL {patch} {basis} (B1+B2, {prior}): chi2/dof A=1 R>=2 {a['chi2_per_dof']:.2f}, "
            f"first-round {b['chi2_per_dof']:.2f} -> {win}")
    if wins["A1_R>=2"] == wins["first_round"]:
        tot = {w: sum(f[w]["chi2_per_dof"] for f in fits.values()) for w in ("A1_R>=2", "first_round")}
        model = min(tot, key=tot.get)
    else:
        model = max(wins, key=wins.get)
    body = {"format": IR.FORMAT, "label": "HARDWARE", "device": "emerald", "task": "emerald-redecode",
            "priors": PRIORS, "rule": RULE, "fit_rule": FIT_RULE, "e4_rule": E4_RULE,
            "batches_opened": [1, 2], "runs_rebuilt": len(runs), "rebuild_exact": True,
            "choice": {k: v["choice"] for k, v in table.items()}, "validation": table,
            "fit_model": model, "fit_model_pools": fits, "written_at_commit": git("rev-parse", "HEAD"),
            "note": "Written by `scripts/emerald_redecode.py select` before any B3 number was computed; committed before `report`."}
    B.write_json(SELECTION, body)
    log(f"CHOICE {body['choice']}; fit model {model}; wrote {SELECTION.relative_to(REPO)}")
    return 0


def committed_selection() -> dict[str, Any]:
    rel = str(SELECTION.relative_to(REPO))
    need(bool(git("ls-files", rel)), f"{rel} is not committed: commit select's output before report")
    need(subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO).returncode == 0, f"{rel} has uncommitted changes")
    sel = json.loads(SELECTION.read_text())
    need(sel["rule"] == RULE and sel["fit_rule"] == FIT_RULE and sel["e4_rule"] == E4_RULE, "the rules differ from the selection's")
    return sel


# ---------------------------------------------------------------------------
# report: everything, with the committed choice.
# ---------------------------------------------------------------------------


def chosen_prior(sel: dict[str, Any], spec: Any) -> str:
    return sel["choice"][f"{spec.params['patch']}_{spec.params['basis']}"]


def model_numbers(x: dict[str, Any], prior: str, cache: dict[str, Any]) -> dict[str, Any]:
    """The prior's own circuit, sampled as ``batch_run.reference`` does (seed 2026, 20,000 shots), and decoded."""
    noisy = x[prior]["noisy"]
    key = hashlib.sha256(str(noisy).encode()).hexdigest()
    if key not in cache:
        raw = noisy.compile_sampler(seed=SEED).sample(MODEL_SHOTS).astype(np.uint8)
        dets, obs = D.detection_events(x["spec"].stim, raw)
        fail = D.decode(noisy, dets)[:, 0] != obs[:, 0]
        cache[key] = {"detection_fraction": float(dets.mean()), "per_t": per_t(dets, x), "p_l": float(fail.mean()),
                      "failures": int(fail.sum()), "shots": MODEL_SHOTS}
    return cache[key]


def det_meta(x: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with open(x["dataset"] / "detectors.csv") as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["round"]) for r in rows]), np.array([r["type"] for r in rows])


def per_t(dets: np.ndarray, x: dict[str, Any]) -> dict[str, float]:
    t, typ = det_meta(x)
    out = {f"t{k}": float(dets[:, t == k].mean()) for k in sorted(set(t.tolist()))}
    for ty in ("X", "Z"):
        for k in sorted(set(t[typ == ty].tolist())):
            out[f"{ty}t{k}"] = float(dets[:, (t == k) & (typ == ty)].mean())
    return out


def fits_for(r, k, n) -> dict[str, Any]:
    f = B.fit_both(r, k, n)
    f.pop("_fits", None)
    f["A1_R>=2"] = fit_r2(r, k, n)
    f["first_round"] = fit_first_round(r, k, n)
    return f


def inflated(fit: dict[str, Any]) -> tuple[float, float]:
    """eps (A = 1) and its curvature s.e. inflated by sqrt(chi2/dof) (at least 1): the batch review's method."""
    fa = fit["fixed_A"]
    return fa["eps"], fa["eps_se_curvature"] * math.sqrt(max(fa["chi2_per_dof"], 1.0))


def write_records(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list] = defaultdict(list)
    for x in runs:
        groups[(x["exp"], x["group"])].append(x)
    out = []
    for (exp, gid), rs in sorted(groups.items()):
        run_id = gid + SUFFIX
        need(not (ROOT / "ler" / exp / run_id).exists(), f"ler/{exp}/{run_id} exists; records are never overwritten")
        points = [{"circuit": x["circuit"], "dataset": B._rel(x["dataset"], ROOT), "rounds": x["spec"].rounds,  # noqa: SLF001
                   "shots": len(x["obs"]), "failures": int(x["compiled"]["fail"].sum()), "raw_failures": int(x["obs"].sum())}
                  for x in rs]
        b = rs[0]["batch"]
        split = "test" if b == SU.SPLIT["test_batch"] else "finetune+validation"
        _, _, title = B.ler_group(rs[0]["spec"], b)
        path, res = IR.write_ler(ROOT, exp, run_id, points, title=f"HARDWARE (emerald): {title}, compiled 1,036 ns window",
                                 label=run_id)
        B.tag_json(path / "ler_fits.json", label="HARDWARE", device="emerald", batch=b, split=split, priors=PRIORS["compiled"],
                   decoder="pymatching on the noisy copy of the circuit that ran", noise=rs[0]["compiled"]["note"],
                   zero_failure_points=[p["rounds"] for p in points if p["failures"] == 0],
                   redecode_of=f"ler/{exp}/{gid} (priors {PRIORS['fallback']})")
        problems = IR.check_csv(path / "ler_points.csv", "ler/ler_points.csv")
        for x in rs:
            der = ROOT / "derived" / exp / x["run_id"]
            problems += IR.check_run(x["dataset"], der)
        need(not problems, f"{exp}/{run_id}: {problems}")
        out.append({"experiment": exp, "run_id": run_id, "points": len(points), "shots": sum(p["shots"] for p in points),
                    "eps_fixed_A": res["fixed_A"]["eps"], "eps_free_A": res["free_A"]["eps"], "check": "pass"})
    return out


def cmd_report(args: argparse.Namespace) -> int:
    sel = committed_selection()
    log(f"SELECTION (committed): {sel['choice']}; fit model {sel['fit_model']}")
    runs = decode_all((1, 2, 3))
    records = write_records(runs)
    log(f"WROTE {len(records)} LER records ({SUFFIX}); each ler_points.csv passes check_csv and every dataset it names "
        "passes check_run")
    rep: dict[str, Any] = {"format": IR.FORMAT, "label": "HARDWARE", "device": "emerald", "priors": PRIORS,
                           "selection": str(SELECTION.relative_to(REPO)), "choice": sel["choice"], "fit_model": sel["fit_model"],
                           "records": records, "bootstrap": BOOT, "seed": SEED}
    # The pooled fits per prior and the chosen one.
    pools = {"finetune_b1b2": [1, 2], "test_b3": [3], "all_b1b2b3": [1, 2, 3]}
    fits: dict[str, Any] = {}
    for patch, basis in CASES:
        for prior in ("fallback", "compiled"):
            for pname, bs in pools.items():
                r, k, n = pool(e1(runs, basis, patch, bs), prior)
                fits[f"{basis}_{patch}_{pname}_{prior}"] = {"basis": basis, "patch": patch, "pool": pname, "prior": prior,
                                                           **fits_for(r, k, n)}
        log(f"fits {patch} {basis} done")
    for key in list(fits):
        f = fits[key]
        if f["prior"] == sel["choice"][f"{f['patch']}_{f['basis']}"]:
            fits[key.rsplit("_", 1)[0] + "_chosen"] = {**f, "prior": f"chosen ({f['prior']})"}
    rep["fits"] = fits
    # X against Z.
    rep["x_vs_z"] = []
    for patch in ("eA", "eB"):
        for prior in ("fallback", "compiled", "chosen"):
            for pname in ("finetune_b1b2", "test_b3", "all_b1b2b3"):
                fx, fz_ = fits[f"X_{patch}_{pname}_{prior}"], fits[f"Z_{patch}_{pname}_{prior}"]
                (ex, sx), (ez, sz) = inflated(fx), inflated(fz_)
                alt = sel["fit_model"]
                row = {"patch": patch, "prior": prior, "pool": pname, "eps_X": ex, "eps_Z": ez, "ratio": ex / ez,
                       "z_inflated": (ex - ez) / math.hypot(sx, sz),
                       f"ratio_{alt}": fx[alt]["eps"] / fz_[alt]["eps"], "per_R": []}
                for i, rr in enumerate(fx["rounds"]):
                    kx, nx, kz, nz = fx["failures"][i], fx["shots"][i], fz_["failures"][i], fz_["shots"][i]
                    wx, wz = wilson(kx, nx), wilson(kz, nz)
                    row["per_R"].append({"R": rr, "X": kx / nx, "Z": kz / nz,
                                         "verdict": "X<Z" if wx[1] < wz[0] else ("X>Z" if wx[0] > wz[1] else "overlap")})
                row["count"] = {v: sum(p["verdict"] == v for p in row["per_R"]) for v in ("X<Z", "X>Z", "overlap")}
                rep["x_vs_z"].append(row)
    # The model, sampled: its own X/Z and the per-round ratios, before (fallback) and after (compiled).
    cache: dict[str, Any] = {}
    ratio_rows, model_pl = [], defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen_model_check = 0
    for x in runs:
        hw = per_t(x["dets"], x)
        bc = json.loads((ROOT / "derived" / x["exp"] / x["run_id"] / "batch_checks.json").read_text())
        row = {"run_id": x["run_id"], "exp": x["exp"], "batch": x["batch"], "circuit": x["circuit"],
               "patch": x["spec"].params["patch"], "basis": x["spec"].params["basis"], "rounds": x["spec"].rounds,
               "delay_ns": x["spec"].params.get("delay_ns", 0.0), "hw_detection_fraction": float(x["dets"].mean()), "hw": hw}
        for prior in ("fallback", "compiled"):
            mn = model_numbers(x, prior, cache)
            row[prior] = {"detection_fraction": mn["detection_fraction"], "ratio": row["hw_detection_fraction"] / mn["detection_fraction"],
                          "ratio_per_t": {t: hw[t] / v for t, v in mn["per_t"].items() if v > 0}, "model_p_l": mn["p_l"]}
            if x["batch"] in (1, 2, 3):
                model_pl[(x["exp"], row["patch"], row["basis"], row["delay_ns"], prior)][x["spec"].rounds] = [mn["failures"], mn["shots"]]
        committed = (bc.get("model_with_idle") or {}).get("detection_fraction")
        need(committed is None or row["fallback"]["detection_fraction"] == committed,
             f"{x['run_id']}: the fallback model's detection fraction {row['fallback']['detection_fraction']} != batch_checks' {committed}")
        seen_model_check += committed is not None
        ratio_rows.append(row)
    log(f"MODEL: the fallback model's detection fraction equals batch_checks.json's on {seen_model_check} of {len(runs)} runs")
    rep["model_check"] = {"runs": len(runs), "equal_to_batch_checks": seen_model_check}
    rep["ratios"] = ratio_rows
    model_xz = []
    for patch in ("eA", "eB"):
        for prior in ("fallback", "compiled"):
            eps = {}
            for basis in ("X", "Z"):
                d = model_pl[("E1", patch, basis, 0.0, prior)]
                r = np.array(sorted(d))
                f = stats.fit_eps(r, [d[i][0] for i in r], [d[i][1] for i in r], free_amplitude=False, bootstrap=0)
                eps[basis] = f.eps
            model_xz.append({"patch": patch, "prior": prior, "eps_X": eps["X"], "eps_Z": eps["Z"], "ratio": eps["X"] / eps["Z"],
                             "shots_per_R": MODEL_SHOTS})
    rep["model_x_vs_z"] = model_xz
    # E4: the delay plus the window.
    e4 = []
    for basis in ("Z", "X"):
        for delay in (1000.0, 2000.0, 5000.0):
            rs = [x for x in runs if x["exp"] == "E4" and x["spec"].params["basis"] == basis and x["spec"].params["delay_ns"] == delay]
            row = {"basis": basis, "delay_ns": delay, "chosen": sel["choice"][f"eA_{basis}"]}
            for prior in ("fallback", "compiled"):
                r, k, n = pool(rs, prior)
                f = stats.fit_eps(r, k, n, free_amplitude=False, bootstrap=BOOT, seed=SEED)
                rat = [rw[prior]["ratio"] for rw in ratio_rows if rw["exp"] == "E4" and rw["basis"] == basis and rw["delay_ns"] == delay]
                row[prior] = {"idle_ns": WINDOW_NS[prior] + delay, "rounds": r.tolist(), "failures": k.tolist(), "shots": n.tolist(),
                              "eps": f.eps, "eps_ci_bootstrap": list(f.eps_ci_bootstrap), "chi2_per_dof": f.chi2_per_dof,
                              "detection_ratio": [min(rat), max(rat)]}
            e4.append(row)
    rep["e4"] = e4
    rep["qb10"] = qb10(runs, sel)
    B.write_json(REPORT, rep)
    write_points(runs)
    log(f"wrote {REPORT.relative_to(REPO)} and {POINTS.relative_to(REPO)}")
    return 0


def write_points(runs: list[dict[str, Any]]) -> None:
    with open(POINTS, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["basis", "patch", "batch", "rounds", "shots", "failures_fallback", "failures_compiled", "raw_failures",
                    "validation_shots", "validation_failures_fallback", "validation_failures_compiled", "circuit", "run_id"])
        for x in sorted(e1(runs), key=lambda x: (x["spec"].params["basis"], x["spec"].params["patch"], x["batch"], x["spec"].rounds)):
            v = x["split"] == "validation"
            w.writerow([x["spec"].params["basis"], x["spec"].params["patch"], x["batch"], x["spec"].rounds, len(x["obs"]),
                        int(x["fallback"]["fail"].sum()), int(x["compiled"]["fail"].sum()), int(x["obs"].sum()), int(v.sum()),
                        int(x["fallback"]["fail"][v].sum()), int(x["compiled"]["fail"][v].sum()), x["circuit"], x["run_id"]])


def qb10(runs: list[dict[str, Any]], sel: dict[str, Any]) -> dict[str, Any]:
    """eB's QB10 (an X-type ancilla): its readout bias from the raw memory, and a sensitivity decode on B1+B2 only."""
    out: dict[str, Any] = {"qubit": QB10}
    per_round = {"Z": defaultdict(lambda: [0, 0]), "X": defaultdict(lambda: [0, 0])}
    others_first = []
    for x in e1(runs, patch="eB"):
        spec = x["spec"]
        meas_q = C.measured_qubits(spec.stim)
        anc = [q for q, r in spec.roles.items() if r == "ancilla"]
        col = [j for j, q in enumerate(meas_q) if spec.physical[q] == QB10]
        for i, j in enumerate(col[:-1] if len(col) > spec.rounds else col):
            per_round[spec.params["basis"]][i + 1][0] += int(x["raw"][:, j].sum())
            per_round[spec.params["basis"]][i + 1][1] += len(x["raw"])
        if spec.params["basis"] == "Z":
            for q in anc:
                if spec.stabilizers[q].pauli == "X" and spec.physical[q] != QB10:
                    j = next(j for j, mq in enumerate(meas_q) if mq == q)
                    others_first.append(float(x["raw"][:, j].mean()))
    out["p1_by_round"] = {b: {str(k): v[0] / v[1] for k, v in sorted(d.items())} for b, d in per_round.items()}
    m_z = per_round["Z"][1][0] / per_round["Z"][1][1]
    m_x = per_round["X"][1][0] / per_round["X"][1][1]
    out["z_first_mean"], out["x_first_misread"] = m_z, m_x
    out["other_x_type_first_mean_range"] = [min(others_first), max(others_first)]
    out["asymmetry_p01_minus_p10"] = 2 * (0.5 - m_z)
    cal = runs[0]["cal"]
    c = cal.qubits[QB10]
    out["calibration"] = {k: c.get(k) for k in ("p1_given_0", "p0_given_1", "p1_given_0_terminal", "p0_given_1_terminal", "t1_ns",
                                                "error_1q", "readout_leakage")}
    p_sym = m_x + (0.5 - m_z)  # (P(0|1) + P(1|0)) / 2, with P(1|0) ~ the X-basis misread
    out["sensitivity_record_flip"] = p_sym
    sens = {}
    for basis in ("Z", "X"):
        prior = sel["choice"][f"eB_{basis}"]
        base, alt = [], []
        for x in e1(runs, basis, "eB", (1, 2)):
            spec = x["spec"]
            s = next(q for q, n in spec.physical.items() if n == QB10)
            nm, _ = dev.hardware_noise(x["cal"], spec, windows=WINDOWS[prior])
            qr = {q: dict(v) for q, v in nm.qubit_rates.items()}
            qr[s]["record_flip_probability"] = p_sym
            if "final_record_flip_probability" in qr[s]:
                qr[s]["final_record_flip_probability"] = p_sym
            noisy = spec.noisy(type(nm)(**{**{f: getattr(nm, f) for f in nm.__dataclass_fields__}, "qubit_rates": qr}))
            base.append(x[prior]["fail"])
            alt.append(D.decode(noisy, x["dets"])[:, 0] != x["obs"])
        fb, fa = np.concatenate(base), np.concatenate(alt)
        mc = mcnemar(fa, fb)
        sens[basis] = {"prior": prior, "shots": len(fb), "failures": int(fb.sum()), "failures_qb10": int(fa.sum()),
                       "qb10_only": mc["a_only"], "base_only": mc["b_only"], "mcnemar_p": mc["p"]}
    out["sensitivity_b1b2"] = sens
    out["note"] = ("Sensitivity on B1+B2 only (finetune side), never on B3: QB10's record flip set to (P(0|1) + P(1|0)) / 2 "
                   "from the raw memory, on every readout of QB10.")
    return out


# ---------------------------------------------------------------------------
# doc
# ---------------------------------------------------------------------------


def _ci(ci, nd=4) -> str:
    return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def findings(sel: dict[str, Any], rep: dict[str, Any]) -> list[str]:
    """The doc's summary, computed from the two JSON files (no notes file)."""
    f, alt, v = rep["fits"], sel["fit_model"], sel["validation"]
    worse = [k for k, x in v.items() if x["compiled"]["failures"] > x["fallback"]["failures"] and x["mcnemar"]["p"] < 0.05]
    out = [f"- **The compiled window decodes worse, so the batch's prior stays.** On finetune validation the compiled prior fails "
           f"more often on {', '.join(worse)} (McNemar p < 0.05), and "
           + "; ".join(f"on {k} {x['compiled']['failures']} against {x['fallback']['failures']} (p {x['mcnemar']['p']:.2f})"
                       for k, x in v.items() if k not in worse)
           + f"; the rule chooses fallback on "
           f"{sum(c == 'fallback' for c in sel['choice'].values())} of 4. B3 agrees after the fact: compiled ε (A = 1) is "
           + ", ".join(f"{p} {b} {f[f'{b}_{p}_test_b3_compiled']['fixed_A']['eps']:.4f} against {f[f'{b}_{p}_test_b3_fallback']['fixed_A']['eps']:.4f}"
                       for p, b in CASES) + ". The headline B3 numbers of `docs/batches.md` therefore stand.",
           "- **It fits the detection statistics far better.** Over every E1 circuit the hardware's detection fraction is "
           + "; ".join(f"{p} {b} {min(r['fallback']['ratio'] for r in rs):.2f}–{max(r['fallback']['ratio'] for r in rs):.2f}× the "
                       f"fallback model and {min(r['compiled']['ratio'] for r in rs):.2f}–{max(r['compiled']['ratio'] for r in rs):.2f}× the compiled one"
                       for p, b in CASES for rs in [[r for r in rep['ratios'] if r['exp'] == 'E1' and r['patch'] == p and r['basis'] == b]])
           + ". The bulk rounds land near 1; round 1 (t = 0, no idle before it) stays at 1.35–1.42× in the Z basis, and the "
           "last rounds of long circuits stay above 1. A better-matched model that decodes worse is garnet-timing's `b_pauli` "
           "story again: the decoder needs the right relative weights, not the right totals.",
           f"- **The fit form matters more than the prior.** The first-round model wins on all four finetune pools; on B3 it "
           f"raises ε by 17–22% over the challenge's A = 1 form (amplitude 1.07–1.12), well outside the bootstrap intervals.",
           ]
    xz = {(r["patch"], r["prior"], r["pool"]): r for r in rep["x_vs_z"]}
    m = {(r["patch"], r["prior"]): r["ratio"] for r in rep["model_x_vs_z"]}
    out.append(f"- **X against Z.** eA: X < Z under both priors and every pool (ε_X/ε_Z "
               f"{min(xz[('eA', p, q)]['ratio'] for p in ('fallback', 'compiled') for q in ('finetune_b1b2', 'test_b3')):.2f}–"
               f"{max(xz[('eA', p, q)]['ratio'] for p in ('fallback', 'compiled') for q in ('finetune_b1b2', 'test_b3')):.2f}, "
               f"z {min(xz[('eA', p, q)]['z_inflated'] for p in ('fallback', 'compiled') for q in ('finetune_b1b2', 'test_b3')):+.1f} to "
               f"{max(xz[('eA', p, q)]['z_inflated'] for p in ('fallback', 'compiled') for q in ('finetune_b1b2', 'test_b3')):+.1f} inflated), "
               f"against a model X/Z of {m[('eA', 'fallback')]:.2f} (fallback) and {m[('eA', 'compiled')]:.2f} (compiled): a real excess "
               f"of the errors a Z-basis memory sees. eB: {xz[('eB', 'fallback', 'test_b3')]['ratio']:.2f} with fallback and "
               f"{xz[('eB', 'compiled', 'test_b3')]['ratio']:.2f} compiled on B3, |z| ≤ 0.5, no round count separates, and the model "
               f"itself gives {m[('eB', 'fallback')]:.2f} / {m[('eB', 'compiled')]:.2f}: not established. 'Bit-flip-heavy' holds for eA only.")
    q = rep["qb10"]
    out.append(f"- **QB10** reads 1 on {q['z_first_mean']:.3f} of its random round-1 readouts (asymmetry ≈ {q['asymmetry_p01_minus_p10']:.2f}, "
               f"against a calibration error of {q['calibration']['p0_given_1']}). Giving the decoder that readout error on B1+B2 changes "
               f"nothing in Z and {q['sensitivity_b1b2']['X']['failures']} → {q['sensitivity_b1b2']['X']['failures_qb10']} in X "
               f"(p {q['sensitivity_b1b2']['X']['mcnemar_p']:.2f}): it doesn't move eB's numbers.")
    return out


def cmd_doc(args: argparse.Namespace) -> int:
    sel = json.loads(SELECTION.read_text())
    rep = json.loads(REPORT.read_text())
    f = rep["fits"]
    alt = sel["fit_model"]
    L = ["# Emerald's batch, re-decoded with its compiled readout window", "",
         "Generated by `scripts/emerald_redecode.py doc` from `InfoRecord/emerald/ler/summary/redecode_selection.json` "
         "(written by `select`, committed before any B3 number) and `redecode_report.json` (`report`). "
         f"**fallback** is the batch's prior, `{PRIORS['fallback']}`: `hardware_noise(cal, spec, windows=None)`, the 360 ns "
         f"`measure_ns` window. **compiled** is `{PRIORS['compiled']}`: `hardware_noise(cal, spec)` reading "
         "`data/calibration/emerald/pulse/compile.23e12f6b-….json`, 1,036 ns on every eA and eB ancilla. "
         "Everything else in the prior is the same. All intervals are 95%: Wilson for P_L, a seeded bootstrap "
         f"({BOOT}, seed {SEED}) for ε.", "",
         "## The rebuild", "",
         f"Before either prior was trusted, every one of the {sel['runs_rebuilt']} B1+B2 runs (`select`) and then all "
         f"{rep['model_check']['runs']} runs of B1–B3 (`report`) were rebuilt from the committed raw memory: the native circuit "
         "equals `placement.json`'s, `memory.npz` equals `raw_meas.npy`, the detection events and observable equal the "
         "dataset's, the split equals `suite.split`, and the fallback prior gives back the dataset's decoder note, its DEM "
         "size and its per-shot `pm_fail` exactly (so every committed failure count). The fallback model's detection "
         f"fraction (seed {SEED}, {MODEL_SHOTS:,} shots) equals `batch_checks.json`'s on "
         f"{rep['model_check']['equal_to_batch_checks']} of {rep['model_check']['runs']} runs. The pilot EP0 is not part of "
         "the batch and is not re-decoded here (`docs/pulla-check.md` has it).", "",
         "## What it says", "",
         *findings(sel, rep), "",
         "## The selection (finetune validation only)", "",
         f"**Rule** (in the script and in the selection file): {sel['rule']}", "",
         f"**E4:** {sel['e4_rule']}", "",
         "| patch | basis | validation shots | fallback P_L | compiled P_L | compiled only / fallback only | McNemar p | choice |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in sel["validation"].items():
        L.append(f"| {v['patch']} | {v['basis']} | {v['validation_shots']:,} | {v['fallback']['failures']} = {v['fallback']['p_l']:.4f} "
                 f"{_ci(v['fallback']['wilson95'])} | {v['compiled']['failures']} = {v['compiled']['p_l']:.4f} "
                 f"{_ci(v['compiled']['wilson95'])} | {v['mcnemar']['compiled_only']} / {v['mcnemar']['fallback_only']} | "
                 f"{v['mcnemar']['p']:.3f} | **{v['choice']}** |")
    L += ["", f"**Fit model** (MINOR 1): {sel['fit_rule']}", "",
          "| patch | basis | prior | χ²/dof, A = 1 on R ≥ 2 | χ²/dof, first-round term | winner |", "|---|---|---|---|---|---|"]
    for k, v in sel["fit_model_pools"].items():
        L.append(f"| {k.split('_')[0]} | {k.split('_')[1]} | {v['prior']} | {v['A1_R>=2']['chi2_per_dof']:.2f} | "
                 f"{v['first_round']['chi2_per_dof']:.2f} | {v['winner']} |")
    L += ["", f"**Chosen fit model: `{alt}`.**", "",
          "## B3 (test): the headline", "",
          "ε_L per round on B3 alone (22,000 shots per patch and basis, 11 round counts). The challenge's form is A = 1 on "
          f"every R. The model dependence is the chosen alternative (`{alt}`) beside it, and χ²/dof says how far the bootstrap "
          "interval, which assumes the model, understates.", "",
          "| patch | basis | prior | ε, A = 1 [bootstrap] | χ²/dof | ε, A free (≤ 1) | ε, A = 1 on R ≥ 2 | ε, first-round term (ε₁; A) | model dependence |",
          "|---|---|---|---|---|---|---|---|---|"]
    for patch, basis in CASES:
        for prior in ("chosen", "fallback", "compiled"):
            x = f[f"{basis}_{patch}_test_b3_{prior}"]
            fa, fr, r2, fr1 = x["fixed_A"], x["free_A"], x["A1_R>=2"], x["first_round"]
            dep = x[alt]["eps"] - fa["eps"]
            L.append(f"| {patch} | {basis} | {x['prior']} | {'**' if prior == 'chosen' else ''}{fa['eps']:.4f}"
                     f"{'**' if prior == 'chosen' else ''} {_ci(fa['eps_ci_bootstrap'])} | {fa['chi2_per_dof']:.1f} | "
                     f"{fr['eps']:.4f} (A {fr['amplitude']:.3f}) | {r2['eps']:.4f} {_ci(r2['eps_ci_bootstrap'])} | "
                     f"{fr1['eps']:.4f} {_ci(fr1['eps_ci_bootstrap'])} ({fr1['eps_1']:.4f}; {fr1['amplitude']:.3f}) | "
                     f"{dep:+.4f} ({dep / fa['eps']:+.0%}) |")
    L += ["", "### B3 P_L(R), both priors (Wilson 95%)", ""]
    for patch, basis in CASES:
        xf, xc = f[f"{basis}_{patch}_test_b3_fallback"], f[f"{basis}_{patch}_test_b3_compiled"]
        cells = "; ".join(f"R={r}: {kf}→{kc}/{n} = {kc / n:.4f} {_ci(ci)}" for r, kf, kc, n, ci in
                          zip(xc["rounds"], xf["failures"], xc["failures"], xc["shots"], xc["wilson95"]))
        L.append(f"- **{patch} {basis}** (fallback→compiled failures, compiled P_L): {cells}")
    L += ["", "### The pooled fits", "", "| patch | basis | pool | ε fallback, A = 1 | ε compiled, A = 1 | ε chosen, A = 1 | "
          f"ε chosen, {alt} |", "|---|---|---|---|---|---|---|"]
    for patch, basis in CASES:
        for pname in ("finetune_b1b2", "test_b3", "all_b1b2b3"):
            a, b_, c = (f[f"{basis}_{patch}_{pname}_{p}"] for p in ("fallback", "compiled", "chosen"))
            L.append(f"| {patch} | {basis} | {pname} | {a['fixed_A']['eps']:.4f} {_ci(a['fixed_A']['eps_ci_bootstrap'])} | "
                     f"{b_['fixed_A']['eps']:.4f} {_ci(b_['fixed_A']['eps_ci_bootstrap'])} | {c['fixed_A']['eps']:.4f} | "
                     f"{c[alt]['eps']:.4f} {_ci(c[alt]['eps_ci_bootstrap'])} |")
    L += ["", "## X against Z (MINOR 2)", "",
          "ε_X/ε_Z with A = 1; z is the difference over the curvature errors inflated by √(χ²/dof) (the batch review's "
          f"method); the ratio under `{alt}` beside it; model-free, the round counts where the Wilson intervals of P_L "
          "separate. The baseline is the decoder model's own X/Z: each prior's noisy circuit, sampled (seed 2026, "
          f"{MODEL_SHOTS:,} shots per R, all 11 R), decoded with its own DEM, fitted with A = 1.", "",
          f"| patch | prior | pool | ε_X / ε_Z | z (inflated) | ratio, {alt} | per R: X<Z / X>Z / overlap |", "|---|---|---|---|---|---|---|"]
    for row in rep["x_vs_z"]:
        c = row["count"]
        L.append(f"| {row['patch']} | {row['prior']} | {row['pool']} | {row['ratio']:.3f} | {row['z_inflated']:+.1f} | "
                 f"{row[f'ratio_{alt}']:.3f} | {c['X<Z']} / {c['X>Z']} / {c['overlap']} |")
    L += ["", "| patch | prior | model ε_X | model ε_Z | model X/Z |", "|---|---|---|---|---|"]
    for m in rep["model_x_vs_z"]:
        L.append(f"| {m['patch']} | {m['prior']} | {m['eps_X']:.4f} | {m['eps_Z']:.4f} | {m['ratio']:.3f} |")
    L += ["", "## The model against the hardware, before and after", "",
          "Hardware detection fraction over the model's (seed 2026, 20,000 shots), per E1 circuit pooled over B1–B3; t = 0 is "
          "round 1 (Z-type in the Z basis, X-type in the X basis), the last t the final data detectors.", "",
          "| patch | basis | prior | whole circuit, range over R | R = 3: t = 0 / 1 / 2 / 3 | R = 10: t = 0 / 1 / 5 / 9 / 10 |",
          "|---|---|---|---|---|---|"]
    for patch, basis in CASES:
        for prior in ("fallback", "compiled"):
            rows = [r for r in rep["ratios"] if r["exp"] == "E1" and r["patch"] == patch and r["basis"] == basis]
            whole = [r[prior]["ratio"] for r in rows]

            def mean_t(R, ts):
                rr = [r for r in rows if r["rounds"] == R]
                return " / ".join(f"{np.mean([r[prior]['ratio_per_t'][f't{t}'] for r in rr]):.2f}" for t in ts)
            L.append(f"| {patch} | {basis} | {prior} | {min(whole):.2f}–{max(whole):.2f} | {mean_t(3, (0, 1, 2, 3))} | "
                     f"{mean_t(10, (0, 1, 5, 9, 10))} |")
    L += ["", "## E4: the delay plus the window (eA, B2)", "",
          "The idle per round is now 1,036 ns + the delay (it was 360 ns + the delay). ε with A = 1, 4 round counts, "
          "8,000 shots per delay and basis.", "",
          "| basis | delay | chosen | idle, fallback → compiled (ns) | ε fallback | ε compiled | detection / model, fallback → compiled |",
          "|---|---|---|---|---|---|---|"]
    for row in rep["e4"]:
        a, b_ = row["fallback"], row["compiled"]
        L.append(f"| {row['basis']} | {row['delay_ns']:g} ns | {row['chosen']} | {a['idle_ns']:g} → {b_['idle_ns']:g} | "
                 f"{a['eps']:.4f} {_ci(a['eps_ci_bootstrap'])} | {b_['eps']:.4f} {_ci(b_['eps_ci_bootstrap'])} | "
                 f"{a['detection_ratio'][0]:.2f}–{a['detection_ratio'][1]:.2f} → {b_['detection_ratio'][0]:.2f}–{b_['detection_ratio'][1]:.2f} |")
    q = rep["qb10"]
    L += ["", "## eB's QB10 (MINOR 5)", "",
          f"- **Size.** QB10 is an X-type ancilla of eB. In the Z basis its round-1 readout is random, and it reads 1 on "
          f"{q['z_first_mean']:.3f} of shots (B1–B3 pooled), against {q['other_x_type_first_mean_range'][0]:.3f}–"
          f"{q['other_x_type_first_mean_range'][1]:.3f} for eB's other X-type ancillas. If the projection is 50/50, that is "
          f"P(0|1) − P(1|0) ≈ {q['asymmetry_p01_minus_p10']:.3f}. In the X basis its round-1 readout is deterministic, and it "
          f"misreads {q['x_first_misread']:.4f}. The calibration has P(1|0) {q['calibration']['p1_given_0']}, P(0|1) "
          f"{q['calibration']['p0_given_1']}, T1 {q['calibration']['t1_ns'] / 1e3:.1f} µs: the bias is not in the calibration.",
          f"- P(read 1) by round, Z basis: {', '.join(f'{k}: {v:.3f}' for k, v in q['p1_by_round']['Z'].items())}; X basis: "
          f"{', '.join(f'{k}: {v:.3f}' for k, v in q['p1_by_round']['X'].items())}. Without reset, an X-type ancilla in the Z basis "
          "reads s₁ (random) in odd rounds: those stay below 0.5 in every round, so the bias persists. The X-basis misread "
          "includes the stabilizer's gate errors, so it bounds P(1|0) from above, and the sensitivity flip below is an upper-end value.",
          f"- **Does it move eB?** {q['note']} Record flip {q['sensitivity_record_flip']:.4f}:"]
    for b, s in q["sensitivity_b1b2"].items():
        L.append(f"  - {b} ({s['prior']}): {s['failures']} → {s['failures_qb10']} of {s['shots']:,} "
                 f"(QB10-prior only {s['qb10_only']}, base only {s['base_only']}, McNemar p {s['mcnemar_p']:.3f}).")
    L += ["", "## What changed from `docs/batches.md`", ""]
    for patch, basis in CASES:
        a, c = f[f"{basis}_{patch}_test_b3_fallback"], f[f"{basis}_{patch}_test_b3_chosen"]
        L.append(f"- {patch} {basis}, B3 ε (A = 1): {a['fixed_A']['eps']:.4f} (batch, fallback) → **{c['fixed_A']['eps']:.4f}** "
                 f"({c['prior']}); with the {alt} model {c[alt]['eps']:.4f}.")
    L += ["", "## Files and how to run", "",
          f"- New LER records (compiled): `InfoRecord/emerald/ler/{{E1,E4}}/<batch run id>{SUFFIX}/` ({len(rep['records'])} records).",
          "- `InfoRecord/emerald/ler/summary/redecode_selection.json`, `redecode_report.json`, `redecode_E1_points.csv`.", "",
          "```", "uv sync --locked && uv run pytest tests/test_emerald_redecode.py",
          "flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/emerald_redecode.py select",
          "git add InfoRecord/emerald/ler/summary/redecode_selection.json && git commit   # before report",
          "flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 uv run python scripts/emerald_redecode.py report",
          "uv run python scripts/emerald_redecode.py doc", "```", ""]
    DOC.write_text("\n".join(L) + "\n")
    log(f"wrote {DOC.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select", help="B1+B2: the rebuild, both decodes, the choice (never opens B3)")
    sub.add_parser("report", help="all batches, with the committed choice")
    sub.add_parser("doc", help="docs/emerald-redecode.md")
    args = ap.parse_args()
    return {"select": cmd_select, "report": cmd_report, "doc": cmd_doc}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
