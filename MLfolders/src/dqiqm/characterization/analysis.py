"""From per-shot records to the table: counts per circuit, the C1/C2/C4 fits, and the table rows.

The input is what ``dqiqm.hw.save_raw`` writes: per circuit, a (shots, 17) uint8 array whose
column j is ``patch.qubits[j]``, with the circuits' designs in batch order (``job_plan.json`` keeps
the order). The same code reads the closure test's simulated records and, later, the hardware's.
No shot is dropped and nothing is post-selected.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dqiqm.characterization import fit as F
from dqiqm.characterization.circuits import C2_PSI, Design
from dqiqm.characterization.patch import DEVICE, Patch, Priors
from dqiqm.experiments import device as dev


def load_job(job_dir: str | Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    """``meta.json`` and the per-circuit arrays of one saved job (``hw.save_raw``'s layout)."""
    job_dir = Path(job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    with np.load(job_dir / "memory.npz") as z:
        arrays = [z[f"c{i}"] for i in range(meta["circuits"])]
    return meta, arrays


def ones_per_qubit(arrays: Sequence[np.ndarray], n_qubits: int) -> list[tuple[int, np.ndarray]]:
    """(shots, ones per column) per circuit."""
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[1] != n_qubits:
            raise ValueError(f"a record of shape {a.shape}, expected (shots, {n_qubits})")
        out.append((int(a.shape[0]), a.sum(axis=0).astype(np.int64)))
    return out


# ---------------------------------------------------------------------------
# C1 (and C4's isolated C1).
# ---------------------------------------------------------------------------


def c1_data(designs: Sequence[Design], arrays: Sequence[np.ndarray], patch: Patch, priors: Priors, *,
            experiment: str = "C1") -> dict[str, F.C1Data]:
    counts = ones_per_qubit(arrays, len(patch.qubits))
    per: dict[str, list] = defaultdict(list)
    for d, (shots, ones) in zip(designs, counts):
        if d.experiment != experiment or d.params.get("family") != "prx_amplification":
            continue
        seq = tuple((float(ph), int(k)) for ph, k in d.params["sequence"])
        for q in d.params["driven"]:
            per[q].append((seq, shots, int(ones[patch.index(q)]), d.name))
    out = {}
    for q in patch.qubits:
        if q not in per:
            continue
        rows = per[q]
        pq = priors.qubits[q]
        out[q] = F.C1Data(q, tuple(r[0] for r in rows), np.array([r[1] for r in rows], float),
                          np.array([r[2] for r in rows], float), tuple(r[3] for r in rows),
                          t1_ns=pq["t1_ns"], t2_ns=pq["t2_ns"], gate_ns=pq["prx_ns"])
    return out


def c1_fits(data: Mapping[str, F.C1Data], priors: Priors, *, fixed: Mapping[str, Mapping[str, float]] | None = None
            ) -> dict[str, F.Fit]:
    return {q: F.fit_c1(d, lam0=2 * priors.qubits[q]["error_1q"], fixed=(fixed or {}).get(q)) for q, d in data.items()}


# ---------------------------------------------------------------------------
# C2 (and C4's isolated C2).
# ---------------------------------------------------------------------------


def c2_points(designs: Sequence[Design], arrays: Sequence[np.ndarray], patch: Patch, *, experiment: str = "C2"
              ) -> dict[tuple[str, str, str], list[F.PhasePoint]]:
    """Per (orientation, target, control): the phase points, one per (s, n)."""
    counts = ones_per_qubit(arrays, len(patch.qubits))
    groups: dict[tuple, dict[int, tuple[int, int]]] = defaultdict(dict)
    for d, (shots, ones) in zip(designs, counts):
        if d.experiment != experiment or d.params.get("family") != "conditional_ramsey":
            continue
        p = d.params
        for t, c in zip(p["targets"], p["controls"]):
            groups[(p["orientation"], t, c, p["s"], p["n"])][p["psi_index"]] = (int(ones[patch.index(t)]), shots)
    out: dict[tuple[str, str, str], list[F.PhasePoint]] = defaultdict(list)
    for (o, t, c, s, n), by_psi in sorted(groups.items(), key=lambda kv: (kv[0][0], dev.qubit_sort_key(kv[0][1]), kv[0][3], kv[0][4])):
        if sorted(by_psi) != list(range(len(C2_PSI))):
            raise ValueError(f"{o} {t}<-{c} s={s} n={n}: analysis phases {sorted(by_psi)} incomplete")
        ones = [by_psi[k][0] for k in range(len(C2_PSI))]
        shots = [by_psi[k][1] for k in range(len(C2_PSI))]
        out[(o, t, c)].append(F.phase_point(s, n, ones, shots))
    return dict(out)


def c2_fits(points: Mapping[tuple[str, str, str], Sequence[F.PhasePoint]]) -> dict[tuple[str, str, str], F.Fit]:
    return {k: F.fit_c2(v, f"{k[1]}<-{k[2]}") for k, v in points.items()}


def couplers(fits: Mapping[tuple[str, str, str], F.Fit]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per coupler (sorted pair): delta (both orientations combined), and each qubit's local phase."""
    by_pair: dict[tuple[str, str], dict[str, F.Fit]] = defaultdict(dict)
    for (o, t, c), f in fits.items():
        by_pair[dev.pair_key(t, c)][t] = f
    out = {}
    for pair, per_target in sorted(by_pair.items()):
        ds = [f.values["delta"] for f in per_target.values()]
        # each orientation's interval half-width / 1.96 is its (inflated) standard error
        ses = [(f.ci["delta"][1] - f.ci["delta"][0]) / (2 * F.Z95) for f in per_target.values()]
        mean, se, chi2 = F.combine(ds, ses) if len(ds) > 1 else (ds[0], ses[0], 0.0)
        out[pair] = {
            "delta": mean,
            "delta_se": se,
            "delta_ci": (mean - F.Z95 * se, mean + F.Z95 * se),
            "delta_by_target": {t: f.values["delta"] for t, f in per_target.items()},
            "delta_consistency_chi2": chi2,
            "phi": {t: f.values["phi_t"] for t, f in per_target.items()},
            "phi_ci": {t: f.ci["phi_t"] for t, f in per_target.items()},
            "phi_se": {t: f.se["phi_t"] * f.inflation for t, f in per_target.items()},
            "fits": {t: f for t, f in per_target.items()},
        }
    return out


# ---------------------------------------------------------------------------
# Table rows.
# ---------------------------------------------------------------------------


def c1_rows(fits: Mapping[str, F.Fit], priors: Priors, *, experiment: str, context: str, common: Mapping[str, Any],
            injected: Mapping[str, tuple[float, float]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for q, f in fits.items():
        pq = priors.qubits[q]
        for k, par in enumerate(("eps", "eta")):
            rows.append({
                **common,
                "locus": q, "gate": "prx", "implementation": "drag_crf_sx", "experiment": experiment, "context": context,
                "parameter": par, "shots": f.extra["shots"], "fit_value": f.values[par], "fit_unit": "rad",
                "fit_ci_low": f.ci[par][0], "fit_ci_high": f.ci[par][1], "se": f.se[par] * f.inflation,
                "chi2_per_dof": f.chi2_per_dof, "rb_error": pq["error_1q"], "t1_us": pq["t1_ns"] / 1e3,
                "t2star_us": pq["t2_ns"] / 1e3,
                "injected": None if injected is None else injected.get(q, (0.0, 0.0))[k],
                "notes": (f"lam={f.values['lam']:.4g} a0={f.values['a0']:.4g} a1={f.values['a1']:.4g}"
                          + (f"; at bound: {','.join(f.extra['at_bound'])}" if f.extra["at_bound"] else "")),
            })
    return rows


def c2_rows(per_coupler: Mapping[tuple[str, str], Mapping[str, Any]], priors: Priors, *, experiment: str, context: str,
            common: Mapping[str, Any], injected: Mapping[tuple[str, str], tuple[float, float, float]] | None = None
            ) -> list[dict[str, Any]]:
    rows = []
    for pair, r in per_coupler.items():
        fits = r["fits"]
        shots = sum(sum(p["shots"] for p in f.extra["points"]) for f in fits.values())
        chi = [f.chi2_per_dof for f in fits.values() if f.dof > 0]
        pc = priors.couplers[pair]
        inj = None if injected is None else injected.get(pair, (0.0, 0.0, 0.0))
        base = {**common, "locus": f"{pair[0]}__{pair[1]}", "gate": "cz", "implementation": "slepian_crf",
                "experiment": experiment, "context": context, "fit_unit": "rad", "rb_error": pc["cz_error"]}
        rows.append({**base, "parameter": "delta", "shots": shots, "fit_value": r["delta"],
                     "fit_ci_low": r["delta_ci"][0], "fit_ci_high": r["delta_ci"][1], "se": r["delta_se"],
                     "chi2_per_dof": max(chi) if chi else None, "injected": None if inj is None else inj[0],
                     "notes": "both orientations: " + ", ".join(f"target {t}: {v:+.5f}" for t, v in r["delta_by_target"].items())
                     + f"; consistency chi2 {r['delta_consistency_chi2']:.2f} (1 dof)"})
        for t in sorted(r["phi"], key=dev.qubit_sort_key):
            f = fits[t]
            rows.append({**base, "parameter": f"phi_{t}", "shots": sum(p["shots"] for p in f.extra["points"]),
                         "fit_value": r["phi"][t], "fit_ci_low": r["phi_ci"][t][0], "fit_ci_high": r["phi_ci"][t][1],
                         "se": r["phi_se"][t], "chi2_per_dof": f.chi2_per_dof if f.dof > 0 else None,
                         "injected": None if inj is None else (inj[1] if t == pair[0] else inj[2]),
                         "notes": f"target {t}, control {[q for q in pair if q != t][0]}"})
    return rows


def common_fields(priors: Priors, *, timestamp_utc: str, simulated: bool, seed: int | None, status: str) -> dict[str, Any]:
    return {"timestamp_utc": timestamp_utc, "device": DEVICE, "calibration_set_id": priors.calibration_set_id,
            "seed": seed, "simulated": simulated, "change_status": status}


def coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many 95% intervals hold the injected value, per parameter kind."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("injected") is None:
            continue
        kind = "phi" if str(r["parameter"]).startswith("phi_") else str(r["parameter"])
        lo, hi, v = float(r["fit_ci_low"]), float(r["fit_ci_high"]), float(r["injected"])
        o = out.setdefault(kind, {"n": 0, "covered": 0, "max_abs_z": 0.0, "z": []})
        o["n"] += 1
        o["covered"] += int(lo <= v <= hi)
        se = float(r["se"]) if r.get("se") not in (None, "") else float("nan")
        z = (float(r["fit_value"]) - v) / se if se and math.isfinite(se) and se > 0 else float("nan")
        o["z"].append(z)
    for o in out.values():
        z = np.array([x for x in o.pop("z") if math.isfinite(x)])
        o["fraction"] = o["covered"] / o["n"] if o["n"] else float("nan")
        o["mean_z"] = float(z.mean()) if len(z) else float("nan")
        o["rms_z"] = float(np.sqrt(np.mean(z ** 2))) if len(z) else float("nan")
        o["max_abs_z"] = float(np.abs(z).max()) if len(z) else float("nan")
    return out
