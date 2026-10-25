"""Re-decode the Garnet pilot's committed shots under the old and the new decoder priors (offline).

    uv run python scripts/garnet_reprior.py [--json PATH] [--doc]

Job ``01a17000-6394-755f-a58b-b4f9c01893b0`` (P0, ``E1_d3_Z_r03_pA``, 1,000 shots), read from its raw
``memory.npz`` (never written). The circuit is rebuilt with ``suite.build`` from the committed snapshot
and checked against ``placement.json``'s ``native_sha256``. Each prior is a noisy copy of that circuit;
PyMatching decodes the hardware shots with it, and Stim samples it (100,000 shots, seed 2026) for the
model's per-round detection fractions and its own P_L.

The priors (calibration: the pilot's before snapshot, as the pilot decoded with):

* ``a_old``: ``calibrated_noise(idle_ns=556)``, symmetric ``DEPOLARIZE1`` idle, ``measure``'s readout
  everywhere, and the idle after the last round. That is exactly the pilot's decoder: its text must
  hash to :data:`OLD_DECODER_SHA256` (built from main ``811930e``), and it must give 209 failures;
* ``b_ramsey``: ``device.hardware_noise(t2="ramsey")``, the default hardware prior: the compiled 1,096 ns
  readout window, a symmetric ``DEPOLARIZE1`` idle, ``measure_fidelity``'s unheralded errors on the final
  layer, no idle after the last round. (Round 1 of garnet-timing had the biased idle here, in the wrong
  frame: that row is now ``x_pauli_stim_frame``.);
* ``c_echo``: the same with the echo T2;
* ``b_pauli`` and ``c_pauli``: the same with ``idle_channel="pauli"``, the biased twirl in each data qubit's
  hardware frame (``device.readout_frames``: x <-> z on Stim 1, 3, 5, 12 and 17 here);
* ``x_pauli_stim_frame``: round 1's ``b_ramsey``, the biased twirl in Stim's frame (wrong on hardware), kept so
  the review's numbers reproduce;
* ablations, one fix at a time on top of ``a_old`` (labelled ``x_…``): they say which fix moves what.

Nothing here needs the network or a token. Under a minute of CPU; run it under the sim lock anyway.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dqiqm import decode as D
from dqiqm import stats
from dqiqm.experiments import circuits as C
from dqiqm.experiments import device as dev
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
JOB_DIR = REPO / "InfoRecord" / "raw" / "20261024-garnet-pilot" / "01a17000-6394-755f-a58b-b4f9c01893b0"
DATASET = REPO / "InfoRecord" / "datasets" / "E1" / "p0_E1_d3_Z_r03_pA"
PLACEMENT = REPO / "InfoRecord" / "circuits" / "E1" / "placement.json"
BEFORE = REPO / "data" / "calibration" / "garnet" / "a18a947f-0408-4628-81ea-1715637e434f.20261024T221752Z.json"
DOC = REPO / "docs" / "garnet-design.md"
BEGIN, END = "<!-- reprior:begin -->", "<!-- reprior:end -->"
CIRCUIT = "E1_d3_Z_r03_pA"
#: sha256 of ``str(decoder circuit)`` of the pilot's own decode (garnet-pilot, main 811930e).
OLD_DECODER_SHA256 = "bec21d8ca17bf9720bc38ba6f12112526a8f084af370218ea7c27519218ca4e6"
OLD_FAILURES = 209
SIM_SHOTS = 100_000
SEED = 2026


def rounds_and_types(spec) -> tuple[np.ndarray, np.ndarray]:
    """Each detector's round t (its third coordinate) and type ("X" or "Z", from its ancilla)."""
    coords = spec.stim.get_detector_coordinates()
    t = np.array([int(coords[i][2]) for i in range(spec.stim.num_detectors)])
    xy_x = {tuple(spec.stim.get_final_qubit_coordinates()[q]) for q in spec.x_ancillas}
    typ = np.array(["X" if (coords[i][0], coords[i][1]) in xy_x else "Z" for i in range(spec.stim.num_detectors)])
    return t, typ


def per_round(dets: np.ndarray, t: np.ndarray, typ: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"all": {}, "X": {}, "Z": {}}
    for k in sorted(set(t.tolist())):
        out["all"][str(k)] = float(dets[:, t == k].mean())
        for ty in ("X", "Z"):
            m = (t == k) & (typ == ty)
            if m.any():
                out[ty][str(k)] = float(dets[:, m].mean())
    return out


def evaluate(name: str, dec, spec, dets_hw: np.ndarray, obs_hw: np.ndarray, t: np.ndarray, typ: np.ndarray,
             info: dict[str, Any]) -> dict[str, Any]:
    assert dec.num_detectors == spec.stim.num_detectors and dec.num_measurements == spec.stim.num_measurements
    dem = dec.detector_error_model(decompose_errors=True)
    pred = D.decode(dec, dets_hw)
    fails = pred[:, 0] != obs_hw[:, 0]
    k, n = int(fails.sum()), len(dets_hw)
    lo, hi = stats.wilson(k, n)
    raw = dec.compile_sampler(seed=SEED).sample(SIM_SHOTS).astype(np.uint8)
    dets_m, obs_m = D.detection_events(spec.stim, raw)
    km = int((D.decode(dec, dets_m)[:, 0] != obs_m[:, 0]).sum())
    mlo, mhi = stats.wilson(km, SIM_SHOTS)
    rate_m = np.clip(dets_m.mean(axis=0), 1e-9, 1 - 1e-9)
    hw_k = dets_hw.sum(axis=0)
    # per-detector binomial log-likelihood of the hardware's detection counts under the model's rates
    ll = float(np.sum(hw_k * np.log(rate_m) + (n - hw_k) * np.log(1 - rate_m)))
    rounds_hw, rounds_m = per_round(dets_hw, t, typ), per_round(dets_m, t, typ)
    return {
        "name": name,
        **info,
        "decoder_sha256": hashlib.sha256(str(dec).encode()).hexdigest(),
        "dem_errors": dem.num_errors,
        "graphlike_distance": len(dec.shortest_graphlike_error()),
        "hw_failures": k, "hw_shots": n, "hw_p_l": k / n, "hw_p_l_wilson95": [float(lo), float(hi)],
        "model_failures": km, "model_shots": SIM_SHOTS, "model_p_l": km / SIM_SHOTS,
        "model_p_l_wilson95": [float(mlo), float(mhi)],
        "model_detection_fraction": float(dets_m.mean()),
        "per_round_model": rounds_m, "per_round_hw": rounds_hw,
        "per_round_abs_error_sum": float(sum(abs(rounds_m["all"][r] - rounds_hw["all"][r]) for r in rounds_hw["all"])),
        "detector_loglik": ll,
        "_fails": fails,
    }


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """McNemar on the same shots: failures of b only, of a only, and the exact two-sided p-value."""
    only_b, only_a = int((b & ~a).sum()), int((a & ~b).sum())
    m = only_a + only_b
    tail = sum(math.comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2**m if m else 1.0
    return {"only_this": only_b, "only_a_old": only_a, "mcnemar_p": min(1.0, 2 * tail)}


def build_priors(cal: dev.Calibration, spec) -> list[tuple[str, Any, dict[str, Any]]]:
    # the pilot's rule (garnet_pilot.readout_window_ns): the slowest ancilla's measure duration, the probe alone
    old_idle = max(float(cal.qubits[spec.physical[q]]["measure_ns"]) for q, r in spec.roles.items() if r == "ancilla")
    old_src = "measure_ns (the pilot's rule: the probe alone)"
    args = dict(pairs=spec.cx_pairs, x_ancillas=spec.x_ancillas)

    def legacy(nm):  # the pre-fix circuit: an idle after the last round too
        return dataclasses.replace(nm, idle_after_last_round=True)

    old = legacy(dev.calibrated_noise(cal, spec.physical, spec.roles, idle_ns=old_idle, **args))
    nb, ib = dev.hardware_noise(cal, spec, t2="ramsey")
    nc, ic = dev.hardware_noise(cal, spec, t2="echo")
    pb, pib = dev.hardware_noise(cal, spec, t2="ramsey", idle_channel="pauli")
    pc, pic = dev.hardware_noise(cal, spec, t2="echo", idle_channel="pauli")
    frames = dev.readout_frames(spec)
    w = ib["readout_window_ns"]
    base = {"idle_ns": old_idle, "idle_channel": "depolarize", "t2": "ramsey", "final_readout": "measure",
            "idle_after_last_round": True}
    out = [("a_old", old, base | {"readout_window_source": old_src})]
    out.append(("b_ramsey", nb, ib))
    out.append(("c_echo", nc, ic))
    out.append(("b_pauli", pb, pib))
    out.append(("c_pauli", pc, pic))
    stim_frame = dev.calibrated_noise(cal, spec.physical, spec.roles, idle_ns=w, idle_channel="pauli", final_readout=True, **args)
    out.append(("x_pauli_stim_frame", stim_frame, {**pib, "idle_frame": "stim"}))
    ablations = [
        ("x_no_last_idle", dev.calibrated_noise(cal, spec.physical, spec.roles, idle_ns=old_idle, **args),
         base | {"idle_after_last_round": False}),
        ("x_window_1096", legacy(dev.calibrated_noise(cal, spec.physical, spec.roles, idle_ns=w, **args)),
         base | {"idle_ns": w}),
        ("x_pauli", legacy(dev.calibrated_noise(cal, spec.physical, spec.roles, idle_ns=old_idle, idle_channel="pauli",
                                                frames=frames, **args)),
         base | {"idle_channel": "pauli", "idle_frame": pib["idle_frame"]}),
        ("x_final_readout", legacy(dev.calibrated_noise(cal, spec.physical, spec.roles, idle_ns=old_idle, final_readout=True, **args)),
         base | {"final_readout": "measure_fidelity (unheralded)"}),
    ]
    return out + ablations


def channel(x: dict[str, Any]) -> str:
    if x["idle_channel"] != "pauli":
        return x["idle_channel"]
    return "pauli, Stim frame (wrong on hw)" if x.get("idle_frame") == "stim" else "pauli, hw frame"


def table(results: list[dict[str, Any]]) -> str:
    rounds = list(results[0]["per_round_hw"]["all"])
    hw = results[0]["per_round_hw"]["all"]
    lines = [
        "| priors | idle (ns) | idle channel | T2 | final readout | idle after last round | DEM errors | d | hardware P_L (1,000 shots, R = 3) | model P_L (100,000 sim shots) | "
        + " | ".join(f"t = {r}" for r in rounds) + " | Σ\\|model − hw\\| | detector log-lik | vs a_old: fails only here / only in a_old (McNemar p) |",
        "|" + "---|" * (13 + len(rounds)),
        "| **hardware** | | | | | | | | | | " + " | ".join(f"**{hw[r]:.3f}**" for r in rounds) + " | | | |",
    ]
    for x in results:
        m = x["per_round_model"]["all"]
        lo, hi = x["hw_p_l_wilson95"]
        mlo, mhi = x["model_p_l_wilson95"]
        lines.append(
            f"| `{x['name']}` | {x['idle_ns']:g} | {channel(x)} | {x['t2']} | {x['final_readout'].split(' ')[0]} | "
            f"{'yes' if x['idle_after_last_round'] else 'no'} | {x['dem_errors']} | {x['graphlike_distance']} | "
            f"{x['hw_failures']}/{x['hw_shots']} = **{x['hw_p_l']:.3f}** [{lo:.3f}, {hi:.3f}] | "
            f"{x['model_p_l']:.4f} [{mlo:.4f}, {mhi:.4f}] | "
            + " | ".join(f"{m[r]:.3f}" for r in rounds)
            + f" | {x['per_round_abs_error_sum']:.3f} | {x['detector_loglik']:.0f} | "
            + ("" if x["name"] == "a_old" else f"{x['vs_a_old']['only_this']} / {x['vs_a_old']['only_a_old']} (p = {x['vs_a_old']['mcnemar_p']:.2f})")
            + " |")
    return "\n".join(lines)


def ratio_table(results: list[dict[str, Any]]) -> str:
    """Hardware over model per round, as the pilots review states it (final data detectors 0.71x for the old priors)."""
    rounds = list(results[0]["per_round_hw"]["all"])
    hw = results[0]["per_round_hw"]["all"]
    lines = ["| priors | " + " | ".join(f"t = {r}" for r in rounds) + " |", "|" + "---|" * (1 + len(rounds))]
    for x in results:
        m = x["per_round_model"]["all"]
        lines.append(f"| `{x['name']}` | " + " | ".join(f"{hw[r] / m[r]:.2f}×" for r in rounds) + " |")
    return "\n".join(lines)


def type_table(results: list[dict[str, Any]]) -> str:
    keys = [(ty, r) for ty in ("Z", "X") for r in results[0]["per_round_hw"][ty]]
    hw = results[0]["per_round_hw"]
    lines = ["| priors | " + " | ".join(f"{ty} t = {r}" for ty, r in keys) + " |", "|" + "---|" * (1 + len(keys)),
             "| **hardware** | " + " | ".join(f"**{hw[ty][r]:.3f}**" for ty, r in keys) + " |"]
    for x in results:
        m = x["per_round_model"]
        lines.append(f"| `{x['name']}` | " + " | ".join(f"{m[ty][r]:.3f}" for ty, r in keys) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="write the full results here")
    ap.add_argument("--doc", action="store_true", help=f"write the tables into {DOC.relative_to(REPO)} between {BEGIN} and {END}")
    args = ap.parse_args()

    suite = SU.build(dev.live_calibration(dev.DEFAULT_SNAPSHOT), experiments=("E1",))
    spec = suite[CIRCUIT]
    want = json.loads(PLACEMENT.read_text())[CIRCUIT]["native_sha256"]
    if C.native_sha256(spec.hardware) != want:
        raise SystemExit(f"{CIRCUIT}: rebuilt native_sha256 differs from placement.json: not the circuit that ran")
    raw, meta = D.load_memory(JOB_DIR, 0, circuit=spec.stim)
    if not np.array_equal(raw, np.load(DATASET / "raw_meas.npy")):
        raise SystemExit("memory.npz c0 differs from the dataset's raw_meas.npy")
    dets, obs = D.detection_events(spec.stim, raw)
    if not np.array_equal(dets, np.load(DATASET / "det_events.npy")):
        raise SystemExit("detection events differ from the dataset's det_events.npy")
    t, typ = rounds_and_types(spec)
    cal = dev.live_calibration(BEFORE)
    if cal.calibration_set_id != meta["calibration_set_id"]:
        raise SystemExit(f"the before snapshot is {cal.calibration_set_id}, the job ran on {meta['calibration_set_id']}")

    results = [evaluate(name, spec.noisy(nm), spec, dets, obs, t, typ, info) for name, nm, info in build_priors(cal, spec)]
    old = results[0]
    for x in results[1:]:
        x["vs_a_old"] = paired(old["_fails"], x["_fails"])
    for x in results:
        del x["_fails"]
    if old["decoder_sha256"] != OLD_DECODER_SHA256 or old["hw_failures"] != OLD_FAILURES:
        raise SystemExit(f"a_old does not reproduce the pilot's decode: sha {old['decoder_sha256']}, "
                         f"{old['hw_failures']} failures (want {OLD_DECODER_SHA256}, {OLD_FAILURES})")
    main = results[:5]
    best_fit = max(main, key=lambda x: x["detector_loglik"])
    best_pl = min(main, key=lambda x: x["hw_p_l"])
    worse = [x["name"] for x in results[1:] if x["vs_a_old"]["mcnemar_p"] < 0.05 and x["vs_a_old"]["only_this"] > x["vs_a_old"]["only_a_old"]]
    md = "\n".join([
        BEGIN,
        f"Generated by `scripts/garnet_reprior.py`: job `{meta['job_id']}` ({CIRCUIT}, {len(raw):,} shots, R = 3, calibration set "
        f"`{cal.calibration_set_id}`), re-decoded from its committed `memory.npz` with PyMatching. Priors from the pilot's before "
        f"snapshot (`{BEFORE.relative_to(REPO)}`). Model columns: the prior's own noisy circuit, {SIM_SHOTS:,} Stim shots, seed "
        f"{SEED}. P_L intervals are Wilson 95%. `a_old` is the pilot's own decoder, byte for byte (sha256 `{OLD_DECODER_SHA256[:12]}…`). "
        "`b_ramsey` and `c_echo` are the default hardware prior (symmetric idle); `b_pauli` and `c_pauli` the biased switch, in each "
        "data qubit's hardware frame; `x_pauli_stim_frame` is garnet-timing round 1's `b_ramsey`, biased in Stim's frame. The "
        "other `x_…` rows change one thing from `a_old` at a time. 'Detector log-lik' is the binomial log-likelihood of the hardware's "
        "per-detector detection counts under the model's per-detector rates, the detectors taken as independent (higher fits "
        "better). The McNemar column compares failures shot by shot with `a_old` on the same 1,000 shots.",
        "",
        "Detection fraction per detector round t (t = 0: round 1, Z-type only; t = 3: the final data detectors, Z-type only):",
        "",
        table(results),
        "",
        "Hardware over model, per round (t = 3 is the final data detectors):",
        "",
        ratio_table(results),
        "",
        "By stabilizer type:",
        "",
        type_table(results),
        "",
        f"Of the candidate priors (`a_old`, `b_ramsey`, `c_echo`, `b_pauli`, `c_pauli`): the best fit to the detection "
        f"statistics is `{best_fit['name']}`, the lowest hardware P_L `{best_pl['name']}`"
        + ("; they disagree, so the detector log-likelihood alone does not choose a decoder" if best_fit is not best_pl else "")
        + ". Decoding significantly worse than `a_old` (McNemar p < 0.05): "
        + (", ".join(f"`{n}`" for n in worse) if worse else "none") + ".",
        END,
    ])
    print(md)
    if args.json:
        args.json.write_text(json.dumps({"job_id": meta["job_id"], "circuit": CIRCUIT, "results": results}, indent=1) + "\n")
        print(f"wrote {args.json}")
    if args.doc:
        text = DOC.read_text()
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            DOC.write_text(head + md + tail)
        else:
            raise SystemExit(f"{DOC} has no {BEGIN} ... {END} section")
        print(f"wrote {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
