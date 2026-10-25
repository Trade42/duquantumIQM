"""Every physics check of dqiqm.surface, printed as tables: the surface-gen PR's numbers.

    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
        uv run python scripts/surface_check.py

(about a minute of CPU on 4 workers at the default 10^6 shots per logical-error-rate row;
``--shots 100000`` is enough for a quick look). It writes the tables and every number, with
its seed, to data/sim/surface-gen/surface_check.{md,json}, and exits 1 if a hard check fails.

1. Structure, for d in {3, 5}, rounds in {1, 2, 3, 5, 10}, both bases and both reset variants:
   the contract's instructions and measurement order; detector, measurement and observable
   counts against the formula; 1,000 noiseless shots with no detection event, no nonzero raw
   detector parity and the observable 0; the DEM builds (decompose_errors=True) and its
   graphlike distance is d; and, without reset, adding an X_ERROR after every ancilla M gives
   exactly the with-reset DEM.
2. Against stim.Circuit.generated (with reset, the challenge's noise, d = 3, rounds 3 and 5):
   the same detector count and distance, the same circuit (Z: the flattened text; X: the DEM),
   the same samples at the same seed, and PyMatching LERs with independent seeds that agree by
   a two-proportion z-test, |z| < 1.96 (the brief's criterion since 18:15 UTC). The table also
   shows whether the two 95% Wilson intervals overlap, and the first brief's criterion (each
   P_L inside the other's interval), which fails 17% of the time even for one circuit.
3. The no-reset trade-off: P_L and the per-round eps_L with and without reset, with and
   without the idle option, a sweep of the idle rate, and record-only readout flips (M(p)).
   Reported, not a hard check.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymatching
import stim

from dqiqm.surface import NoiseModel, geometry, memory_circuit

REPO = Path(__file__).resolve().parents[1]
SEED = 2026
P = 0.003
DEFAULT_NOISE = dict(  # challenge/surface_code.py:DEFAULT_NOISE
    after_clifford_depolarization=P,
    after_reset_flip_probability=P,
    before_measure_flip_probability=P,
    before_round_data_depolarization=P,
)
STRUCTURE = {"QUBIT_COORDS", "R", "H", "CX", "M", "MR", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"}
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}
RECORD_FLIP = 0.01
CHUNK = 100_000
Z95 = 1.959963984540054


# ─── statistics ──────────────────────────────────────────────────────────────


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The 95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_round(p_l: float, rounds: int) -> float:
    """eps_L from 2 P_L = 1 - (1 - 2 eps_L)^rounds (monotone, so it maps interval ends too)."""
    return 0.5 * (1 - (1 - 2 * min(p_l, 0.5)) ** (1 / rounds))


# ─── circuits and sampling (run in worker processes) ─────────────────────────


def build(spec: dict) -> stim.Circuit:
    if spec["source"] == "stim":
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{spec['basis'].lower()}",
            distance=spec["d"],
            rounds=spec["rounds"],
            **spec["noise"],
        )
    return memory_circuit(spec["d"], spec["rounds"], spec["basis"], spec["no_reset"], NoiseModel(**spec["noise"]))


def sample_ler(spec: dict) -> dict:
    """Decode `shots` seeded shots with PyMatching from the circuit's own DEM."""
    circuit = build(spec)
    matcher = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    sampler = circuit.compile_detector_sampler(seed=spec["seed"])
    errors, done, digest = 0, 0, hashlib.sha256()
    t0 = time.process_time()
    while done < spec["shots"]:
        n = min(CHUNK, spec["shots"] - done)
        det, obs = sampler.sample(n, separate_observables=True)
        digest.update(np.packbits(det, axis=1).tobytes())
        digest.update(np.packbits(obs, axis=1).tobytes())
        errors += int(np.any(matcher.decode_batch(det) != obs, axis=1).sum())
        done += n
    return {**spec, "errors": errors, "samples_sha256": digest.hexdigest(), "cpu_s": round(time.process_time() - t0, 2)}


# ─── 1. structure ────────────────────────────────────────────────────────────


def records(circuit: stim.Circuit):
    """Each detector's and the observable's absolute measurement indices."""
    dets, obs, n = [], [], 0
    for inst in circuit.flattened():
        targets = inst.targets_copy()
        if inst.name in ("M", "MR"):
            n += len(targets)
        elif inst.name == "DETECTOR":
            dets.append([n + t.value for t in targets])
        elif inst.name == "OBSERVABLE_INCLUDE":
            obs += [n + t.value for t in targets]
    return dets, obs


def contract_ok(circuit: stim.Circuit, d: int, rounds: int, no_reset: bool, noisy: bool) -> bool:
    g = geometry(d)
    names = [inst.name for inst in circuit]
    allowed = STRUCTURE | (NOISE if noisy else set())
    first_tick = names.index("TICK")
    measured = [t.value for inst in circuit if inst.name in ("M", "MR") for t in inst.targets_copy()]
    return (
        set(names) <= allowed
        and all(i < first_tick for i, n in enumerate(names) if n == "R")
        and ("MR" in names) == (not no_reset)
        and measured == sorted(g.ancillas) * rounds + sorted(g.data)
        and set(circuit.get_final_qubit_coordinates()) == set(g.coords)
        and all(len(v) == 3 for v in circuit.get_detector_coordinates().values())
        and circuit.num_observables == 1
    )


def with_flip_after_ancilla_m(circuit: stim.Circuit, p: float) -> stim.Circuit:
    insts = list(circuit)
    final = max(i for i, inst in enumerate(insts) if inst.name in ("M", "MR"))
    out = stim.Circuit()
    for i, inst in enumerate(insts):
        out.append(inst)
        if inst.name == "M" and i != final:
            out.append("X_ERROR", inst.targets_copy(), p)
    return out


def structure_row(d: int, rounds: int, basis: str, no_reset: bool, seed: int) -> dict:
    clean = memory_circuit(d, rounds, basis, no_reset)
    noisy = memory_circuit(d, rounds, basis, no_reset, DEFAULT_NOISE)
    det, obs = clean.compile_detector_sampler(seed=seed).sample(1000, separate_observables=True)
    raw = clean.compile_sampler(seed=seed).sample(1000).astype(np.uint8)
    dets, observable = records(clean)
    raw_nonzero = sum(int((raw[:, idx].sum(axis=1) % 2).sum()) for idx in dets)
    raw_obs = int((raw[:, observable].sum(axis=1) % 2).sum())
    dem = noisy.detector_error_model(decompose_errors=True)
    row = dict(
        d=d,
        rounds=rounds,
        basis=basis,
        reset="no" if no_reset else "MR",
        contract=contract_ok(clean, d, rounds, no_reset, False) and contract_ok(noisy, d, rounds, no_reset, True),
        detectors=clean.num_detectors,
        detectors_formula=rounds * (d * d - 1),
        measurements=clean.num_measurements,
        measurements_formula=rounds * (d * d - 1) + d * d,
        observables=clean.num_observables,
        noiseless_shots=1000,
        noiseless_seed=seed,
        detection_events=int(det.sum()),
        raw_parity_nonzero=raw_nonzero,
        observable_nonzero=raw_obs + int(obs.sum()),
        distance=len(dem.shortest_graphlike_error()),
    )
    if no_reset:
        reset_dem = memory_circuit(d, rounds, basis, False, DEFAULT_NOISE).detector_error_model(decompose_errors=True)
        flipped = with_flip_after_ancilla_m(noisy, P).detector_error_model(decompose_errors=True)
        row["flip_after_M_is_MR_dem"] = flipped.approx_equals(reset_dem, atol=1e-12)
    row["pass"] = bool(
        row["contract"]
        and row["detectors"] == row["detectors_formula"]
        and row["measurements"] == row["measurements_formula"]
        and row["observables"] == 1
        and row["detection_events"] == row["raw_parity_nonzero"] == row["observable_nonzero"] == 0
        and row["distance"] == d
        and row.get("flip_after_M_is_MR_dem", True)
    )
    return row


# ─── tables ──────────────────────────────────────────────────────────────────


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "NO"
        return str(v)

    cells = [[fmt(r.get(key, "")) for key, _ in columns] for r in rows]
    widths = [max(len(head), *(len(c[i]) for c in cells)) for i, (_, head) in enumerate(columns)]
    lines = ["| " + " | ".join(head.ljust(w) for (_, head), w in zip(columns, widths)) + " |"]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(cell, widths)) + " |" for cell in cells]
    return "\n".join(lines)


def ler_text(errors: int, shots: int) -> str:
    lo, hi = wilson(errors, shots)
    return f"{errors / shots:.5f} [{lo:.5f}, {hi:.5f}]"


def eps_text(errors: int, shots: int, rounds: int) -> str:
    lo, hi = wilson(errors, shots)
    return f"{per_round(errors / shots, rounds):.5f} [{per_round(lo, rounds):.5f}, {per_round(hi, rounds):.5f}]"


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", type=int, default=1_000_000, help="shots per logical-error-rate row (>= 10^5)")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (at most 4 on this machine)")
    ap.add_argument("--out", type=Path, default=REPO / "data/sim/surface-gen")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()
    if args.workers > 4:
        ap.error("at most 4 workers: the machine is shared")
    t_start = time.time()
    out: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        out.append(text)

    say("# surface-gen: physics checks of dqiqm.surface.memory_circuit")
    say()
    say(f"stim {stim.__version__}, pymatching {pymatching.__version__}, numpy {np.__version__}; seed base {SEED}; "
        f"{args.shots:,} shots per LER row; noise p = {P} on the four challenge knobs unless stated.")
    say()

    # 1. structure
    cases = list(itertools.product((3, 5), (1, 2, 3, 5, 10), "ZX", (True, False)))
    structure = [structure_row(d, r, b, nr, SEED + i) for i, (d, r, b, nr) in enumerate(cases)]
    say("## 1. Structure, noiseless runs, DEM and distance")
    say()
    say("Noiseless: 1,000 shots per row; 'events' counts detection events (Stim, against its reference sample); "
        "'raw≠0' counts detector parities computed from the raw measurements that are not 0; 'obs≠0' counts shots "
        "whose observable is not 0 (raw or as a flip). Distance: len(shortest_graphlike_error()) of the DEM "
        "(decompose_errors=True) under the challenge's noise. 'flip=MR': without reset, an X_ERROR(p) after every "
        "ancilla M gives exactly the with-reset DEM (approx_equals, atol 1e-12).")
    say()
    say(table(structure, [
        ("d", "d"), ("rounds", "R"), ("basis", "basis"), ("reset", "reset"), ("contract", "contract"),
        ("detectors", "dets"), ("detectors_formula", "R(d²-1)"), ("measurements", "meas"),
        ("measurements_formula", "R(d²-1)+d²"), ("observables", "obs"), ("detection_events", "events"),
        ("raw_parity_nonzero", "raw≠0"), ("observable_nonzero", "obs≠0"), ("distance", "distance"),
        ("flip_after_M_is_MR_dem", "flip=MR"), ("pass", "pass"),
    ]))
    say()

    # 2 and 3: the sampling jobs, all seeded up front
    seed = iter(itertools.count(SEED + 1000))
    stim_rows, jobs = [], []
    for r, b in itertools.product((3, 5), "ZX"):
        base = dict(d=3, rounds=r, basis=b, no_reset=False, noise=DEFAULT_NOISE, shots=args.shots)
        s_ours, s_stim = next(seed), next(seed)
        row = dict(d=3, rounds=r, basis=b, jobs=dict(
            ours=dict(base, source="ours", seed=s_ours),
            stim_same_seed=dict(base, source="stim", seed=s_ours),
            stim=dict(base, source="stim", seed=s_stim),
        ))
        stim_rows.append(row)
        jobs += row["jobs"].values()

    trade_rows = []
    for d, r, b, idle in itertools.product((3, 5), (3, 5), "ZX", (0.0, P)):
        trade_rows.append(dict(d=d, rounds=r, basis=b, idle=idle))
    for idle in (0.0015, 0.006, 0.012):
        trade_rows.append(dict(d=3, rounds=5, basis="Z", idle=idle, sweep=True))
    for d in (3, 5):  # appended last, so every earlier row keeps its seed
        trade_rows.append(dict(d=d, rounds=5, basis="Z", idle=0.0, record=RECORD_FLIP))
    for row in trade_rows:
        row.setdefault("record", 0.0)
        noise = dict(DEFAULT_NOISE, idle_during_measurement=row["idle"], record_flip_probability=row["record"])
        base = dict(d=row["d"], rounds=row["rounds"], basis=row["basis"], noise=noise, shots=args.shots, source="ours")
        row["jobs"] = dict(no_reset=dict(base, no_reset=True, seed=next(seed)), reset=dict(base, no_reset=False, seed=next(seed)))
        jobs += row["jobs"].values()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(sample_ler, jobs))
    by_key = {json.dumps({k: v for k, v in j.items()}, sort_keys=True): res for j, res in zip(jobs, results)}

    def result(job):
        return by_key[json.dumps(job, sort_keys=True)]

    # 2. against Stim
    for row in stim_rows:
        ours_c = memory_circuit(3, row["rounds"], row["basis"], False, DEFAULT_NOISE)
        stim_c = build(row["jobs"]["stim"])
        row["detectors"] = f"{ours_c.num_detectors} / {stim_c.num_detectors}"
        d_ours = len(ours_c.detector_error_model(decompose_errors=True).shortest_graphlike_error())
        d_stim = len(stim_c.detector_error_model(decompose_errors=True).shortest_graphlike_error())
        row["distance"] = f"{d_ours} / {d_stim}"
        if row["basis"] == "Z":
            row["same_circuit"] = str(ours_c.flattened()) == str(stim_c.flattened())
            row["same_how"] = "flattened text"
        else:
            row["same_circuit"] = ours_c.detector_error_model(decompose_errors=True).approx_equals(
                stim_c.detector_error_model(decompose_errors=True).flattened(), atol=1e-12)
            row["same_how"] = "DEM approx_equals"
        ours, same, theirs = (result(row["jobs"][k]) for k in ("ours", "stim_same_seed", "stim"))
        row["same_samples"] = ours["samples_sha256"] == same["samples_sha256"]
        n = args.shots
        lo_o, hi_o = wilson(ours["errors"], n)
        lo_s, hi_s = wilson(theirs["errors"], n)
        p_o, p_s = ours["errors"] / n, theirs["errors"] / n
        row["within"] = lo_s <= p_o <= hi_s and lo_o <= p_s <= hi_o
        row["overlap"] = lo_s <= hi_o and lo_o <= hi_s
        pooled = (ours["errors"] + theirs["errors"]) / (2 * n)
        z = (p_o - p_s) / math.sqrt(2 * pooled * (1 - pooled) / n) if 0 < pooled < 1 else 0.0
        row["z"] = round(z, 2)
        row["z_ok"] = abs(z) < Z95
        row["ler_ours"] = f"{ler_text(ours['errors'], n)} (seed {ours['seed']})"
        row["ler_stim"] = f"{ler_text(theirs['errors'], n)} (seed {theirs['seed']})"
        row["pass"] = bool(row["detectors"].split(" / ")[0] == row["detectors"].split(" / ")[1]
                           and d_ours == d_stim == 3 and row["same_circuit"] and row["same_samples"] and row["z_ok"])
        row["results"] = dict(ours=ours, stim_same_seed=same, stim=theirs)
    say("## 2. Against stim.Circuit.generated (with reset, challenge noise, d = 3)")
    say()
    say(f"P_L with its 95% Wilson interval, {args.shots:,} shots each, decoded by PyMatching from each circuit's own DEM. "
        "'same samples': with one seed, ours and Stim's give bit-identical detection events and observable flips "
        "(SHA-256 of all shots). With independent seeds: z is the two-proportion z-score, and '|z|<1.96' is the brief's "
        "criterion, the one 'pass' uses; 'overlap': the two Wilson intervals intersect; 'within (old)': each P_L lies "
        "inside the other's interval, the first brief's criterion, shown for the record. For two independent estimates "
        "of one rate, it holds only when |z| < 1.96/sqrt(2) = 1.39, 83% of the time per row.")
    say()
    say(table(stim_rows, [
        ("d", "d"), ("rounds", "R"), ("basis", "basis"), ("detectors", "dets ours/Stim"), ("distance", "distance ours/Stim"),
        ("same_circuit", "same circuit"), ("same_how", "by"), ("same_samples", "same samples"),
        ("ler_ours", "P_L ours"), ("ler_stim", "P_L Stim"), ("z", "z"), ("z_ok", "|z|<1.96"), ("overlap", "overlap"), ("within", "within (old)"), ("pass", "pass"),
    ]))
    say()

    # 3. the trade-off
    for row in trade_rows:
        a, b = result(row["jobs"]["no_reset"]), result(row["jobs"]["reset"])
        n, r = args.shots, row["rounds"]
        row["pl_no_reset"] = ler_text(a["errors"], n)
        row["pl_reset"] = ler_text(b["errors"], n)
        row["eps_no_reset"] = eps_text(a["errors"], n, r)
        row["eps_reset"] = eps_text(b["errors"], n, r)
        row["ratio"] = f"{b['errors'] / a['errors']:.3f}" if a["errors"] else "-"
        lo_a, hi_a = wilson(a["errors"], n)
        lo_b, hi_b = wilson(b["errors"], n)
        row["lower"] = "no reset" if hi_a < lo_b else ("reset" if hi_b < lo_a else "tie (intervals overlap)")
        row["results"] = dict(no_reset=a, reset=b)
        noise = dict(DEFAULT_NOISE, idle_during_measurement=row["idle"], record_flip_probability=row["record"])
        row["same_dem"] = memory_circuit(row["d"], r, row["basis"], True, noise).detector_error_model(
            decompose_errors=True).approx_equals(
            memory_circuit(row["d"], r, row["basis"], False, noise).detector_error_model(decompose_errors=True), atol=1e-12)
    columns = [
        ("d", "d"), ("rounds", "R"), ("basis", "basis"), ("idle", "idle"), ("record", "record"), ("pl_no_reset", "P_L no reset"),
        ("pl_reset", "P_L reset (MR)"), ("eps_no_reset", "eps_L/round no reset"), ("eps_reset", "eps_L/round reset"),
        ("ratio", "reset/no-reset"), ("same_dem", "same DEM"), ("lower", "lower P_L"),
    ]
    say("## 3. The no-reset trade-off")
    say()
    say(f"Our circuits, challenge noise p = {P}, {args.shots:,} shots each, independent seeds; P_L and eps_L per round "
        "with 95% Wilson intervals. 'idle' is idle_during_measurement: DEPOLARIZE1 on the data during each ancilla "
        "readout (both variants) and X_ERROR on the live ancillas after M (no reset only). 'record' is "
        "record_flip_probability: M(p) on every readout (an X_ERROR before MR with reset). 'same DEM': the two "
        "variants' DEMs are approx_equal. 'lower P_L' names the variant whose interval lies wholly below the other's.")
    say()
    say(table([r for r in trade_rows if not r.get("sweep") and not r["record"]], columns))
    say()
    say("Idle sweep (d = 3, R = 5, Z basis; the reset-flip rate stays 0.003):")
    say()
    sweep = sorted([r for r in trade_rows if r["d"] == 3 and r["rounds"] == 5 and r["basis"] == "Z" and not r["record"]],
                   key=lambda r: r["idle"])
    say(table(sweep, columns))
    say()
    say(f"Record-only readout flips, M({RECORD_FLIP}), on top of the challenge noise (R = 5, Z basis, no idle). Without "
        "reset a flipped bit fires D_k and D_(k+2); with MR it is a state flip before the reset and fires D_k and D_(k+1). "
        "Compare with the idle = 0 rows above:")
    say()
    say(table([r for r in trade_rows if r["record"]], columns))
    say()

    hard = all(r["pass"] for r in structure) and all(r["pass"] for r in stim_rows)
    say(f"Hard checks: {'PASS' if hard else 'FAIL'} ({sum(r['pass'] for r in structure)}/{len(structure)} structure rows, "
        f"{sum(r['pass'] for r in stim_rows)}/{len(stim_rows)} Stim rows). Wall time {time.time() - t_start:.0f} s, "
        f"{os.cpu_count()} CPUs visible, {args.workers} workers.")

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "surface_check.md").write_text("\n".join(out) + "\n")
        strip = lambda rows: [{k: v for k, v in r.items() if k != "jobs"} for r in rows]
        (args.out / "surface_check.json").write_text(json.dumps(dict(
            versions=dict(stim=stim.__version__, pymatching=pymatching.__version__, numpy=np.__version__),
            seed_base=SEED, shots=args.shots, noise=DEFAULT_NOISE, hard_checks_pass=hard,
            structure=structure, against_stim=strip(stim_rows), trade_off=strip(trade_rows),
        ), indent=1) + "\n")
        print(f"wrote {args.out / 'surface_check.md'} and surface_check.json")
    return 0 if hard else 1


if __name__ == "__main__":
    sys.exit(main())
