"""Offline Garnet rehearsal: d=3 no-reset memory, three decoders, two simulators, no network.

Layout and noise come from iqm-client's bundled IQMFakeGarnet (a fixed error profile, not
a live calibration). Garnet's 30 couplers are embedded as a 5x5 grid with missing corners,
and challenge.run_on_hardware.prepare_qiskit places and compiles the code (native, zero SWAPs).

Two sources of shots, decoded by the same three decoders on the same shots:
  stim  the decoder circuit itself: Garnet-calibrated Pauli noise (matched to the
        calibrated decoders);
  aer   the compiled native circuit on IQMFakeGarnet, i.e. IQM's own noise model,
        which no decoder was built from (a hardware stand-in).
Decoders: PyMatching with uniform weights, PyMatching with calibrated weights (correlated),
and the neural decoder trained only on fresh stim shots from the calibrated circuit.

    python -m iyanu.garnet_sim --rounds 1 2 3 4 --shots 20000
"""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
import numpy as np


def block_network():
    """Refuse every outbound connection in this process (the rehearsal is offline)."""
    def refuse(*_a, **_k):
        raise RuntimeError("garnet_sim is offline: network access is blocked.")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse


def fake_garnet():
    """IQMFakeGarnet with Qiskit's default scheduling stage.

    iqm-client's default stage (iqm_default_scheduling) rebuilds the circuit with its qubits
    reordered (an 'ancilla' register first), so circuit.find_bit(q).index is no longer the
    physical index and prepare_qiskit's coupler check reads the wrong pairs (a 2-qubit CZ
    with initial_layout [0, 3] comes back on indices (18, 19)). The default stage keeps
    indices physical; prepare_qiskit then verifies every CZ against the couplers.
    """
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    class Garnet(type(IQMFakeGarnet())):
        def get_scheduling_stage_plugin(self):
            return "default"

    backend = IQMFakeGarnet()
    backend.__class__ = Garnet
    return backend


NATIVE = {"r", "cz", "measure", "barrier"}


def couplings(backend):
    return sorted({tuple(sorted((backend.index_to_qubit_name(a), backend.index_to_qubit_name(b))))
                   for a, b in backend.coupling_map.get_edges()})


def grid_from_couplings(edges):
    """Embed a square-lattice coupler graph as a grid (QB names, None for gaps), or refuse.

    Backtracking: every coupler is a unit grid edge. First try with grid neighbours exactly the
    coupler neighbours (an induced subgraph); if that fails (a chip with disabled couplers, such
    as Emerald), allow grid neighbours without a coupler, preferring cells that need none. Any
    such embedding serves, since placement re-validates every pair on the live couplers.
    """
    try:
        return _embed(edges, strict=True)
    except ValueError:
        return _embed(edges, strict=False)


def _embed(edges, *, strict):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    key = lambda q: int(q[2:])
    start = min(adj, key=key)
    order, seen = [start], {start}
    for q in order:  # BFS, so every later qubit has a placed neighbour
        for n in sorted(adj[q]-seen, key=key):
            seen.add(n)
            order.append(n)
    pos, cells = {}, {}

    def place(i):
        if i == len(order):
            return True
        q = order[i]
        placed = {n for n in adj[q] if n in pos}
        options = [(0, 0)] if not placed else sorted(
            {(pos[n][0]+dr, pos[n][1]+dc) for n in placed for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))})
        def near(cell):
            return {cells[c] for c in ((cell[0]+1, cell[1]), (cell[0]-1, cell[1]),
                                       (cell[0], cell[1]+1), (cell[0], cell[1]-1)) if c in cells}
        options = sorted(options, key=lambda c: len(near(c)-placed))
        for cell in options:
            nb = near(cell)
            if cell in cells or not placed <= nb or (strict and nb != placed):
                continue
            pos[q], cells[cell] = cell, q
            if place(i+1):
                return True
            del pos[q], cells[cell]
        return False

    if len(order) != len(adj) or not place(0):
        raise ValueError("Coupler graph does not embed as a square grid.")
    for a, b in edges:
        (r1, c1), (r2, c2) = pos[a], pos[b]
        if abs(r1-r2)+abs(c1-c2) != 1:
            raise ValueError(f"Coupler {a}-{b} is not a grid unit edge.")
    minr, minc = min(r for r, _ in pos.values()), min(c for _, c in pos.values())
    rows = max(r for r, _ in pos.values())-minr+1
    cols = max(c for _, c in pos.values())-minc+1
    grid = [[None]*cols for _ in range(rows)]
    for q, (r, c) in pos.items():
        grid[r-minr][c-minc] = q
    return grid


def snapshot_from_fake(backend, *, measure_ns=1000.):
    """CalibrationSnapshot from the fake backend's error profile.

    Qiskit depolarizing parameter lam -> average infidelity lam*(d-1)/d (1q: lam/2, 2q: 3lam/4),
    which NoiseModel maps back to the same Pauli channel. The profile has no readout duration
    or leakage: measure_ns is an assumption, reset is unused (no-reset circuit), leakage zero.
    """
    from challenge.calibration import CalibrationSnapshot, QubitCalibration
    ep = backend.error_profile
    sq = ep.single_qubit_gate_depolarizing_error_parameters["prx"]
    qubits = {q: QubitCalibration(sq[q]/2, (ep.readout_errors[q]["0"]+ep.readout_errors[q]["1"])/2, 0.,
                                  float(ep.t1s[q]), min(float(ep.t2s[q]), 2*float(ep.t1s[q])))
              for q in ep.t1s}
    edges = {CalibrationSnapshot.edge_key(a, b): 0.75*lam
             for (a, b), lam in ep.two_qubit_gate_depolarizing_error_parameters["cz"].items()}
    durations = {"1q": ep.single_qubit_gate_durations["prx"], "2q": ep.two_qubit_gate_durations["cz"],
                 "measure": measure_ns, "reset": measure_ns}
    return CalibrationSnapshot(qubits, edges, durations, dict(
        source="iqm-client IQMFakeGarnet error profile (bundled, not a live calibration)",
        assumed_measure_ns=measure_ns))


def aer_shots(backend, prepared, shots, seed):
    from challenge.internal_helpers import counts_to_measurement_array
    from challenge.extract_syndromes import extract_syndromes
    ideal = prepared["stim_circuit"]
    counts = backend.run(prepared["circuit"], shots=shots, seed_simulator=seed).result().get_counts()
    raw = counts_to_measurement_array(counts, ideal.num_measurements, shots)
    s = extract_syndromes(raw, ideal, print_summary=False)
    return np.asarray(s["det_events"], dtype=bool), np.asarray(s["obs_flips"], dtype=bool)


def failures(pred, obs):
    return int(np.any(pred != obs, axis=1).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--shots", type=int, default=20_000, help="test shots per round count, per simulator")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--uniform-p", type=float, default=0.005, help="weights of the uniform PyMatching baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="results/iyanu/garnet_sim")
    a = ap.parse_args(argv)
    block_network()

    from challenge.run_on_hardware import prepare_qiskit
    from challenge.noise_models import NoiseModel, add_noise
    from challenge.decoding import make_matcher, wilson_interval, fit_ler
    from iyanu.neural_gpu import NeuralConfig, train_neural_decoder, paired_comparison

    backend = fake_garnet()
    grid = grid_from_couplings(couplings(backend))
    cal = snapshot_from_fake(backend)
    print("Garnet grid:")
    for row in grid:
        print("  " + " ".join(f"{q or '.':>5}" for q in row))

    decoders = ("pymatching_uniform", "pymatching_calibrated", "neural")
    records, started = [], time.perf_counter()
    for rounds in a.rounds:
        prepared = prepare_qiskit(backend, a.distance, rounds, a.basis, calibration=cal, grid=grid)
        ops = prepared["circuit"].count_ops()
        if set(ops) - NATIVE:
            raise ValueError(f"Non-native operations in compiled circuit: {set(ops)-NATIVE}")
        ideal, dec_circuit = prepared["stim_circuit"], prepared["decoder_circuit"]
        placement = prepared["placement"].stim_to_physical
        uniform_circuit = add_noise(ideal, NoiseModel.uniform(a.uniform_p), qubit_map=placement)
        m_uni, m_cal = make_matcher(uniform_circuit), make_matcher(dec_circuit)
        nn = train_neural_decoder(dec_circuit, NeuralConfig(steps=a.steps, seed=a.seed+rounds, device=a.device))
        predict = {"pymatching_uniform": lambda d: m_uni.decode_batch(d, enable_correlations=True),
                   "pymatching_calibrated": lambda d: m_cal.decode_batch(d, enable_correlations=True),
                   "neural": nn.decode_batch}
        test_seed = a.seed+1_000_003+rounds
        sources = {"stim": dec_circuit.compile_detector_sampler(seed=test_seed).sample(a.shots, separate_observables=True)}
        t = time.perf_counter()
        sources["aer"] = aer_shots(backend, prepared, a.shots, test_seed)
        aer_s = time.perf_counter()-t
        rec = dict(rounds=rounds, detectors=ideal.num_detectors, physical_qubits=sorted(set(placement.values())),
                   hardware_report=prepared["report"], aer_seconds=aer_s, neural=nn.metadata(),
                   circuit_sha256=hashlib.sha256(str(prepared["circuit"]).encode()).hexdigest())
        for src, (dets, obs) in sources.items():
            preds = {name: predict[name](dets) for name in decoders}
            rec[src] = dict(shots=len(obs), raw_flip_rate=float(np.any(obs, axis=1).mean()),
                            detection_rate=float(dets.mean()),
                            errors={n: failures(p, obs) for n, p in preds.items()},
                            neural_vs_calibrated=paired_comparison(preds["neural"], preds["pymatching_calibrated"], obs))
        records.append(rec)
        line = "  ".join(f"{src}: " + ", ".join(f"{n.split('_')[-1]} {rec[src]['errors'][n]/a.shots:.4f}"
                                                 for n in decoders) for src in sources)
        print(f"rounds {rounds}: {line}", flush=True)

    fits = {src: {n: fit_ler([r["rounds"] for r in records], [r[src]["errors"][n] for r in records],
                             [r[src]["shots"] for r in records]) for n in decoders}
            for src in ("stim", "aer")} if len(a.rounds) > 1 else {}

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out = Path(a.output)/run_id
    out.mkdir(parents=True, exist_ok=True)
    result = dict(run_id=run_id, device="IQMFakeGarnet (offline)", distance=a.distance, basis=a.basis,
                  no_reset=True, shots_per_point=a.shots, seed=a.seed, uniform_p=a.uniform_p,
                  calibration=cal.metadata, grid=grid, records=records, fits=fits,
                  seconds=time.perf_counter()-started)
    (out/"results.json").write_text(json.dumps(result, indent=2, default=str))
    if fits:
        plot(records, fits, decoders, out/"pl_vs_rounds.png", a)

    print(f"\nP_L per round count ({a.shots} shots each, 95% Wilson):")
    for src in ("stim", "aer"):
        print(f"  [{src}]")
        for n in decoders:
            pts = "  ".join(f"r{r['rounds']}: {r[src]['errors'][n]/a.shots:.4f}"
                            f" [{wilson_interval(r[src]['errors'][n], a.shots)[0]:.4f},"
                            f"{wilson_interval(r[src]['errors'][n], a.shots)[1]:.4f}]" for r in records)
            fit = fits.get(src, {}).get(n)
            eps = f"  eps_L = {fit['epsilon_per_round']:.4f} [{fit['ci95_profile'][0]:.4f}, {fit['ci95_profile'][1]:.4f}]" if fit else ""
            print(f"    {n:<22} {pts}{eps}")
    print("\nneural vs calibrated PyMatching, same shots (only-neural-fails / only-PM-fails, McNemar p):")
    for r in records:
        print(f"  rounds {r['rounds']}: " + "   ".join(
            f"{src} {r[src]['neural_vs_calibrated']['only_a_fails']}/{r[src]['neural_vs_calibrated']['only_b_fails']}"
            f" p={r[src]['neural_vs_calibrated']['mcnemar_exact_p']:.3g}" for src in ("stim", "aer")))
    rep = records[0]["hardware_report"]
    print(f"\nlayout: {rep['physical_qubits']} qubits, routing SWAPs {rep['routing_swaps']}, "
          f"non-native pairs {rep['non_native_pairs']}; wrote {out}")
    return result


def plot(records, fits, decoders, path, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from challenge.decoding import wilson_interval
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    rs = np.array([r["rounds"] for r in records])
    grid_r = np.linspace(rs.min(), rs.max(), 100)
    for ax, src, title in zip(axes, ("stim", "aer"), ("Stim, calibrated Pauli noise", "Aer, IQMFakeGarnet noise")):
        for i, n in enumerate(decoders):
            k = np.array([r[src]["errors"][n] for r in records])
            p = k/a.shots
            ci = np.array([wilson_interval(int(e), a.shots) for e in k])
            x = rs+(i-1)*0.06
            ax.errorbar(x, p, yerr=[p-ci[:, 0], ci[:, 1]-p], fmt="o", capsize=3, label=n)
            eps = fits[src][n]["epsilon_per_round"]
            ax.plot(grid_r, .5*(1-(1-2*eps)**grid_r), lw=1, color=ax.lines[-1].get_color())
        ax.set(title=title, xlabel="QEC rounds", xticks=rs)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("logical error probability $P_L$")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"d={a.distance} no-reset memory on offline Garnet, {a.shots} shots/point, 95% Wilson; "
                 f"fit 2P_L = 1-(1-2eps)^r")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
