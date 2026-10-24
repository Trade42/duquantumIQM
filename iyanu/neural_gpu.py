"""Trial neural decoder on a GPU (CUDA, Apple MPS) or CPU: detection events -> logical flips.

An MLP reads a shot's whole detection-event vector and predicts each observable flip, so it
is an offline full-history decoder, not a causal streaming one. Weights train only on fresh
shots sampled from the noisy decoder circuit; evaluation uses an independent sampler seed
and frozen weights. Every shot gets a prediction: there is no unresolved-shot fallback.

    python -m iyanu.neural_gpu --distance 3 --rounds 3 --p 0.005
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("neural_gpu needs PyTorch: pip install torch "
                          "(CUDA wheels from pytorch.org on a GPU cluster).") from exc
    return torch


def pick_device(name=None):
    """Explicit name, else the fastest available: cuda, then mps, then cpu."""
    torch = _torch()
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device):
    torch = _torch()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class NeuralConfig:
    hidden: int = 256
    layers: int = 3
    steps: int = 3000
    batch: int = 4096
    lr: float = 2e-3
    seed: int = 0
    device: str | None = None


class NeuralDecoder:
    """`decode_batch(det_events) -> (shots, num_observables)` bool, shot order kept."""

    def __init__(self, model, num_detectors, num_observables, device, config, train_info):
        self.model, self.device, self.config = model.eval(), device, config
        self.num_detectors, self.num_observables = num_detectors, num_observables
        self.train_info = train_info

    def to(self, device):
        """Copy of this decoder on another device (same frozen weights)."""
        import copy
        device = pick_device(str(device))
        return NeuralDecoder(copy.deepcopy(self.model).to(device), self.num_detectors,
                             self.num_observables, device, self.config, self.train_info)

    def decode_batch(self, det_events, enable_correlations=None):
        # decode_syndromes passes PyMatching's option; refuse it rather than ignore it.
        if enable_correlations:
            raise ValueError("Correlated matching does not apply to the neural decoder; "
                             "call decode_syndromes(..., correlated=False).")
        torch = _torch()
        dets = np.asarray(det_events)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(f"Expected 2D detection events with {self.num_detectors} columns, "
                             f"got shape {dets.shape}.")
        if dets.dtype != np.bool_ and not np.isin(dets, (0, 1)).all():
            raise ValueError("Detection events must be binary.")
        with torch.inference_mode():
            x = torch.from_numpy(dets.astype(np.float32, copy=False)).to(self.device)
            return (self.model(x) > 0).cpu().numpy()

    def metadata(self):
        return dict(backend="neural_gpu", device=str(self.device), config=asdict(self.config),
                    num_detectors=self.num_detectors, num_observables=self.num_observables,
                    parameters=sum(p.numel() for p in self.model.parameters()),
                    correlations="not applicable (learned model of the full syndrome)",
                    history="offline full-shot", **self.train_info)


def _mlp(num_in, num_out, hidden, layers):
    nn = _torch().nn
    widths = [num_in] + [hidden]*layers
    blocks = [m for a, b in zip(widths, widths[1:]) for m in (nn.Linear(a, b), nn.ReLU())]
    return nn.Sequential(*blocks, nn.Linear(widths[-1], num_out))


def train_neural_decoder(circuit, config=NeuralConfig(), *, log_every=0):
    """Train on fresh shots from `circuit` (noisy Stim circuit); no shot is reused."""
    torch = _torch()
    device = pick_device(config.device)
    torch.manual_seed(config.seed)
    sampler = circuit.compile_detector_sampler(seed=config.seed)
    model = _mlp(circuit.num_detectors, circuit.num_observables, config.hidden, config.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    sample_s, started, loss = 0., time.perf_counter(), float("nan")
    for step in range(config.steps):
        t = time.perf_counter()
        dets, obs = sampler.sample(config.batch, separate_observables=True)
        sample_s += time.perf_counter()-t
        x = torch.from_numpy(dets.astype(np.float32)).to(device)
        y = torch.from_numpy(obs.astype(np.float32)).to(device)
        out = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        out.backward()
        opt.step()
        sched.step()
        if log_every and (step+1) % log_every == 0:
            loss = out.item()
            print(f"step {step+1}/{config.steps}  loss {loss:.4f}", flush=True)
    _sync(device)
    info = dict(train_shots=config.steps*config.batch, train_seconds=time.perf_counter()-started,
                train_sampling_seconds=sample_s, final_loss=out.item())
    return NeuralDecoder(model, circuit.num_detectors, circuit.num_observables, device, config, info)


def latency_ns(decode, dets, *, repeats=2000, warmup=100):
    """End-to-end single-shot latency (numpy in, numpy out), one shot per call."""
    rows = [dets[i % len(dets)][None, :] for i in range(repeats+warmup)]
    for row in rows[:warmup]:
        decode(row)
    times = np.empty(repeats, dtype=np.int64)
    for i, row in enumerate(rows[warmup:]):
        t = time.perf_counter_ns()
        decode(row)
        times[i] = time.perf_counter_ns()-t
    return dict(repeats=repeats, p50_us=float(np.percentile(times, 50))/1e3,
                p99_us=float(np.percentile(times, 99))/1e3, max_us=float(times.max())/1e3)


def throughput(decode, dets, batch):
    decode(dets[:batch])  # warm
    n, started = 0, time.perf_counter()
    for i in range(0, len(dets), batch):
        decode(dets[i:i+batch])
        n += len(dets[i:i+batch])
    seconds = time.perf_counter()-started
    return dict(batch=batch, shots=n, seconds=seconds, shots_per_second=n/seconds)


def paired_comparison(pred_a, pred_b, obs):
    """Same-shot comparison: failures, Wilson intervals, discordant counts, exact McNemar p."""
    from scipy.stats import binomtest
    from challenge.decoding import wilson_interval
    fail_a, fail_b = np.any(pred_a != obs, axis=1), np.any(pred_b != obs, axis=1)
    n, ea, eb = len(obs), int(fail_a.sum()), int(fail_b.sum())
    only_a, only_b = int((fail_a & ~fail_b).sum()), int((~fail_a & fail_b).sum())
    p = 1. if only_a+only_b == 0 else binomtest(only_a, only_a+only_b).pvalue
    return dict(shots=n, a_errors=ea, b_errors=eb, a_ci95=wilson_interval(ea, n),
                b_ci95=wilson_interval(eb, n), only_a_fails=only_a, only_b_fails=only_b,
                mcnemar_exact_p=p)


def main(argv=None):
    from challenge.surface_code import make_stim_circuit
    from challenge.noise_models import noise_suite, add_noise
    from challenge.decoding import make_matcher
    torch = _torch()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--p", type=float, default=0.005, help="uniform noise strength")
    ap.add_argument("--steps", type=int, default=NeuralConfig.steps)
    ap.add_argument("--batch", type=int, default=NeuralConfig.batch)
    ap.add_argument("--hidden", type=int, default=NeuralConfig.hidden)
    ap.add_argument("--layers", type=int, default=NeuralConfig.layers)
    ap.add_argument("--test-shots", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="results/iyanu/neural_gpu")
    a = ap.parse_args(argv)

    ideal = make_stim_circuit(a.distance, a.rounds, a.basis, no_reset=True, gate_basis="cz")
    circuit = add_noise(ideal, noise_suite(a.p)["uniform"])
    config = NeuralConfig(a.hidden, a.layers, a.steps, a.batch, seed=a.seed, device=a.device)
    print(f"d={a.distance} rounds={a.rounds} p={a.p}: {circuit.num_detectors} detectors; "
          f"training on {pick_device(a.device)}", flush=True)
    nn_dec = train_neural_decoder(circuit, config, log_every=max(1, a.steps//10))

    # Held-out shots: a different sampler seed from training (seed), frozen weights.
    test_seed = a.seed+1_000_003
    dets, obs = circuit.compile_detector_sampler(seed=test_seed).sample(a.test_shots, separate_observables=True)
    matcher = make_matcher(circuit, correlated=True)
    mwpm = lambda d: matcher.decode_batch(d, enable_correlations=True)
    acc = paired_comparison(nn_dec.decode_batch(dets), mwpm(dets), obs)

    cpu_dec = nn_dec.to("cpu")
    decoders = {f"neural_{nn_dec.device.type}": nn_dec.decode_batch, "pymatching_correlated": mwpm}
    if nn_dec.device.type != "cpu":
        decoders["neural_cpu"] = cpu_dec.decode_batch
    timing = {name: dict(latency=latency_ns(f, dets),
                         throughput=[throughput(f, dets, b) for b in (1000, 10_000, min(100_000, len(dets)))])
              for name, f in decoders.items()}

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out = Path(a.output)/run_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"circuit.stim").write_text(str(circuit))
    torch.save(nn_dec.model.state_dict(), out/"model.pt")
    record = dict(run_id=run_id, distance=a.distance, rounds=a.rounds, basis=a.basis, no_reset=True,
                  gate_basis="cz", noise=f"uniform p={a.p}", test_seed=test_seed,
                  circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
                  accuracy_a_neural_b_pymatching=acc, timing=timing, neural=nn_dec.metadata(),
                  environment=dict(platform=platform.platform(), processor=platform.processor(),
                                   torch=torch.__version__, torch_threads=torch.get_num_threads()))
    (out/"results.json").write_text(json.dumps(record, indent=2))

    print(f"\nheld-out shots {acc['shots']}:")
    for label, e, ci in (("neural", acc["a_errors"], acc["a_ci95"]), ("pymatching", acc["b_errors"], acc["b_ci95"])):
        print(f"  {label:<11} P_L = {e/acc['shots']:.5f}  [{ci[0]:.5f}, {ci[1]:.5f}]  ({e} failures)")
    print(f"  only neural fails {acc['only_a_fails']}, only pymatching fails {acc['only_b_fails']}, "
          f"McNemar p = {acc['mcnemar_exact_p']:.3g}")
    print("\nsingle-shot latency (us, p50 / p99)   throughput (shots/s at batch 1k / 10k / 100k)")
    for name, t in timing.items():
        lat, thr = t["latency"], t["throughput"]
        print(f"  {name:<22} {lat['p50_us']:8.1f} / {lat['p99_us']:8.1f}     "
              + " / ".join(f"{x['shots_per_second']:,.0f}" for x in thr))
    print(f"\ntraining {nn_dec.train_info['train_shots']:,} shots in {nn_dec.train_info['train_seconds']:.1f} s; "
          f"wrote {out}")
    return record


if __name__ == "__main__":
    main()
