"""Generic training loop for detection-event classifiers (torch imported lazily).

Streamed shots (fresh Stim samples every step) for Pauli models, a pre-generated
pool for the slow leakage sampler. AdamW, BCEWithLogitsLoss on the observable
flips, warmup + cosine learning rate, AMP on CUDA, early stopping on validation
P_L, periodic checkpoints and a hard wall-clock cap.
"""
from __future__ import annotations
import math
import os
import time
import numpy as np

from collections import deque
from .data import (Dataset, DatasetSpec, LEAKAGE_CHUNK, LEAKAGE_STREAM_BASE, Stream, chunk_seed, sample_shots,
                   uses_leakage_sampler)

HARD_CAP_SECONDS = 45 * 60

DEFAULTS = dict(batch_size=8192, lr=2e-3, final_lr_fraction=0.02, weight_decay=1e-4, warmup_fraction=0.03,
                max_train_shots=50_000_000, max_wall_seconds=HARD_CAP_SECONDS, anneal_by_time=False,
                time_budget_seconds=None, eval_every_steps=200, eval_every_seconds=None,
                patience_evals=12, min_progress_to_stop=0.3,
                checkpoint_seconds=120, amp=True, drop_trivial=True, cpu_threads=None, leakage_stream_workers=4)


class StreamSource:
    """Fresh shots every step: no shot is ever seen twice."""

    def __init__(self, spec: DatasetSpec, batch: int, seed: int):
        self.stream, self.batch, self.pool_shots = Stream(spec, seed), batch, None

    def next(self):
        return self.stream.next(self.batch)


class LeakageStreamSource:
    """Fresh leakage-sampler shots, each used once: worker processes sample seeded chunks ahead of the trainer.

    Training is then bound by the sampler's speed, but no shot is ever repeated.
    """

    def __init__(self, spec: DatasetSpec, batch: int, seed: int, workers: int = 4):
        from concurrent.futures import ProcessPoolExecutor
        self.spec, self.batch, self.seed, self.pool_shots = spec, batch, seed, None
        self.executor = ProcessPoolExecutor(max_workers=workers)
        self.pending, self.next_chunk, self.fresh_shots = deque(), 0, 0
        self.dets, self.obs = [], []
        for _ in range(3 * workers):
            self._submit()

    def _submit(self):
        seed = chunk_seed(self.spec, "train", self.seed, LEAKAGE_STREAM_BASE + self.next_chunk)
        self.pending.append(self.executor.submit(sample_shots, self.spec, LEAKAGE_CHUNK, seed))
        self.next_chunk += 1

    def next(self):
        while sum(len(o) for o in self.obs) < self.batch:
            dets, obs, _ = self.pending.popleft().result()
            self._submit()
            self.dets.append(dets)
            self.obs.append(obs)
        dets, obs = np.concatenate(self.dets), np.concatenate(self.obs)
        self.dets, self.obs = [dets[self.batch:]], [obs[self.batch:]]
        self.fresh_shots += self.batch
        return dets[:self.batch], obs[:self.batch]

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


class PoolSource:
    """Reshuffled epochs over a fixed pool (leakage sampler, or a small smoke budget)."""

    def __init__(self, pool: Dataset, batch: int, seed: int):
        self.pool, self.batch, self.pool_shots = pool, min(batch, pool.shots), pool.shots
        self.rng = np.random.default_rng(seed)
        self.order, self.cursor = self.rng.permutation(pool.shots), 0

    def next(self):
        if self.cursor + self.batch > len(self.order):
            self.order, self.cursor = self.rng.permutation(self.pool_shots), 0
        idx = np.sort(self.order[self.cursor:self.cursor + self.batch])
        self.cursor += self.batch
        return self.pool.dets_packed[idx], self.pool.obs[idx]


def batch_source(train, batch: int, seed: int, cfg: dict | None = None):
    if isinstance(train, Dataset):
        return PoolSource(train, batch, seed)
    if uses_leakage_sampler(train):
        return LeakageStreamSource(train, batch, seed, workers=int((cfg or DEFAULTS)["leakage_stream_workers"]))
    return StreamSource(train, batch, seed)


def unpack_bits(packed, width: int):
    """Little-endian bit unpacking on the tensor's own device: (N, bytes) uint8 -> (N, width) uint8."""
    import torch
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :width]


def lr_at(progress: float, cfg: dict) -> float:
    warm = cfg["warmup_fraction"]
    if progress < warm:
        return cfg["lr"] * max(progress / warm, 1e-3)
    frac = min(1., (progress - warm) / max(1e-9, 1 - warm))
    low = cfg["lr"] * cfg["final_lr_fraction"]
    return low + .5 * (cfg["lr"] - low) * (1 + math.cos(math.pi * frac))


def evaluate_logits(net, packed, obs, width: int, *, use_amp: bool, chunk: int = 65536):
    """Validation failures (trivial shots predict 0) and BCE over the nontrivial shots."""
    import torch
    failures, loss_sum, nontrivial = 0, 0., 0
    with torch.no_grad():
        for start in range(0, len(obs), chunk):
            p, y = packed[start:start + chunk], obs[start:start + chunk]
            active = (p != 0).any(dim=1)
            x = unpack_bits(p[active], width).float()
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
                logits = net(x).float()
            pred = torch.zeros_like(y, dtype=torch.bool)
            pred[active] = logits > 0
            failures += int((pred != y.bool()).any(dim=1).sum())
            loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[active].float(), reduction="sum"))
            nontrivial += int(active.sum())
    return failures, loss_sum / max(1, nontrivial * obs.shape[1])


def _save(path, payload):
    import torch
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_classifier(net, *, train, val: Dataset, cfg: dict, device: str, seed: int, log=print,
                     checkpoint_path=None) -> dict:
    """Train ``net`` (detection events -> logits), restore the best validation state, freeze it."""
    import torch
    cfg = {**DEFAULTS, **cfg}
    np.random.seed(seed)
    torch.manual_seed(seed)
    on_cuda = device.startswith("cuda")
    restore = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    if on_cuda:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.use_deterministic_algorithms(True)
        if cfg["cpu_threads"]:
            torch.set_num_threads(int(cfg["cpu_threads"]))
    try:
        return _train(net, torch, train, val, cfg, device, seed, log, checkpoint_path, on_cuda)
    finally:
        torch.use_deterministic_algorithms(restore[0])
        torch.set_num_threads(restore[1])


def _train(net, torch, train, val, cfg, device, seed, log, checkpoint_path, on_cuda):
    cap = min(float(cfg["max_wall_seconds"]), HARD_CAP_SECONDS)
    budget = float(cfg["time_budget_seconds"] or cap)
    use_amp = bool(cfg["amp"]) and on_cuda
    width = val.num_detectors
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    source = batch_source(train, int(cfg["batch_size"]), seed, cfg)
    batch = source.batch
    max_steps = max(1, math.ceil(cfg["max_train_shots"] / batch))
    val_packed = torch.from_numpy(val.dets_packed).to(device)
    val_obs = torch.from_numpy(val.obs).to(device)
    best = dict(failures=None, loss=math.inf, step=0, state=None)
    history, stale, shots_seen = [], 0, 0
    running = torch.zeros((), device=device)
    running_steps, stop_reason = 0, "max_train_shots reached"
    started = last_ckpt = last_eval = time.perf_counter()
    step = 0
    for step in range(1, max_steps + 1):
        elapsed = time.perf_counter() - started
        progress = step / max_steps
        if cfg["anneal_by_time"]:
            progress = max(progress, elapsed / budget)
        lr = lr_at(min(progress, 1.), cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        dets, obs = source.next()
        p = torch.from_numpy(dets).to(device, non_blocking=True)
        y = torch.from_numpy(obs).to(device, non_blocking=True)
        shots_seen += len(obs)
        if cfg["drop_trivial"]:
            active = (p != 0).any(dim=1)
            p, y = p[active], y[active]
            if not len(y):
                continue
        x = unpack_bits(p, width).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = net(x)
        loss = loss_fn(logits.float(), y.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.detach()
        running_steps += 1
        last = progress >= 1 or time.perf_counter() - started > cap
        timed = cfg["eval_every_seconds"] and time.perf_counter() - last_eval >= cfg["eval_every_seconds"]
        if step % cfg["eval_every_steps"] == 0 or timed or last:
            net.eval()
            failures, val_loss = evaluate_logits(net, val_packed, val_obs, width, use_amp=use_amp)
            net.train()
            last_eval = time.perf_counter()
            train_loss = float(running) / max(1, running_steps)
            running.zero_()
            running_steps = 0
            improved = best["failures"] is None or (failures, val_loss) < (best["failures"], best["loss"])
            if improved:
                best.update(failures=failures, loss=val_loss, step=step,
                            state={k: v.detach().clone() for k, v in net.state_dict().items()})
                stale = 0
            else:
                stale += 1
            elapsed = time.perf_counter() - started
            history.append(dict(step=step, shots_seen=shots_seen, seconds=round(elapsed, 2), lr=lr,
                                train_loss=train_loss, val_loss=val_loss, val_failures=failures))
            log(f"    step {step:6d}  shots {shots_seen:12,d}  {elapsed:7.1f}s  lr {lr:.2e}  train {train_loss:.4f}"
                f"  val {val_loss:.4f}  val fails {failures}/{val.shots}{'  *' if improved else ''}")
            if stale >= cfg["patience_evals"] and progress >= cfg["min_progress_to_stop"]:
                stop_reason = f"early stop: no val improvement in {stale} evaluations"
                break
        if checkpoint_path and time.perf_counter() - last_ckpt > cfg["checkpoint_seconds"]:
            _save(checkpoint_path, dict(step=step, state=net.state_dict(), best_step=best["step"],
                                        best_state=best["state"], history=history))
            last_ckpt = time.perf_counter()
        if time.perf_counter() - started > cap:
            stop_reason = f"wall-clock cap {cap:.0f} s"
            break
        if progress >= 1:
            stop_reason = ("time budget reached" if cfg["anneal_by_time"] and step < max_steps
                           else "max_train_shots reached")
            break
    if on_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    if hasattr(source, "close"):
        source.close()
    net.load_state_dict(best["state"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    if checkpoint_path:
        _save(checkpoint_path, dict(step=step, state=net.state_dict(), best_step=best["step"], history=history))
    pool = source.pool_shots
    return dict(steps=step, batch_size=batch, shots_seen=shots_seen,
                epochs=(shots_seen / pool) if pool else None,
                data=("pool" if pool else "fresh leakage-sampler shots, each used once"
                      if isinstance(source, LeakageStreamSource) else "streamed (fresh Stim shots every step)"),
                pool_shots=pool,
                drop_trivial=cfg["drop_trivial"], wall_seconds=wall, device=device,
                amp="float16 autocast + GradScaler" if use_amp else None,
                final_val_loss=best["loss"], final_val_failures=best["failures"],
                final_val_p_l=best["failures"] / val.shots, val_shots=val.shots, best_step=best["step"],
                stop_reason=stop_reason, history=history, config=cfg,
                reproducible="bit-reproducible (CPU, deterministic algorithms)" if not on_cuda
                             else "not bit-reproducible (CUDA kernels)")


# ---------------------------------------------------------------- recurrent decoders (stage 1)

SEQ_DEFAULTS = dict(batch_size=4096, lr=1e-3, final_lr_fraction=0.05, weight_decay=1e-4, warmup_fraction=0.03,
                    max_train_shots=10**11, max_wall_seconds=HARD_CAP_SECONDS, anneal_by_time=True,
                    time_budget_seconds=None, eval_every_seconds=30, eval_every_steps=10**9, patience_evals=12,
                    min_progress_to_stop=0.5, checkpoint_seconds=120, amp=True, grad_clip=1.0, workers=8,
                    leakage_ahead=2, reservoir_chunks=16, l2_to_init=0.0, cpu_threads=None)


def sample_chunk(spec: DatasetSpec, n: int, seed: int, need_raw: bool):
    """Worker task: (packed dets, obs, packed raw or None) for one seeded chunk."""
    from .data import sample_shots, sample_with_raw
    if not need_raw:
        dets, obs, _ = sample_shots(spec, n, seed)
        return dets, obs, None
    dets, obs, _, raw = sample_with_raw(spec, n, seed)
    return dets, obs, raw


class RawStream:
    """Fresh shots of a Pauli-model spec with their raw record (measurement sampler, then m2d)."""

    def __init__(self, spec: DatasetSpec, seed: int):
        from .data import _asymmetric, _circuits, _flip_rng, STREAM_CHUNK
        ideal, noisy = _circuits(spec)
        self.asym = _asymmetric(spec)
        stream_seed = chunk_seed(spec, "train", seed, STREAM_CHUNK - 1)
        self.sampler = (self.asym[0] if self.asym else noisy).compile_sampler(seed=stream_seed)
        self.converter, self.rng = ideal.compile_m2d_converter(), _flip_rng(stream_seed)

    def next(self, n: int):
        from .data import pack
        from .hardware import apply_readout_flips
        raw = self.sampler.sample(n)
        if self.asym:
            apply_readout_flips(raw, self.asym[1], self.asym[2], self.rng)
        dets, obs = self.converter.convert(measurements=raw, separate_observables=True)
        return pack(dets.view(np.uint8)), obs.view(np.uint8), pack(raw.view(np.uint8))


class MixtureSource:
    """Batches from a sequence.Mixture: a component by weight, then R (curriculum first), fresh shots.

    Pauli components stream in-process (Stim). Leakage-sampler components come from worker
    processes, one seeded chunk per batch, kept ``leakage_ahead`` chunks ahead per spec; when the
    trainer would wait, it reuses a random chunk from that spec's reservoir of recent chunks, and
    counts it (``reused_shots``).
    """

    def __init__(self, mixture, batch: int, seed: int, cfg: dict, need_raw: bool):
        from concurrent.futures import ProcessPoolExecutor
        self.mixture, self.batch, self.seed, self.need_raw = mixture, batch, seed, need_raw
        self.rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x5EC,)))
        self.weights = np.array([c.weight for c in mixture.components], dtype=float)
        self.weights /= self.weights.sum()
        self.streams, self.pending, self.reservoir, self.counter = {}, {}, {}, {}
        self.fresh_shots = self.reused_shots = 0
        self.wait_seconds = 0.0
        self.ahead, self.reservoir_chunks = int(cfg["leakage_ahead"]), int(cfg["reservoir_chunks"])
        leaky = [s for s in mixture.specs() if uses_leakage_sampler(s)]
        self.executor = ProcessPoolExecutor(max_workers=int(cfg["workers"])) if leaky else None
        for spec in leaky:
            self.pending[spec] = deque()
            self.reservoir[spec] = deque(maxlen=self.reservoir_chunks)
            self.counter[spec] = 0
            for _ in range(self.ahead):
                self._submit(spec)
        self.pool_shots = None

    def _submit(self, spec):
        seed = chunk_seed(spec, "train", self.seed, LEAKAGE_STREAM_BASE + self.counter[spec])
        self.counter[spec] += 1
        self.pending[spec].append(self.executor.submit(sample_chunk, spec, self.batch, seed, self.need_raw))

    def allowed_rounds(self, progress: float):
        m = self.mixture
        early = [r for r in m.rounds if r <= m.curriculum_max_rounds]
        return early if progress < m.curriculum_fraction and early else list(m.rounds)

    def next(self, progress: float):
        component = self.mixture.components[int(self.rng.choice(len(self.weights), p=self.weights))]
        rounds = self.allowed_rounds(progress)
        spec = component.spec(self.mixture.distance, int(rounds[int(self.rng.integers(len(rounds)))]), self.mixture.basis)
        if spec not in self.pending:
            if spec not in self.streams:
                self.streams[spec] = RawStream(spec, self.seed) if self.need_raw else Stream(spec, self.seed)
            out = self.streams[spec].next(self.batch)
            self.fresh_shots += self.batch
            return (spec, *out) if self.need_raw else (spec, *out, None)
        queue = self.pending[spec]
        if queue[0].done() or not self.reservoir[spec]:
            started = time.perf_counter()
            chunk = queue.popleft().result()
            self.wait_seconds += time.perf_counter() - started
            self._submit(spec)
            self.reservoir[spec].append(chunk)
            self.fresh_shots += len(chunk[1])
        else:
            chunk = self.reservoir[spec][int(self.rng.integers(len(self.reservoir[spec])))]
            self.reused_shots += len(chunk[1])
        return (spec, *chunk)

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def describe(self) -> dict:
        return dict(data="mixture: fresh streamed shots (Pauli components), worker-sampled leakage chunks with a "
                         "reservoir reused only while the trainer would wait", fresh_shots=self.fresh_shots,
                    reused_shots=self.reused_shots, leakage_wait_seconds=self.wait_seconds,
                    mixture=self.mixture.describe())


class DatasetMixSource:
    """Batches from fixed Datasets (fine-tuning on real shots): a dataset with probability by size, epochs reshuffled."""

    def __init__(self, datasets, batch: int, seed: int, need_raw: bool):
        self.datasets, self.need_raw = list(datasets), need_raw
        if need_raw and any(d.raw_packed is None for d in self.datasets):
            raise ValueError("The raw-record variant needs datasets with their raw record")
        self.rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xF1,)))
        sizes = np.array([d.shots for d in self.datasets], dtype=float)
        self.p = sizes / sizes.sum()
        self.batch = int(batch)
        self.order = [self.rng.permutation(d.shots) for d in self.datasets]
        self.cursor = [0] * len(self.datasets)
        self.pool_shots = int(sizes.sum())
        self.fresh_shots, self.reused_shots, self.wait_seconds = 0, 0, 0.0

    def next(self, progress: float):
        k = int(self.rng.choice(len(self.datasets), p=self.p))
        d = self.datasets[k]
        b = min(self.batch, d.shots)
        if self.cursor[k] + b > d.shots:
            self.order[k], self.cursor[k] = self.rng.permutation(d.shots), 0
        idx = np.sort(self.order[k][self.cursor[k]:self.cursor[k] + b])
        self.cursor[k] += b
        self.fresh_shots += b
        return d.spec, d.dets_packed[idx], d.obs[idx], (d.raw_packed[idx] if self.need_raw else None)

    def describe(self) -> dict:
        return dict(data="fixed datasets (fine-tuning), sampled by size, reshuffled epochs", pool_shots=self.pool_shots,
                    datasets=[dict(id=d.manifest.get("dataset_id"), split=d.split, shots=d.shots,
                                   label=d.manifest.get("label", d.manifest.get("source"))) for d in self.datasets])


def evaluate_sequence(decoder, vals, *, use_amp: bool) -> tuple[int, int, float, list]:
    """Validation failures, shots and mean BCE (non-trivial shots), per set too."""
    import torch
    from .sequence import gather, gather_raw
    failures = shots = 0
    loss_sum, active_total, per = 0.0, 0, []
    with torch.no_grad():
        for v in vals:
            lay = decoder.layouts.get(v["rounds"])
            x = unpack_bits(v["packed"], v["width"])
            active = (x != 0).any(dim=1)
            pred = torch.zeros_like(v["obs"], dtype=torch.bool)
            if bool(active.any()):
                r, f = gather(x[active], lay)
                raw = gather_raw(unpack_bits(v["raw"], v["raw_width"])[active], lay) if v["raw"] is not None else None
                with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
                    logits = decoder.forward(decoder.net, r, f, raw, lay)[0].float()
                pred[active] = logits > 0
                loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, v["obs"][active].float(), reduction="sum"))
                active_total += int(active.sum())
            fail_v = int((pred != v["obs"].bool()).any(dim=1).sum())
            failures += fail_v
            shots += len(v["obs"])
            per.append(dict(id=v["id"], rounds=v["rounds"], failures=fail_v, shots=len(v["obs"])))
    return failures, shots, loss_sum / max(1, active_total), per


def train_sequence(decoder, *, train, val, cfg: dict, device: str, seed: int, log=print, checkpoint_path=None) -> dict:
    """Train a SequenceDecoder on a Mixture (pre-training) or a list of Datasets (fine-tuning)."""
    import torch
    cfg = {**SEQ_DEFAULTS, **cfg}
    np.random.seed(seed)
    torch.manual_seed(seed)
    on_cuda = device.startswith("cuda")
    restore = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    if on_cuda:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.use_deterministic_algorithms(True)
        if cfg["cpu_threads"]:
            torch.set_num_threads(int(cfg["cpu_threads"]))
    try:
        return _train_sequence(decoder, torch, train, val, cfg, device, seed, log, checkpoint_path, on_cuda)
    finally:
        torch.use_deterministic_algorithms(restore[0])
        torch.set_num_threads(restore[1])


def _train_sequence(decoder, torch, train, val, cfg, device, seed, log, checkpoint_path, on_cuda):
    from .sequence import Mixture, gather, gather_raw, layout_for
    net = decoder.net
    cap = min(float(cfg["max_wall_seconds"]), HARD_CAP_SECONDS)
    budget = float(cfg["time_budget_seconds"] or cap)
    use_amp = bool(cfg["amp"]) and on_cuda
    need_raw = decoder.uses_raw
    batch = int(cfg["batch_size"])
    source = (MixtureSource(train, batch, seed, cfg, need_raw) if isinstance(train, Mixture)
              else DatasetMixSource(train, batch, seed, need_raw))
    vals = []
    for v in val:
        if need_raw and v.raw_packed is None:
            raise ValueError("The raw-record variant needs validation sets with their raw record")
        vals.append(dict(id=v.manifest.get("dataset_id"), rounds=v.spec.rounds, width=v.num_detectors,
                         packed=torch.from_numpy(v.dets_packed).to(device), obs=torch.from_numpy(v.obs).to(device),
                         raw=None if v.raw_packed is None else torch.from_numpy(v.raw_packed).to(device),
                         raw_width=v.circuit.num_measurements))
    val_shots = sum(len(v["obs"]) for v in vals)
    net.to(device).train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    l2 = float(cfg["l2_to_init"])
    init = [p.detach().clone() for p in params] if l2 else None
    max_steps = max(1, math.ceil(cfg["max_train_shots"] / batch))
    best = dict(failures=None, loss=math.inf, step=0, state=None)
    history, stale, shots_seen, step = [], 0, 0, 0
    running = torch.zeros((), device=device)
    running_steps, stop_reason = 0, "max_train_shots reached"
    started = last_ckpt = last_eval = time.perf_counter()
    try:
        for step in range(1, max_steps + 1):
            elapsed = time.perf_counter() - started
            progress = step / max_steps
            if cfg["anneal_by_time"]:
                progress = max(progress, elapsed / budget)
            lr = lr_at(min(progress, 1.), cfg)
            for group in opt.param_groups:
                group["lr"] = lr
            spec, dets, obs, raw = source.next(progress)
            lay = decoder.layouts.get(spec.rounds)
            p = torch.from_numpy(np.ascontiguousarray(dets)).to(device, non_blocking=True)
            y = torch.from_numpy(np.ascontiguousarray(obs)).to(device, non_blocking=True).float()
            shots_seen += len(obs)
            x = unpack_bits(p, lay["num_detectors"])
            active = (x != 0).any(dim=1)            # Empty syndromes take the decoder's fast path.
            if not bool(active.any()):
                continue
            r, f = gather(x[active], lay)
            y = y[active]
            rb = None
            if need_raw:
                rt = torch.from_numpy(np.ascontiguousarray(raw)).to(device, non_blocking=True)
                width = layout_for(decoder.distance, spec.rounds, decoder.basis)["num_measurements"]
                rb = gather_raw(unpack_bits(rt, width)[active], lay)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                main, aux = decoder.forward(net, r, f, rb, lay)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(main.float(), y)
            extra = decoder.aux_loss(aux, y, r, f, lay)
            if extra is not None:
                loss = loss + extra
            if l2:
                loss = loss + l2 * sum(((q - q0) ** 2).sum() for q, q0 in zip(params, init))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            running += loss.detach()
            running_steps += 1
            now = time.perf_counter()
            last = progress >= 1 or now - started > cap
            if now - last_eval >= cfg["eval_every_seconds"] or step % cfg["eval_every_steps"] == 0 or last:
                net.eval()
                failures, n_val, val_loss, per = evaluate_sequence(decoder, vals, use_amp=use_amp)
                net.train()
                last_eval = time.perf_counter()
                train_loss = float(running) / max(1, running_steps)
                running.zero_()
                running_steps = 0
                improved = best["failures"] is None or (failures, val_loss) < (best["failures"], best["loss"])
                if improved:
                    best.update(failures=failures, loss=val_loss, step=step, per=per,
                                state={k: v.detach().clone() for k, v in net.state_dict().items()})
                    stale = 0
                else:
                    stale += 1
                elapsed = time.perf_counter() - started
                history.append(dict(step=step, shots_seen=shots_seen, seconds=round(elapsed, 2), lr=lr,
                                    train_loss=train_loss, val_loss=val_loss, val_failures=failures,
                                    fresh=source.fresh_shots, reused=source.reused_shots))
                log(f"    step {step:6d}  shots {shots_seen:12,d}  {elapsed:7.1f}s  lr {lr:.2e}  train {train_loss:.4f}"
                    f"  val {val_loss:.4f}  val fails {failures}/{n_val} ({failures/max(1, n_val):.4f})"
                    f"{'  *' if improved else ''}  [reused {source.reused_shots:,}, wait {source.wait_seconds:.0f}s]")
                if stale >= cfg["patience_evals"] and progress >= cfg["min_progress_to_stop"]:
                    stop_reason = f"early stop: no val improvement in {stale} evaluations"
                    break
            if checkpoint_path and time.perf_counter() - last_ckpt > cfg["checkpoint_seconds"]:
                _save(checkpoint_path, dict(step=step, state=net.state_dict(), best_step=best["step"],
                                            best_state=best["state"], history=history))
                last_ckpt = time.perf_counter()
            if time.perf_counter() - started > cap:
                stop_reason = f"wall-clock cap {cap:.0f} s"
                break
            if progress >= 1:
                stop_reason = ("time budget reached" if cfg["anneal_by_time"] and step < max_steps
                               else "max_train_shots reached")
                break
    finally:
        if hasattr(source, "close"):
            source.close()
    if on_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    if best["state"] is None:
        raise RuntimeError("Training ended before the first validation")
    net.load_state_dict(best["state"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    if checkpoint_path:
        _save(checkpoint_path, dict(step=step, state=net.state_dict(), best_step=best["step"], history=history))
    return dict(steps=step, batch_size=batch, shots_seen=shots_seen, **source.describe(),
                epochs=(shots_seen / source.pool_shots) if source.pool_shots else None,
                wall_seconds=wall, device=device, amp="float16 autocast + GradScaler" if use_amp else None,
                final_val_loss=best["loss"], final_val_failures=best["failures"], val_shots=val_shots,
                final_val_p_l=best["failures"] / max(1, val_shots), best_step=best["step"], val_per_set=best.get("per"),
                stop_reason=stop_reason, history=history, config=cfg, l2_to_init=l2 or None,
                reproducible="bit-reproducible (CPU, deterministic algorithms)" if not on_cuda
                             else "not bit-reproducible (CUDA kernels)")
