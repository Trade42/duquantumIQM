"""Space-time convolutional decoders on a GPU (CUDA, Apple MPS) or CPU.

Detection events sit on a (rounds+1) x (d+1) x (d+1) lattice (Stim detector coordinates
x/2, y/2, t), so 3D convolutions see local error patterns the same way everywhere.

ConvDecoder     full decoder: lattice -> 3D conv trunk -> logical flip logits.
ConvPredecoder  pre-decoder: the same trunk scores every graphlike edge of the decoder's
                detector error model; confident edges whose detectors are all lit are applied
                (each a local maximum at its detectors, so applied edges never overlap), and
                PyMatching decodes the residual. Prediction = edge observables XOR matching.

Both train only on fresh simulated shots from the noisy decoder circuit (the pre-decoder on
the DEM sampler, which also returns which error mechanisms fired); evaluation shots are
independent. Both are offline full-history decoders.
"""
from __future__ import annotations
import time
from dataclasses import asdict, dataclass
import numpy as np

from iyanu.neural_gpu import _sync, _torch, pick_device


class SpaceTimeLayout:
    """Detector index <-> (t, row, col) on the lattice; channel 1 marks real detector sites."""

    def __init__(self, circuit):
        coords = circuit.get_detector_coordinates()
        if len(coords) != circuit.num_detectors or any(len(c) < 3 for c in coords.values()):
            raise ValueError("Every detector needs (x, y, t) coordinates.")
        xs = sorted({c[0] for c in coords.values()})
        ys = sorted({c[1] for c in coords.values()})
        ts = sorted({c[2] for c in coords.values()})
        xi, yi, ti = ({v: i for i, v in enumerate(a)} for a in (xs, ys, ts))
        self.shape = (len(ts), len(ys), len(xs))
        self.num_detectors = circuit.num_detectors
        flat = [ti[coords[k][2]]*len(ys)*len(xs) + yi[coords[k][1]]*len(xs) + xi[coords[k][0]]
                for k in range(circuit.num_detectors)]
        if len(set(flat)) != len(flat):
            raise ValueError("Two detectors share a lattice site.")
        self.flat_index = np.asarray(flat, dtype=np.int64)
        mask = np.zeros(int(np.prod(self.shape)), dtype=np.float32)
        mask[self.flat_index] = 1
        self.mask = mask.reshape(self.shape)

    def to_lattice(self, x, torch):
        """(batch, detectors) float tensor -> (batch, 2, T, H, W): events and site mask."""
        b = x.shape[0]
        grid = torch.zeros((b, int(np.prod(self.shape))), device=x.device, dtype=x.dtype)
        grid[:, torch.as_tensor(self.flat_index, device=x.device)] = x
        mask = torch.as_tensor(self.mask, device=x.device).expand(b, *self.shape)
        return torch.stack((grid.view(b, *self.shape), mask), dim=1)


def _trunk(channels, layers):
    nn = _torch().nn
    blocks, c_in = [], 2
    for _ in range(layers):
        blocks += [nn.Conv3d(c_in, channels, 3, padding=1), nn.GELU()]
        c_in = channels
    return nn.Sequential(*blocks)


@dataclass(frozen=True)
class ConvConfig:
    channels: int = 32
    layers: int = 4
    head: int = 256
    steps: int = 3000
    batch: int = 2048
    lr: float = 2e-3
    seed: int = 0
    device: str | None = None
    threshold: float = 0.5  # pre-decoder only: minimum edge probability to apply


def _validated(dets, num_detectors):
    dets = np.asarray(dets)
    if dets.ndim != 2 or dets.shape[1] != num_detectors:
        raise ValueError(f"Expected 2D detection events with {num_detectors} columns, got shape {dets.shape}.")
    if dets.dtype != np.bool_ and not np.isin(dets, (0, 1)).all():
        raise ValueError("Detection events must be binary.")
    return dets


def _chunks(dets, size=4096):
    """Decode in fixed-size pieces: the pre-decoder's edge features are shots x edges x 3C."""
    return [dets[i:i+size] for i in range(0, len(dets), size)]


def _optimizer(model, config):
    torch = _torch()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    return opt, torch.optim.lr_scheduler.OneCycleLR(opt, config.lr, total_steps=config.steps, pct_start=0.1)


def _no_correlations(enable_correlations, name):
    if enable_correlations:
        raise ValueError(f"Correlated matching does not apply to {name}; call decode_syndromes(..., correlated=False).")


class ConvDecoder:
    """`decode_batch(det_events) -> (shots, num_observables)` bool."""

    def __init__(self, circuit, config=ConvConfig()):
        torch, nn = _torch(), _torch().nn
        self.config, self.device = config, pick_device(config.device)
        self.layout = SpaceTimeLayout(circuit)
        self.num_detectors, self.num_observables = circuit.num_detectors, circuit.num_observables
        t, h, w = self.layout.shape
        torch.manual_seed(config.seed)
        self.model = nn.Sequential(_trunk(config.channels, config.layers), nn.Conv3d(config.channels, 8, 1),
                                   nn.GELU(), nn.Flatten(), nn.Linear(8*t*h*w, config.head), nn.GELU(),
                                   nn.Linear(config.head, self.num_observables)).to(self.device)
        self.train_info = {}

    def logits(self, x):
        return self.model(self.layout.to_lattice(x, _torch()))

    def train(self, circuit, *, log_every=0):
        torch = _torch()
        sampler = circuit.compile_detector_sampler(seed=self.config.seed)
        opt, sched = _optimizer(self.model, self.config)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        started, sample_s = time.perf_counter(), 0.
        for step in range(self.config.steps):
            t = time.perf_counter()
            dets, obs = sampler.sample(self.config.batch, separate_observables=True)
            sample_s += time.perf_counter()-t
            x = torch.from_numpy(dets.astype(np.float32)).to(self.device)
            y = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            loss = loss_fn(self.logits(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log_every and (step+1) % log_every == 0:
                print(f"  conv step {step+1}/{self.config.steps} loss {loss.item():.4f}", flush=True)
        _sync(self.device)
        self.model.eval()
        self.train_info = dict(train_shots=self.config.steps*self.config.batch,
                               train_seconds=time.perf_counter()-started, train_sampling_seconds=sample_s,
                               final_loss=loss.item())
        return self

    def decode_batch(self, det_events, enable_correlations=None):
        _no_correlations(enable_correlations, "ConvDecoder")
        torch = _torch()
        dets = _validated(det_events, self.num_detectors)
        with torch.inference_mode():
            return np.concatenate([(self.logits(torch.from_numpy(c.astype(np.float32)).to(self.device)) > 0).cpu().numpy()
                                   for c in _chunks(dets)])

    def metadata(self):
        return dict(backend="conv_full", device=str(self.device), config=asdict(self.config),
                    lattice=self.layout.shape, parameters=sum(p.numel() for p in self.model.parameters()),
                    history="offline full-shot", **self.train_info)


def dem_edges(dem):
    """Graphlike edges of a decomposed DEM, merged by detector set.

    Returns (edges, error_to_edges): edges is a list of dicts with detectors (1 or 2),
    observable mask and combined prior; error_to_edges[i] lists the edges that DEM error i
    flips (its decomposition components, mod 2). Where components with the same detectors
    disagree on observables, the more probable observable mask is kept and the count reported.
    """
    import math
    by_dets, error_to_edges = {}, []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        components, dets, obs = [], [], 0
        for t in inst.targets_copy() + [None]:
            if t is None or t.is_separator():
                components.append((tuple(sorted(dets)), obs))
                dets, obs = [], 0
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs |= 1 << t.val
        flipped = set()
        for key, o in components:
            if not key:
                continue
            if len(key) > 2:
                raise ValueError("DEM is not graphlike; use decompose_errors=True.")
            entry = by_dets.setdefault(key, {"weights": {}})
            entry["weights"][o] = entry["weights"].get(o, 0.) + p
            flipped ^= {key}
        error_to_edges.append(flipped)
    keys = sorted(by_dets)
    index = {k: i for i, k in enumerate(keys)}
    edges, conflicts = [], 0
    for k in keys:
        w = by_dets[k]["weights"]
        conflicts += len(w) > 1
        obs = max(w, key=w.get)
        prior = 1-math.prod(1-q for q in w.values())
        edges.append(dict(detectors=k, observables=obs, prior=prior))
    return edges, [sorted(index[k] for k in f) for f in error_to_edges], conflicts


class ConvPredecoder:
    """Conv edge scorer + PyMatching on the residual; `decode_batch` like ConvDecoder."""

    def __init__(self, circuit, config=ConvConfig()):
        import scipy.sparse as sp
        from challenge.decoding import make_matcher
        torch, nn = _torch(), _torch().nn
        self.config, self.device = config, pick_device(config.device)
        self.layout = SpaceTimeLayout(circuit)
        self.num_detectors, self.num_observables = circuit.num_detectors, circuit.num_observables
        self.dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        self.matcher = make_matcher(circuit, correlated=True)
        edges, error_to_edges, self.observable_conflicts = dem_edges(self.dem)
        n_e, n_d = len(edges), self.num_detectors
        self.num_edges = n_e
        rows = [i for i, e in enumerate(edges) for _ in e["detectors"]]
        cols = [d for e in edges for d in e["detectors"]]
        # Edge x detector incidence (GF(2)) and DEM error -> edge map for training labels.
        self.incidence = sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(n_e, n_d))
        er = [i for i, f in enumerate(error_to_edges) for _ in f]
        ec = [e for f in error_to_edges for e in f]
        self.error_map = sp.csr_matrix((np.ones(len(er), np.int32), (er, ec)), shape=(len(error_to_edges), n_e))
        self.edge_obs = np.array([[e["observables"] >> k & 1 for k in range(self.num_observables)]
                                  for e in edges], dtype=np.uint8)
        u = np.array([e["detectors"][0] for e in edges])
        v = np.array([e["detectors"][1] if len(e["detectors"]) == 2 else -1 for e in edges])
        self.u, self.v = torch.as_tensor(u, device=self.device), torch.as_tensor(v, device=self.device)
        self.boundary = torch.as_tensor(v < 0, device=self.device)
        self.prior_logit = torch.as_tensor(np.log([e["prior"]/(1-e["prior"]) for e in edges]),
                                           dtype=torch.float32, device=self.device)
        c = config.channels
        torch.manual_seed(config.seed)
        self.trunk = _trunk(c, config.layers).to(self.device)
        self.edge_embed = nn.Embedding(n_e, c).to(self.device)
        self.boundary_vec = nn.Parameter(torch.zeros(c, device=self.device))
        self.head = nn.Sequential(nn.Linear(3*c+3, config.head // 2), nn.GELU(), nn.Linear(config.head // 2, 1)).to(self.device)
        self.params = [*self.trunk.parameters(), *self.edge_embed.parameters(), self.boundary_vec, *self.head.parameters()]
        self.train_info = {}

    def edge_logits(self, x):
        torch = _torch()
        b = x.shape[0]
        feats = self.trunk(self.layout.to_lattice(x, torch))            # (B, C, T, H, W)
        feats = feats.flatten(2)[:, :, torch.as_tensor(self.layout.flat_index, device=x.device)]  # (B, C, D)
        feats = feats.transpose(1, 2)                                     # (B, D, C)
        fu = feats[:, self.u]
        fv = torch.where(self.boundary[None, :, None], self.boundary_vec.expand(b, self.num_edges, -1),
                         feats[:, self.v.clamp(min=0)])
        xu = x[:, self.u]
        xv = torch.where(self.boundary[None], torch.ones_like(xu), x[:, self.v.clamp(min=0)])
        z = torch.cat((fu+fv, fu*fv, self.edge_embed.weight.expand(b, -1, -1),
                       xu[..., None], xv[..., None], self.boundary[None, :, None].float().expand(b, -1, 1)), dim=-1)
        return self.head(z).squeeze(-1)+self.prior_logit

    def train(self, *, log_every=0):
        torch = _torch()
        sampler = self.dem.compile_sampler(seed=self.config.seed)
        opt, sched = _optimizer(torch.nn.ParameterList(self.params), self.config)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        started, sample_s = time.perf_counter(), 0.
        for step in range(self.config.steps):
            t = time.perf_counter()
            dets, _, errs = sampler.sample(self.config.batch, return_errors=True)
            labels = (self.error_map.T @ errs.T.astype(np.int32)).T % 2  # (B, edges)
            sample_s += time.perf_counter()-t
            x = torch.from_numpy(dets.astype(np.float32)).to(self.device)
            y = torch.from_numpy(labels.astype(np.float32)).to(self.device)
            loss = loss_fn(self.edge_logits(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log_every and (step+1) % log_every == 0:
                print(f"  predecoder step {step+1}/{self.config.steps} loss {loss.item():.5f}", flush=True)
        _sync(self.device)
        self.train_info = dict(train_shots=self.config.steps*self.config.batch,
                               train_seconds=time.perf_counter()-started, train_sampling_seconds=sample_s,
                               final_loss=loss.item(), edges=self.num_edges,
                               observable_conflicts=self.observable_conflicts)
        return self

    def select_edges(self, x):
        """Edges to apply: probability >= threshold, all detectors lit, local max at each detector."""
        torch = _torch()
        b = x.shape[0]
        score = torch.sigmoid(self.edge_logits(x))
        lit = (x[:, self.u] > 0) & (self.boundary[None] | (x[:, self.v.clamp(min=0)] > 0))
        score = torch.where(lit & (score >= self.config.threshold), score, torch.zeros_like(score))
        score = score + 1e-7*torch.arange(self.num_edges, device=x.device)/self.num_edges  # break ties
        best = torch.zeros((b, self.num_detectors), device=x.device)
        best = best.scatter_reduce(1, self.u.expand(b, -1), score, "amax")
        vb = self.v.clamp(min=0).expand(b, -1)
        best = best.scatter_reduce(1, vb, torch.where(self.boundary[None], torch.zeros_like(score), score), "amax")
        ok_u = score >= best.gather(1, self.u.expand(b, -1))
        ok_v = self.boundary[None] | (score >= best.gather(1, vb))
        return (score > 1e-6) & lit & ok_u & ok_v

    def decode_batch(self, det_events, enable_correlations=None, *, return_stats=False):
        _no_correlations(enable_correlations, "ConvPredecoder (its PyMatching stage is always correlated)")
        torch = _torch()
        dets = _validated(det_events, self.num_detectors).astype(bool, copy=False)
        with torch.inference_mode():
            applied = np.concatenate([self.select_edges(torch.from_numpy(c.astype(np.float32)).to(self.device)).cpu().numpy()
                                      for c in _chunks(dets)]) if len(dets) else np.zeros((0, self.num_edges), bool)
        # Applied edges never share a detector, so each column sum is 0 or 1.
        residual = dets ^ (self.incidence.T @ applied.T.astype(np.uint8)).T.astype(bool)
        obs_pre = (applied.astype(np.int64) @ self.edge_obs) % 2
        need = residual.any(axis=1)
        matched = np.zeros((len(dets), self.num_observables), dtype=np.uint8)
        if need.any():
            matched[need] = self.matcher.decode_batch(residual[need], enable_correlations=True)
        pred = (obs_pre ^ matched).astype(bool)
        if not return_stats:
            return pred
        return pred, dict(events_in=float(dets.sum(1).mean()), events_left=float(residual.sum(1).mean()),
                          cleared_fraction=float(1-need.mean()), edges_applied=float(applied.sum(1).mean()))

    def metadata(self):
        return dict(backend="conv_predecoder+pymatching", device=str(self.device), config=asdict(self.config),
                    lattice=self.layout.shape,
                    parameters=sum(p.numel() for p in self.params), matcher="PyMatching, correlated, same DEM",
                    history="offline full-shot", **self.train_info)
