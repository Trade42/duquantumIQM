"""A measured-history recurrent residual decoder, with correlated MWPM prior.

One recurrent architecture handles every round count. The input consists only
of measured detection events, their fixed coordinates/masks, and a prediction
computed from those events. No observable labels or hidden flags enter decode.
Optional torch is imported only when constructing/training a recurrent model.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import stim

from .hybrid import binary_rows


def layout(circuit):
    coords = circuit.get_detector_coordinates()
    if len(coords) != circuit.num_detectors or not coords or any(len(v) != 3 for v in coords.values()):
        raise ValueError("Require three-dimensional coordinates for every detector.")
    spatial = sorted({tuple(v[:2]) for v in coords.values()})
    # d=3 has eight stabilizer positions. Initialization/terminal checks expose
    # only four; their absent entries are masked, never treated as zero data.
    canonical = [(0., 4.), (2., 0.), (2., 2.), (2., 4.), (4., 2.), (4., 4.), (4., 6.), (6., 2.)]
    if not set(spatial) <= set(canonical):
        raise ValueError("History decoder currently supports the distance-three patch only.")
    cells = [(int(coords[i][2]), canonical.index(tuple(coords[i][:2]))) for i in range(circuit.num_detectors)]
    if len(set(cells)) != len(cells) or any(t < 0 for t, _ in cells):
        raise ValueError("Ambiguous detector coordinates.")
    return cells


def encode(dets, cells, prior):
    dets = binary_rows(dets, len(cells))
    prior = binary_rows(prior, 1, "Prior predictions")
    if len(prior) != len(dets):
        raise ValueError("Prediction and syndrome shot counts differ.")
    steps = max(t for t, _ in cells)+1
    x = np.zeros((len(dets), steps, 20), np.float32)
    for j, (t, site) in enumerate(cells):
        x[:, t, site] = dets[:, j]
        x[:, t, 8+site] = 1
    x[:, :, 16] = np.arange(steps)[None, :]/8
    x[:, 0, 17] = 1
    x[:, -1, 18] = 1
    x[:, :, 19] = prior
    return x


def network(hidden=32):
    import torch

    class ResidualGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(20, hidden, batch_first=True, bidirectional=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(2*hidden, hidden), torch.nn.Tanh(),
                                            torch.nn.Linear(hidden, 1))
            self.head[-1].bias.data.fill_(-2.)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(torch.cat((h[-2], h[-1]), dim=1))

    return ResidualGRU()


class HistoryDecoder:
    def __init__(self, module, circuit, dem, *, threshold=.5, metadata=None):
        from darawork.decoders.ensemble_decoder import MatchingDecoder
        if circuit.num_observables != 1 or dem.num_observables != 1 or dem.num_detectors != circuit.num_detectors:
            raise ValueError("History prior requires one observable and matching detector dimensions.")
        if not 0 < threshold < 1:
            raise ValueError("Correction threshold must lie strictly between zero and one.")
        self.module, self.cells = module, layout(circuit)
        self.prior = MatchingDecoder(dem, correlated=True)
        self.dem, self.circuit = dem, circuit
        self.threshold, self.metadata = threshold, metadata or {}

    def decode_batch(self, det_events, *, enable_correlations=None):
        if enable_correlations is False:
            raise ValueError("History model requires its correlated MWPM prior.")
        import torch
        dets = binary_rows(det_events, len(self.cells))
        if not len(dets):
            return np.zeros((0, 1), bool)
        self.module.eval()
        prior = self.prior.decode(dets)
        x = encode(dets, self.cells, prior)
        with torch.inference_mode():
            probability = np.concatenate([self.module(torch.from_numpy(x[i:i+2048])).sigmoid().numpy()
                                          for i in range(0, len(x), 2048)])
        return prior ^ (probability > self.threshold)


def train(records, validation, *, steps=1200, seed=7319, hidden=32):
    """Fit on explicit records; checkpoint/threshold on separate tuning records.

    Each record is (circuit, DEM, dets, observable labels). Sampling is balanced
    across round counts rather than dominated by short archived experiments.
    This validation is development data; independent confirmation is mandatory.
    """
    import torch
    from darawork.decoders.ensemble_decoder import MatchingDecoder
    if not records or not validation:
        raise ValueError("Independent training and development records required.")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    module = network(hidden)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def convert(rec):
        circuit, dem, dets, obs = rec
        obs = binary_rows(obs, 1, "Training labels")
        pred = MatchingDecoder(dem).decode(dets)
        return torch.from_numpy(encode(dets, layout(circuit), pred)), torch.from_numpy((obs ^ pred).astype(np.float32)), pred, obs

    training = [convert(r) for r in records]
    valid = [convert(r) for r in validation]
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    for step in range(steps):
        x, y, _, _ = training[int(rng.integers(len(training)))]
        indices = rng.integers(len(x), size=min(512, len(x)))
        module.train()
        loss = loss_fn(module(x[indices]), y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 2.)
        optimizer.step()
        if (step+1) % 100 == 0:
            module.eval()
            with torch.inference_mode():
                value = sum(float(loss_fn(module(x), y)) for x, y, _, _ in valid)/len(valid)
            history.append([step+1, value])
            if value < best:
                best, best_state = value, {k: v.detach().clone() for k, v in module.state_dict().items()}
    if best_state is None:
        raise ValueError("Training requires at least 100 steps.")
    module.load_state_dict(best_state)
    module.eval()
    return module, dict(hidden=hidden, seed=seed, steps=steps, training_seconds=time.perf_counter()-started,
                        train_shots=sum(len(r[2]) for r in records), tuning_shots=sum(len(r[2]) for r in validation),
                        checkpoint_rule="minimum tuning binary cross entropy", history=history,
                        inputs="measured detector history, fixed coordinate mask, correlated MWPM prediction",
                        inference="offline complete measured history", hidden_leakage_inputs=False,
                        approximate_disjoint_errors=True, correlations="bidirectional GRU plus correlated MWPM")


def save(module, metadata, path):
    path = Path(path)
    np.savez_compressed(path.with_suffix(".npz"), **{k: v.detach().numpy() for k, v in module.state_dict().items()})
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf8")


def load(path):
    import torch
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    module = network(metadata["hidden"])
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as weights:
        module.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files})
    module.eval()
    return module, metadata
