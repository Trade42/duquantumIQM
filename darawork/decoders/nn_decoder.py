"""Convolutional recurrent decoder: per-round CNN -> ConvGRU over rounds -> pooled MLP.

Input per round t on the (d+1)x(d+1) ancilla grid (detector coords x/2, y/2):
events on memory-basis stabilizers, events on the other type, a mask of which
detectors exist at (site, t) for each type (no-reset: the other type has no
round-0/1 detectors), and normalized x, y. Spatially fully convolutional and
recurrent in time, so one model takes any odd d and any round count.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import stim

CHANNELS = 6


def grid_layout(circuit: stim.Circuit):
    """Per detector (t, type, iy, ix), plus constant mask/coordinate channels."""
    coords = circuit.get_detector_coordinates()
    k = np.array(sorted(coords))
    c = np.array([coords[i] for i in k])
    x, y, t = (c[:, 0] / 2).astype(int), (c[:, 1] / 2).astype(int), c[:, 2].astype(int)
    first = {}
    for xi, yi, ti in zip(x, y, t):
        first[xi, yi] = min(first.get((xi, yi), ti), ti)
    kind = np.array([0 if first[xi, yi] == 0 else 1 for xi, yi in zip(x, y)])  # 0: memory basis.
    H, W, T = y.max() + 1, x.max() + 1, t.max() + 1
    const = np.zeros((T, 4, H, W), np.float32)
    const[t, kind, y, x] = 1  # Channels 2, 3 of the input: which detectors exist.
    const[:, 2] = (np.arange(W) / (W - 1))[None, :]
    const[:, 3] = (np.arange(H) / (H - 1))[:, None]
    return dict(index=(t, kind, y, x), const=const, shape=(T, H, W), num_detectors=len(k))


def to_grid(dets, layout):
    T, H, W = layout["shape"]
    t, kind, y, x = layout["index"]
    X = np.zeros((len(dets), T, CHANNELS, H, W), np.float32)
    X[:, t, kind, y, x] = dets
    X[:, :, 2:] = layout["const"]
    return X


def build_model(hidden=64):
    import torch
    from torch import nn

    class ConvGRU(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.zr = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
            self.h = nn.Conv2d(2 * c, c, 3, padding=1)

        def forward(self, x, h):
            z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
            return (1 - z) * h + z * torch.tanh(self.h(torch.cat([x, r * h], 1)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(CHANNELS, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
            self.gru = ConvGRU(hidden)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, x):  # x: (B, T, C, H, W) -> logit of a logical flip.
            B, T, _, H, W = x.shape
            h = x.new_zeros(B, hidden, H, W)
            for t in range(T):
                h = self.gru(self.enc(x[:, t]), h)
            site = (x[:, -1, 2:4].sum(1, keepdim=True) > 0).float()  # Stabilizer sites.
            mean = (h * site).sum((2, 3)) / site.sum((2, 3)).clamp_min(1)
            peak = (h - 1e4 * (1 - site)).amax((2, 3))
            return self.head(torch.cat([mean, peak], 1)).squeeze(1)

    return Net()


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(sample, configs, *, steps=4000, batch=512, lr=2e-3, hidden=64, seed=0, log_every=250, out=None, init=None):
    """sample(distance, rounds, basis, shots, seed) -> (ideal_circuit, dets, obs).

    Training seeds start at 2**40 so they never coincide with benchmark test seeds.
    """
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev, model = device(), (NNDecoder.load(init) if init else build_model(hidden))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    layouts, history, running = {}, [], []
    for step in range(steps):
        cfg = configs[rng.integers(len(configs))]
        ideal, dets, obs = sample(*cfg, batch, 2**40 + seed * 10**7 + step)
        if cfg not in layouts:
            layouts[cfg] = grid_layout(ideal)
        x = torch.as_tensor(to_grid(dets, layouts[cfg]), device=dev)
        y = torch.as_tensor(obs[:, 0].astype(np.float32), device=dev)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        running.append(loss.item())
        if (step + 1) % log_every == 0:
            history.append(dict(step=step + 1, loss=float(np.mean(running))))
            print(f"step {step + 1}: loss {history[-1]['loss']:.4f}", flush=True)
            running = []
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=model.state_dict(), hidden=hidden, configs=configs, history=history), out)
    return model, history


class NNDecoder:
    """A trained model bound to one circuit's detector layout."""

    def __init__(self, model, circuit: stim.Circuit, *, batch_size=4096, name="CNN-GRU"):
        self.model, self.layout, self.batch_size, self.name = model.eval(), grid_layout(circuit), batch_size, name
        if circuit.num_observables != 1:
            raise ValueError("NN head predicts a single observable.")

    @staticmethod
    def load(path):
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["hidden"])
        model.load_state_dict(ckpt["state"])
        return model.to(device())

    def logits(self, det_events):
        import torch
        dets = np.asarray(det_events, bool)
        if dets.shape[1] != self.layout["num_detectors"]:
            raise ValueError("Detector count does not match the bound circuit.")
        dev = next(self.model.parameters()).device
        out = []
        with torch.no_grad():
            for s in range(0, len(dets), self.batch_size):
                x = torch.as_tensor(to_grid(dets[s:s + self.batch_size], self.layout), device=dev)
                out.append(self.model(x).cpu().numpy())
        return np.concatenate(out)

    def decode(self, det_events):
        return (self.logits(det_events) > 0)[:, None]


def main(argv=None):
    import argparse
    from ..benchmark import nn_path, sample
    p = argparse.ArgumentParser(description="Train the CNN-GRU decoder on one noise model.")
    p.add_argument("--noise", default="garnet_echo")
    p.add_argument("--distances", type=int, nargs="+", default=[3])
    p.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--bases", nargs="+", default=["Z", "X"])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--hardware", metavar="JOB_DIR", help="train on a real Garnet job's training split")
    p.add_argument("--test-shots", type=int, default=5000, help="held out per circuit; never trained on")
    p.add_argument("--init", help="start from this checkpoint (fine-tune)")
    p.add_argument("--lr", type=float, default=2e-3)
    a = p.parse_args(argv)
    if a.hardware:
        from ..benchmark import hardware_data
        data, _ = hardware_data(a.hardware)
        configs = [(3, r, b) for b, r in sorted(data)]

        def draw(d, r, b, n, s):
            ideal, _, dets, obs = data[b, r]
            rows = np.random.default_rng(s).integers(0, len(dets) - a.test_shots, n)  # Training split only.
            return ideal, dets[rows], obs[rows]
        out = nn_path("garnet_hw")
        train(draw, configs, steps=a.steps, batch=a.batch, hidden=a.hidden, lr=a.lr, out=out, init=a.init)
    else:
        configs = [(d, r, b) for d in a.distances for r in a.rounds for b in a.bases]
        out = nn_path(a.noise)
        train(lambda d, r, b, n, s: sample(a.noise, d, r, b, n, s), configs, steps=a.steps, batch=a.batch,
              hidden=a.hidden, lr=a.lr, out=out, init=a.init)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
