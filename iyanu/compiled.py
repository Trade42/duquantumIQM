"""Compile a trained GPU decoder into a lookup table with an MLP fallback, for nanosecond CPU decoding.

The fine-tuned ensemble (mean of MLP and conv flip probabilities) is evaluated once, on the GPU, on
  dense:  every syndrome, when there are at most `dense_max` detectors (2**24 entries = 2 MiB of bits);
  sparse: every distinct syndrome in the training shots plus a large simulated sample, otherwise.
A dense table answers every shot with one memory read. A sparse table (sorted keys + bits) answers
the syndromes it holds; the rest go to the fine-tuned MLP, exported as float32 weights. Test shots
are never used to build a table, so the table's hit rate on them is honest.

The key of a shot is sum_j det_j << j (detector j is bit j), the same as finetune_hw._keys.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def keys(dets):
    dets = np.asarray(dets)
    if dets.shape[1] > 63:
        raise ValueError("at most 63 detectors")
    return dets.astype(np.uint64) @ (np.uint64(1) << np.arange(dets.shape[1], dtype=np.uint64))


def unpack(k, n):
    """uint64 keys -> (len, n) uint8 detection events."""
    return ((np.asarray(k, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def ensemble_prob(mlp, conv, dets, *, chunk=1 << 16):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(dets), chunk):
            x = torch.from_numpy(np.asarray(dets[i:i+chunk], dtype=np.float32)).to(conv.device)
            p = (torch.sigmoid(mlp.model(x.to(mlp.device))).to(conv.device) + torch.sigmoid(conv.logits(x)))/2
            out.append(p[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def mlp_weights(mlp):
    """[(W, b), ...] float32, Linear layers in order (ReLU between, none after the last)."""
    import torch
    return [(m.weight.detach().float().cpu().numpy(), m.bias.detach().float().cpu().numpy())
            for m in mlp.model if isinstance(m, torch.nn.Linear)]


def mlp_forward(layers, dets):
    h = np.asarray(dets, dtype=np.float32)
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers)-1:
            h = np.maximum(h, 0)
    return h[:, 0] > 0


def compile_ensemble(mlp, conv, n_det, path, *, cover_dets=None, dense_max=24, chunk=1 << 20):
    """Write the compiled decoder to `path` (.npz) and return its metadata."""
    layers = mlp_weights(mlp)
    arrays = {f"w{i}": w for i, (w, _) in enumerate(layers)} | {f"b{i}": b for i, (_, b) in enumerate(layers)}
    if n_det <= dense_max:
        bits = np.zeros(1 << n_det, dtype=bool)
        for start in range(0, 1 << n_det, chunk):
            k = np.arange(start, min(start+chunk, 1 << n_det), dtype=np.uint64)
            bits[start:start+len(k)] = ensemble_prob(mlp, conv, unpack(k, n_det)) > 0.5
        arrays["dense"] = np.packbits(bits, bitorder="little")
        meta = dict(mode="dense", n_det=n_det, entries=1 << n_det)
    else:
        k = np.unique(keys(cover_dets))
        bits = ensemble_prob(mlp, conv, unpack(k, n_det)) > 0.5
        arrays["keys"] = k.astype(np.uint32 if n_det <= 32 else np.uint64)
        arrays["bits"] = bits.astype(np.uint8)
        meta = dict(mode="sparse", n_det=n_det, entries=int(len(k)))
    meta["mlp_layers"] = [list(w.shape) for w, _ in layers]
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return meta


class Compiled:
    """Python reference for the C decoder: same table, same fallback, same answers."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        n = len(self.meta["mlp_layers"])
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
        if self.meta["mode"] == "dense":
            self.dense = np.unpackbits(z["dense"], bitorder="little").astype(bool)
        else:
            self.keys, self.bits = z["keys"].astype(np.uint64), z["bits"].astype(bool)
        self.last_hit = 1.0

    def decode_batch(self, dets):
        k = keys(dets)
        if self.meta["mode"] == "dense":
            self.last_hit = 1.0
            return self.dense[k.astype(np.int64)][:, None]
        i = np.clip(np.searchsorted(self.keys, k), 0, len(self.keys)-1)
        hit = self.keys[i] == k
        out = np.zeros(len(k), dtype=bool)
        out[hit] = self.bits[i[hit]]
        if (~hit).any():
            out[~hit] = mlp_forward(self.layers, np.asarray(dets)[~hit])
        self.last_hit = float(hit.mean())
        return out[:, None]
