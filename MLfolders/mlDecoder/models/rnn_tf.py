"""rnn_tf: AlphaQubit-lite, a recurrent transformer over stabilizer tokens.

After Bausch, Senior, Heras et al., "Learning high-accuracy error decoding for quantum
processors", Nature 635, 834 (2024) (arXiv:2310.05900). See docs/rnn_tf.md.

- Each stabilizer is a token with a persistent state (width ``d_model``).
- Each round, its input embedding (a linear map of its detection event, the detector-exists mask
  and, with raw_record, its raw ancilla readout) plus a learned embedding of the stabilizer (its
  index, its X/Z type and Fourier features of its (x, y)) and of the round's kind is added to the
  state, the sum scaled by 0.707 (AlphaQubit's combination), and ``layers`` transformer blocks
  update the states: pre-LayerNorm multi-head self-attention over the stabilizer tokens with a
  learned attention bias from each pair's offset, Manhattan distance and same-type bit, then a gated
  dense block (GELU gate).
- The final data-qubit events (t = R) enter the same way, through their own embedding.
- Readout: LayerNorm, mean and max pooling over the tokens, an MLP, one logit.
- Auxiliary task (AlphaQubit's next-stabilizer prediction): from each round's states, predict the
  next round's detection events; weight ``aux_weight``.
- One set of weights for every R (tokens are positional), per (device patch, basis).

Not carried over from AlphaQubit (lite): the dilated 2D convolutions inside each layer, the
row/column readout network, soft I/Q inputs, ensembling by default, and its scale of training.
"""
from __future__ import annotations
import numpy as np
import stim

from .. import data
from ..sequence import SequenceDecoder, layout_for

DEFAULT_MODEL = dict(d_model=64, heads=4, layers=3, ff_mult=2, aux_weight=0.02, fourier=4, raw_record=False)


def build(circuit: stim.Circuit, config: dict) -> "TransformerDecoder":
    lay = data.detector_layout(circuit)
    basis = next(iter(set(lay["type"][lay["round"] == 0])))
    return TransformerDecoder(lay["distance"], basis, config, family=config.get("family"))


def _net(xy: np.ndarray, types: np.ndarray, D: int, heads: int, layers: int, mult: int, n_fourier: int, raw: bool):
    import torch
    from torch import nn
    F = nn.functional
    S = len(xy)
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0))
    centred = (xy - xy.mean(0)) / span
    freqs = np.pi * 2.0 ** np.arange(n_fourier)
    fourier = np.concatenate([np.sin(centred[:, :, None] * freqs), np.cos(centred[:, :, None] * freqs)], -1).reshape(S, -1)
    diff = (xy[:, None, :] - xy[None, :, :]) / span
    pair = np.concatenate([diff, np.abs(diff).sum(-1, keepdims=True), (types[:, None] == types[None, :])[..., None],
                           np.eye(S)[..., None]], -1).astype(np.float32)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
            self.up, self.down = nn.Linear(D, 2 * mult * D), nn.Linear(mult * D, D)

        def forward(self, h, bias):
            n = h.shape[0]
            q, k, v = self.qkv(self.ln1(h)).view(n, S, 3, heads, D // heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype).unsqueeze(0))
            h = h + self.proj(a.transpose(1, 2).reshape(n, S, D))
            u, g = self.up(self.ln2(h)).chunk(2, dim=-1)
            return h + self.down(u * F.gelu(g))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fourier", torch.from_numpy(fourier.astype(np.float32)))
            self.register_buffer("pair", torch.from_numpy(pair))
            self.register_buffer("types", torch.from_numpy(types.astype(np.int64)))
            self.inp = nn.Linear(3 if raw else 2, D)
            self.fin = nn.Linear(2, D)
            self.index = nn.Parameter(torch.randn(S, D) * 0.02)
            self.type_emb = nn.Embedding(2, D)
            self.kind_emb = nn.Embedding(4, D)
            self.pos = nn.Sequential(nn.Linear(fourier.shape[1], D), nn.GELU(), nn.Linear(D, D))
            self.bias = nn.Sequential(nn.Linear(pair.shape[-1], 32), nn.GELU(), nn.Linear(32, heads))
            self.blocks = nn.ModuleList(Block() for _ in range(layers))
            self.ln_out = nn.LayerNorm(D)
            self.readout = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 1))
            self.next = nn.Linear(D, 1)

        def forward(self, rounds, final, raw_bits, exists, kinds):
            n, r, s = rounds.shape
            static = self.index + self.type_emb(self.types) + self.pos(self.fourier)
            bias = self.bias(self.pair).permute(2, 0, 1)
            h = torch.zeros((n, s, D), dtype=static.dtype, device=rounds.device)
            nexts = []
            for t in range(r):
                feats = [rounds[:, t], exists[t].unsqueeze(0).expand(n, s)]
                if raw:
                    feats.append(raw_bits[:, t])
                e = self.inp(torch.stack(feats, dim=-1)) + static + self.kind_emb(kinds[t])
                h = (h + e) * 0.7071
                for block in self.blocks:
                    h = block(h, bias)
                nexts.append(self.next(h).squeeze(-1))
            e = self.fin(torch.stack([final, exists[r].unsqueeze(0).expand(n, s)], dim=-1)) + static + self.kind_emb.weight[3]
            h = (h + e) * 0.7071
            for block in self.blocks:
                h = block(h, bias)
            h = self.ln_out(h)
            return self.readout(torch.cat([h.mean(1), h.amax(1)], dim=-1)), torch.stack(nexts, dim=1)

    return Net()


class TransformerDecoder(SequenceDecoder):
    name = "rnn_tf"
    FORMAT = "mlDecoder.rnn_tf/1"
    DEFAULT_MODEL = DEFAULT_MODEL

    def make_net(self):
        m = self.model_cfg
        lay = layout_for(self.distance, 1, self.basis)
        if int(m["d_model"]) % int(m["heads"]):
            raise ValueError("d_model must be a multiple of heads")
        return _net(lay["xy"], lay["types"], int(m["d_model"]), int(m["heads"]), int(m["layers"]), int(m["ff_mult"]),
                    int(m["fourier"]), self.uses_raw)

    def forward(self, net, rounds, final, raw, lay):
        main, nexts = net(rounds, final, raw, lay["exists"], lay["kinds"])
        return main, dict(next=nexts)

    def aux_loss(self, aux, y, rounds, final, lay):
        import torch
        w = float(self.model_cfg["aux_weight"])
        if not w:
            return None
        target = torch.cat([rounds[:, 1:], final.unsqueeze(1)], dim=1)
        mask = lay["exists"][1:].unsqueeze(0).expand_as(target)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(aux["next"].float(), target, reduction="none")
        return w * (loss * mask).sum() / mask.sum().clamp(min=1)

    def describe_architecture(self) -> str:
        m = self.model_cfg
        return (f"recurrent transformer: {self.distance**2 - 1} stabilizer tokens x {m['d_model']}, {m['layers']} blocks "
                f"per round ({m['heads']} heads, learned pair bias, gated MLP x{m['ff_mult']}), mean+max readout; "
                f"next-stabilizer auxiliary loss (weight {m['aux_weight']})"
                + (", raw ancilla readouts as inputs (offline, full measured record)" if self.uses_raw else ""))
