"""rnn_lstm: two stacked LSTMs over the rounds, a main head that adds the final data-qubit events, an auxiliary head.

Follows Varbanov, Serra-Peralta, Byfield & Terhal, "Neural network decoder for near-term surface-code
experiments" (Phys. Rev. Research 2025; arXiv:2307.03280), which nearly exactly follows Baireuther,
O'Brien, Tarasinski & Beenakker, Quantum 2, 48 (2018). See docs/rnn_lstm.md.

- Per round r = 0..R-1, the network reads each stabilizer's detection event, the mask of which
  detectors the circuit has at that round, and the round's kind (first, second, bulk: the no-reset
  circuit's first two rounds have different detector definitions). Optionally (raw_record) it also
  reads each ancilla's raw readout, the offline full measured record.
- Two LSTM layers of width ``hidden``; the second's last hidden state, after a ReLU, feeds both heads.
- The main head adds the final data-qubit events (t = R) and their mask; the auxiliary head sees the
  recurrent state only. Loss = BCE(main) + aux_weight * BCE(aux), as in Varbanov et al. (w = 0.5).
  Only the main head decodes.
- One model per (device patch, basis), trained on a mixture of R: the LSTM runs over any R.
"""
from __future__ import annotations
import stim

from .. import data
from ..sequence import SequenceDecoder

DEFAULT_MODEL = dict(hidden=128, head=128, dropout=0.05, aux_weight=0.5, raw_record=False)


def build(circuit: stim.Circuit, config: dict) -> "LSTMDecoder":
    lay = data.detector_layout(circuit)
    basis = next(iter(set(lay["type"][lay["round"] == 0])))
    return LSTMDecoder(lay["distance"], basis, config, family=config.get("family"))


def _net(S: int, hidden: int, head: int, dropout: float, raw: bool):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.S = S
            self.lstm = nn.LSTM(S * (3 if raw else 2) + 3, hidden, num_layers=2, batch_first=True)
            self.main = nn.Sequential(nn.Linear(hidden + 2 * S, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1))
            self.aux = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1))

        def forward(self, rounds, final, raw_bits, exists, kinds):
            n, r, s = rounds.shape
            parts = [rounds, exists[:r].unsqueeze(0).expand(n, r, s)]
            if raw:
                parts.append(raw_bits)
            parts.append(nn.functional.one_hot(kinds, 3).to(rounds.dtype).unsqueeze(0).expand(n, r, 3))
            out, _ = self.lstm(torch.cat(parts, dim=-1))
            h = torch.relu(out[:, -1])
            main = self.main(torch.cat([h, final, exists[r].unsqueeze(0).expand(n, s)], dim=-1))
            return main, self.aux(h)

    return Net()


class LSTMDecoder(SequenceDecoder):
    name = "rnn_lstm"
    FORMAT = "mlDecoder.rnn_lstm/1"
    DEFAULT_MODEL = DEFAULT_MODEL

    def make_net(self):
        m = self.model_cfg
        return _net(self.distance**2 - 1, int(m["hidden"]), int(m["head"]), float(m["dropout"]), self.uses_raw)

    def forward(self, net, rounds, final, raw, lay):
        main, aux = net(rounds, final, raw, lay["exists"], lay["kinds"])
        return main, dict(aux_logit=aux)

    def aux_loss(self, aux, y, rounds, final, lay):
        import torch
        w = float(self.model_cfg["aux_weight"])
        if not w:
            return None
        return w * torch.nn.functional.binary_cross_entropy_with_logits(aux["aux_logit"].float(), y)

    def describe_architecture(self) -> str:
        m = self.model_cfg
        return (f"2-layer LSTM ({m['hidden']}) over rounds, main head with final data events "
                f"({m['head']}, ReLU), auxiliary head (weight {m['aux_weight']})"
                + (", raw ancilla readouts as inputs (offline, full measured record)" if self.uses_raw else ""))
