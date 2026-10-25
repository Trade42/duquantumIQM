# The cascade: LSTM first, Relay-BP for the doubtful shots, MWPM as the safety net

`mlDecoder/cascade.py` and `mlDecoder/relay.py`. This is Sal's design (2026-10-25): the recurrent decoder becomes the fast
first layer of a two-layer decoder.

1. **Layer 1, `rnn_lstm`,** decodes every shot and gives P(flip) = p. Its confidence is c = |2p − 1|. A shot with no
   detection event has p = 0, so c = 1.
2. **Layer 2, Relay-BP,** decodes the shots with c < τ. It runs on the decoder prior's detector error model:
   calibrated_coherence on the device patch, with the readout-window extension, asymmetric readout and QND flips, or
   uniform weights where those win on validation (below). Hyperedges are kept whole, since BP needs no matching graph.
3. **Layer 3, correlated MWPM,** decodes the shots Relay-BP does not converge on. Every shot gets a decision, and
   `decode_detailed` says which layer made each one.

**τ is tuned on the validation split only,** by the one-standard-error rule (dq-orch-3, 2026-10-25, fixed before the
test batch was opened). τ* is the τ with the fewest validation failures (ties to the smaller τ). τ is the smallest
τ ≤ τ* whose failures exceed τ*'s by at most one standard error of the paired difference, √(b + c), where b and c count
the shots on which only one of the two fails. So a deferral is kept only where it beats paired noise. τ* is recorded
beside τ, and only τ is evaluated on the test split, which never sees either. The validation split also served early
stopping, so numbers on it lean optimistic; the test batch is the honest number.

**Layers 2 and 3 share one prior, chosen on validation before τ** (dq-orch-3, 2026-10-25, before the test batch was
opened). Per device patch and basis, the calibrated prior stays unless correlated MWPM with uniform weights (the team's
uniform model at p = 0.001, the "uniform" reference) fails fewer validation shots by more than √(b + c), the same paired
test. τ is then tuned under the chosen prior. Emerald's compiled readout window was measured at 1,036 ns on all 30 eA
and eB qubits (calset 23e12f6b, mini PC, 2026-10-25), so the window the prior uses is right.

Each fine-tuning run records:
- the whole validation trade-off curve: τ against deferred fraction, failures, P_L, and the mean latency implied by the
  measured per-shot costs;
- layer 1's calibration (reliability bins, ECE).

## Relay-BP

T. Müller, T. Alexander, M. E. Beverland et al. (IBM), "Improved belief propagation is sufficient for real-time decoding of
quantum memory", arXiv:2506.01779 (2025). Relay-BP chains memory-BP runs whose per-error memory strengths are disordered:
drawn uniformly from (−0.24, 0.66), so negative values are included. It relays each leg's marginals into the next, and
stops once `stop_nconv` = 5 legs have found a valid correction, keeping the most likely one.

We use IBM's `relay-bp` 0.2.2 (Rust; Apache-2.0; pinned in `requirements-ml.txt`) with:

| Setting | Value |
|---|---|
| `gamma0` | 0.1 |
| `pre_iter` | 80 |
| `num_sets` | 60 |
| `set_max_iter` | 60 |
| `stop_nconv` | 5 |

**One change.** The package redraws the disordered strengths from its own advancing RNG on every call, so about 0.5% of a
shot's decisions depended on which shots were decoded before it. We draw them once per decoder from a seeded numpy
generator and pass them as `explicit_gammas`. Decisions are then deterministic and batch-independent (tested).

**Validation.** All rows are SIMULATED: 20,000 shots of the patch's calibrated_coherence model, decoded with its own
prior. BP+OSD is `ldpc` 2.4.1 `BpOsdDecoder` (min-sum, 100 iterations, OSD-CS order 10) on the first 4,000 of the same
shots, with correlated MWPM on those 4,000 beside it. Times are per shot, one thread, one Python call per batch
(BP+OSD: one call per shot).

| Device patch, basis, R | corr MWPM | Relay-BP | unconverged | BP+OSD-CS10 (MWPM, same 4k) | µs: MWPM / Relay-BP / BP+OSD |
|---|---:|---:|---:|---:|---|
| Garnet pA Z 3 | 0.0941 | 0.0935 | 0 | 0.0930 (0.0932) | 3 / 166 / 99 |
| Garnet pA X 3 | 0.2675 | 0.2657 | 0 | 0.2777 (0.2755) | 3 / 144 / 120 |
| Garnet pA Z 10 | 0.2952 | 0.2979 | 0 | 0.3053 (0.3090) | 26 / 4,082 / 4,423 |
| Emerald eA Z 3 | 0.0612 | 0.0587 | 0 | 0.0592 (0.0585) | 3 / 223 / 129 |
| Emerald eA X 3 | 0.1731 | 0.1686 | 0 | 0.1737 (0.1708) | 4 / 207 / 119 |
| Emerald eA Z 10 | 0.2172 | 0.2217 | 0 | 0.2230 (0.2172) | 17 / 3,637 / 4,091 |

The paper reports Relay-BP comparable to matching on surface codes, and that is what we see. Relay-BP is slightly ahead
of correlated MWPM at R = 3 and slightly behind at R = 10. One standard error is about 0.002–0.003 at R = 3 and about
0.003 at R = 10, so most gaps are within 2σ (Emerald eA X R = 3 and Z R = 10 sit near 2.5σ). It is
about 50–200× slower per shot than PyMatching here. In the cascade only the deferred shots pay that cost.

## Preview (SIMULATED dev)

Garnet pA Z, the pre-trained LSTM (synthetic only). τ = 0.10 was tuned on 3 × 10,000 dev validation shots
(R = 2, 4, 8). The test shots are 30,000 per R from an independent stream.

| R | LSTM | Relay-BP | corr MWPM | cascade | deferred | batched µs/shot: cascade / Relay-BP |
|---|---:|---:|---:|---:|---:|---|
| 3 | 0.0899 | 0.0927 | 0.0939 | **0.0890** | 2.7% | 7 / 21 |
| 6 | 0.1916 | 0.1944 | 0.1919 | **0.1869** | 8.4% | 36 / 115 |
| 10 | 0.2976 | 0.3032 | 0.2972 | **0.2923** | 21.1% | 113 / 362 |

Batched times use Relay-BP's parallel mode; latency at batch 1 is reported on the hardware runs.

**The cost.** A deferred shot costs layer 1 plus Relay-BP, about 0.2–4 ms on one CPU thread from R = 3 to R = 10. The
mean latency is layer 1's plus (deferred fraction × Relay-BP's). The tail latency is Relay-BP's, so p99 is reported
separately.

## Using it

```python
from mlDecoder import final
decoder = final.load("garnet", "pA", "Z", cascade=True)   # tau and the prior are stored with the weights
out = decoder.decode_detailed(det_events)                  # predictions, layer (0 LSTM, 1 Relay-BP, 2 MWPM), p, confidence
```
