# rnn_tf: AlphaQubit-lite, a recurrent transformer decoder

`mlDecoder/models/rnn_tf.py`, on the shared machinery in `mlDecoder/sequence.py`.

## Paper

J. Bausch, A. W. Senior, F. J. H. Heras et al., "Learning high-accuracy error decoding for quantum processors",
*Nature* 635, 834 (2024); arXiv:2310.05900 ("AlphaQubit"). Its key points:

- each stabilizer is a token with a persistent state (320 wide on Sycamore);
- each cycle, the stabilizer's embedding (events and measurements, optionally soft I/Q and leakage information, plus
  a learned index embedding) is added to the state, and the sum is scaled by 0.707;
- three "syndrome transformer" layers per cycle: multi-head attention with a learned attention bias built from each
  pair's offset, Manhattan distance and same-basis bit, a gated dense block, and dilated 3×3 convolutions on the 2D
  grid;
- the final round has its own embedding. The readout scatters the states to 2D, pools along rows or columns
  perpendicular to the logical observable, then applies a residual network;
- an auxiliary loss predicts the next stabilizers;
- pre-training used up to 2×10⁹ samples, from detector error models fitted to the experiment's detection events;
- fine-tuning used 19,880 experimental samples with weight decay relative to the pre-trained weights, early stopping on
  5,120 held-out samples, and an ensemble of 20 seeds (averaged logits);
- on Sycamore d=3 it reached 2.901% LER per round, against 3.028% for the tensor-network decoder.

## Design here (lite)

- 8 stabilizer tokens at d=3. Per round, the input embedding is a linear map of [event, exists mask], plus
  [raw ancilla readout] with `raw_record`. To it are added:
  - a learned per-stabilizer index embedding;
  - an X/Z type embedding;
  - an MLP of Fourier features of the stabilizer's (x, y);
  - an embedding of the round's kind (first, second, bulk, final).

  The sum is added to the state, and the state scaled by 0.707.
- **Per round:** 3 pre-LayerNorm blocks (default width 64): 4-head self-attention with an additive bias, then a gated
  GELU dense block (×2 widening). The bias comes from an MLP over the pair features: offset (dx, dy), Manhattan
  distance, same type, and self.
- **Final step:** the data-qubit events (t = R) enter through their own embedding, then the same blocks.
- **Readout:** LayerNorm, mean and max pooling over the tokens, an MLP, one logit.
- **Auxiliary loss:** next-stabilizer prediction (BCE on the next round's events, masked to existing detectors),
  weight 0.02.
- **One set of weights for every R,** per (device patch, basis). R comes from the input width, and an empty syndrome
  predicts no flip.

**Left out (lite):**
- the dilated convolutions;
- the row/column readout;
- soft I/Q and leakage inputs (InfoRecord carries hard bits);
- ensembling by default;
- AlphaQubit's scale of pre-training: here minutes on a laptop GPU, not 2×10⁹ samples.

## Training

The same as `rnn_lstm` (`docs/rnn_lstm.md`): pre-training on the Garnet noise mixture with a curriculum on R, then
fine-tuning on real shots with L2 towards the pre-trained weights.

## Limits

- **Speed.** The Python loop over rounds runs 3 blocks per round, so batch-1 latency grows with R; it is reported, not
  hidden.
- d = 3 only. Tokens are positional, so other d would need training on mixtures, but there is no hardware above d=3.
