# rnn_lstm: a recurrent LSTM decoder

`mlDecoder/models/rnn_lstm.py`, on the shared machinery in `mlDecoder/sequence.py`.

## Papers

- B. M. Varbanov, M. Serra-Peralta, D. Byfield and B. M. Terhal, "Neural network decoder for near-term surface-code
  experiments", *Phys. Rev. Research* (2025), arXiv:2307.03280. This is the architecture followed here: two stacked
  LSTM layers, then a ReLU. The *main* head reads the recurrent state with the final defects inferred from the data
  qubits; the *auxiliary* head reads the recurrent state only. Loss = BCE(main) + 0.5·BCE(aux). Hidden sizes were 64
  (d=3) to 128 (d=7), with Adam and early stopping. On Google's Sycamore data it was trained on simulated data only,
  because 6.5×10⁵ experimental shots were too few to train on. Its LER was about 15% higher on experimental data than on
  simulated data at d=3.
- P. Baireuther, T. E. O'Brien, B. Tarasinski and C. W. J. Beenakker, "Machine-learning-assisted correction of
  correlated qubit errors in a topological code", *Quantum* 2, 48 (2018). This introduced the two-headed LSTM decoder
  that Varbanov et al. follow. It beats matching by learning correlated (e.g. Y) errors, and needs no noise model.

## Design here

- **Input per round** r = 0..R−1, per stabilizer (8 at d=3, ordered as `data.detector_layout`):
  - the detection event;
  - the mask of which detectors exist at that round (round 0 has only the basis-type checks);
  - the round's kind (first, second, bulk). The no-reset circuit's first two rounds have different detector
    definitions.

  With `raw_record: true` the input also carries each ancilla's raw readout. That is the offline, full measured
  record: allowed by the team's guide, since it is measured and not hidden. In a no-reset circuit it tells whether
  the ancilla sat in |1⟩, which matters under T1 decay and asymmetric readout.
- **Body:** a 2-layer LSTM (hidden 128); the second layer's last state, after a ReLU.
- **Heads:** main = MLP(hidden + final events + final mask → 128 → 1); aux = MLP(hidden → 128 → 1); dropout 0.05.
  Only the main head decodes.
- **One model per (device patch, basis), for every R:** R comes from the input width, (d² − 1)·R detectors, and any
  other width is refused. A shot with no detection event predicts no flip without running the network, as matching
  does.

## Training

`train.train_sequence`, through `SequenceDecoder.fit`:

- **Pre-training** on a `sequence.Mixture` of noise components (`configs/rnn_lstm_full.json`):
  - the team's calibrated models on the Garnet patch, with mlDecoder's readout-window extension, asymmetric readout,
    QND flips and T2 variants;
  - the plain `leakage` model at p = 0.003;
  - R in the hardware set {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20}.

  A curriculum shows only R ≤ 5 for the first 15% of the time. Each batch is fresh:
  - Pauli components stream from Stim in-process;
  - leakage components come from worker processes, one seeded chunk per batch. While the trainer would wait, it reuses
    a recent chunk, and counts the reuse.

  AdamW, warmup then cosine, annealed over a fixed time budget, with AMP on CUDA, gradient clipping, and early stopping
  on validation P_L.
- **Fine-tuning** (`python -m mlDecoder.finetune`) on real shots: the `finetune` split of batches 1 and 2, early
  stopping on `validation`, never on test. The weights are pulled towards the pre-trained ones (`l2_to_init`), as in
  AlphaQubit's fine-tuning.
- CPU runs are bit-reproducible from the seed (tested). CUDA runs are not.

## Limits

- d = 3 only. That is the only distance the hardware has; the design itself is not tied to d.
- The LSTM gets no explicit spatial structure: stabilizers are a flat vector per round.
- Pre-training noise is a model. The real device's bulk rounds are noisier than every calibrated build
  (`docs/hardware.md`), so fine-tuning on real shots is expected to matter.
