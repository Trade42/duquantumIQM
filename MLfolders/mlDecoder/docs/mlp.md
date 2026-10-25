# `mlp`: the feed-forward baseline

`models/mlp.py`. The simplest learned decoder, and the floor every later backend must clear.

## Design

- **Input:** the detection events of one shot, as floats (0/1), in the circuit's detector order. No observables, no leakage flags, no raw measurements.
- **Network:** `Linear(D, w) → ReLU`, repeated for 2–4 hidden layers, then `Linear(w, num_observables)`: one logit per observable. The width is `w = max(min_width, width_factor × D)`, so it scales with the detector count D = R(d² − 1).
  - `mlp_full.json`: 3 hidden layers; width 6×D at d = 3 and 4×D at d = 5, 7 (minimum 128).
  - `mlp_smoke.json`: 2 hidden layers, width 8×D.
- **One model per (d, R, basis, noise).** Nothing is shared across round counts or distances: this is the baseline's defining limit.
- **Empty syndromes:** a shot with no detection event predicts "no flip" without running the network. That is what matching does too. Such shots are dropped from training (`drop_trivial`), because their label is always 0 below distance-many faults, and at p = 0.001 they are most shots at small R. Chamberland and Ronagh dropped them too.
- **Training** (`train.py`): `BCEWithLogitsLoss` on the observable flips, AdamW, linear warmup then cosine decay.
  - Pauli models stream fresh Stim shots every step, so no shot is seen twice and overfitting can't happen.
  - The leakage sampler (`challenge.leakage.sample_leakage`) is pure numpy and slow (about 3k–220k shots/s here), so leakage models train from a pre-generated pool (2M shots at d=3, 1M at d=5, 500k at d=7).
  - Float16 autocast with a GradScaler on CUDA. Early stopping on validation P_L (ties broken by validation loss), restoring the best state, then freezing.
  - Checkpoints go to the run folder. The 45-minute cap is hard-coded as the ceiling.
  - In the full config the learning rate anneals over a fixed time budget per model (d=3 50 s, d=5 30 s, d=7 25 s). The whole 72-model grid's training then fits within 45 minutes.
- **Reproducibility:** CPU runs are bit-reproducible from the seed (`torch.use_deterministic_algorithms`, seeded Stim streams; tested in `tests/test_mlp.py`). CUDA runs are not, because the kernels aren't deterministic.

## The papers

- **C. Chamberland and P. Ronagh, "Deep neural decoders for near term fault-tolerant experiments", Quantum Sci. Technol. 3, 044002 (2018), arXiv:1802.06441.** Rotated surface code d = 3 and 5 (up to 3 and 6 syndrome rounds) under full circuit-level depolarizing noise, p = 10⁻⁴ to 2×10⁻³.
  - The networks predict a logical correction on top of a simple decoder (a lookup table, or a naive "pure error" decoder). Separate X and Z networks, each a 2-class softmax, which is equivalent to one BCE logit.
  - Compared FFNN, LSTM, 3D CNN and softmax regression. At d = 3 a 1-hidden-layer FFNN was best; at d = 5, 2 hidden layers on top of the lookup table. A third layer didn't help. ReLU hidden units, widths searched over 100–1000.
  - Fixed datasets of 2×10⁶ (d=3) and 2×10⁷ (d=5) samples per p, with all-zero samples dropped; RMSProp; Bayesian hyperparameter search.
  - Pseudo-thresholds improved about 1.2× over the lookup table at d = 3 and 5. There was no comparison with MWPM.
  - Their stated limits: deep classifiers struggle with sparse data at low p, where the networks "learn zeros". Training was less successful at small p, and scaling to larger distances is "a major challenge". Training at a higher p sometimes generalised better to low p.
- **G. Torlai and R. G. Melko, "Neural decoder for topological codes", Phys. Rev. Lett. 119, 030501 (2017), arXiv:1610.04238.** Toric code L = 4 and 6, code-capacity phase flips, perfect syndromes, one round.
  - A generative restricted Boltzmann machine learns p(error, syndrome). Decoding clamps the syndrome and Gibbs-samples until a consistent error chain appears. About 10⁵ training samples per p.
  - They report failure rates "almost identical" to MWPM below threshold. It shows a neural decoder can match MWPM at small sizes, but its sampling decoder doesn't carry over to a feed-forward classifier.

What we take from them: raw detection events in, a single logit out, ReLU layers, dropping trivial samples. We go beyond them on two counts: our data are circuit-level no-reset detection events, and we compare against correlated MWPM on identical shots. Neither paper does either.

## Expected limits (stated before the runs)

- An MLP has no notion of locality or of rounds. Its input grows as R(d² − 1), and the number of distinct syndromes it must learn grows exponentially. Expect parity with matching only at d = 3 and small R, and losses at d = 5 and 7. The budget is a few tens of seconds per model, so that is a test of sample efficiency, not of capacity.
- On leakage, the MLP sees the same detection events as MWPM, but it is trained on the leakage sampler's own statistics, while MWPM's DEM is static and Pauli. That gives it a chance of an edge there, limited by the pool size.

## Results (stage 0, simulated)

Full numbers are in `RESULTS.md`; the summary is in `README.md`. Against the expectations above:

- **d = 3:**
  - On `leakage`, the MLP's ε_L is 0.68–0.72× correlated MWPM's.
  - On `uniform` it is 0.85–0.91×, not the tie expected: correlated MWPM is not maximum-likelihood at d = 3 either.
  - Both replicate on an independent test2 set.
- **d = 5, 7:** it loses everywhere, as expected.
- **Leakage pools are the limit.** The leakage models overfit their reused pools: training-set P_L is up to 19σ below test, while validation P_L tracks test. Fresh leakage shots (each used once, 600 s) improve d3 R10 Z from P_L 0.0745 to 0.0671. A 10-minute budget on the same pool doesn't: 0.0759, early-stopped at 300 s.
- **Precision:** the run scores on CUDA with float16 autocast. A float32 CPU decode differs on about 2 shots per 100k.
