# iyanu: trial GPU neural decoder

A small MLP that reads a shot's detection events and predicts the logical flip, trained on
fresh Stim shots from the repo's no-reset circuit (`challenge.surface_code` + `challenge.noise_models`).
It runs on CUDA, Apple MPS or CPU, and is compared with correlated PyMatching on the same
held-out shots, for accuracy and for latency/throughput. Not wired into `challenge.experiment`.

Needs PyTorch on top of the package (`pip install -e '.[test]' torch`).

```
python -m iyanu.neural_gpu --distance 3 --rounds 3 --p 0.005   # writes results/iyanu/neural_gpu/<run-id>/
python -m pytest -q iyanu                                     # 5 tests; skipped without torch
```

First result (Apple M3, d=3, 3 rounds, uniform p=0.005, 200,000 held-out shots, seed 0):

| Decoder | P_L (95% Wilson) | Single-shot latency p50 / p99 | Shots/s, batch 10k |
|---|---|---|---|
| Neural, MPS GPU | 0.0731 [0.0719, 0.0742] | 389 / 468 µs | 3.2M |
| Neural, CPU | same model | 24 / 26 µs | 2.0M |
| PyMatching, correlated | 0.0743 [0.0731, 0.0754] | 2.7 / 7.7 µs | 0.92M |

Paired on the same shots: 2,505 fail only for the neural decoder and 2,745 only for PyMatching
(exact McNemar p = 0.001). One seed and one noise model, so this is a first look, not a result.
The GPU wins on batched throughput and loses on single-shot latency (transfer and launch overhead).
