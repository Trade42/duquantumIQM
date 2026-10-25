# ML telescope for real-time surface-code decoding

The telescope replaces a history-indexed lookup table with an LSTM, sending
uncertain predictions to Relay-BP and using correlated MWPM for nonconvergence.
It reuses a fixed set of weights as the syndrome history grows, addressing the
lookup table's exponential storage cost for real-time decoding.

On Emerald, the ML pipeline reduces one-to-three-round fitted LER by **82.4% in X
and 30.1% in Z** relative to vanilla memory. See
[ML_DECODER.md](../../docs/ML_DECODER.md) for results and decoding cost per round,
and [UPSTREAM.md](UPSTREAM.md) for the imported `MLDecodeTest` work.

The four frozen models cover Garnet pA and Emerald eA in X/Z. Each has 243,714
parameters and approximately 488 KiB of float16 checkpoint data. They use binary
detection events only, preserve shot order, and infer the round count from input
width `(shots, 8*rounds)`. Training and hardware evaluation are distance three;
evaluation spans eleven round counts from 1 to 20.

## Evidence and vanilla comparison

From the repository root, with the two core packages installed:

```powershell
python -m MLfolders.mlDecoder.evidence
python -m MLfolders.mlDecoder.compare_emerald
```

The audit verifies imported hashes, all 154,000 B3 raw-record conversions,
test splits, saved failure counts and LER fits using the core Python packages.
The comparison recomputes the Emerald R1–3 vanilla and telescope fits and
normalizes recorded decoding times by round count. B3 contains 55,000 shots per
basis on Garnet and 22,000 per basis on Emerald.

## Isolated ML runtime

Use Python 3.13+ in a separate environment. From the repository root:

```powershell
py -3.13 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install -e ".[test]" -e ./ethan
python -m pip install "torch>=2.13,<3"
```

PyTorch is sufficient for the LSTM and the three shipped cascades with `tau=0`.
To execute Relay-BP on deferred shots, also install the upstream-pinned package:

```powershell
python -m pip install "relay-bp==0.2.2"
```

Building Relay-BP from source requires a Rust/native toolchain.
`requirements-ml.txt` records the original training environment.

On Linux, use the corresponding `bin/activate` path. Commands below run from the
repository root so the `MLfolders` package is importable.

```python
from MLfolders.mlDecoder import final

decoder = final.load("garnet", "pA", "Z")
flips = decoder.decode_batch(det_events)  # binary (shots, 1)

cascade = final.load("emerald", "eA", "X", cascade=True)
detail = cascade.decode_detailed(det_events)  # predictions, layer, p, confidence
```

Use the model's detector frame. The packaged API consumes measured detector
histories, including final data-boundary events, and returns logical flips in
shot order.

## Recompute inference

```powershell
python -m MLfolders.mlDecoder.final list
python -m MLfolders.mlDecoder.evidence --decode
python -m pytest -q --import-mode=importlib MLfolders/mlDecoder/tests
```

The committed [CPU replay](results_record/promoted_cpu_replay.json) records
inference with the shipped models. The
[Emerald comparison](results_record/emerald_vanilla_r1_r3.json) records the
vanilla reference, telescope scores and decoding cost per round.

## Training work

The recurrent architecture, synthetic mixture sampler, training loop, calibration
and readout-window model, fine-tuning/selection code, and selected configurations
are retained. For example:

```powershell
python -m MLfolders.mlDecoder.run --backend rnn_lstm --config MLfolders/mlDecoder/configs/rnn_lstm_full.json --groups garnet_pA_Z garnet_pA_X
```

Re-running original hardware fine-tuning additionally requires the full B1/B2
InfoRecord at `MLDECODER_INFORECORD` and a pre-training checkpoint. The included
B3 records are test data only. Experiments should write new run directories.
The acquisition-era noise models used for training live in `_reference/`.
