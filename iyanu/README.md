# Decoder research modules

These modules provide optional inference and characterization tools alongside
the [surface-code submission](../README.md).

| Module | Function |
|---|---|
| `neural_gpu.py` | Detector-vector MLP and accelerated training |
| `conv_decoders.py` | Space/time convolutions and local predecoding |
| `ldpc_decoders.py` | Union-find and belief-propagation/OSD adapters |
| `compare_decoders.py` | Common-shot simulation comparisons |
| `garnet_sim.py` | Offline native-layout and noise rehearsal |
| `finetune_hw.py` | Simulation pretraining and independent hardware-label fine-tuning |
| `circuits.py` | Circuit variants (ours, IQM scheduling, vanilla H-CZ-H) verified on the serialized request |
| `compiled.py` | Compile the fine-tuned GPU ensemble into lookup tables with an MLP fallback |
| `latency/` | C decoder and latency benchmark for the compiled tables (`decode_bench.c`, `prepare.py`) |
| `qpu.py`, `hw_run.py` | Guarded IQM Resonance submission (shot cap, CZ checks on the request, ledger, raw data) |

The challenge's evaluated X/Z pipeline and canonical inference interface are
described in [Methods](../docs/METHODS.md) and
[Reproduction](../docs/REPRODUCIBILITY.md).

Hardware LER results for these decoders (Emerald and Garnet, GPU ensemble fine-tuned on
hardware shots: Emerald Z 4.00%, X 6.20% per round) and the nanosecond lookup-table
decoding latency are in [RESULTS.md](RESULTS.md); raw hardware data is under
[archive/iyanu/hw_data](../archive/iyanu/hw_data).
