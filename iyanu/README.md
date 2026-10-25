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

The challenge's evaluated X/Z pipeline and canonical inference interface are
described in [Methods](../docs/METHODS.md) and
[Reproduction](../docs/REPRODUCIBILITY.md).
