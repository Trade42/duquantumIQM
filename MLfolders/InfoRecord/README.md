# Held-out B3 memory records

This tree contains only the pA/eA B3 holdout from MLDecodeTest, with original
arrays, manifests, shot tables and source circuits. It contains 55,000 shots per
basis on Garnet and 22,000 per basis on Emerald: 154,000 shots in 44 cases.
No B3 record is used for training or threshold selection. Training/validation
raw records remain upstream; their aggregate histories are retained in the ML
result JSON files. The imported model weights are frozen.

Use `python -m MLfolders.mlDecoder.evidence` for the aggregate evidence audit,
and `--decode` for a fresh CPU replay of LSTM predictions against these records.
