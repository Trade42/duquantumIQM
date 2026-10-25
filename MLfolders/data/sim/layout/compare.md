# d=3 on IQMFakeAphrodite: ours against Qiskit's automatic layout

Written by `scripts/layout_compare.py`; the full record is `compare.json`.

| basis | rounds | method | swaps | extra_cz | cz | cz_per_round | r | r_per_round | measure | depth | cz_depth | readout_layers | duration_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Z | 1 | ours | 0 | 0 | 24 | 24 | 38 | 38 | 17 | 9 | 4 | 1 | n/a |
| Z | 1 | ours, no barriers | 0 | 0 | 24 | 24 | 38 | 38 | 17 | 9 | 4 | 5 | n/a |
| Z | 1 | qiskit L1 | 0 | 0 | 24 | 24 | 38 | 38 | 17 | 9 | 4 | 5 | n/a |
| Z | 1 | qiskit L3 | 0 | 0 | 24 | 24 | 38 | 38 | 17 | 9 | 4 | 5 | n/a |
| Z | 3 | ours | 0 | 0 | 72 | 24 | 106 | 35.3333 | 33 | 27 | 12 | 3 | n/a |
| Z | 3 | ours, no barriers | 0 | 0 | 72 | 24 | 106 | 35.3333 | 33 | 25 | 12 | 14 | n/a |
| Z | 3 | qiskit L1 | 0 | 0 | 72 | 24 | 106 | 35.3333 | 33 | 25 | 12 | 14 | n/a |
| Z | 3 | qiskit L3 | 0 | 0 | 72 | 24 | 106 | 35.3333 | 33 | 25 | 12 | 14 | n/a |
| Z | 5 | ours | 0 | 0 | 120 | 24 | 174 | 34.8 | 49 | 45 | 20 | 5 | n/a |
| Z | 5 | ours, no barriers | 0 | 0 | 120 | 24 | 174 | 34.8 | 49 | 39 | 20 | 22 | n/a |
| Z | 5 | qiskit L1 | 0 | 0 | 120 | 24 | 174 | 34.8 | 49 | 39 | 20 | 22 | n/a |
| Z | 5 | qiskit L3 | 0 | 0 | 120 | 24 | 174 | 34.8 | 49 | 39 | 20 | 22 | n/a |
| X | 1 | ours | 0 | 0 | 24 | 24 | 36 | 36 | 17 | 9 | 4 | 1 | n/a |
| X | 1 | ours, no barriers | 0 | 0 | 24 | 24 | 36 | 36 | 17 | 9 | 4 | 4 | n/a |
| X | 1 | qiskit L1 | 0 | 0 | 24 | 24 | 36 | 36 | 17 | 9 | 4 | 4 | n/a |
| X | 1 | qiskit L3 | 0 | 0 | 24 | 24 | 36 | 36 | 17 | 9 | 4 | 4 | n/a |
| X | 3 | ours | 0 | 0 | 72 | 24 | 104 | 34.6667 | 33 | 27 | 12 | 3 | n/a |
| X | 3 | ours, no barriers | 0 | 0 | 72 | 24 | 104 | 34.6667 | 33 | 25 | 12 | 14 | n/a |
| X | 3 | qiskit L1 | 0 | 0 | 72 | 24 | 104 | 34.6667 | 33 | 25 | 12 | 14 | n/a |
| X | 3 | qiskit L3 | 0 | 0 | 72 | 24 | 104 | 34.6667 | 33 | 25 | 12 | 14 | n/a |
| X | 5 | ours | 0 | 0 | 120 | 24 | 172 | 34.4 | 49 | 45 | 20 | 5 | n/a |
| X | 5 | ours, no barriers | 0 | 0 | 120 | 24 | 172 | 34.4 | 49 | 39 | 20 | 22 | n/a |
| X | 5 | qiskit L1 | 0 | 0 | 120 | 24 | 172 | 34.4 | 49 | 39 | 20 | 22 | n/a |
| X | 5 | qiskit L3 | 0 | 0 | 120 | 24 | 172 | 34.4 | 49 | 39 | 20 | 22 | n/a |
