# data

| Path | What | In git |
|---|---|---|
| `calibration/<device>/<calibration_set_id>.json` | calibration snapshots from Resonance (architecture, fidelities), one file per calibration set | yes |
| `hw/<YYYYMMDD>-<task>/<job id>/` | the raw result of one hardware job, written by `dqiqm.hw` | yes |
| `sim/` | small simulation result tables that a PR reports | yes |
| `sim/cache/` | sweep caches; regenerable, so ignored | no |

## Raw data is never edited

Hardware results are written once and never changed afterwards. The analysis reads them; corrections and derived numbers go in new files. A job that has to be pulled again goes into a new directory, and the old one stays as it is.

## One hardware job: `hw/<YYYYMMDD>-<task>/<job id>/`

`<YYYYMMDD>` is the UTC day the job was submitted.

- `meta.json`: `job_id`, `task`, `device`, `calibration_set_id`, `qubit_mapping` (circuit qubit index to device qubit name), `shots`, `circuits`, `circuit_names`, `success`, `commit` (`git rev-parse HEAD` at submission), `purpose`, `saved_utc` and `timeline` (the server's status timestamps).
- `memory.npz`: per-shot measurements, one `uint8` array `c<i>` of shape `(shots, clbits)` per circuit, in batch order. Column `j` is circuit classical bit `j`, so the Qiskit bitstring order is already undone (Qiskit prints clbit 0 rightmost). It is missing when the job failed.

Load it with:

```python
import json, numpy as np
meta = json.load(open(f"{d}/meta.json"))
bits = np.load(f"{d}/memory.npz")["c0"]   # shots x clbits
```

To check committed data against the server, re-pull the job with `dqiqm.hw.retrieve(job_id, device)`, write it to a scratch directory with `dqiqm.hw.save_raw(...)`, and compare the arrays.
