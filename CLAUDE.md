# duquantumIQM

## The project

This is our entry to the IQM challenge of DuQuantum 2026 (<https://github.com/svijaymurugan/DuQuantum-IQM-Challenge-2026>). It runs a rotated surface-code memory experiment without resets on IQM Emerald, through IQM Resonance. It decodes the syndromes offline, then fits the logical error rate per round, ε_L, from the logical error probability P_L at several round counts: 2·P_L = 1 − (1 − 2ε_L)^rounds.

Emerald has no mid-circuit reset, so an ancilla is never reset between rounds. After round k it holds the XOR of every syndrome so far, and a detector compares measurements two rounds apart: m_k ⊕ m_(k−2), not m_k ⊕ m_(k−1).

The judges weight:

| Criterion | Weight |
|---|---|
| Theoretical correctness: a correct surface code and a working pipeline | 30% |
| Sophistication: architecture-aware circuits and a provably low SWAP count | 30% |
| Logical error rate statistics | 20% |
| Flexibility and advanced functionality (compilation, noise models, pulse level, ...) | 20% |

## Commands

```
uv sync              # builds .venv from pyproject.toml + uv.lock (Python 3.13)
uv run pytest        # offline; hardware tests carry @pytest.mark.hw and are deselected
uv run python scripts/<script>.py
```

Add a dependency with `uv add <pkg>`, and commit `pyproject.toml` and `uv.lock` together. The pins at the top of `pyproject.toml` are the challenge's own `requirements.txt` (iqm-client 35.0.3, qrisp 0.9.9, qiskit 2.1.2, ...): move them only for a stated reason.

## Layout

| Path | What |
|---|---|
| `challenge/` | the organisers' provided code and iqm_utils, verbatim: never edit it; copy what you need into `src/dqiqm/` and say where it came from |
| `src/dqiqm/` | the package |
| `src/dqiqm/hw.py` | the only path to IQM hardware |
| `tests/` | offline tests; `conftest.py` blocks every network socket |
| `scripts/` | the scripts that produce a PR's numbers, runnable headless (later tasks add them) |
| `data/` | `calibration/`, `hw/` (raw hardware results), `sim/`; see `data/README.md` |
| `.env.example` | the token's variable name; the real `.env` is ignored by git |

## Hardware rules

- **One path.** `dqiqm.hw` is the only module that may build a backend able to submit to Resonance (`https://resonance.iqm.tech`). Transpile against `hw.backend(device)`, which is read-only (its `run` refuses). Submit with `hw.submit(...)`. Never call `IQMProvider(...).get_backend()`, `IQMClient.submit_*` or a qrisp `IQMBackend(...)` anywhere else, and never in a test.
- **Not `challenge/`.** The organisers' code builds its own backends, with no guard. Never run `challenge/run_on_hardware.py`: it submits jobs (`run_qiskit`, `run_qrisp`). Never import `challenge/iqm_utils/get_calibration_data.py`: it connects at import, to the old `resonance.meetiqm.com` address. Only `dqiqm.hw` may connect or submit; copy the logic you need into it.
- **Arming.** `hw.submit` refuses before any network call unless `DQ_QPU=run` and `DQ_QPU_TASK=<task>` are set. The task's allowance comes from `DQ_QPU_MAX_JOBS` and `DQ_QPU_MAX_SHOTS`, where shots means total shots (circuits × shots, summed). If they are unset, the allowance is zero.
- **Ledger.** Usage is counted from the task's lines in the QPU ledger (`DQ_QPU_LEDGER`, default `/home/sal/duquantumIQM/orchestration/qpu-ledger.md`). `hw.submit` builds the run request once, records the physical qubits that request sends, and submits that same request. The moment a job id comes back, it appends the job's line, before it waits for the result. Lines are never edited. A correction is a new row, and it keeps the `N x M` form (two integers) in `circuits x shots`: a row that won't parse makes `hw.submit` refuse for every task. A correction row counts towards its task's jobs and shots.
- **Raw data.** Every job's raw result goes in `data/hw/<YYYYMMDD>-<task>/<job id>/`: `meta.json` (with each circuit's logical-to-device qubit map and the qubits it was sent to) plus per-shot memory. It is committed and never edited.
- **Reading** needs the token but no arming: `hw.retrieve(job_id, device)`, `hw.list_devices()`, `hw.architecture(device)`, `hw.calibration_set_id(device)` and `hw.calibration_fidelities(device, calibration_set_id)`, which records the calibration set its figures describe.
- **Token.** `IQM_TOKEN` comes from the environment or from `.env`, loaded by `hw` on first connection. Never print it, log it or commit it.
- **Order.** Simulate first. Send a pilot of one circuit at no more than 1,000 shots, then the batch.
- **Qrisp.** In iqm-client 35.0.3, `DetectorExperiment.batched_compute_LER` calls `backend.run(batch, shots=...)` once. The qrisp `IQMBackend.run` returns `run_async(...).result().all_counts` and drops the job, so the job ids never reach the caller. A qrisp hardware path must therefore subclass `IQMBackend` inside `hw.py` and override `run_async`: check the allowance, call `super().run_async`, then write the ledger line from `job.job_id()`. The per-shot data is in `job._iqm_job.result()`. Until that subclass exists, Qrisp runs only on Stim (`StimBackend()` or no backend).

## Review checklist

A reviewer checks each item the change touches.

### Physics

- **Noiseless circuit.** It has zero detection events, and every observable equals its expected value on every shot. In Qrisp, check each `parity(..., expectation=...)` by hand on a small case: Qrisp does not infer it.
- **Detectors and observables.** Their counts match the design. The detector error model builds (`decompose_errors=True` for a matching decoder). The claimed distance holds: `len(circuit.shortest_graphlike_error())` for graphlike codes, or a stated search for the rest.
- **No reset.** The detectors compare measurements two rounds apart (m_k ⊕ m_(k−2)), and the first two rounds and the final data-qubit detectors are right.
- **Code claims.** For `[[n,k,d]]`, the stabilizers commute (`Hx·Hzᵀ = 0 mod 2`), and k = n − rank(Hx) − rank(Hz) over GF(2).
- **Decoders.** The decoder's priors come from the same noise model the PR says it simulates. A decoder tuned on test data is a finding. A detector error model used on hardware data matches the circuit that ran.
- **Hardware circuits:**
  - only native ops: `r` (prx), `cz`, `measure`, `delay`, `barrier`, and `reset` only on a device with classical control;
  - every two-qubit gate on a coupler of the device architecture for the recorded physical qubits, with the SWAP count stated;
  - the calibration set recorded.
- **Statistics.** Every logical error rate states the shots, the rounds and a 95% interval (Wilson), per round where rounds are compared. Any post-selection is stated and counted. No shots or runs are dropped silently.
- **Plots.** Labelled axes with units, error bars, and the data file each plot comes from.

### Code

- Each "Done when" of the task is met.
- Nothing outside the task's owned paths changed.
- No secret, token or `.env` is committed: `gh pr diff <n> | grep -nE 'IQM_TOKEN *= *[^[:space:]]|[Bb]earer '` finds nothing.
- No test or import reaches the network or hardware, and `uv run pytest` passes from a clean clone after `uv sync --locked`.
- Hardware results are re-pulled by job id (`hw.retrieve`), never re-run, and equal the committed raw data. Every job id is in the ledger with the same device, qubits and shots, within the allowance.
- Seeds are fixed, so the numbers reproduce.
- Main would still build and test with the change merged.

A BLOCKER is wrong physics or numbers, a leaked secret, a path to unallowed hardware spend, or code that doesn't run. A MAJOR is an unmet "Done when", a missing check, or a result that can't be reproduced. Everything else is MINOR.
