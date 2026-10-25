# pulseCalibration

The gate characterization of IQM Garnet (`garnet-calib`) and the read-only check of pulse-level access through IQM Pulla (`pulla-check`), as a folder that runs on its own.

- **garnet-calib** measures, at circuit level, the coherent errors of the gates our surface code uses on Garnet's pilot patch (17 qubits, 24 couplers): per qubit the PRX amplitude and axis errors, per coupler the CZ conditional phase and the local Z phases, each with a 95% interval. A closure test on a simulated stand-in shows the analysis recovers injected errors. Its pulse-level track is designed, not run.
- **pulla-check** shows our token can open Pulla on Garnet and Emerald, read every gate's pulse parameters and compile circuits to pulse schedules locally. It submitted nothing. Its Emerald compile gives the real readout window, and `--compare` re-decodes the Emerald pilot (EP0) offline under the fallback and the compiled window.

One pulse-level job has since run, on Emerald: `emerald-pulse` sent EP0's circuit as a pulse playlist we compiled with Pulla (job `01a170f0-8c09-74b8-b7d7-ed83d24602bc`). It is not part of this folder; its script, record and write-up are in `../MLfolders/` (`docs/emerald-pulse.md`).

Everything here except this README is a byte copy of the same path in `../MLfolders/`, our full project (from our development repository at commit `f60d9f1`). It carries the part of the `dqiqm` package these two need, and the same `pyproject.toml` and `uv.lock`.

## Run

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is pinned in `.python-version`.

```
uv sync --locked
uv run pytest                                          # offline, no token
uv run python scripts/garnet_calib_build.py --check    # rebuilds the circuits and compares them with InfoRecord/characterization/
uv run python scripts/garnet_calib_report.py           # the closure test's tables, from the committed results
uv run python scripts/pulla_check.py --report          # the committed Pulla results, no network
uv run python scripts/pulla_check.py --compare         # Emerald's pilot EP0 under the fallback and the compiled readout window, offline
```

The closure test (`scripts/garnet_calib_closure.py`, about 25 minutes) and the compensation check (`scripts/garnet_calib_compensate.py`; its docstring says about 3 minutes, but on our shared machine it ran for over 20 minutes on 4 cores) are simulations. Their committed outputs are in `InfoRecord/characterization/sim/`. A live `scripts/pulla_check.py` run needs an IQM Resonance token in `.env` (see `.env.example`), and makes only GET requests.

## What is here

| Path | What |
|---|---|
| `docs/garnet-calib.md` | the characterization: the experiments and why each, the closure numbers, the job plan, the pulse-level track |
| `docs/pulla-check.md` | pulse-level access through Pulla, and Garnet's pulse parameters |
| `src/dqiqm/characterization/` | the characterization package |
| `src/dqiqm/`, the rest | what it imports: `layout`, `surface`, `decode`, `stats`, `hw` and `experiments.{device, memory, circuits}` |
| `scripts/` | the build, closure, compensation and report scripts, and `pulla_check.py` |
| `tests/` | offline tests; `conftest.py` blocks every network socket |
| `InfoRecord/characterization/` | the circuits, the job and pulse plans, and the simulated closure results |
| `InfoRecord/circuits/`, `InfoRecord/calibration/garnet/` | the surface-code pilot's placement and job plan, and the calibration snapshot |
| `InfoRecord/emerald/` | for `--compare`: Emerald's R = 3 circuits and placement, its calibration snapshot, and the pilot EP0's raw job, dataset and checks |
| `data/calibration/` | the live Garnet snapshot (calibration set `a18a947f`), the Emerald snapshot taken before EP0, and pulla-check's outputs for Garnet and Emerald |

The other documents these two name, and the rest of the project (the surface-code memory on Garnet and Emerald, the decoding and the hardware runs), are in `../MLfolders/`.
