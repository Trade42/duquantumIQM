# pulla-check: pulse-level access through IQM Pulla

This task was **read-only and spent zero credits**. It submitted nothing (no circuit job, no pulse job, no `:mock` or facade), booked no timeslot, never set `DQ_QPU*`, and added no row to the QPU ledger. Data fetched 2026-10-24 between 21:51 and 21:52 UTC; Emerald's compile (emerald-compile, same guard, same rules) at 2026-10-25 01:27 UTC.

## The answer

| Device | Does `Pulla(...)` work with our token? | Requests it made |
|---|---|---|
| **garnet** | **Yes.** The constructor succeeds. | `GET quantum-computers`, `GET .../garnet/artifacts/settings`, `GET .../garnet/artifacts/chip-design-records`, `GET .../garnet/artifacts/duts`: all 200 |
| **emerald** | **Yes.** The constructor succeeds. | the same four GETs on `emerald`: all 200 |

**What this proves.** Our token can do the following, on both devices:
- read the station-control settings tree, the chip design record, the chip label and full calibration sets (2,575 observations on Garnet, 7,011 on Emerald);
- build IQM's standard circuit-to-pulse compiler;
- compile circuits **locally** into pulse schedules.

Every pulse parameter of every gate is visible to us.

**What it does not prove.** It does not show that Resonance would *accept* a pulse-level job from our account. Submitting one is `Pulla.submit_playlist`, which issues `POST jobs/<device>/run`. That call was forbidden here and was never made. The challenge text says "To access PulLA you need to be a part of a paid organisation", and that restriction could be enforced at submission. That stays unknown until someone with an allowance tries.

## Run it

```
uv run python scripts/pulla_check.py             # live: needs IQM_TOKEN (.env); writes new result files
uv run python scripts/pulla_check.py --devices emerald   # live, Emerald only (emerald-compile ran this)
uv run python scripts/pulla_check.py --report    # no network, no token: prints the committed results
uv run python scripts/pulla_check.py --compare [--doc]   # no network, no token: the Emerald comparison below
uv run pytest tests/test_pulla_check.py tests/test_emerald_compile.py   # offline
```

The live run takes under a minute (most of it the local compiles), and the report under a second. Neither needs the sim lock.

**The guard.** Before any `Pulla` exists, `install_guard()` replaces three things:
- `Pulla.submit_playlist`, with a function that raises `ReadOnlyViolation`;
- `submit_run`, `submit_sweep`, `submit_circuits`, `_submit_job`, `cancel_job` and `delete_job` on the server client, likewise;
- the client's single HTTP seam, `_IQMServerClient._send_request`, with a wrapper that refuses any method other than GET before it is sent, and logs each GET's relative path and status.

The tests check each of these offline. Devices are an allowlist (`garnet`, `emerald`), so a `:mock` or facade device is refused. The token reaches the IQM client only through the environment (`hw._load_token()`), and the script never reads it.

**Calls made.** These are the brief's allowed calls:
- the constructor;
- `get_chip_label`, `get_chip_topology`, `get_calibration_stash(<calset>)` and `get_standard_compiler(...)`;
- `dqiqm.hw.calibration_set_id(device)`, to pin the calibration set.

Every request of the run is listed in a `requests` field: Garnet's in `access.json` and `compile.*.json`, Emerald's in `access.json` and its parameter file. All of them are GETs, and all returned 200.

## Files: `data/calibration/<device>/pulse/`

A file is never overwritten. A later run gets `.<YYYYMMDDTHHMMSSZ>` in its name. The committed files were written by the script before its own commit, so their `dqiqm_commit` is the base, `6516d55`.

| File | What |
|---|---|
| `garnet/pulse/access.json`, `emerald/pulse/access.json` | the constructor's outcome. On failure it would hold the error class, its chain, the HTTP status and a scrubbed message; never a header or token |
| `garnet/pulse/a18a947f-….json` | Garnet's gate and readout calibration, **per locus, for the pilot patch**: 17 qubits and 24 couplers. Also the default implementations, their classes and waveforms, each qubit's drive frequency, and the readout characterization |
| `garnet/pulse/compile.a18a947f-….json` | per-qubit readout windows, the pilot compiled locally (with and without barriers), the merging test, and the same test through qiskit's transpiler |
| `garnet/pulse/pilot/` | `E1_d3_Z_r03_pA.qasm`, `E1_d3_Z_r03_pA_nobar.qasm` and their `placement.json` entries, vendored verbatim from branch `garnet-design` at `037ad05`. The script checks each file's sha256 and op counts against `placement.json` |
| `emerald/pulse/23e12f6b-….json` | Emerald's default implementations, overrides and classes, with every parameter's median, min and max **over all loci** |
| `emerald/pulse/compile.23e12f6b-….json` | emerald-compile: per-qubit readout windows of eA and eB's 30 qubits, their readout calibration per qubit, the four R = 3 circuits compiled locally, and the request log (below) |
| `emerald/pulse/access.20261025T012734Z.json`, `emerald/pulse/23e12f6b-….20261025T012740Z.json` | emerald-compile's live run: the same constructor outcome and the same parameter summary: equal to the first two files in every field but `utc` and `dqiqm_commit` |

**What is kept, and what is not:**
- Kept: only `gates.<prx|cz|measure|measure_fidelity|reset_wait>.*`, `characterization.gate_properties.measure*.*`, `gate_definitions.*.default_implementation` and overrides, and `controllers.<qubit>.drive.frequency` (the PRX carrier).
- Values: only numbers and bools. An array keeps its length, and string values are dropped and counted (51 on Garnet's patch).
- Not saved: no other `controllers.*` field, and nothing of the settings tree or the chip design record.
- Every file passes a regex check for URLs, IPs, hostnames and auth words before it is written, and a test runs the same check on the committed files.

## Garnet: what the pulses are (calibration set `a18a947f-0408-4628-81ea-1715637e434f`)

| Gate | Implementation (class) | Pulse family |
|---|---|---|
| `prx` | `drag_crf_sx` (`PRX_DRAGCosineRiseFallSX`) | One **16 ns** √X pulse on the drive channel. I is a raised cosine (`CosineRiseFall`, full width 16 ns, rise = full width / 2, so no plateau). Q is its unit-peak derivative (`CosineRiseFallDerivative`), the DRAG quadrature, scaled by `amplitude_q`. `rz_before` and `rz_after` are virtual-Z corrections folded into the pulse phase. **`prx(π/2)` is one pulse (16 ns). `prx(π)` and any other non-zero angle are two pulses (32 ns), as R_z·SX·R_z·SX·R_z.** `prx(0)` is one zero-amplitude 16 ns pulse. |
| `cz` | `slepian_crf` (`CZ_Slepian_CRF`) | **40 ns.** The coupler flux pulse on `TC-a-b` is a `Slepian` (full width 34 ns, flat 0). The qubit flux pulse is a `CosineRiseFall` (full width 38 ns, rise 2 ns) on the *first* qubit of the calibration locus. Both qubits get a virtual RZ. |
| `measure` | `constant` (`Measure_Constant`) | A constant probe pulse at the readout frequency (5.06–6.10 GHz, amplitude 0.020–0.044), with thresholded integration on calibrated weights. Probe length = integration length = 204–556 ns. **The readout window is acquisition delay (≈465 ns) + integration length, rounded to 4 ns, + 76 ns of integration dead time: 744–1,100 ns.** |
| `measure_fidelity` | `constant` (`Measure_Constant`) | The same form, with a shorter probe (88–396 ns, integration +100 ns). Window 728–1,036 ns. The compiler uses it for **terminal** measurements (see below). |
| `reset_wait` | `reset_wait` | A 298 µs relaxation wait, prepended to every shot. |

`fast_constant` (200 ns probe, 300 ns integration) is also calibrated for `measure` on every pilot qubit, but it is not the default.

**β** below is the textbook DRAG coefficient: Q(t) = β·dI/dt, so β = (amplitude_q / amplitude_i) · full_width / π.

**Per qubit** (the 17 pilot qubits). The two error columns are the stash's `characterization.gate_properties.<gate>.constant.<q>.single_shot_10_error` and `_01_error`. On all 17 qubits these equal the quality-metric set's **heralded** `error_1_to_0` and `error_0_to_1` (`:herald=measure.constant`), checked against resonance-check's snapshot of the same calibration set. They are post-selected figures; the unheralded ones are below, under the pilot.

| qubit | probe line | drive (GHz) | PRX amp I | PRX amp Q (DRAG) | Q/I | β (ns) | rz_before = rz_after (rad) | acq. delay (ns) | measure probe / integration (ns) | measure window (ns) | heralded P(0\|1) / P(1\|0) measure | measure_fidelity probe / integration (ns) | measure_fidelity window (ns) | heralded P(0\|1) / P(1\|0) measure_fidelity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB1 | PL-1 | 4.5991 | 0.2172 | -0.0475 | -0.219 | -1.11 | 0.0567 | 465.5 | 364 / 364 | 904 | 0.0182 / 0.0032 | 168 / 268 | 808 | 0.0125 / 0.0036 |
| QB2 | PL-1 | 4.6300 | 0.2881 | -0.0536 | -0.186 | -0.95 | 0.0600 | 465.5 | 556 / 556 | 1096 | 0.0745 / 0.0016 | 120 / 220 | 760 | 0.0320 / 0.0045 |
| QB4 | PL-2 | 4.7174 | 0.2141 | -0.0449 | -0.210 | -1.07 | 0.0749 | 466.5 | 268 / 268 | 812 | 0.0109 / 0.0038 | 152 / 252 | 796 | 0.0107 / 0.0014 |
| QB5 | PL-1 | 4.5104 | 0.2812 | -0.0503 | -0.179 | -0.91 | 0.0570 | 465.5 | 556 / 556 | 1096 | 0.0260 / 0.0014 | 88 / 188 | 728 | 0.0063 / 0.0009 |
| QB6 | PL-1 | 4.6613 | 0.2577 | -0.0487 | -0.189 | -0.96 | 0.0806 | 465.5 | 412 / 412 | 952 | 0.0124 / 0.0114 | 116 / 216 | 756 | 0.0056 / 0.0017 |
| QB7 | PL-1 | 4.5118 | 0.2353 | -0.0416 | -0.177 | -0.90 | 0.0933 | 465.5 | 556 / 556 | 1096 | 0.0394 / 0.0039 | 168 / 268 | 808 | 0.0401 / 0.0038 |
| QB8 | PL-3 | 4.6453 | 0.1224 | -0.0231 | -0.189 | -0.96 | 0.0525 | 464.5 | 476 / 476 | 1016 | 0.0480 / 0.0008 | 148 / 248 | 788 | 0.0202 / 0.0002 |
| QB9 | PL-2 | 4.5347 | 0.1041 | -0.0208 | -0.200 | -1.02 | 0.0803 | 466.5 | 508 / 508 | 1052 | 0.0227 / 0.0066 | 200 / 300 | 844 | 0.0245 / 0.0017 |
| QB10 | PL-2 | 4.7702 | 0.1152 | -0.0242 | -0.210 | -1.07 | 0.0454 | 466.5 | 316 / 316 | 860 | 0.0130 / 0.0020 | 92 / 192 | 736 | 0.0081 / 0.0034 |
| QB11 | PL-2 | 4.5028 | 0.2455 | -0.0460 | -0.187 | -0.95 | 0.1078 | 466.5 | 484 / 484 | 1028 | 0.0291 / 0.0038 | 200 / 300 | 844 | 0.0338 / 0.0072 |
| QB12 | PL-1 | 4.6920 | 0.2769 | -0.0457 | -0.165 | -0.84 | 0.0428 | 465.5 | 492 / 492 | 1032 | 0.0491 / 0.0029 | 344 / 444 | 984 | 0.0281 / 0.0025 |
| QB13 | PL-3 | 4.5069 | 0.1258 | -0.0249 | -0.198 | -1.01 | 0.0807 | 464.5 | 204 / 204 | 744 | 0.0116 / 0.0060 | 92 / 192 | 732 | 0.0069 / 0.0015 |
| QB14 | PL-3 | 4.6306 | 0.1647 | -0.0266 | -0.162 | -0.82 | 0.0243 | 464.5 | 420 / 420 | 960 | 0.0098 / 0.0031 | 180 / 280 | 820 | 0.0064 / 0.0029 |
| QB15 | PL-3 | 4.5160 | 0.1290 | -0.0237 | -0.184 | -0.94 | 0.0996 | 464.5 | 492 / 492 | 1032 | 0.0422 / 0.0031 | 200 / 300 | 840 | 0.0372 / 0.0080 |
| QB16 | PL-2 | 4.6233 | 0.1065 | -0.0189 | -0.178 | -0.90 | 0.0670 | 466.5 | 556 / 556 | 1100 | 0.0250 / 0.0029 | 128 / 228 | 772 | 0.1767 / 0.0016 |
| QB19 | PL-3 | 4.6442 | 0.1990 | -0.0346 | -0.174 | -0.89 | 0.0692 | 464.5 | 524 / 524 | 1064 | 0.0309 / 0.0154 | 396 / 496 | 1036 | 0.0159 / 0.0007 |
| QB20 | PL-2 | 4.5164 | 0.1240 | -0.0234 | -0.188 | -0.96 | 0.0369 | 466.5 | 524 / 524 | 1068 | 0.0081 / 0.0026 | 172 / 272 | 816 | 0.0802 / 0.0076 |

**Per coupler** (the 24 pilot couplers). Every CZ is 40 ns. The qubit flux pulse goes on the first qubit of the calibration locus: CZ QB1–QB2 is calibrated as `QB2__QB1`, and the compiled merge test shows its flux pulses on `QB2__flux.awg` and `TC-1-2__flux.awg`. The Slepian shape parameters (`coupling_strength_normalized`, `frequency_initial_normalized`, `frequency_to_minimize_normalized`, `lambda_1`, `lambda_2`) are in the JSON.

| CZ locus (calibration order) | coupler | coupler: amplitude, full width (ns) | qubit flux: amplitude, full width, rise (ns) | rz first, second (rad) |
|---|---|---|---|---|
| QB2__QB1 | TC-1-2 | 0.2746, 34 | 0.2188, 38, 2 | 1.917, -2.102 |
| QB4__QB1 | TC-1-4 | 0.1495, 34 | 0.0629, 38, 2 | -2.894, -2.167 |
| QB2__QB5 | TC-2-5 | 0.1367, 34 | 0.0646, 38, 2 | 1.477, -2.518 |
| QB4__QB5 | TC-4-5 | 0.2764, 34 | -0.0762, 38, 2 | -2.813, -2.333 |
| QB4__QB9 | TC-4-9 | 0.1395, 34 | -0.0385, 38, 2 | -2.187, -2.356 |
| QB6__QB5 | TC-5-6 | 0.2672, 34 | 0.0149, 38, 2 | 2.448, -2.272 |
| QB10__QB5 | TC-5-10 | 0.1175, 34 | -0.1368, 38, 2 | 2.112, -2.194 |
| QB6__QB7 | TC-6-7 | 0.2415, 34 | 0.0175, 38, 2 | 2.040, -2.282 |
| QB6__QB11 | TC-6-11 | 0.1130, 34 | 0.0078, 38, 2 | -2.612, -2.336 |
| QB12__QB7 | TC-7-12 | 0.1074, 34 | -0.0377, 38, 2 | 2.645, -2.248 |
| QB8__QB9 | TC-8-9 | -0.1119, 34 | 0.0687, 38, 2 | 0.231, -2.438 |
| QB8__QB13 | TC-8-13 | -0.2380, 34 | 0.0248, 38, 2 | 0.790, -2.344 |
| QB10__QB9 | TC-9-10 | 0.1855, 34 | -0.1070, 38, 2 | 2.972, -2.101 |
| QB14__QB9 | TC-9-14 | 0.1224, 34 | 0.0927, 38, 2 | 3.054, -2.287 |
| QB10__QB11 | TC-10-11 | 0.2369, 34 | -0.1447, 38, 2 | -2.668, -2.442 |
| QB10__QB15 | TC-10-15 | 0.1192, 34 | -0.1264, 38, 2 | 0.211, -2.407 |
| QB12__QB11 | TC-11-12 | 0.2025, 34 | -0.0550, 38, 2 | -1.501, -2.161 |
| QB16__QB11 | TC-11-16 | 0.1071, 34 | 0.0437, 38, 2 | 1.841, -2.269 |
| QB14__QB13 | TC-13-14 | -0.1000, 34 | 0.0491, 38, 2 | 3.133, -2.194 |
| QB14__QB15 | TC-14-15 | -0.0918, 34 | 0.0683, 38, 2 | 0.349, -2.371 |
| QB16__QB15 | TC-15-16 | 0.1677, 34 | 0.0586, 38, 2 | -0.979, -2.245 |
| QB19__QB15 | TC-15-19 | 0.1912, 34 | 0.0345, 38, 2 | -2.742, -2.251 |
| QB16__QB20 | TC-16-20 | 0.1035, 34 | 0.0598, 38, 2 | -1.311, -2.313 |
| QB19__QB20 | TC-19-20 | 0.1597, 34 | 0.0328, 38, 2 | -2.353, -2.351 |

## Emerald (calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`), summary only

It has the same families as Garnet, with these differences:
- **`prx`:** `drag_crf_sx`, 24 ns (full width 22 ns).
- **`cz`:** `slepian_crf`, 40 ns median (max 56 ns).
- **`measure`:** `constant`, with a 360 ns probe, 500 ns integration and 460 ns acquisition delay.
- **`reset_wait`:** 400 µs.
- **Terminal readout (`measure_fidelity`):** `shelved_constant` (`Shelved_Measure_Constant`), a shelving readout. Its median single-shot errors are P(0|1) 0.004 and P(1|0) 0.001, against 0.011 and 0.001 for `measure`.
- **Per-locus overrides:** four couplers around QB50 override the CZ to `crf_acstarkcrf` (`CZ_CRF_ACStarkCRF`, 60 ns): QB44–QB50, QB49–QB50, QB50–QB51 and QB50–QB54. Only two of them (QB44–QB50 and QB50–QB51) have that implementation calibrated. The other two are among the five uncalibrated couplers that resonance-check found.
- **Drive frequencies:** 3.80–4.21 GHz.

## Emerald's circuits, compiled locally (emerald-compile)

**Why.** `device.hardware_noise(cal, spec)` takes each mid-circuit readout's idle from the compile file. Until this task only Garnet had one, and the lookup read Garnet's folder only. Emerald's priors therefore fell back on `measure_ns`, 360 ns, which is the probe alone. That is what the batch has been decoded with so far.

**The run.** `uv run python scripts/pulla_check.py --devices emerald`, once, at 2026-10-25 01:27 UTC, at commit `9699986`. It was read-only:
- Every request went through the GET-only guard: 11 GETs, all 200, listed in the compile file's `requests`. They were the constructor's four, `quantum-computers`, `calibration-sets/emerald/default/dynamic-quantum-architecture` (to name the set), `calibration-sets/emerald/23e12f6b-…` (the stash), and four of `artifacts/duts`.
- No `DQ_QPU*` was set, and nothing went in the ledger.
- The live calibration set was **`23e12f6b-c23c-4736-8736-b0142989c0e6`**, the one the circuits were placed on. The script compiles only on that set. On any other, the parameter file records the live set and the script stops with exit code 2.

**What was compiled.** The batch's d = 3, R = 3 memory circuits: `E1_d3_Z_r03_eA`, `E1_d3_X_r03_eA`, `E1_d3_Z_r03_eB` and `E1_d3_X_r03_eB`.
- Each is read from `InfoRecord/emerald/circuits/E1/` and checked against its `placement.json` entry: sha256, op counts, qubits and zero SWAPs. Each of its 24 CZ pairs is confirmed by `ChipTopology.get_coupler_for`.
- They are compiled with Pulla's standard compiler on the pinned stash, as Garnet's pilot was: body, full, and body with neighbourhood 0.
- Windows are recorded for all **30 qubits** of eA and eB, which share QB19, QB20, QB27 and QB28. The file also keeps each qubit's `measure`, `measure_fidelity` and `prx_12` calibration, so the windows can be recomputed.

**The window: 1,036 ns on every one of the 30 qubits**, for `measure` and `measure_fidelity` alike.
- It is IQM's arithmetic: acquisition delay 460 ns + integration 500 ns (rounded to 4 ns) + 76 ns of integration dead time. All 60 windows recompute from the recorded parameters with 0 mismatches.
- Emerald's readout is uniform on these qubits, so unlike Garnet the window doesn't depend on which qubit is slowest. Against the 360 ns probe, it is **2.88×**.

**The compiled schedule: 3,956 ns, against our 1,848 ns**, for all four circuits.
- Gates: 256 ns per round, the same as ours: four 40 ns CZ layers and four 24 ns PRX layers.
- Readout: each round's 8 ancilla readouts are multiplexed into one 1,036 ns trigger per probe line. eA's ancillas sit on 5 lines (PL-1, 2, 3, 4, 8) and eB's on 5 (PL-1, 2, 5, 6, 7). The triggers start at 256, 1,548 and 2,880 ns.
- So 3 × 256 + 3 × 1,036 + 80 = 3,956 ns. The full compile adds the 400 µs reset wait (403,956 ns), and neighbourhood 0 changes nothing.
- **The extra 80 ns is the shelved terminal readout.** Emerald's `measure_fidelity` is `shelved_constant`, and every terminal measurement is converted to it (after the circuit stage: `measure` 16, `measure_fidelity` 18).
  - Shelving plays a 40 ns `prx_12(π)` (`modulated_drag_crf`) before the trigger and another after it (`do_prx_12` true, `second_prx_12_offset` 0).
  - `probe_all` makes the terminal layer read **all 54 qubits** of the chip: 54 acquisitions on all 8 lines, of which 37 are probe only. Each of the 54 gets both `prx_12` pulses: 214 IQPulses = 106 PRX + 108 `prx_12` (212 = 104 + 108 for X).
  - So the terminal layer lasts 40 + 1,036 + 40 = 1,116 ns. `measure_fidelity_window_ns` (1,036) is the trigger alone. It enters only durations, never an idle prior: nothing idles through the terminal layer.
- The probe pulse in the schedule is 376 ns, against the 360 ns `duration` of the calibration.

**The lookup** (`src/dqiqm/experiments/device.py`).
- `compiled_readout_windows(calset, directory=None, *, device=None)` reads `data/calibration/<device>/pulse/compile.<calset>.json` (`device.compile_dir`). With no device, it reads whichever device's folder holds the file. With a device, it refuses a file that names another one.
- `hardware_noise` passes `cal.device`, so Emerald's priors read Emerald's file, and Garnet's read Garnet's.
- `COMPILE_DIR` is still Garnet's folder, and a `directory` argument still works.
- Garnet's prior is **byte for byte unchanged**: `tests/test_emerald_compile.py` pins the sha256 of the noisy circuit as it was on main `79092cf`. The window is 1,096 ns, from the same file and with the same source string.

### The comparison: the Emerald pilot under both priors

On the pilot's record only (EP0, on the fine-tune side). The batch's records are not touched; they are re-decoded after the batch merges.

<!-- emerald-compile:begin -->
Generated by `scripts/pulla_check.py --compare`: job `01a17017-0aef-7503-a36d-6d3151635fdf` (EP0, `E1_d3_Z_r03_eA`, 1,000 shots, R = 3, calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`), re-decoded from its committed `memory.npz` with PyMatching. Priors from the pilot's before snapshot (`data/calibration/emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261024T224404Z.json`); compiled windows from `data/calibration/emerald/pulse/compile.23e12f6b-c23c-4736-8736-b0142989c0e6.json`. Model columns: each prior's own noisy circuit, 100,000 Stim shots, seed 2026. P_L intervals are Wilson 95%.

**The readout window, per patch.** Fallback: the slowest ancilla's `measure` duration, the probe alone (`device.readout_layer_ns` without compile data). Compiled: the slowest ancilla's `measure_window_ns`. Compiled layers: each readout layer's longest trigger in the compiled body (rounds 1, 2, then the terminal layer).

| patch | ancillas | fallback (ns) | compiled mid-circuit window (ns) | slowest ancilla | compiled / fallback | terminal `measure_fidelity` window, slowest of 17 (ns) | compiled layers, Z / X (ns) | our schedule → compiled body, Z / X (ns) |
|---|---|---|---|---|---|---|---|---|
| eA | QB14, QB19, QB21, QB28, QB30, QB37, QB39, QB44 | 360 | **1,036** | all 8 | 2.88× | 1,036 | 1,036, 1,036, 1,036 / 1,036, 1,036, 1,036 | 1,848 → 3,956 / 1,848 → 3,956 |
| eB | QB5, QB8, QB10, QB17, QB19, QB26, QB28, QB33 | 360 | **1,036** | all 8 | 2.88× | 1,036 | 1,036, 1,036, 1,036 / 1,036, 1,036, 1,036 | 1,848 → 3,956 / 1,848 → 3,956 |

**`device.hardware_noise(cal, spec)`'s info dict**, before (no Emerald compile file: the `measure_ns` fallback, which is what `windows=None` gives) and after:

| key | eA before | eA after | eB before | eB after |
|---|---|---|---|---|
| `readout_window_ns` | `360.0` | `1036.0` | `360.0` | `1036.0` |
| `readout_window_source` | `measure_ns (no compile data: probe only)` | `compiled measure_window_ns (data/calibration/emerald/pulse/compile.23e12f6b-c23c-4736-8736-b0142989c0e6.json)` | `measure_ns (no compile data: probe only)` | `compiled measure_window_ns (data/calibration/emerald/pulse/compile.23e12f6b-c23c-4736-8736-b0142989c0e6.json)` |
| `delay_ns` | `0.0` | `0.0` | `0.0` | `0.0` |
| `idle_ns` | `360.0` | `1036.0` | `360.0` | `1036.0` |
| `idle_channel` | `depolarize` | `depolarize` | `depolarize` | `depolarize` |
| `idle_frame` | `None` | `None` | `None` | `None` |
| `t2` | `ramsey` | `ramsey` | `ramsey` | `ramsey` |
| `final_readout` | `measure_fidelity (unheralded)` | `measure_fidelity (unheralded)` | `measure_fidelity (unheralded)` | `measure_fidelity (unheralded)` |
| `idle_after_last_round` | `False` | `False` | `False` | `False` |
| `family` | `surface_memory` | `surface_memory` | `surface_memory` | `surface_memory` |
| `calibration_set_id` | `23e12f6b-c23c-4736-8736-b0142989c0e6` | `23e12f6b-c23c-4736-8736-b0142989c0e6` | `23e12f6b-c23c-4736-8736-b0142989c0e6` | `23e12f6b-c23c-4736-8736-b0142989c0e6` |

**EP0 decoded, and the detection fraction per detector round t** (t = 0: round 1, Z-type only; t = 3: the final data detectors). `pilot_record` is the pilot's own decoder, rebuilt (it must give the record's 129 failures, 294 DEM errors and model P_L 0.09154); `fallback_360` is today's batch prior; `compiled` is the prior with this task's compile file.

| priors | idle (ns) | final readout | idle after last round | DEM errors | d | hardware P_L (1,000 shots, R = 3) | ε single point (R = 3) | model P_L | t = 0, model | t = 1, model | t = 2, model | t = 3, model | Σ\|model − hw\| | detector log-lik | vs fallback_360: fails only here / only there (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hardware** | | | | | | | | | **0.0853** | **0.1674** | **0.1722** | **0.0622** | | | |
| `pilot_record` | 360 | measure | yes | 294 | 3 | 129/1000 = **0.129** [0.110, 0.151] | 0.0473 [0.0396, 0.0566] | 0.0915 [0.0898, 0.0933] | 0.0612 | 0.1213 | 0.1335 | 0.1248 | 0.171 | -9694 | 5 / 9 (p = 0.42) |
| `fallback_360` | 360 | measure_fidelity | no | 294 | 3 | 133/1000 = **0.133** [0.113, 0.155] | 0.0490 [0.0411, 0.0584] | 0.0712 [0.0697, 0.0729] | 0.0612 | 0.1213 | 0.1315 | 0.0815 | 0.130 | -9630 |  |
| `compiled` | 1036 | measure_fidelity | no | 294 | 3 | 136/1000 = **0.136** [0.116, 0.159] | 0.0502 [0.0422, 0.0597] | 0.1373 [0.1352, 0.1395] | 0.0612 | 0.1735 | 0.1898 | 0.0936 | 0.079 | -9500 | 18 / 15 (p = 0.73) |

Hardware over model, per round:

| priors | t = 0 | t = 1 | t = 2 | t = 3 |
|---|---|---|---|---|
| `pilot_record` | 1.39× | 1.38× | 1.29× | 0.50× |
| `fallback_360` | 1.39× | 1.38× | 1.31× | 0.76× |
| `compiled` | 1.39× | 0.96× | 0.91× | 0.67× |

By stabilizer type (detection fraction):

| priors | Z t = 0 | Z t = 1 | Z t = 2 | Z t = 3 | X t = 1 | X t = 2 |
|---|---|---|---|---|---|---|
| **hardware** | **0.0853** | **0.1817** | **0.1860** | **0.0622** | **0.1530** | **0.1585** |
| `pilot_record` | 0.0612 | 0.1274 | 0.1406 | 0.1248 | 0.1153 | 0.1263 |
| `fallback_360` | 0.0612 | 0.1274 | 0.1392 | 0.0815 | 0.1153 | 0.1238 |
| `compiled` | 0.0612 | 0.1798 | 0.1989 | 0.0936 | 0.1672 | 0.1808 |
<!-- emerald-compile:end -->

**What it says:**
- **The rebuild is exact.** `pilot_record` is the pilot's own decoder (`pre_fix_noise(emerald.calibrated_noise(idle_ns=360))`), and it gives back the record exactly: 129 failures, 294 DEM errors, and the record's model numbers at seed 2026. garnet-timing had left this unverified for Emerald.
- **The compiled window closes the bulk deficit.** At t = 1 and 2, the hardware over the model is 1.38× and 1.31× with the 360 ns fallback, and 0.96× and 0.91× with the compiled 1,036 ns. Both detector types move, Z-type and X-type alike. The compiled prior fits best: detector log-likelihood −9,500 against −9,630, and Σ|model − hw| 0.079 against 0.130. Its model P_L, 0.137, matches the hardware's 0.136. The fallback's model predicts 0.071.
- **Decoding doesn't separate them.** The hardware P_L is 0.133 with the fallback and 0.136 compiled, with 18 against 15 discordant shots (McNemar p = 0.73). That is the same picture as Garnet's `b_ramsey` against `a_old`. It is one Z-basis circuit at 1,000 shots, so choose per basis on the batch's finetune validation shots, by decoded P_L, as garnet-timing set out.
- **What the window leaves unexplained:**
  - Round 0 (t = 0) stays at 1.39×. No idle comes before round 1, so the window can't move it. Those are excess flips in the first round's Z-type checks: state preparation, the first gates, or the first readout.
  - The final data detectors (t = 3) are still over-predicted: 0.67× compiled, against 0.76× with the fallback and 0.50× for the pilot's own prior. The longer window in round 2 raises the model's t = 3 rate a little.
  - These are notes for the decoder work, not fixes made here.

## The pilot circuit, compiled locally

The pilot is `E1_d3_Z_r03_pA`: d = 3, Z basis, 3 rounds, 17 qubits, 24 couplers, 106 `r(π/2, ±π/2)`, 72 CZ, 33 measurements and 17 barriers. The variant without barriers is `_nobar`. Each is compiled with Pulla's standard compiler on the pinned calibration set. "Body" means the same compiler with its `prepend_reset` pass removed, so the 298 µs reset wait is left out.

| Variant | Our schedule | Compiled body | Body with neighborhood 0 | Full compile (with the reset wait) | Readout triggers |
|---|---|---|---|---|---|
| with barriers | **2,340 ns** | **3,900 ns** | 3,900 ns | 301,900 ns | 9 |
| no barriers | 2,260 ns | **7,748 ns** | 7,748 ns | 305,748 ns | 19 |

**Where the 1,560 ns (barrier version) come from: the readout window, not the gates.**

| | gates per round | readout per round |
|---|---|---|
| our model, ×3 | 224 ns | 556 ns (the longest probe pulse in the patch) |
| compiled | 224 ns, the same | 1,096 / 1,096 / 1,036 ns |

The compiled readout triggers start at 224, 1,544 and 2,864 ns. Each round's eight ancilla readouts are **multiplexed into one trigger per probe line**: 3+3+2 acquisitions in rounds 1–2. Round 3 has 7+7+6 = 20: the 17 final readouts, plus the three probe-only qubits. Each trigger is as long as the longest window on its line. PL-1's mid-circuit trigger carries ancillas QB1, QB5 and QB7, and QB5 and QB7 have 1,096 ns windows.

So a round is 224 + 1,096 ns. Every qubit, data qubits included, idles through the whole window. Our model counts the 556 ns probe pulse. The compiled window adds the ≈465 ns acquisition delay and the 76 ns integration dead time: 1,096 − 556 = 540 ns per mid-circuit round, and 1,036 − 556 = 480 ns for the final readout. The gate part is the same in both: four 40 ns CZ layers and four 16 ns PRX layers, with barrier layers of 56, 56, 40, 56 and 16 ns.

**Neighborhood 0 changes nothing.** That was a diagnostic run with `timebox_resolution.resolve_timeboxes.neighborhood = 0`. So neighbouring-coupler blocking costs this circuit nothing.

**How barriers come out.**
- A barrier becomes zero-duration `Block` instructions (442 in this schedule) on the control channels of its qubits. Under the hard-boundary scheduler they force those channels to a common time. They add no time of their own.
- The barriers also make the readout fast. IQM's `multiplex_readout` pass merges only **consecutive** measure instructions, with no other instruction on those qubits in between; IQM's own comment calls it "fragile".
  - With barriers, a round's eight measurements are consecutive, so each round is one trigger per probe line.
  - In the `_nobar` QASM, each ancilla's measurement is followed at once by that ancilla's next gate. Every measurement then gets its own trigger (19 triggers, 1 acquisition each until the final readout), and the triggers queue on the three shared probe lines.
  - **Without barriers the compiled schedule is 2.1× longer** (7,748 ns against 3,900 ns), although our ASAP model says it is 80 ns shorter.

**Is anything merged?**
- Gates: no. The schedule has 106 `IQPulse`s, one per `r(π/2)`, and 144 `FluxPulse`s and 144 `VirtualRZ`s, two of each per CZ.
- Readout: yes. The 33 measurements are merged into 9 triggers by the readout multiplexing described above. That is the only merging.

**The terminal measurements change.** After the circuit stage, the circuit has `measure` 16 and `measure_fidelity` 18:
- Rounds 1–2 (16 ancilla readouts) stay `measure`.
- The round-3 ancilla readouts and the data readout are *terminal*: nothing follows them on those qubits. `subscribe_and_probe` (`convert_terminal_measurements=True`) turns them into `measure_fidelity`.
- One more `measure_fidelity`, probe only with no data returned, goes on QB3, QB17 and QB18 (`probe_all=True`).

So the last round is read out with `measure_fidelity`'s calibration and errors, not `measure`'s.

The unheralded errors of the two readouts, from the quality-metric set in resonance-check's snapshot (`data/calibration/garnet/a18a947f-….json` → `quality_metric_set`: `metrics.ssro.<gate>.constant.<q>.error_1_to_0` and `.error_0_to_1`):

| qubit | role | measure P(0\|1) / P(1\|0) | measure_fidelity P(0\|1) / P(1\|0) | measure_fidelity, heralded P(0\|1) |
|---|---|---|---|---|
| QB1 | ancilla | 0.0276 / 0.0046 | 0.0188 / 0.0052 | 0.0125 |
| QB2 | data | 0.0912 / 0.0124 | 0.0382 / 0.0132 | 0.0320 |
| QB4 | data | 0.0120 / 0.0068 | 0.0134 / 0.0042 | 0.0107 |
| QB5 | ancilla | 0.0290 / 0.0082 | 0.0090 / 0.0046 | 0.0063 |
| QB6 | data | 0.0184 / 0.0142 | 0.0106 / 0.0106 | 0.0056 |
| QB7 | ancilla | 0.0414 / 0.0100 | 0.0410 / 0.0110 | 0.0401 |
| QB8 | data | 0.0352 / 0.0044 | 0.0240 / 0.0062 | 0.0202 |
| QB9 | ancilla | 0.0358 / 0.0238 | 0.0258 / 0.0180 | 0.0245 |
| QB10 | data | 0.0192 / 0.0176 | 0.0118 / 0.0184 | 0.0081 |
| QB11 | ancilla | 0.0878 / 0.0166 | 0.0674 / 0.0260 | 0.0338 |
| QB12 | data | 0.0596 / 0.0164 | 0.0410 / 0.0144 | 0.0281 |
| QB13 | ancilla | 0.0208 / 0.0270 | 0.0168 / 0.0226 | 0.0069 |
| QB14 | data | 0.0466 / 0.0122 | 0.0196 / 0.0134 | 0.0064 |
| QB15 | ancilla | 0.0498 / 0.0066 | 0.0390 / 0.0150 | 0.0372 |
| QB16 | data | 0.0332 / 0.0232 | 0.0152 / 0.0180 | 0.1767 |
| QB19 | data | 0.0264 / 0.0234 | 0.0200 / 0.0104 | 0.0159 |
| QB20 | ancilla | 0.0174 / 0.0180 | **0.0898** / 0.0204 | 0.0802 |

- Unheralded, `measure_fidelity` is as good as or better than `measure` on 16 of the 17 qubits.
- The exception is **QB20**, an ancilla whose round-3 readout gets converted: P(0|1) is 0.090 against 0.017. Its heralded figure (0.080) agrees.
- QB16's heralded `measure_fidelity` P(0|1) of 0.177 contradicts its own unheralded 0.015, and QB16 is the only qubit where heralding makes the figure *worse*. Treat it as an outlier of that calibration measurement, not as a readout property.

**Caveat.** Our circuit jobs are compiled by the server, not by us. Pulla's `get_standard_compiler` builds IQM's compiler from the station's own settings and the standard stages. We cannot see which stage options the server uses for circuit jobs, so these schedules show what IQM's compiler does by default, not a measured run.

## The gate-merging question

N consecutive `prx(π/2)` on QB2, and N consecutive CZs on QB1–QB2, compiled to schedules, with and without a barrier between each gate. "Gates" is the time to the first readout. "Body" includes the terminal readout that `probe_all` adds.

| Gate | N | Barriers | Pulses | Gates (ns) | Body (ns) |
|---|---|---|---|---|---|
| prx | 1 | no | 1 | 16 | 1,052 |
| prx | 5 | no | 5 | 80 | 1,116 |
| prx | 21 | no | 21 | 336 | 1,372 |
| prx | 1 | yes | 1 | 16 | 1,052 |
| prx | 5 | yes | 5 | 80 | 1,116 |
| prx | 21 | yes | 21 | 336 | 1,372 |
| cz | 1 | no | 2 (coupler + qubit flux) | 40 | 1,076 |
| cz | 5 | no | 10 | 200 | 1,236 |
| cz | 21 | no | 42 | 840 | 1,876 |
| cz | 1 | yes | 2 | 40 | 1,076 |
| cz | 5 | yes | 10 | 200 | 1,236 |
| cz | 21 | yes | 42 | 840 | 1,876 |

**IQM's compiler neither merges nor cancels repeated gates.** The challenge's extension text pictures PulLA "fusing or reshaping pulses". The standard compiler does none of that, so any fusing would be our own implementation or pass.
- N `prx(π/2)` are N pulses, exactly 16 ns each. N CZs are 2N flux pulses (plus 2N virtual RZs), exactly 40 ns each.
- Barriers make no difference to a chain on one qubit or one pair.
- Five `prx(π/2)` are not shortened to one, and 21 CZs are not reduced to 1.

**Qiskit's transpiler does merge them**, before IQM ever sees the circuit. This was checked offline, against `IQMFakeGarnet` with `initial_layout=[0, 1]`. The gate counts after `transpile`:

| Chain | Barriers | level 0 | level 1 | level 2 | level 3 |
|---|---|---|---|---|---|
| 5 × r(π/2) | no | 5 | **1** | **1** | **1** |
| 21 × r(π/2) | no | 21 | **1** | **1** | **1** |
| 5 × CZ | no | 5 | **1** | **1** | **1** |
| 21 × CZ | no | 21 | **1** | **1** | **1** |
| 5 × r(π/2) | yes | 5 | 5 | 5 | 5 |
| 21 × r(π/2) | yes | 21 | 21 | 21 | 21 |
| 5 × CZ | yes | 5 | 5 | 5 | 5 |
| 21 × CZ | yes | 21 | 21 | 21 | 21 |

**For a gate-repetition experiment:**
- Put a barrier between the repeated gates, or transpile at `optimization_level=0`.
- Check the transpiled circuit's `count_ops()` before it goes to `hw.submit`.
- Keep the angle at π/2 if each repetition should be one pulse. A `prx(π)` is already two pulses.

## What a pulse-level experiment could change (our own jobs only)

All of these change only the run definition that one job would carry. Nothing writes calibration.

- **A different or custom gate implementation:**
  - `Compiler.add_implementation(op_name, impl_name, impl_class, set_as_default=...)`;
  - `Compiler.set_default_implementation(gate, impl)`;
  - `Compiler.set_default_implementation_for_loci(gate, impl, loci)`;
  - `get_standard_compiler(gate_definitions=...)`.

  The library already ships many classes in `iqm.pulse.gates`: `PRX_DRAGGaussian`, `PRX_FastDrag`, `PRX_HdDrag`, `CZ_CRF`, `CZ_TruncatedGaussianSmoothedSquare`, ...
- **PRX amplitude, phase or DRAG for one job.** Take `settings = compiler.get_settings(circuits=...)` and edit, for example, `settings.gates.prx.drag_crf_sx.QB2.amplitude_i`, `.amplitude_q` (DRAG) or `.rz_before` / `.rz_after`. Then call `compiler.compile(circuits, settings=settings)`. Another way is a local stash: `LatestFromStash(PullaStash({...}))` with edited observations, passed as `loading_rules`.
- **Rabi (an amplitude sweep):** `compiler.compile(circuits, sweeps=...)`, with `exa.common.control.sweep.sweep.Sweep` over `settings.gates.prx.drag_crf_sx.<q>.amplitude_i`. A hard sweep gives one playlist segment per point.
- **Ramsey detuning:**
  - Through Pulla: sweep the drive frequency (`settings.controllers.<q>.drive.frequency`).
  - With no pulse access at all, a circuit job does it: `prx(π/2)`, `delay(t)`, then `prx(π/2)` with phase 2π·Δ·t (virtual detuning).
- **CZ overrides.** These are sweepable like the PRX parameters:
  - `settings.gates.cz.slepian_crf.<QBa__QBb>.coupler.amplitude`;
  - `.qubit.amplitude`;
  - `.duration`;
  - `.rz.<q>`.
- **Dynamical decoupling:**
  - Through Pulla: the stage option `dynamical_decoupling.apply_dd_strategy.dd_is_disabled = False`, or a `DDStrategy` (`iqm.cpc.compiler.dd`) in the compile context under `"DDStrategy"`. `iqm.pulla.utils_dd.generate_phases_of_urn_sequence(n)` gives the URn phases for a custom sequence.
  - **No Pulla is needed for DD.** A circuit job takes `CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=DDStrategy(...))` from `iqm.iqm_client.models`, passed as `hw.submit(..., circuit_compilation_options=...)`, which goes to `IQMBackend.create_run_request`.
  - The same object also carries `heralding_mode`, `active_reset_cycles` (which also stops the terminal `measure_fidelity` conversion) and `max_circuit_duration_over_t2`.
  - In the compiled pilot, a measured qubit's drive channel is *blocked* during its readout, but every other qubit's channel just *waits*. So the data qubits' ~1.1 µs idles during ancilla readout are exactly where DD would go.

**What Pulla cannot do, or what we did not establish, from our account:**
- **Write calibration.** Neither Pulla nor the server client has a call that writes a calibration set. Every change above lives in one job's run definition and nowhere else.
- **Submit a pulse-level job.** This is unverified: `submit_playlist` was never called. It may be refused for a non-paid organisation.
- **Book a timeslot.** `use_timeslot=True` books dedicated time and costs money. It is out of bounds.
- **Use `dqiqm.hw` for it.** `hw` has no path for pulse-level jobs. `Pulla.submit_playlist` → `submit_run` would bypass `hw`'s arming, allowance and ledger. A pulse-level task needs an `hw.submit_playlist` wrapper first, with the same guard, ledger line and raw-data saving as `hw.submit`.
- **Choose a non-default implementation from qiskit**, such as `measure` → `fast_constant`. The qiskit serializer writes no `implementation`. `CircuitOperation` has the field, but IQM calls it experimental, and whether the server honours it in a circuit job is untested.

## What this changes for the project

1. **A Garnet d=3 round is about 1.32 µs, not 0.78 µs.** Each mid-circuit readout window is about 1.1 µs, set by the slowest measured qubit on each probe line (PL-1: ancillas QB5 and QB7 at 1,096 ns). Every qubit idles through it. A noise model's per-round idle should use the compiled window, which `compile.*.json` → `readout_windows` gives per qubit, not the probe length.
2. **Keep a barrier before each measurement layer.** Without one, the compiled pilot is 2.1× longer.
3. **The final round is read out with `measure_fidelity`**, whose errors differ from `measure`'s. Decoder priors for the round-3 ancilla readout and the data readout should use the unheralded `metrics.ssro.measure_fidelity.constant.<q>.*` figures, and the mid-circuit rounds `measure`'s, unless the server's settings say otherwise. On this calibration set that is mostly a small improvement. QB20 is the exception: P(0|1) 0.090 against 0.017. The stash's `characterization` readout figures are the heralded ones, so they are not the right priors for unheralded data.
4. **Repeated-gate experiments need barriers, or level-0 transpilation.** IQM won't merge the gates, but qiskit will.
5. **An Emerald d=3 round is about 1.29 µs (256 + 1,036 ns), not 0.62 µs** (emerald-compile). Every ancilla's window is 1,036 ns, and `hardware_noise` now reads it from Emerald's own compile file: on the pilot, the bulk rounds' hardware-over-model ratio goes from about 1.35× to 0.94×.
6. **DD, heralding and active reset are circuit-job options.** They are reachable through `hw.submit` now, with no pulse access.
