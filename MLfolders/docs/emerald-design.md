# Emerald design: the Garnet suite on the challenge's device

Sal, 2026-10-24 about 21:05 UTC: "yes run on emerald too and you can also train and gain data from
emerald as well as you are doing for garnet … just do it quicker as well … if its the same as
garnet but just a bit different". This is the Garnet design (`docs/garnet-design.md`) applied to
IQM Emerald: E1 (the d=3 no-reset memory, the main ML dataset) on Emerald's two best patches, E4
(the delay sweep) on the best one, the record under `InfoRecord/emerald/`, and the hardware job
plan for Sal's approval. Everything here is offline: no Resonance access, no job.

Code: `src/dqiqm/experiments/emerald.py` (it wraps the Garnet modules, which are not edited),
`scripts/emerald_dryrun.py`, `scripts/emerald_plan.py`, `scripts/emerald_report.py`,
`tests/test_emerald_design.py`.

## The device

Live snapshot of 2026-10-24 19:13 UTC, calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`
(resonance-check), byte-copied into `InfoRecord/emerald/calibration/emerald/`.

- **54 qubits, 90 static couplers, 85 with a CZ.** The 5 without one: **QB6–QB7, QB15–QB23,
  QB24–QB32, QB49–QB50, QB50–QB54**. Every placement here is searched on the 85
  (`emerald.architecture`: the static grid, `layout.Architecture.with_couplers`), and every circuit
  is checked against them (`emerald.check_live_couplers`: native ops only, every `cz` on a
  calibrated coupler, none on the 5, no SWAP). That check is needed because `layout.check_hardware`
  and IQM's `validate_circuit` see the stand-in's 90 couplers.
- **Figures** (resonance-check): T1 median 46.3 µs (worst 6.5 µs, QB35); T2 echo median 36.7 µs;
  readout median 0.0118 (worst QB3: P(0|1) 0.151); CZ IRB median 0.0035 (worst 0.0538, QB35–QB43).
- **Durations:** prx 24 ns, CZ 40 ns, measure 360 ns (medians), **reset_wait 400 µs** (Garnet: 298 µs).
- **Readout figures** are the default `measure` (`constant`) `error_0_to_1` / `error_1_to_0`, not
  the heralded figure (`device.calibration_from_snapshot`, unchanged).

## The stand-in

`IQMFakeAphrodite().copy_with_error_profile(<live 23e12f6b profile>)` (`emerald.fake_backend`).
Aphrodite has exactly Emerald's 54 qubit names and 90 couplers. Two adaptations, both in the
stand-in only:

1. **The 5 CZ-less couplers are padded.** `copy_with_error_profile` refuses a profile without all
   90 couplers ("missing loci"). They get a depolarizing parameter of 1.0: no circuit uses them
   (checked on every circuit), and one that did would decode at chance.
2. **T2 is capped at T1, not 2·T1, on 5 qubits** (QB8, QB30, QB37, QB39, QB50, whose Ramsey T2 is
   above T1). With T2 > T1, Aer's thermal relaxation is a general Kraus channel, and Aer 0.17.2's
   matrix-product-state method fails on it (`QuantumError: Kraus is empty` on the pilot circuit, a
   segfault on a one-qubit circuit; the shipped profile, whose T2 < T1 everywhere, runs). The cap adds a
   little dephasing on those 5 qubits. The dry run's decoder priors keep the 2·T1 cap
   (`device.t2_ns`), so on those qubits they understate the stand-in's Z error per gate by
   2×10⁻⁵ (QB8, QB39, QB50) to 2.3×10⁻⁴ (QB30, per CZ; QB37 2.0×10⁻⁴), against a CZ error of
   about 5×10⁻³ (1.25 × the median IRB 0.0035).

Like Garnet's, the stand-in has **no idle noise** (Aer puts none on `delay` or idle qubits), and its
gate noise is about twice the benchmarked figure (garnet-design review, MINOR 4). So E4 is checked
for structure only, and **`sim/` is not pre-training data**.

## Placements: eA and eB

On the 85 calibrated couplers the d=3 code fits with zero SWAPs in **60 placements on 15 qubit
sets** (4 orientations each). With all 90 it would be 92 on 23 sets; the 5 CZ-less couplers rule
out 8 sets.

- **eA** = the placement with the lowest mean P_L(R = 3, Z and X) under calibrated Stim noise
  (`memory.rank`, PyMatching, 50,000 shots per basis and placement), the same rule as Garnet's pA.
- **eB** = the best placement on **a different qubit set**. Garnet has one set, so its pB is a
  re-orientation; Emerald has 15, so eB is a real spatial repeat. eA and eB share 4 qubits
  (QB19, QB20, QB27, QB28); they run in separate jobs.

The full ranking is in the table below and in `job_plan.json: choices.E1` (every placement).

<!-- report:begin -->
### The placements

Ranked by mean P_L at R = 3 (Z and X), PyMatching under calibrated Stim noise, 50,000 shots per basis and placement (`memory.rank` on the 85 calibrated couplers); one row per qubit set, its best orientation.

| rank | patch | qubits | best orientation | X-type ancillas | P_L Z | P_L X | mean |
|---|---|---|---|---|---|---|---|
| 1 | eA | QB13, QB14, QB19, QB20, QB21, QB22, QB27, QB28, QB29, QB30, QB31, QB36, QB37, QB38, QB39, QB44, QB45 | 1 | QB14, QB28, QB30, QB44 | 0.0464 | 0.0487 | 0.0475 |
| 2 | eB | QB4, QB5, QB8, QB9, QB10, QB11, QB16, QB17, QB18, QB19, QB20, QB25, QB26, QB27, QB28, QB33, QB34 | 2 | QB8, QB10, QB26, QB28 | 0.0530 | 0.0477 | 0.0503 |
| 3 |  | QB3, QB4, QB9, QB10, QB11, QB12, QB16, QB17, QB18, QB19, QB20, QB24, QB25, QB26, QB27, QB34, QB35 | 5 | QB3, QB17, QB19, QB35 | 0.0501 | 0.0529 | 0.0515 |
| 4 |  | QB5, QB6, QB9, QB10, QB11, QB12, QB17, QB18, QB19, QB20, QB21, QB26, QB27, QB28, QB29, QB34, QB35 | 1 | QB6, QB18, QB20, QB34 | 0.0567 | 0.0535 | 0.0551 |
| 5 |  | QB11, QB12, QB19, QB20, QB21, QB22, QB26, QB27, QB28, QB29, QB30, QB34, QB35, QB36, QB37, QB44, QB45 | 5 | QB11, QB27, QB29, QB45 | 0.0556 | 0.0558 | 0.0557 |
| 6 |  | QB4, QB5, QB10, QB11, QB12, QB13, QB17, QB18, QB19, QB20, QB21, QB25, QB26, QB27, QB28, QB35, QB36 | 7 | QB4, QB18, QB20, QB36 | 0.0639 | 0.0482 | 0.0560 |
| 7 |  | QB17, QB18, QB25, QB26, QB27, QB28, QB32, QB33, QB34, QB35, QB36, QB40, QB41, QB42, QB43, QB48, QB49 | 6 | QB26, QB28, QB40, QB42 | 0.0594 | 0.0542 | 0.0568 |
| 8 |  | QB1, QB2, QB3, QB4, QB5, QB6, QB9, QB10, QB11, QB12, QB13, QB18, QB19, QB20, QB21, QB26, QB27 | 3 | QB2, QB10, QB12, QB26 | 0.0668 | 0.0556 | 0.0612 |
| 9 |  | QB10, QB11, QB16, QB17, QB18, QB19, QB24, QB25, QB26, QB27, QB28, QB33, QB34, QB35, QB36, QB41, QB42 | 1 | QB11, QB25, QB27, QB41 | 0.0586 | 0.0650 | 0.0618 |
| 10 |  | QB5, QB6, QB11, QB12, QB13, QB14, QB18, QB19, QB20, QB21, QB22, QB26, QB27, QB28, QB29, QB36, QB37 | 4 | QB12, QB14, QB26, QB28 | 0.0626 | 0.0620 | 0.0623 |
| 11 |  | QB11, QB12, QB17, QB18, QB19, QB20, QB25, QB26, QB27, QB28, QB29, QB34, QB35, QB36, QB37, QB42, QB43 | 1 | QB12, QB26, QB28, QB42 | 0.0785 | 0.0560 | 0.0673 |
| 12 |  | QB10, QB11, QB18, QB19, QB20, QB21, QB25, QB26, QB27, QB28, QB29, QB33, QB34, QB35, QB36, QB43, QB44 | 5 | QB10, QB26, QB28, QB44 | 0.0805 | 0.0618 | 0.0712 |
| 13 |  | QB12, QB13, QB18, QB19, QB20, QB21, QB26, QB27, QB28, QB29, QB30, QB35, QB36, QB37, QB38, QB43, QB44 | 3 | QB13, QB27, QB29, QB43 | 0.0638 | 0.0821 | 0.0730 |
| 14 |  | QB19, QB20, QB25, QB26, QB27, QB28, QB33, QB34, QB35, QB36, QB37, QB42, QB43, QB44, QB45, QB48, QB49 | 3 | QB20, QB34, QB36, QB48 | 0.0811 | 0.0679 | 0.0745 |
| 15 |  | QB19, QB20, QB27, QB28, QB29, QB30, QB34, QB35, QB36, QB37, QB38, QB42, QB43, QB44, QB45, QB50, QB51 | 7 | QB19, QB35, QB37, QB51 | 0.0784 | 0.0977 | 0.0880 |

### Structure (every circuit)

| experiment | circuits | detectors | measurements | observables | SWAPs | CZ on calibrated couplers | DEM errors | graphlike distance |
|---|---|---|---|---|---|---|---|---|
| E1 | 44 | 8–160 | 17–169 | [1] | [0] | all (24–480 per circuit) | 29–2849 | [3] |
| E4 | 24 | 16–64 | 25–73 | [1] | [0] | all (48–192 per circuit) | 149–1049 | [3] |

**Noiseless, through the hardware path** (`checks/noiseless.json`): all 68 circuits, 32 shots each, Aer without noise → `hw.save_raw` → `decode.load_memory` → `detection_events`: **0 detection events, 0 observable flips**, with up to 49 random columns per circuit.

**Detectors against the team's `make_stim_circuit`** (`checks/crosscheck.json`, team-main `01951f8`, loaded read-only): 68 circuits (E1 and E4), detectors equal in order (record sets), `get_detector_coordinates()` equal, observable equal: **pass True**.

**E1 decoded against 'always 0'** (`checks/decode.json`; PyMatching on the noisy copy of the circuit that ran, `emerald.calibrated_noise(..., idle_ns=0)`). Pooled: P_L = 0.0991 [0.0942, 0.1042] decoded, against 0.2397 [0.2326, 0.2469] undecoded (13,672 shots): **better: True**.

| patch | basis | R | shots | P_L PyMatching [Wilson 95%] | P_L always 0 [Wilson 95%] |
|---|---|---|---|---|---|
| eA | X | 1 | 288 | 0.0174 [0.0074, 0.0400] | 0.0729 [0.0482, 0.1089] |
| eA | X | 2 | 288 | 0.0278 [0.0141, 0.0538] | 0.1319 [0.0977, 0.1759] |
| eA | X | 3 | 288 | 0.0625 [0.0399, 0.0966] | 0.1285 [0.0947, 0.1721] |
| eA | X | 4 | 288 | 0.0660 [0.0426, 0.1007] | 0.1875 [0.1466, 0.2366] |
| eA | X | 5 | 288 | 0.0625 [0.0399, 0.0966] | 0.2083 [0.1654, 0.2589] |
| eA | X | 6 | 288 | 0.0868 [0.0595, 0.1250] | 0.2951 [0.2454, 0.3502] |
| eA | X | 8 | 288 | 0.0799 [0.0538, 0.1170] | 0.3229 [0.2716, 0.3789] |
| eA | X | 10 | 288 | 0.1424 [0.1067, 0.1874] | 0.3507 [0.2979, 0.4074] |
| eA | X | 12 | 288 | 0.1493 [0.1128, 0.1951] | 0.3576 [0.3045, 0.4145] |
| eA | X | 15 | 288 | 0.2292 [0.1844, 0.2811] | 0.4201 [0.3646, 0.4778] |
| eA | X | 20 | 288 | 0.2326 [0.1876, 0.2848] | 0.4271 [0.3713, 0.4848] |
| eA | Z | 1 | 288 | 0.0139 [0.0054, 0.0352] | 0.0833 [0.0566, 0.1210] |
| eA | Z | 2 | 288 | 0.0243 [0.0118, 0.0493] | 0.1389 [0.1037, 0.1836] |
| eA | Z | 3 | 1288 | 0.0365 [0.0276, 0.0482] | 0.1413 [0.1233, 0.1614] |
| eA | Z | 4 | 288 | 0.0521 [0.0318, 0.0841] | 0.2049 [0.1623, 0.2552] |
| eA | Z | 5 | 288 | 0.0764 [0.0510, 0.1129] | 0.1736 [0.1342, 0.2216] |
| eA | Z | 6 | 288 | 0.0799 [0.0538, 0.1170] | 0.1875 [0.1466, 0.2366] |
| eA | Z | 8 | 288 | 0.1042 [0.0739, 0.1448] | 0.2292 [0.1844, 0.2811] |
| eA | Z | 10 | 288 | 0.1424 [0.1067, 0.1874] | 0.2917 [0.2422, 0.3466] |
| eA | Z | 12 | 288 | 0.1250 [0.0917, 0.1682] | 0.2639 [0.2163, 0.3177] |
| eA | Z | 15 | 288 | 0.1910 [0.1498, 0.2403] | 0.3542 [0.3012, 0.4110] |
| eA | Z | 20 | 288 | 0.2257 [0.1812, 0.2774] | 0.3750 [0.3211, 0.4322] |
| eB | X | 1 | 288 | 0.0035 [0.0006, 0.0194] | 0.0521 [0.0318, 0.0841] |
| eB | X | 2 | 288 | 0.0139 [0.0054, 0.0352] | 0.0972 [0.0681, 0.1369] |
| eB | X | 3 | 288 | 0.0382 [0.0215, 0.0671] | 0.1424 [0.1067, 0.1874] |
| eB | X | 4 | 288 | 0.0417 [0.0240, 0.0714] | 0.1667 [0.1281, 0.2140] |
| eB | X | 5 | 288 | 0.0521 [0.0318, 0.0841] | 0.2153 [0.1717, 0.2663] |
| eB | X | 6 | 288 | 0.0590 [0.0372, 0.0925] | 0.1910 [0.1498, 0.2403] |
| eB | X | 8 | 288 | 0.1215 [0.0887, 0.1643] | 0.2604 [0.2131, 0.3140] |
| eB | X | 10 | 288 | 0.1632 [0.1250, 0.2103] | 0.2882 [0.2389, 0.3430] |
| eB | X | 12 | 288 | 0.1667 [0.1281, 0.2140] | 0.3264 [0.2748, 0.3825] |
| eB | X | 15 | 288 | 0.1771 [0.1373, 0.2253] | 0.3368 [0.2847, 0.3932] |
| eB | X | 20 | 288 | 0.2153 [0.1717, 0.2663] | 0.4514 [0.3949, 0.5091] |
| eB | Z | 1 | 288 | 0.0139 [0.0054, 0.0352] | 0.0972 [0.0681, 0.1369] |
| eB | Z | 2 | 288 | 0.0347 [0.0190, 0.0627] | 0.1250 [0.0917, 0.1682] |
| eB | Z | 3 | 288 | 0.0451 [0.0266, 0.0757] | 0.1458 [0.1097, 0.1913] |
| eB | Z | 4 | 288 | 0.0556 [0.0345, 0.0883] | 0.2083 [0.1654, 0.2589] |
| eB | Z | 5 | 288 | 0.0799 [0.0538, 0.1170] | 0.2292 [0.1844, 0.2811] |
| eB | Z | 6 | 288 | 0.0833 [0.0566, 0.1210] | 0.2604 [0.2131, 0.3140] |
| eB | Z | 8 | 288 | 0.1215 [0.0887, 0.1643] | 0.2951 [0.2454, 0.3502] |
| eB | Z | 10 | 288 | 0.2153 [0.1717, 0.2663] | 0.4132 [0.3578, 0.4708] |
| eB | Z | 12 | 288 | 0.1771 [0.1373, 0.2253] | 0.3785 [0.3244, 0.4357] |
| eB | Z | 15 | 288 | 0.1806 [0.1404, 0.2291] | 0.4201 [0.3646, 0.4778] |
| eB | Z | 20 | 288 | 0.2917 [0.2422, 0.3466] | 0.4410 [0.3848, 0.4987] |

**LER records** (`sim/ler/<experiment>/<run id>/`, 2P = 1 − A(1 − 2ε)^R by maximum likelihood, `dqiqm.stats.fit_eps`, 95% bootstrap intervals; `ler_points.csv` has P_L per R with Wilson intervals). At these shot counts the intervals are wide: the point is the pipeline.

| exp | run | ε_L (A = 1) [95%] | χ²/dof | ε_L (free A) [95%] | A | shots | rounds with 0 failures |
|---|---|---|---|---|---|---|---|
| E1 | X_eA_b1 | 0.0149 [0.0121, 0.0184] | 0.81 | 0.0128 [0.0083, 0.0171] | 0.971 | 1,056 |  |
| E1 | X_eA_b2 | 0.0160 [0.0132, 0.0195] | 1.46 | 0.0160 [0.0128, 0.0194] | 1.000 | 1,056 | [1] |
| E1 | X_eA_b3 | 0.0167 [0.0137, 0.0206] | 0.79 | 0.0167 [0.0125, 0.0204] | 1.000 | 1,056 |  |
| E1 | X_eB_b1 | 0.0165 [0.0137, 0.0204] | 0.36 | 0.0165 [0.0123, 0.0203] | 1.000 | 1,056 |  |
| E1 | X_eB_b2 | 0.0127 [0.0103, 0.0157] | 0.80 | 0.0127 [0.0103, 0.0157] | 1.000 | 1,056 | [1] |
| E1 | X_eB_b3 | 0.0134 [0.0108, 0.0166] | 1.41 | 0.0134 [0.0108, 0.0166] | 1.000 | 1,056 | [2, 1] |
| E1 | Z_eA_b1 | 0.0143 [0.0118, 0.0176] | 0.69 | 0.0143 [0.0112, 0.0175] | 1.000 | 1,056 | [1] |
| E1 | Z_eA_b2 | 0.0135 [0.0110, 0.0167] | 0.37 | 0.0135 [0.0106, 0.0166] | 1.000 | 1,056 | [1] |
| E1 | Z_eA_b3 | 0.0151 [0.0124, 0.0187] | 1.80 | 0.0148 [0.0110, 0.0182] | 0.996 | 1,056 |  |
| E1 | Z_eB_b1 | 0.0190 [0.0158, 0.0232] | 0.70 | 0.0190 [0.0141, 0.0230] | 1.000 | 1,056 |  |
| E1 | Z_eB_b2 | 0.0198 [0.0163, 0.0242] | 1.68 | 0.0198 [0.0161, 0.0241] | 1.000 | 1,056 |  |
| E1 | Z_eB_b3 | 0.0158 [0.0130, 0.0193] | 1.26 | 0.0158 [0.0129, 0.0193] | 1.000 | 1,056 | [1] |
| E4 | X_eA_t1000ns_b2 | 0.0149 [0.0072, 0.0247] | 1.04 | 0.0062 [0.0000, 0.0217] | 0.925 | 192 |  |
| E4 | X_eA_t2000ns_b2 | 0.0110 [0.0047, 0.0190] | 1.44 | 0.0023 [0.0000, 0.0137] | 0.925 | 192 |  |
| E4 | X_eA_t5000ns_b2 | 0.0136 [0.0060, 0.0232] | 4.06 | 0.0136 [0.0000, 0.0232] | 1.000 | 192 | [3] |
| E4 | Z_eA_t1000ns_b2 | 0.0217 [0.0122, 0.0338] | 0.41 | 0.0217 [0.0086, 0.0338] | 1.000 | 192 |  |
| E4 | Z_eA_t2000ns_b2 | 0.0097 [0.0035, 0.0176] | 0.89 | 0.0097 [0.0019, 0.0175] | 1.000 | 192 | [2] |
| E4 | Z_eA_t5000ns_b2 | 0.0123 [0.0047, 0.0217] | 0.31 | 0.0069 [0.0000, 0.0189] | 0.953 | 192 |  |

**Derived features on the stand-in** (means over runs, `derived/*/*/summary.json`):

| experiment | detection rate | same-stab lag 1 | same-stab lag 2 | cross-stab lag 0 |
|---|---|---|---|---|
| E1 | 0.0873 | 0.0167 | 0.0094 | 0.0064 |
| E4 | 0.0821 | 0.0142 | 0.0068 | 0.0066 |

Dry run: 15 jobs, 157 dataset runs, 18 LER records, shots per circuit {"P0": 1000, "E1": 96, "E4": 48}; 7.6 min wall on 4 cores under the sim lock.
<!-- report:end -->

## The experiments

| | what | circuits | why |
|---|---|---|---|
| **E1** | `surface.memory_circuit(3, R, basis, no_reset=True)`, Z and X, R ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20}, on eA and eB | 44 | the main ML dataset and the challenge's LER; two patches separate qubit effects from code effects |
| **E4** | E1 with `delay(t)` on every qubit after each round's readout but the last, t ∈ {1, 2, 5} µs, R ∈ {2, 3, 5, 8}, Z and X, on eA | 24 | T1 (Z) and T2 (X) between rounds: the decoherence contribution, and the idle terms the priors need |

- Hardware target only, native ops on calibrated couplers only, zero SWAPs, barriers as Garnet's
  default (each round's readout one parallel layer). There is no barrier A/B on Emerald: Garnet's P1
  decides it, and if it says "off", rebuild with `barriers=False`.
- **E1's detectors are identical, in order, records and coordinates, to the team's
  `make_stim_circuit`**, and so are E4's (`sim/checks/crosscheck.json`; method of
  `orchestration/scratch/gen-crosscheck/crosscheck.py`, compared per detector in order). The source
  is the same `surface.memory_circuit` call as Garnet's.
- **Not built:** E3 (readout characterisation). `readout.spec` reads every qubit of the backend and
  names its patch `all20`, so on Emerald it needs a copy restricted to a patch with new names: not
  quick. E2 and E5 are skipped, as the brief says.

## d = 5: no

**No zero-SWAP d=5 placement exists on Emerald's 85 calibrated couplers.** On all 90 couplers there
is exactly one 49-qubit set (in 4 orientations): QB1–QB6, QB9–QB45, QB48–QB53. Every orientation
needs **3 of the CZ-less couplers: QB15–QB23, QB24–QB32 and QB49–QB50** (`sim/checks/d5.json`,
`emerald.d5_feasibility`). So d=5 on Emerald waits for a calibration set that calibrates those three
couplers. No d=5 circuits are built.

## Dynamical decoupling: an A/B to run on E1 (described, not built)

The challenge names DD explicitly, and IQM's client offers it as a compilation option:
`iqm.iqm_client.CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=DDStrategy(...))`
(iqm-client 35.0.3). The server then fills idle windows with DD pulse sequences; the Stim meaning
of the circuit is unchanged (DD is the identity), so the same `.stim`, detectors and decoder apply.

- **Where it can help here:** the data qubits idle through every ancilla readout (about 360 ns per
  round), and every qubit through E4's delays.
- **Design:** E1 Z and X at R ∈ {3, 8, 15} on eA, 2,000 shots each, as 4 jobs in ABBA order (DD on,
  off, off, on), back to back, so drift cancels to first order. Compilation options apply to a
  whole run request, so A and B cannot share a job. Then E4 at 5 µs, on and off, where DD should
  matter most. The X basis (T2) is where the effect should show; the priors for DD shots use the
  echo T2 (`device.t2_ns(..., "echo")`).
- **What `hw.py` needs:** nothing to *send* it: `hw.submit(..., circuit_compilation_options=...)`
  already passes `**run_options` to `IQMBackend.create_run_request`. What is missing is the
  **record**: `save_raw`'s `meta.json` does not store the compilation options, so DD-on and DD-off
  raw data could not be told apart afterwards. `submit` should write the options it was given
  (`dd_mode`, `dd_strategy`, and any other `CircuitCompilationOptions` field) into `meta.json`, and
  the ledger purpose should name them. The InfoRecord manifest would carry them via
  `write_dataset(..., extra=...)` (a new `MANIFEST` key, `compilation`, for whoever owns
  `inforecord.py`).

## Decoder priors

`emerald.calibrated_noise(cal, spec, idle_ns=...)` is `device.calibrated_noise(..., idle_ns=0)` plus
the readout window's idle counted **once**: data qubits get the T1/T2 twirl's p_xy + p_z as
`idle_during_measurement`, ancillas p_xy, and nothing goes before round 1. This is the fix the
garnet-design review asked for (MAJOR 1), in our own module, so the Emerald pilot bands do not
depend on the Garnet fix round; with `idle_ns=0` it is `device.calibrated_noise` exactly (tested).
The dry run decodes with `idle_ns=0` (the stand-in has no idle). For hardware data use the readout
window plus any delay.

## The dry run (SIMULATED)

`scripts/emerald_dryrun.py`, under the lock: every job of the plan below, at small shot counts, in
its planned circuit order, through the hardware path (Aer `Result` → `hw.save_raw` → `decode.load_memory`
→ `inforecord.write_dataset` / `write_derived` / `write_ler`), into `InfoRecord/emerald/sim/`.
The tables above (between the report markers) are written by `scripts/emerald_report.py` from
`sim/checks/` and `sim/ler/`.

The garnet-design checks not repeated here: the MPS-against-statevector check (a 54-qubit
statevector is out of reach; Garnet's showed identical records on its pilot circuit), the
injected-error feature check (the writer and `temporal` are device-independent; Garnet's
`checks/features.json` covers them) and the size table (the E1 circuits are Garnet's, so its
per-million-shot sizes apply).

## The hardware job plan (for Sal's approval)

Smaller jobs than Garnet's: Emerald is the challenge's device, so its queue will be busier. One job
per (patch, basis) of E1, 11 circuits × 2,000 shots (22,000 shots, about 10 s of QPU), against
Garnet's 22 × 5,000. Three batches; batch 3 whole is the **test** set (Garnet's `split`:
finetune / validation / test).

<!-- plan:begin -->
Generated by `scripts/emerald_plan.py` from calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6` (68 recorded circuits, every `native_sha256` checked).

**Estimated QPU time of a job = Σ over its circuits of shots × (circuit duration + reset_wait) + circuits × 0.1 s.** reset_wait is Emerald's own passive-reset wait, 400 µs (`gates.reset_wait` in the snapshot; Garnet's is 298 µs). Circuit durations come from Emerald's calibrated durations (prx 24 ns, CZ 40 ns, measure 360 ns median; each readout layer takes its slowest qubit's), with barriers as sync points (`circuits.schedule`): the pilot circuit is 1.85 µs. The 'pessimistic' column charges 1.92 ms per shot, Garnet's CLOPS bound (no Emerald CLOPS figure was checked).

| job | phase | exp | circuits | shots / circuit | total shots | circuit (µs), min–max (mean) | per shot (µs) = mean + reset_wait | shots × per shot (s) | + circuits × 0.1 s | **est. QPU (s)** | pessimistic (s) | what |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EP0 | pilot | E1 | 1 | 1,000 | 1,000 | 1.8–1.8 (1.8) | 401.8 | 0.4 | 0.1 | **0.5** | 2 | sanity checks before anything else |
| E-B1-1-E1 | batch 1 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eA, Z basis |
| E-B1-2-E1 | batch 1 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eA, X basis |
| E-B1-3-E1 | batch 1 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eB, Z basis |
| E-B1-4-E1 | batch 1 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eB, X basis |
| E-B2-1-E1 | batch 2 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eA, Z basis |
| E-B2-2-E1 | batch 2 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eA, X basis |
| E-B2-3-E1 | batch 2 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eB, Z basis |
| E-B2-4-E1 | batch 2 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eB, X basis |
| E-B2-5-E4 | batch 2 | E4 | 12 | 2,000 | 24,000 | 2.2–39.9 (12.1) | 412.1 | 9.9 | 1.2 | **11.1** | 48 | delay sweep, patch eA, Z basis |
| E-B2-6-E4 | batch 2 | E4 | 12 | 2,000 | 24,000 | 2.2–39.9 (12.1) | 412.1 | 9.9 | 1.2 | **11.1** | 48 | delay sweep, patch eA, X basis |
| E-B3-1-E1 | batch 3 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eA, Z basis |
| E-B3-2-E1 | batch 3 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eA, X basis |
| E-B3-3-E1 | batch 3 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eB, Z basis |
| E-B3-4-E1 | batch 3 | E1 | 11 | 2,000 | 22,000 | 0.6–12.3 (4.8) | 404.8 | 8.9 | 1.1 | **10.0** | 44 | memory, patch eB, X basis |
| **total** | | | 157 | | **313,000** | | | 127 | 16 | **143 s (2.4 min)** | **619** | |

| E1 split | shots |
|---|---|
| finetune (batches 1, 2 and the pilot, 80% of shot indices) | 141,600 |
| validation (the same jobs, 20% of shot indices; load as finetune or validation, never as test) | 35,400 |
| test (batch 3, whole) | 88,000 |
| **total E1** | **265,000** |

**The pilot (EP0: `E1_d3_Z_r03_eA`, 1 job, 1 circuit, 1,000 shots), before anything else.** Stop and report if any check fails.
1. `memory.npz` `c0` is (1000, 33), and the `calibration_set_id` is recorded.
2. At most 5 all-zero rows. The stand-in gives 1, the priors 0.59–1.19 per 1,000.
3. The X-type ancillas' round-1 bits, clbits `x_cols` = [0, 2, 5, 7] (QB14, QB30, QB28, QB44), pooled mean in [0.4, 0.6]. The stand-in gives 0.513, the priors 0.500.
4. The Z-type ancillas' round-1 bits, clbits `z_cols` = [1, 3, 4, 6] (QB21, QB39, QB19, QB37), pooled mean under 0.3. The stand-in gives 0.056, the priors 0.061 without idle, 0.061 with the 360 ns readout idle.
5. The detection fraction is in [0.040, 0.232]: half the stand-in's 0.080, up to twice the with-idle prior's 0.116 (no-idle prior 0.088).
6. The job timeline gives the real per-shot overhead. Re-estimate the plan from it before E-B1.

The full bands, with how each number is computed, are in `InfoRecord/emerald/circuits/job_plan.json: pilot`.
<!-- plan:end -->

**Batching:**
- E-B1, E-B2 and E-B3 at least 2 h apart (across a calibration change if one comes).
- Each batch opened and closed by a calibration snapshot (E6, no QPU time), its package written next to it.
- Rebuild the suite from the "before" snapshot of each batch (`emerald.build(<snapshot>)`): a new
  calibration set can move eA and eB, or calibrate the CZ-less couplers. Compare every circuit's
  `native_sha256` with `circuits/<exp>/placement.json` before submitting, and submit in
  `job_plan.json` order.
- One hardware worker at a time; Emerald and Garnet jobs never in parallel from our side.

**Analysis of the hardware data:** as Garnet's (`docs/garnet-design.md`), with
`data_root=InfoRecord/emerald/raw` and priors from `emerald.calibrated_noise(<"before" calibration>,
spec, idle_ns=<readout window + delay>)`; LER records per (basis, patch, batch).

## What `hw.py` needs for Emerald (not edited)

1. **Nothing for the path:** `hw.submit(..., device="emerald", calibration_set_id=<the snapshot's>,
   data_root=REPO / "InfoRecord" / "emerald" / "raw")` writes
   `InfoRecord/emerald/raw/<YYYYMMDD>-<task>/<job id>/`; `_connect` passes `quantum_computer=device`.
2. **No hardcoded "garnet" in `hw.py`** (`grep -in garnet src/dqiqm/hw.py` finds nothing). The
   Garnet defaults are elsewhere: `device.calibration_from_snapshot`'s `snap.get("device", "garnet")`
   (an Emerald snapshot names its device, so it is not hit) and `inforecord.write_architecture`'s
   `garnet.json` file name (wrapped here as `emerald.json`).
3. **`list_devices()` fails** with more than one device on the token (resonance-check finding 1):
   `IQMClient(SERVER_URL)` with no `quantum_computer` raises "Quantum computer not selected".
   Build the client with a listed device, or call `get_quantum_computers()` without resolving one.
4. **Record the compilation options** in `meta.json` (the DD A/B above).
5. Garnet's list still applies: refuse any op outside {prx, cz, measure, barrier, delay}, keep the
   timeline, a one-call E6 snapshot (here into `InfoRecord/emerald/calibration/emerald/`), and
   record each job's circuit order.

## What is left

- E3 on Emerald (a patch-restricted copy of `readout.spec`, with `eA` names), if wanted.
- The DD A/B, after the `hw.py` record change.
- d=5, if a calibration set calibrates QB15–QB23, QB24–QB32 and QB49–QB50.
- A hardware ingest script (Garnet's "what is left" applies).
