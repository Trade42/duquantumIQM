# Garnet: the experiment suite, the InfoRecord, and the hardware job plan

garnet-design task, 2026-10-24. Sal's requests: collect all the data Garnet gives, from circuits
designed to carry as much information as possible (especially temporal correlations), with every
column an ML decoder could use, recorded in `InfoRecord/`. The ML plan is hybrid (Sal, 19:47 UTC):
pre-train on synthetic data from Garnet's real calibration, then fine-tune and test on real
Garnet shots.

This document covers three things:
- the experiments, and why each one adds information;
- what the offline dry run showed;
- the hardware job plan, for Sal's approval.

**Devices.** `InfoRecord/` is Garnet's record. Emerald's will be `InfoRecord/emerald/`, with the identical layout inside it (another task builds it). Every run's `manifest.json` names its `device`; `placement.json` and the `ler/` records don't carry the key yet (their device is their root), which is on the backlog.

The record's format and column dictionary are in `InfoRecord/README.md`.

Everything here was built from the live calibration of 2026-10-24 19:13 UTC, set
`a18a947f-0408-4628-81ea-1715637e434f`. That snapshot is `data/calibration/garnet/…20261024T191313Z.json`, copied byte for byte into
`InfoRecord/calibration/garnet/`. A new calibration set can move the placements. Before each
batch, rebuild the suite from the snapshot taken just before it (E6).

## The device, as the suite sees it

- 20 qubits and 30 couplers, all calibrated, on this grid (`InfoRecord/architecture/garnet.json`):

  ```
     .  QB1  QB2    .    .
   QB3  QB4  QB5  QB6  QB7
   QB8  QB9 QB10 QB11 QB12
  QB13 QB14 QB15 QB16 QB17
     . QB18 QB19 QB20    .
  ```

- Median figures:
  - T1 37 µs; T2 Ramsey 9.1 µs; T2 echo 19.9 µs;
  - readout P(1|0) 1.2% and P(0|1) 3.3% (worst P(0|1): QB2 at 9.1%);
  - CZ IRB error 0.50% (worst: QB16–QB20 at 1.2%);
  - durations: prx 16 ns, CZ 40 ns, measure 480 ns (556 ns on QB2, QB5, QB7 and QB16);
  - `reset_wait`: 298 µs.
- **The d=3 code fits with zero SWAPs on exactly one set of 17 qubits.** That set is all but QB3, QB17 and QB18.
  - There are 4 placements, the code graph's automorphisms.
  - Two make {QB5, QB7, QB13, QB15} the X-type ancillas; two make {QB1, QB9, QB11, QB20} X-type.
- **So there is no second patch.** The record's two "patches" are:
  - **pA:** the placement with the lowest P_L under calibrated Stim noise. It is ranked over 50,000 seeded shots per basis (`memory.rank`). pA is orientation 4, with X-type ancillas {QB5, QB7, QB13, QB15}: mean P_L(R=3) 0.0826 against 0.0852 and 0.0853 for the next two, about 2σ apart (`circuits/job_plan.json: choices`).
  - **pB:** the best placement with the other X-type set (orientation 7, {QB1, QB9, QB11, QB20}). Every ancilla changes its check type, which separates qubit effects from check-type effects. That is the only spatial variation Garnet allows.
  - Both use the worst CZ (QB16–QB20), and both read the qubit with the worst mean readout error (QB11) as an ancilla.
- **There is no Hamiltonian cycle,** so the longest loop is shorter than the longest line. The longest line has 19 qubits. The calibration score picks one of 566 lines (`repetition.best_path`).
- There are 4 3×3 blocks for the stability patch (`stability.best_block`).

## The stand-in, and what it cannot show

The stand-in is `IQMFakeGarnet().copy_with_error_profile(...)` with the live noise
(`dqiqm.experiments.device`):
- per-qubit T1, Ramsey T2 (capped at 2·T1), 1Q error (RB → depolarizing parameter 2r), CZ error (IRB → 4r/3);
- the asymmetric readout, taken from the default `measure`'s `error_0_to_1` and `error_1_to_0` (not iqm-qubit-selector's heralded figure).

The T2 is Ramsey because no circuit has dynamical decoupling.

Aer applies three kinds of noise: a depolarizing channel and thermal relaxation over each gate's duration, and a classical assignment error on each readout. **It puts no noise on `delay` or on idle qubits.** So offline:
- E4's idles are checked for structure only;
- E3's repeated readouts show record flips but no state flips. The analysis is checked on Stim runs with injected state flips instead (below).

The decoder priors (`device.calibrated_noise`) map the same calibration onto each circuit's Stim qubits. With `idle_ns=0` they match the stand-in. For hardware data, `idle_ns` is the readout window (plus any delay), so T1 and T2 between gates enter the priors. The idle is counted once per mid-circuit readout window, after the ancillas' readout: the whole T1/T2 twirl on a data qubit, its X/Y part on an ancilla. There is none after the last round, whose ancillas are read in the data's layer (garnet-timing; before it, `surface` put one there too, R windows for R − 1, which for E4 carried the whole delay). `device.hardware_noise(cal, spec)` is the hardware prior in one call: the compiled readout window, a symmetric idle (the biased twirl is a switch, in each data qubit's hardware frame), and `measure_fidelity`'s errors on the final readout (see "Decoder priors for hardware", below).

The laptop pre-trains on synthetic shots built from the calibration package with these idle terms, not on `sim/` (no idle noise, and about twice the benchmarked gate noise), and picks the placement by each shot's `patch` (`InfoRecord/README.md`, "Pre-training the ML decoder").

## The experiments

All circuits are no-reset, built from Stim sources and placed with `dqiqm.layout`. They are `target="hardware"` only, and the plan and `InfoRecord/circuits/` assert it. Every circuit is checked:
- zero SWAPs, one `cz` per CX, each on a coupler;
- measurement j in clbit j on its placed qubit;
- each qubit's sequence equal to the logical circuit's (`layout.check_hardware`);
- exact Stim equivalence (`layout.check_equivalent`);
- IQM's own `validate_circuit`.

Circuits with delays are checked against their delay-free twin (`circuits.check_delays`).

The X of a |1⟩ preparation is `r(π, 0)`, prepended to the converted X-free source, because the converter has no X. The Stim source *with* the X is the reference for every check.

### E1: d=3 surface-code memory, the main ML dataset

- **Circuits:**
  - `surface.memory_circuit(3, R, basis, no_reset=True)`, Z and X;
  - R ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20};
  - on pA and pB: 44 circuits;
  - plus the barrier-free twin of Z R=3 pA, built for the barrier A/B (P1), which is dropped: its files stay, and no job runs it.
- **Detectors** are those of the team's `make_stim_circuit`, in order and coordinates. The laptop's pipeline reads the shots as they are.
- **Why:** this is the experiment the ML decoder is for, at every R the ε_L fit needs. Three time-separated batches give the drift. pB gives the check-type swap.
- **What it measures:**
  - P_L(R) and ε_L, both fit forms, per (basis, patch, batch) in `ler/`;
  - p_ij by lag: k→k+1 for gate and state errors, k→k+2 for no-reset record flips;
  - rates by round and by shot block;
  - streaks (leakage);
  - syndrome-weight distributions.

### E2: repetition codes on the 19-qubit line

- **Circuits:**
  - bit-flip in |0…0⟩ and |1…1⟩, and phase-flip in |+…+⟩;
  - distance 10, 9 ancillas;
  - R ∈ {1, 2, 3, 5, 8, 12, 20, 30, 50}: 27 circuits;
  - detectors in `surface.py`'s no-reset style.
- **Why:**
  - Long runs on one line give the temporal correlations and leakage streaks, with up to 50 readouts per ancilla.
  - |1…1⟩ against |0…0⟩ is the T1 asymmetry: decay pushes data toward |0⟩.
  - Phase-flip against bit-flip separates dephasing from relaxation.
  - Each ancilla's record-flip signature (k→k+2) is cleanly separated from its neighbours.

### E3: readout characterisation

All 20 qubits are read in parallel in each readout layer, as in the codes. There are 22 circuits:
- **Assignment** (one readout each): all-0, all-1, and 16 seeded random states. They give per-qubit P(1|0) and P(0|1), the correlation matrix of misassignments (correlated readout errors), and the neighbour-state dependence.
- **Repeated readout without reset** (15 in a row): all-0, all-1 and the two checkerboards.
  - D_1 = m_1 ⊕ prep and D_k = m_k ⊕ m_(k−1).
  - A record flip fires the adjacent pair (D_k, D_(k+1)); a state flip, such as T1 during the readout window, fires D_k alone.
  - `temporal.readout_flips` estimates the two separately per qubit. That is the surface-gen finding the no-reset decoder needs: in the code, record flips fire D_k and D_(k+2).
- **Why:** these are the per-qubit readout priors (record_flip_probability, idle_during_measurement) for the hardware DEM, measured in the same batch as the codes.

### E4: delay sweep on E1

- **Circuits:**
  - E1 with `delay(t)` on every qubit after each round's readout but the last;
  - t ∈ {1, 2, 5} µs (t = 0 is E1);
  - Z and X, R ∈ {2, 3, 5, 8}, on pA: 24 circuits.
- **Why:** the slope of ε_L against t is the decoherence contribution per µs of idle, Z (T1) against X (T2). It calibrates `idle_during_measurement` and the data idle on hardware, and it tests whether the decoder's idle priors scale right.

### E5: the stability experiment (Gidney, Quantum 6, 786, 2022)

- **Layout:** the smallest surface-code stability patch, on the best 3×3 block. Four data qubits sit on the edge midpoints, a weight-4 check in the centre, and weight-2 checks of the other type on the corners. Every data qubit is in exactly two corner checks, so the corner checks multiply to the identity.
  - The observable is that product in round 1. Space-like errors cannot flip it, only time-like ones can.
  - Stim's graphlike distance is R under circuit noise without record flips, for R = 2 to 12, both duals.
  - **With record flips it is ⌈R/2⌉.** Without reset a misread bit m_k fires D_k and D_(k+2), so one record flip spans two rounds of the time direction. Every real readout has record flips, so on hardware the no-reset stability experiment has half the time-like distance of a with-reset one (with reset, a misread fires D_k and D_(k+1)). E5 measures that; E1's distance stays 3.
- **Circuits:** both duals (corner checks Z or X), R ∈ {2, 3, 4, 6, 8, 12}: 12 circuits.
- **Schedule:** the 4-layer CX schedule is found by search (`stability.schedule`). The centre must meet each corner's two shared data qubits in the same order. The noiseless Stim check proves it.
- **Why:** it isolates exactly the time-like errors (misreads, ancilla flips) a memory experiment mixes with space-like ones. P_L falls as R grows. It is a direct, decoder-level measurement of the timelike edge weights, on the same readout chain as E1.

### E6: calibration snapshots

- **Before and after every batch:** `scripts/resonance_check.py`'s snapshot (read-only `dqiqm.hw` helpers, no QPU time).
- **Where:** `InfoRecord/calibration/garnet/<UTC>-<calset>.json`, each with its pre-training package (`scripts/garnet_calibration_package.py`, offline).
- **Record links:** every dataset's manifest names its `calibration.before` and `.after`, and `qubits.csv` holds the "before" figures. The hardware task runs these.

### Optional, not in the plan (each needs a mentor's OK)

- **E7, mid-circuit reset:** measure + `cc_prx` (both devices list `cc_prx` and `reset_wait`). It is the with-reset variant on hardware, for the reset/no-reset trade-off surface-gen measured in simulation. The challenge asks for no reset, so it stays out of the main plan.
- **E8, leakage heralding:** `measure_012` is served on QB1, QB3, QB9 and QB14. QB1 and QB9 are E1 ancillas, so a variant that reads them with `measure_012` would herald |2⟩ directly and label the streak features.

## The dry run (`scripts/garnet_dryrun.py`, `InfoRecord/sim/`)

<!-- results:begin -->

All numbers are **SIMULATED**: IQMFakeGarnet+profile(a18a947f), calibration set `a18a947f-0408-4628-81ea-1715637e434f`, Aer MPS, seeded. Shots per circuit: {"P0": 1000, "P1": 256, "E1": 128, "E2": 64, "E3": 256, "E4": 128, "E5": 256}. 14 jobs, 264 dataset runs, 23 LER records; 8.7 min wall on 4 cores under the sim lock.

**Structure** (`checks/structure.json`; distance = `len(shortest_graphlike_error())` of the noisy copy):

| experiment | circuits | detectors | measurements | observables | SWAPs | DEM errors | graphlike distance |
|---|---|---|---|---|---|---|---|
| E1 | 45 | 8–160 | 17–169 | [1] | 0 | 29–2849 | [3] |
| E2 | 27 | 18–459 | 19–460 | [1] | 0 | 39–1901 | [10] |
| E3 | 22 | 20–300 | 20–300 | [0] | 0 | n/a | n/a |
| E4 | 24 | 16–64 | 25–73 | [1] | 0 | 149–1049 | [3] |
| E5 | 12 | 7–57 | 14–64 | [1] | 0 | 72–515 | [1, 2, 3, 4, 6] |

E5's graphlike distance under the calibrated noise, (R, distance), Z dual (the X dual is the same): [(2, 1), (3, 2), (4, 2), (6, 3), (8, 4), (12, 6)], that is ⌈R/2⌉. With no record flips in the model (uniform circuit noise) it is R (tests/test_garnet_experiments.py::test_distances). Without reset a misread bit m_k fires D_k and D_(k+2): one record flip spans two rounds of the time direction, so the stability experiment's time-like distance halves; E1's distance stays 3. Every detector count is the designed one (tests/test_garnet_experiments.py::test_counts).

**Noiseless, through the hardware path** (`checks/noiseless.json`): all 130 hardware circuits, 32 shots each, Aer without noise → `hw.save_raw` → `decode.load_memory` → `detection_events`: **0 detection events, 0 observable flips**. The records are not trivially quiet: up to 49 random columns per circuit.

**MPS against statevector** (`checks/mps_vs_statevector.json`, E1_d3_Z_r02_pA, 200 noisy shots, seed 2026): identical records: **True**; detection fraction 0.0884 (MPS, 1.49 s) and 0.0884 (statevector, 12.81 s).

**E1 decoded against 'always 0'** (`checks/decode.json`): PyMatching on the noisy copy of the circuit that ran (`calibrated_noise(..., idle_ns=0)`, the stand-in's own noise). Pooled over every E1 run: P_L = 0.1374 [0.1325, 0.1424] decoded, against 0.2722 [0.2658, 0.2786] undecoded (18,408 shots): **better: True**.

| basis | R | shots | P_L PyMatching [Wilson 95%] | P_L always 0 [Wilson 95%] |
|---|---|---|---|---|
| X | 1 | 768 | 0.0299 [0.0200, 0.0445] | 0.1341 [0.1118, 0.1600] |
| X | 2 | 768 | 0.0560 [0.0418, 0.0746] | 0.1380 [0.1154, 0.1642] |
| X | 3 | 768 | 0.0729 [0.0566, 0.0935] | 0.1693 [0.1444, 0.1974] |
| X | 4 | 768 | 0.1042 [0.0845, 0.1278] | 0.2201 [0.1922, 0.2507] |
| X | 5 | 768 | 0.1172 [0.0963, 0.1419] | 0.2188 [0.1910, 0.2493] |
| X | 6 | 768 | 0.1159 [0.0951, 0.1405] | 0.2344 [0.2058, 0.2656] |
| X | 8 | 768 | 0.1589 [0.1347, 0.1864] | 0.2891 [0.2581, 0.3221] |
| X | 10 | 768 | 0.2044 [0.1774, 0.2344] | 0.3034 [0.2719, 0.3368] |
| X | 12 | 768 | 0.2135 [0.1860, 0.2439] | 0.3112 [0.2795, 0.3448] |
| X | 15 | 768 | 0.2695 [0.2394, 0.3020] | 0.3776 [0.3440, 0.4124] |
| X | 20 | 768 | 0.2747 [0.2443, 0.3074] | 0.4115 [0.3772, 0.4466] |
| Z | 1 | 768 | 0.0208 [0.0129, 0.0336] | 0.1328 [0.1106, 0.1587] |
| Z | 2 | 768 | 0.0586 [0.0441, 0.0775] | 0.1901 [0.1639, 0.2194] |
| Z | 3 | 2280 | 0.0671 [0.0575, 0.0781] | 0.2237 [0.2071, 0.2412] |
| Z | 4 | 768 | 0.0820 [0.0646, 0.1036] | 0.2630 [0.2331, 0.2953] |
| Z | 5 | 768 | 0.1081 [0.0880, 0.1320] | 0.2565 [0.2269, 0.2886] |
| Z | 6 | 768 | 0.0990 [0.0798, 0.1221] | 0.2747 [0.2443, 0.3074] |
| Z | 8 | 768 | 0.1589 [0.1347, 0.1864] | 0.3438 [0.3110, 0.3780] |
| Z | 10 | 768 | 0.2005 [0.1737, 0.2303] | 0.3620 [0.3288, 0.3966] |
| Z | 12 | 768 | 0.2109 [0.1836, 0.2412] | 0.3919 [0.3580, 0.4269] |
| Z | 15 | 768 | 0.2695 [0.2394, 0.3020] | 0.4023 [0.3682, 0.4374] |
| Z | 20 | 768 | 0.2682 [0.2381, 0.3007] | 0.4349 [0.4002, 0.4702] |

**LER records** (`ler/<experiment>/<run id>/`): the fit 2P = 1 − A(1 − 2ε)^R by maximum likelihood (`dqiqm.stats.fit_eps`), 95% bootstrap intervals (1,000 resamples); each record also has `ler_points.csv` (P_L per R with Wilson intervals) and `ler.png`. At these shot counts the intervals are wide: the point is the pipeline, not the numbers.

| exp | run | ε_L (A = 1) [95%] | χ²/dof | ε_L (free A) [95%] | A | χ²/dof | shots |
|---|---|---|---|---|---|---|---|
| E1 | X_pA_b1 | 0.0215 [0.0183, 0.0256] | 0.77 | 0.0179 [0.0131, 0.0228] | 0.952 | 0.36 | 1,408 |
| E1 | X_pA_b2 | 0.0247 [0.0209, 0.0290] | 1.41 | 0.0198 [0.0144, 0.0259] | 0.938 | 1.07 | 1,408 |
| E1 | X_pA_b3 | 0.0235 [0.0199, 0.0273] | 0.87 | 0.0221 [0.0169, 0.0264] | 0.982 | 0.92 | 1,408 |
| E1 | X_pB_b1 | 0.0248 [0.0214, 0.0288] | 0.65 | 0.0248 [0.0211, 0.0288] | 1.000 | 0.72 | 1,408 |
| E1 | X_pB_b2 | 0.0250 [0.0213, 0.0293] | 1.37 | 0.0250 [0.0203, 0.0289] | 1.000 | 1.52 | 1,408 |
| E1 | X_pB_b3 | 0.0249 [0.0214, 0.0294] | 1.18 | 0.0199 [0.0142, 0.0258] | 0.937 | 0.69 | 1,408 |
| E1 | Z_pA_b1 | 0.0236 [0.0200, 0.0277] | 1.62 | 0.0236 [0.0192, 0.0275] | 1.000 | 1.80 | 1,408 |
| E1 | Z_pA_b2 | 0.0239 [0.0203, 0.0279] | 1.09 | 0.0238 [0.0182, 0.0274] | 0.998 | 1.20 | 1,408 |
| E1 | Z_pA_b3 | 0.0201 [0.0171, 0.0237] | 1.10 | 0.0193 [0.0149, 0.0231] | 0.989 | 1.16 | 1,408 |
| E1 | Z_pB_b1 | 0.0219 [0.0186, 0.0258] | 1.60 | 0.0200 [0.0151, 0.0248] | 0.974 | 1.60 | 1,408 |
| E1 | Z_pB_b2 | 0.0232 [0.0199, 0.0273] | 0.64 | 0.0227 [0.0176, 0.0265] | 0.993 | 0.70 | 1,408 |
| E1 | Z_pB_b3 | 0.0232 [0.0199, 0.0271] | 0.69 | 0.0224 [0.0175, 0.0264] | 0.988 | 0.74 | 1,408 |
| E2 | bit_0_b1 | 0.0000 [0.0000, 0.0000] | 0.00 | 0.0000 [0.0000, 0.0000] | 1.000 | 0.00 | 576 |
| E2 | bit_1_b1 | 0.0000 [0.0000, 0.0000] | 0.00 | 0.0000 [0.0000, 0.0000] | 1.000 | 0.00 | 576 |
| E2 | phase_p_b1 | 0.0000 [0.0000, 0.0000] | 0.00 | 0.0000 [0.0000, 0.0000] | 1.000 | 0.00 | 576 |
| E4 | X_pA_t1000ns_b2 | 0.0189 [0.0134, 0.0257] | 2.35 | 0.0189 [0.0134, 0.0257] | 1.000 | 3.52 | 512 |
| E4 | X_pA_t2000ns_b2 | 0.0270 [0.0200, 0.0353] | 1.22 | 0.0214 [0.0061, 0.0331] | 0.951 | 1.51 | 512 |
| E4 | X_pA_t5000ns_b2 | 0.0293 [0.0221, 0.0378] | 1.59 | 0.0275 [0.0072, 0.0372] | 0.984 | 2.34 | 512 |
| E4 | Z_pA_t1000ns_b2 | 0.0139 [0.0091, 0.0195] | 2.29 | 0.0139 [0.0091, 0.0195] | 1.000 | 3.43 | 512 |
| E4 | Z_pA_t2000ns_b2 | 0.0195 [0.0139, 0.0265] | 2.17 | 0.0101 [0.0000, 0.0210] | 0.920 | 1.21 | 512 |
| E4 | Z_pA_t5000ns_b2 | 0.0232 [0.0169, 0.0305] | 1.92 | 0.0232 [0.0109, 0.0305] | 1.000 | 2.88 | 512 |

**Derived features against known injected errors** (`checks/features.json`): Stim runs of E1 Z R=10 and E3's repeated readout with errors of known rate, written and analysed by the record's own writer:

| run | same-stabilizer p_ij, lag 1 | lag 2 | lag 3 | record flips | state flips |
|---|---|---|---|---|---|
| E1 Z R=10, record_flips_0.03 | 0.0017 | 0.0299 | -0.0000 | | |
| E1 Z R=10, ancilla_state_flips_0.03 | 0.0285 | 0.0000 | 0.0001 | | |
| E1 Z R=10, none | 0.0000 | 0.0000 | 0.0000 | | |
| E3 repeat |1>, injected record 0.02, state 0.0 | | | | 0.0201 | -0.0008 |
| E3 repeat |1>, injected record 0.0, state 0.01 | | | | 0.0000 | 0.0101 |
| E3 repeat |1>, injected record 0.02, state 0.01 | | | | 0.0199 | 0.0085 |

Record flips show at k→k+2 (lag 2) only, state flips at k→k+1 only, and the E3 estimator recovers both rates; with both injected (0.02 and 0.01) it gives 0.0199 and 0.0085 (the estimator is first order in the rates).

On the stand-in itself (means over runs, `derived/*/*/summary.json`):

| experiment | detection rate | same-stab lag 1 | same-stab lag 2 | cross-stab lag 0 |
|---|---|---|---|---|
| E1 | 0.1086 | 0.0188 | 0.0185 | 0.0074 |
| E2 | 0.0609 | 0.0139 | 0.0170 | 0.0013 |
| E4 | 0.1077 | 0.0187 | 0.0205 | 0.0076 |
| E5 | 0.1011 | 0.0162 | 0.0151 | 0.0117 |

E4 on the stand-in, against E1 at the same R in the same batch (detection rate): the stand-in puts no noise on `delay`, so the idles change nothing beyond shot noise. On hardware this is where T1 and T2 show.

| basis | R | E1 | E4 1 µs | E4 2 µs | E4 5 µs |
|---|---|---|---|---|---|
| Z | 2 | 0.1108 | 0.0898 | 0.0874 | 0.0957 |
| Z | 3 | 0.1146 | 0.1068 | 0.1045 | 0.1123 |
| Z | 5 | 0.0924 | 0.1068 | 0.1088 | 0.1172 |
| Z | 8 | 0.1105 | 0.1117 | 0.1139 | 0.1144 |
| X | 2 | 0.0991 | 0.0854 | 0.1035 | 0.1021 |
| X | 3 | 0.1009 | 0.0964 | 0.1178 | 0.1146 |
| X | 5 | 0.1041 | 0.1199 | 0.1076 | 0.1217 |
| X | 8 | 0.1172 | 0.1194 | 0.1152 | 0.1121 |

E3's repeated readouts on the stand-in: record flips 0.0256 per readout (the calibration's mean assignment error over the 20 qubits is 0.0257), state flips -0.0020: the stand-in's readout error is a pure record flip, as Aer models it, and E1's same-stabilizer lag-2 p_ij is its no-reset signature.

**Sizes** (`checks/sizes.json`), per million shots:

| circuit | measurements | detectors | uint8 .npy | bit-packed .npy | shots.csv | shots.csv.gz |
|---|---|---|---|---|---|---|
| E1_d3_Z_r20_pA | 169 | 160 | 330 MB | 43 MB | 142 MB | 21 MB |
| E1_d3_Z_r03_pA | 33 | 24 | 58 MB | 9 MB | 99 MB | 7 MB |

Parquet: not measured; `pyarrow` is not in `uv.lock`, and the brief forbids adding it. Arrays above 2^24 entries (for E1 R=20, about 50,000 shots) are bit-packed automatically.

<!-- results:end -->

## The hardware job plan

**The rules:**
- **Grouping:** one job per circuit group, every circuit of a job at the same shot count (IQM's batch rule).
- **Circuit order:** IQM runs a batch's circuits one after another, so drift inside a job would line up with the circuit order. Each job's order is a seeded random permutation, recorded per shot as `position_in_job`.

**The phases:**
- **P0, the pilot:** E1 Z R=3 pA, 1,000 shots, alone.
- **No P1.** The barrier A/B (the same circuit with barriers on and off, 2 × 4,000 shots) is dropped: pulla-check compiled the pilot both ways with IQM's own compiler, and without barriers the readouts stop multiplexing (19 triggers instead of 9, queued on three probe lines), so the circuit takes **7,748 ns against 3,900 ns** (`docs/pulla-check.md`). Barriers stay on, which is what the layout review asked to decide. The barrier-off circuit `E1_d3_Z_r03_pA_nobar` stays in `circuits/E1/` as built; no job runs it.
- **Batches B1, B2, B3:**
  - at least 2 h apart (across a calibration change if one comes), each opened and closed by an E6 snapshot;
  - every batch: E3 and E1 on pA and pB;
  - B1 adds E2, B2 adds E4, B3 adds E5.
- **The split, fixed now:**
  - **B3, whole, is the test set**, never fine-tuned on;
  - in every other job, a seeded 20% of the shot indices is validation and the rest finetune (`suite.split`);
  - it is recorded per shot (`shots.csv: split`) and per run (`manifest.json: split`);
  - `validation` is the fine-tuning side's held-out 20%: a loader with only finetune and test loads it as finetune, never as test;
  - no job runs the barrier-off circuit, so every hardware shot has `barriers` true; a loader should still filter on it.

**Before any job is submitted,** the hw worker rebuilds the circuits with `suite.build(device.live_calibration(<snapshot>))` and checks each one's `circuits.native_sha256` against `placement.json`; a mismatch stops it. (`stim_sha256` is of the Stim text without the file's trailing newline; `qasm_sha256` is of the file.)

**The pilot's checks**, before anything else is sent:
1. `memory.npz` has shape (1000, 33) and the job's `calibration_set_id` is recorded.
2. At most 5 all-zero rows in 1,000: the stand-in gives 0, the calibrated priors about 0.02–0.06%, a dead readout 100%.
3. The X-type ancillas' round-1 bits (clbits 0, 2, 5, 7: `band.x_cols`) are random: the pooled fraction of 1s is within [0.4, 0.6]. The Z-type ancillas' round-1 bits (clbits 1, 3, 4, 6: `band.z_cols`) are mostly 0: the pooled fraction below 0.3, where the stand-in gives 0.056 and the calibrated priors 0.079 (the same with or without the idle: no readout window comes before round 1's readout). Together these prove that the entangling gates ran and that the bit order is right.
4. The detection fraction is within the band below: from half the stand-in's (0.104), to twice the calibrated-with-idle prediction's (0.172): [0.052, 0.345]. A fraction near 0.5 means a bit-order or circuit error. The pilot passed this band with 0.221 (`InfoRecord/derived/E1/p0_E1_d3_Z_r03_pA/pilot_checks.json`). The band stays as it was built (the pre-fix priors, rebuilt exactly with `idle_after_last_round=True`), since `scripts/garnet_pilot.py` re-reads it; the plan's `pilot` block adds `hardware_prior`, what garnet-timing's default priors expect (detection fraction 0.196).
5. The job timeline (`execution_start` → `execution_end`) over 1,000 shots was to give the true per-shot overhead. It did not: the span is 50.7 µs per shot, less than the 298 µs reset wait, while the raw shots show the qubits did wait about 200 µs or more (`orchestration/reports/pilots-review.md`). So the span does not bracket the shot loop, and the plan stays on reset_wait per shot.

If any check fails, stop and report. Do not spend more shots chasing it.

**The hardware analysis**, per dataset:
1. `decode.load_memory` → `detection_events` with the circuit that ran (`circuits/<exp>/<name>.stim`).
2. PyMatching on its noisy copy, using `device.hardware_noise(<the batch's "before" snapshot>, spec)`: the compiled readout window plus the spec's delay as the idle, a symmetric `DEPOLARIZE1` idle, Ramsey T2, and `measure_fidelity`'s errors on the final layer (E1, E4; the E2 and E5 builders keep `measure`'s and an idle after the last round, which the prior's info records). `idle_channel="pauli"` (the biased twirl, in each data qubit's hardware frame) and `t2="echo"` are the switches. Which prior decodes the batch is chosen per basis **by decoded P_L on the finetune split's validation shots, never on test**, not by the detector log-likelihood alone; see "Decoder priors for hardware".
3. `ler/` per (basis, patch, batch): P_L at every R with shots, failures and Wilson 95%, and the fit 2P = 1 − A(1−2ε)^R in both forms (A = 1, the challenge's formula, and A free) with both intervals, χ² and the plot (`inforecord.write_ler`).
4. `derived/` for every run.

## Decoder priors for hardware (garnet-timing)

**What the priors are now** (`device.hardware_noise`):
- **The readout window is the compiled one.** The idle per mid-circuit readout layer is the slowest compiled `measure_window_ns` among the layer's measured qubits, from pulla-check's `data/calibration/garnet/pulse/compile.<calset>.json` → `readout_windows`: **1,096 ns** for the patch's ancillas (QB5 and QB7), against the 556 ns `measure` probe the priors used before. E4 adds its delay to the same windows. Without compile data for the calibration set, the old rule (the slowest `measure` duration) is used and logged.
- **The terminal layer** compiles to `measure_fidelity`, whose window (`measure_fidelity_window_ns`) is for durations only: it reads every qubit, so nothing idles through it. What it changes is the final readout's record flips: the last round's ancillas and the data take `measure_fidelity`'s **unheralded** errors (`metrics.ssro.measure_fidelity.constant.<q>.error_0_to_1` / `_1_to_0`, the mean of the two, as for `measure`). They are slightly better than `measure`'s on 16 of 17 qubits; QB20, an ancilla, is worse (mean 0.055 against 0.018).
- **The idle on a data qubit is a symmetric `DEPOLARIZE1`** of the twirl's total by default, which is the same in every frame. An ancilla gets the X/Y part as an `X_ERROR`, as before. T2 is Ramsey by default (no dynamical decoupling); `t2="echo"` is the switch.
- **The biased twirl is a switch,** `idle_channel="pauli"`: `PAULI_CHANNEL_1` with the physical (p_xy/2, p_xy/2, p_z) mapped into Stim's frame per data qubit (garnet-design review, MINOR 5; garnet-timing review, BLOCKER 1). The compiler moves each round's leading prx on some data qubits to before the previous readout, so through the window those qubits sit in a rotated frame, where a physical Z (dephasing) acts as a Stim X. `device.readout_frames` derives each data qubit's frame from the instructions the server receives: it injects a physical X, Y or Z after each mid-circuit readout layer of the native circuit and matches the detector-and-observable signature against Stim's at the model's idle. On every E1 and E4 circuit (both patches, both bases) it gives x ↔ z on Stim data 1, 3, 5, 12 and 17 and the identity on 8, 10, 15 and 19; E4's delay lies in the same window and frame. It refuses a readout layer not fenced by barriers (the barrier-off circuit), since its frame is then the compiler's schedule. `tests/test_garnet_timing.py` checks the map on the committed QASM, independently of the translation.
- **No idle after the last round** (garnet-design review 2, new MINOR 1).

**Validated on the pilot** (`scripts/garnet_reprior.py`, offline; the table is regenerated by `--doc`):

<!-- reprior:begin -->
Generated by `scripts/garnet_reprior.py`: job `01a17000-6394-755f-a58b-b4f9c01893b0` (E1_d3_Z_r03_pA, 1,000 shots, R = 3, calibration set `a18a947f-0408-4628-81ea-1715637e434f`), re-decoded from its committed `memory.npz` with PyMatching. Priors from the pilot's before snapshot (`data/calibration/garnet/a18a947f-0408-4628-81ea-1715637e434f.20261024T221752Z.json`). Model columns: the prior's own noisy circuit, 100,000 Stim shots, seed 2026. P_L intervals are Wilson 95%. `a_old` is the pilot's own decoder, byte for byte (sha256 `bec21d8ca17b…`). `b_ramsey` and `c_echo` are the default hardware prior (symmetric idle); `b_pauli` and `c_pauli` the biased switch, in each data qubit's hardware frame; `x_pauli_stim_frame` is garnet-timing round 1's `b_ramsey`, biased in Stim's frame. The other `x_…` rows change one thing from `a_old` at a time. 'Detector log-lik' is the binomial log-likelihood of the hardware's per-detector detection counts under the model's per-detector rates, the detectors taken as independent (higher fits better). The McNemar column compares failures shot by shot with `a_old` on the same 1,000 shots.

Detection fraction per detector round t (t = 0: round 1, Z-type only; t = 3: the final data detectors, Z-type only):

| priors | idle (ns) | idle channel | T2 | final readout | idle after last round | DEM errors | d | hardware P_L (1,000 shots, R = 3) | model P_L (100,000 sim shots) | t = 0 | t = 1 | t = 2 | t = 3 | Σ\|model − hw\| | detector log-lik | vs a_old: fails only here / only in a_old (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hardware** | | | | | | | | | | **0.079** | **0.267** | **0.284** | **0.146** | | | |
| `a_old` | 556 | depolarize | ramsey | measure | yes | 294 | 3 | 209/1000 = **0.209** [0.185, 0.235] | 0.1989 [0.1964, 0.2013] | 0.079 | 0.176 | 0.199 | 0.206 | 0.237 | -12297 |  |
| `b_ramsey` | 1096 | depolarize | ramsey | measure_fidelity | no | 294 | 3 | 211/1000 = **0.211** [0.187, 0.237] | 0.2228 [0.2202, 0.2254] | 0.079 | 0.226 | 0.249 | 0.149 | 0.080 | -11926 | 32 / 30 (p = 0.90) |
| `c_echo` | 1096 | depolarize | echo | measure_fidelity | no | 294 | 3 | 213/1000 = **0.213** [0.189, 0.239] | 0.1228 [0.1208, 0.1249] | 0.069 | 0.162 | 0.190 | 0.144 | 0.212 | -12352 | 37 / 33 (p = 0.72) |
| `b_pauli` | 1096 | pauli, hw frame | ramsey | measure_fidelity | no | 301 | 3 | 264/1000 = **0.264** [0.238, 0.292] | 0.2035 [0.2011, 0.2060] | 0.079 | 0.204 | 0.230 | 0.149 | 0.121 | -11870 | 104 / 49 (p = 0.00) |
| `c_pauli` | 1096 | pauli, hw frame | echo | measure_fidelity | no | 301 | 3 | 211/1000 = **0.211** [0.187, 0.237] | 0.1192 [0.1172, 0.1213] | 0.069 | 0.156 | 0.184 | 0.143 | 0.224 | -12370 | 42 / 40 (p = 0.91) |
| `x_pauli_stim_frame` | 1096 | pauli, Stim frame (wrong on hw) | ramsey | measure_fidelity | no | 301 | 3 | 221/1000 = **0.221** [0.196, 0.248] | 0.1192 [0.1173, 0.1213] | 0.079 | 0.209 | 0.235 | 0.150 | 0.112 | -12023 | 55 / 43 (p = 0.27) |
| `x_no_last_idle` | 556 | depolarize | ramsey | measure | no | 294 | 3 | 211/1000 = **0.211** [0.187, 0.237] | 0.1634 [0.1611, 0.1657] | 0.079 | 0.176 | 0.199 | 0.157 | 0.188 | -12243 | 18 / 16 (p = 0.86) |
| `x_window_1096` | 1096 | depolarize | ramsey | measure | yes | 294 | 3 | 228/1000 = **0.228** [0.203, 0.255] | 0.2917 [0.2889, 0.2945] | 0.079 | 0.226 | 0.248 | 0.252 | 0.184 | -12078 | 41 / 22 (p = 0.02) |
| `x_pauli` | 556 | pauli, hw frame | ramsey | measure | yes | 301 | 3 | 205/1000 = **0.205** [0.181, 0.231] | 0.1783 [0.1759, 0.1807] | 0.079 | 0.164 | 0.186 | 0.201 | 0.256 | -12326 | 22 / 26 (p = 0.67) |
| `x_final_readout` | 556 | depolarize | ramsey | measure_fidelity | yes | 294 | 3 | 209/1000 = **0.209** [0.185, 0.235] | 0.1873 [0.1849, 0.1897] | 0.079 | 0.176 | 0.199 | 0.195 | 0.226 | -12279 | 9 / 9 (p = 1.00) |

Hardware over model, per round (t = 3 is the final data detectors):

| priors | t = 0 | t = 1 | t = 2 | t = 3 |
|---|---|---|---|---|
| `a_old` | 0.99× | 1.52× | 1.43× | 0.71× |
| `b_ramsey` | 0.99× | 1.18× | 1.14× | 0.98× |
| `c_echo` | 1.14× | 1.65× | 1.50× | 1.02× |
| `b_pauli` | 0.99× | 1.31× | 1.24× | 0.98× |
| `c_pauli` | 1.14× | 1.71× | 1.54× | 1.02× |
| `x_pauli_stim_frame` | 0.99× | 1.28× | 1.21× | 0.98× |
| `x_no_last_idle` | 0.99× | 1.52× | 1.43× | 0.93× |
| `x_window_1096` | 0.99× | 1.18× | 1.15× | 0.58× |
| `x_pauli` | 0.99× | 1.63× | 1.52× | 0.73× |
| `x_final_readout` | 0.99× | 1.52× | 1.43× | 0.75× |

By stabilizer type:

| priors | Z t = 0 | Z t = 1 | Z t = 2 | Z t = 3 | X t = 1 | X t = 2 |
|---|---|---|---|---|---|---|
| **hardware** | **0.079** | **0.242** | **0.276** | **0.146** | **0.292** | **0.293** |
| `a_old` | 0.079 | 0.177 | 0.203 | 0.206 | 0.175 | 0.195 |
| `b_ramsey` | 0.079 | 0.228 | 0.257 | 0.149 | 0.224 | 0.241 |
| `c_echo` | 0.069 | 0.166 | 0.200 | 0.144 | 0.158 | 0.180 |
| `b_pauli` | 0.079 | 0.217 | 0.248 | 0.149 | 0.191 | 0.211 |
| `c_pauli` | 0.069 | 0.164 | 0.198 | 0.143 | 0.149 | 0.170 |
| `x_pauli_stim_frame` | 0.079 | 0.163 | 0.199 | 0.150 | 0.254 | 0.272 |
| `x_no_last_idle` | 0.079 | 0.177 | 0.203 | 0.157 | 0.175 | 0.195 |
| `x_window_1096` | 0.079 | 0.228 | 0.252 | 0.252 | 0.224 | 0.243 |
| `x_pauli` | 0.079 | 0.172 | 0.195 | 0.201 | 0.156 | 0.178 |
| `x_final_readout` | 0.079 | 0.177 | 0.207 | 0.195 | 0.175 | 0.192 |

Of the candidate priors (`a_old`, `b_ramsey`, `c_echo`, `b_pauli`, `c_pauli`): the best fit to the detection statistics is `b_pauli`, the lowest hardware P_L `a_old`; they disagree, so the detector log-likelihood alone does not choose a decoder. Decoding significantly worse than `a_old` (McNemar p < 0.05): `b_pauli`, `x_window_1096`.
<!-- reprior:end -->

**What the pilot says** (one Z-basis circuit at 1,000 shots):
- **The final detectors are fixed.** Hardware over model at t = 3 goes from 0.71× (the pilots review's figure, reproduced) to 0.98× (`b_ramsey`). Most of it is fix 3 (`x_no_last_idle`: 0.93×); the terminal readout errors add a little (`x_final_readout`: 0.75× alone).
- **The default prior, `b_ramsey` (symmetric idle), fits the detection statistics well and decodes like the old one.** Detector log-likelihood −11,926 against −12,297 (old) and −12,352 (echo), Σ|model − hw| 0.080 against 0.237 and 0.212; its model P_L, 0.223, matches the hardware's 0.211. Shot by shot against the old priors, 32 against 30 discordant failures, p = 0.90. Echo fits worse: it under-predicts rounds 0–2 (1.14×, 1.65×, 1.50×), as expected with no dynamical decoupling, though it is the closest at t = 3 (1.02×).
- **The biased twirl in the hardware frame (`b_pauli`) fits the detection statistics best and decodes worst.** Its log-likelihood, −11,870, is the highest of every row, and its model P_L (0.204) matches the hardware's; yet it decodes the hardware significantly worse than the old priors: P_L 0.264 [0.238, 0.292] against 0.209, 104 against 49 discordant failures, p < 0.01. The echo version (`c_pauli`) decodes like the old priors (p = 0.91) and fits worst. Round 1's biased prior, in Stim's frame (`x_pauli_stim_frame`), put the dephasing of five data qubits on the wrong Stim Pauli; its numbers (0.221, model 0.119) are kept for the record and mean nothing about the hardware.
- **So the detector log-likelihood alone does not choose a decoder:** here it ranks `b_pauli` first and decoding ranks it last. `hardware_noise` defaults to the symmetric idle, and the batch analysis chooses the prior per basis **by decoded P_L on the finetune split's validation shots, never on test** (P0's shots are already in the finetune pool). The X basis has no hardware data yet.
- **No prior has enough noise in the bulk rounds, and the channel moves the deficit around.** At t = 1, 2 the hardware is 1.18× and 1.14× `b_ramsey`, 1.31× and 1.24× `b_pauli`. By type at t = 1, the hardware's Z-type plus X-type is 0.242 + 0.292; `b_ramsey` gives 0.228 + 0.224, and `b_pauli` 0.217 + 0.191. In the hardware frame, the biased twirl's dephasing falls on Stim X on the five rotated qubits, so it nearly closes the Z-type deficit and widens the X-type one. Something in the readout window flips the data in a way T1/T2 doesn't give, the X-type detectors most. The pilot's lag-1 same-stabilizer correlation, 0.063 against the stand-in's 0.018, points the same way. Leakage, measurement-induced transitions on the neighbours, and CZ errors above IRB are candidates; none is tested.
- **The window without fix 3 over-predicts the final detectors (0.58×) and trends worse on decoding.** The compiled window alone, still with the idle after the last round (`x_window_1096`), has 41 against 22 discordant failures, p = 0.02. That is one of several uncorrected comparisons, so the t = 3 ratio is the stronger evidence. Fix 1 is used only with fix 3.

<!-- plan:begin -->
Generated by `scripts/garnet_plan.py` from calibration set `a18a947f-0408-4628-81ea-1715637e434f`.

**Estimated QPU time of a job = Σ over its circuits of shots × (circuit duration + reset_wait) + circuits × 0.1 s.** reset_wait is the device's passive-reset wait, 298 µs (`gates.reset_wait` in the snapshot): the qubits relax between shots, as there is no active reset. Any other per-shot cost is still unknown (the pilot's timeline did not settle it, below); the 'pessimistic' column instead charges 1.92 ms per shot, the whole per-shot cost implied by Garnet's published CLOPS of 2,600 (100 × 10 × 100 × 5 / 2,600 s over 10⁵ shots). Circuit durations come from the calibrated durations (prx 16 ns, CZ 40 ns, measure 480 ns median; each readout layer takes its slowest qubit's) with barriers as sync points (`circuits.schedule`): e.g. E1 R = 3 is 2.34 µs. That model counts a readout as its `measure` probe; IQM's compiler makes each readout window about twice as long (pulla-check: the pilot compiles to 3.90 µs), which moves a shot by a few µs against the 298 µs reset wait.

**The pilot's timeline is not a planning figure.** Job `01a17000-6394-755f-a58b-b4f9c01893b0`'s execution span is 50.66 ms for 1,000 shots, 50.7 µs per shot (`InfoRecord/derived/E1/p0_E1_d3_Z_r03_pA/pilot_checks.json`), against the 298 µs reset wait planned here. That is less than the wait alone, yet the raw shots show the qubits did wait about 200 µs or more between shots (the round-1 Z-type mean is 0.079, the perfect-start value; a 48 µs wait would give about 0.34: `orchestration/reports/pilots-review.md`). So the span does not bracket the shot loop, and the plan charges the reset wait per shot.

| job | phase | exp | circuits | shots / circuit | total shots | circuit (µs), min–max (mean) | per shot (µs) = mean + reset_wait | shots × per shot (s) | + circuits × 0.1 s | **est. QPU (s)** | pessimistic (s) | what |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | pilot | E1 | 1 | 1,000 | 1,000 | 2.3–2.3 (2.3) | 300.3 | 0.3 | 0.1 | **0.4** | 2 | sanity checks before anything else |
| B1-1-E3 | batch 1 | E3 | 22 | 2,000 | 44,000 | 0.6–8.4 (2.0) | 300.0 | 13.2 | 2.2 | **15.4** | 87 | readout characterisation |
| B1-2-E1 | batch 1 | E1 | 22 | 5,000 | 110,000 | 0.8–15.6 (6.1) | 304.1 | 33.5 | 2.2 | **35.7** | 214 | memory, patch pA |
| B1-3-E1 | batch 1 | E1 | 22 | 5,000 | 110,000 | 0.8–15.6 (6.1) | 304.1 | 33.5 | 2.2 | **35.7** | 214 | memory, patch pB |
| B1-4-E2 | batch 1 | E2 | 27 | 5,000 | 135,000 | 0.7–33.4 (9.7) | 307.7 | 41.5 | 2.7 | **44.2** | 264 | repetition codes |
| B2-1-E3 | batch 2 | E3 | 22 | 2,000 | 44,000 | 0.6–8.4 (2.0) | 300.0 | 13.2 | 2.2 | **15.4** | 87 | readout characterisation |
| B2-2-E1 | batch 2 | E1 | 22 | 5,000 | 110,000 | 0.8–15.6 (6.1) | 304.1 | 33.5 | 2.2 | **35.7** | 214 | memory, patch pA |
| B2-3-E1 | batch 2 | E1 | 22 | 5,000 | 110,000 | 0.8–15.6 (6.1) | 304.1 | 33.5 | 2.2 | **35.7** | 214 | memory, patch pB |
| B2-4-E4 | batch 2 | E4 | 24 | 5,000 | 120,000 | 2.6–41.2 (12.8) | 310.8 | 37.3 | 2.4 | **39.7** | 235 | delay sweep |
| B3-1-E3 | batch 3 | E3 | 22 | 2,000 | 44,000 | 0.6–8.4 (2.0) | 300.0 | 13.2 | 2.2 | **15.4** | 87 | readout characterisation |
| B3-2-E1 | batch 3 | E1 | 22 | 5,000 | 110,000 | 0.8–15.6 (6.1) | 304.1 | 33.5 | 2.2 | **35.7** | 214 | memory, patch pA |
| B3-3-E1 | batch 3 | E1 | 22 | 5,000 | 110,000 | 0.8–15.6 (6.1) | 304.1 | 33.5 | 2.2 | **35.7** | 214 | memory, patch pB |
| B3-4-E5 | batch 3 | E5 | 12 | 5,000 | 60,000 | 1.6–9.6 (4.6) | 302.6 | 18.2 | 1.2 | **19.4** | 117 | stability |
| **total** | | | 262 | | **1,108,000** | | | 338 | 26 | **364 s (6.1 min)** | **2164** | |

| experiment | jobs | circuit runs | shots | est. QPU (s) |
|---|---|---|---|---|
| E1 | 7 | 133 | 661,000 | 214 |
| E2 | 1 | 27 | 135,000 | 44 |
| E3 | 3 | 66 | 132,000 | 46 |
| E4 | 1 | 24 | 120,000 | 40 |
| E5 | 1 | 12 | 60,000 | 19 |

| E1 split | shots |
|---|---|
| finetune (batches 1, 2 and the pilot P0, 80%) | 352,800 |
| validation (the same jobs, 20% of shot indices) | 88,200 |
| test (batch 3, whole) | 220,000 |
| **total E1** | **661,000** |

**Pilot expectations** (E1 Z R=3 pA; 'band' is what the pilot must fall in before anything else runs):

```json
{
 "circuit": "E1_d3_Z_r03_pA",
 "stand_in": {
  "shots": 1000,
  "columns": 33,
  "detection_fraction": 0.10358333333333333,
  "detection_fraction_ci": [
   0.09979140074369819,
   0.1075021471393312
  ],
  "x_ancilla_m1_one": 0.4805,
  "z_ancilla_m1_one": 0.056,
  "all_zero_rows": 0,
  "source": "InfoRecord/sim/datasets/E1/p0_E1_d3_Z_r03_pA"
 },
 "calibrated_stim_no_idle": {
  "shots": 100000,
  "columns": 33,
  "detection_fraction": 0.121295,
  "detection_fraction_ci": [
   0.1208825723614411,
   0.121708639953004
  ],
  "x_ancilla_m1_one": 0.4997025,
  "z_ancilla_m1_one": 0.07947,
  "all_zero_rows": 62,
  "idle_ns": 0.0
 },
 "calibrated_stim_with_idle": {
  "shots": 100000,
  "columns": 33,
  "detection_fraction": 0.17248166666666667,
  "detection_fraction_ci": [
   0.17200421866323318,
   0.17296016312524737
  ],
  "x_ancilla_m1_one": 0.4997025,
  "z_ancilla_m1_one": 0.07947,
  "all_zero_rows": 19,
  "idle_ns": 556.0
 },
 "band": {
  "detection_fraction": [
   0.051791666666666666,
   0.34496333333333334
  ],
  "x_ancilla_m1_one": [
   0.4,
   0.6
  ],
  "z_ancilla_m1_one_max": 0.3,
  "all_zero_rows_max_per_1000": 5,
  "columns": 33,
  "x_cols": [
   0,
   2,
   5,
   7
  ],
  "z_cols": [
   1,
   3,
   4,
   6
  ],
  "note": "x_ancilla_m1_one and z_ancilla_m1_one are pooled means: the fraction of 1s over every shot and every column in x_cols (z_cols) of raw_meas, clbit j = column j. detection_fraction is the detection events over shots x detectors."
 },
 "hardware_prior": {
  "shots": 100000,
  "columns": 33,
  "detection_fraction": 0.19632833333333333,
  "detection_fraction_ci": [
   0.19582627648851783,
   0.19683136229509487
  ],
  "x_ancilla_m1_one": 0.4997025,
  "z_ancilla_m1_one": 0.07947,
  "all_zero_rows": 9,
  "idle_ns": 1096.0,
  "prior": {
   "calibration_set_id": "a18a947f-0408-4628-81ea-1715637e434f",
   "family": "surface_memory",
   "readout_window_ns": 1096.0,
   "readout_window_source": "compiled measure_window_ns (data/calibration/garnet/pulse/compile.a18a947f-0408-4628-81ea-1715637e434f.json)",
   "delay_ns": 0.0,
   "idle_ns": 1096.0,
   "idle_channel": "depolarize",
   "idle_frame": null,
   "t2": "ramsey",
   "final_readout": "measure_fidelity (unheralded)",
   "idle_after_last_round": false
  },
  "note": "garnet-timing's hardware prior (device.hardware_noise), for reference; the band below stays the one the pilot passed, built from calibrated_stim_with_idle (the pre-fix priors)"
 }
}
```
<!-- plan:end -->

## What `dqiqm.hw` needs (not edited here)

- **Nothing for the path.** `hw.submit(..., data_root=REPO/"InfoRecord"/"raw")` and `hw.save_raw(..., data_root=...)` already write `InfoRecord/raw/<YYYYMMDD>-<task>/<job id>/`. The dry run uses exactly that.
- **Recommended:**
  1. Refuse any serialized op outside {prx, cz, measure, barrier, delay} (layout review MINOR 1). The plan's circuits are hardware-target only, but `submit` itself doesn't check.
  2. Return or record the job timeline in a machine-readable form; `meta.json` has it as `timeline`. The pilot's per-shot overhead comes from it.
  3. Fix `list_devices()` (resonance-check finding 1).
  4. Add a helper that snapshots the calibration (resonance-check's format) into `InfoRecord/calibration/garnet/` before and after a job group, so E6 is one call.
