# emerald-dd: dynamical decoupling on and off on IQM Emerald (HARDWARE)

emerald-dd task, 2026-10-25 UTC. A paired A/B of IQM's server-side dynamical decoupling (DD) on the d = 3
no-reset surface-code memory on patch eA of IQM Emerald, as `docs/emerald-design.md` ("Dynamical
decoupling: an A/B to run on E1") designed it. All numbers are **HARDWARE** unless a table says
SIMULATED.

**7 jobs, 81,000 shots, all within the allowance; both segments PASS.**

<!-- conclusion:begin -->
## The answer

**Does DD move the X basis's logical rate? No.** ε_L(X) is 0.0370 [0.0351, 0.0389] with DD and 0.0375 [0.0357, 0.0394]
without: Δε = +0.0005 [−0.0021, +0.0030], against an ABBA drift of A1 − A2 = −0.0022 and B1 − B2 =
+0.0008. Every per-R difference is inside its interval (z −0.6, +0.8, +0.3 at R = 3, 8, 15), under both
decoders. X's raw observable flips do drop with DD (+0.021, +0.029 and +0.020 at R = 3, 8 and 15; z +2.3,
+2.6, +1.8), but its decoded P_L and ε_L do not. The hypothesis, that idle dephasing during the readouts
limits the X basis's logical rate, is not supported.

**DD moves the Z basis.** ε_L(Z) falls from 0.0470 [0.0448, 0.0493] to 0.0396 [0.0377, 0.0415], a 16% cut:
Δε = +0.0074 [+0.0045, +0.0104], about 6× the drift (A1 − A2 = −0.0006, B1 − B2 = −0.0012). P_L is lower
with DD at every R (R = 3: 0.1085 → 0.0862, z +3.4; R = 8: z +3.3; R = 15: z +2.4). The cut holds under
the echo-T2 decoder (Δε = +0.0095 [+0.0069, +0.0121]), and without one: the observable's raw flip rate drops
by 0.051 at R = 3 and 0.060 at R = 8 (z +5.4 each). With DD off the Z basis is worse than X (R = 3: 0.1085 against 0.0755
here; the batches saw the same); with DD the two nearly meet (R = 3: 0.0862 against 0.0790; ε 0.0396 against
0.0370).

**Where it is: one weight-2 stabilizer.** Decoder-free, per stabilizer (the tables below), DD changes
two of eight and leaves the rest within ±0.02:

- **Z(6,2)** (ancilla QB39; data QB31, QB38) fires at 0.27–0.30 per detector at R = 15 with DD off (the
  batches' three DD-off runs: 0.28–0.31), and 0.16–0.17 with DD on. In the Z basis its round-1 detector reads 0.11
  either way; the gap opens from round 2, i.e. after the first readout window. Both of its data
  qubits idle through the readout in a Hadamard frame (`device.readout_frames`), where a physical Z acts as
  Stim's X and flips Z-type detectors and the **Z** logical. That is why a dephasing-type error that DD
  refocuses shows up in the Z basis and not the X basis. The error DD removes sits on **QB31**, not QB38:
  QB38 is shared with Z(4,4), which DD leaves unchanged (0.238 on, 0.236 off in the Z basis; 0.256 on,
  0.261 off in X), while a DD-removable error on QB38 would flip Z(4,4) as often as Z(6,2). QB31 is the
  data qubit of Z(6,2) that no other Z-type stabilizer has. It is not plain T2* decay: the decoder priors'
  model, which already has QB31's and QB38's Ramsey T2 (15.2 µs, 9.3 µs) over the window, gives Z(6,2)
  0.131 (Z basis) and 0.142 (X basis) on Emerald's compiled 1,036 ns readout window (the emerald-dd
  review's re-run of the model, E1 R = 15, 20,000 shots, seed 2026). The hardware's excess over it is
  about 0.14–0.16, and DD removes about 80% of it. DD off is still about 2× the model, and DD on (0.161,
  0.172) stays above the model's echo rate (0.105, 0.113). The model columns in the tables below are the
  360 ns `measure_ns` fallback the records were made with (0.086–0.092: an excess of about 0.2, 60%
  removed). A large, systematic phase on QB31 during the readout window, which an echo cancels, fits; what
  makes it (a measurement-induced shift, a ZZ phase from a neighbour) is beyond these data. The frame explains which
  basis the error lands in, not its size, nor why the 4 identity-frame data qubits give the X basis nothing
  measurable.
- **Z(0,4)** (ancilla QB19; data QB20, QB27) gets worse with DD: 0.12–0.13 → 0.19–0.20. Its cost is
  already there in round 1, before any readout window could matter to the data, and QB19's deterministic
  first readout misreads 0.055–0.070 more often with DD (0.070–0.077 → 0.132–0.140), while the other three
  Z-type ancillas move by under 0.01. Two mechanisms fit, and these data do not choose: DD pulses on QB19
  itself, in the merged gaps it sits out in the CZ layers just before each readout (round 1 included), or
  pulses on its neighbours during its readout. QB39, the other weight-2 Z-type ancilla, shows no such cost
  (its misread moves by under 0.005), which a pulse-error account on the ancilla alone would have to
  explain. The mechanism is open. It costs the Z basis something, and Z(6,2)'s gain outweighs it.

**E4 at 5 µs: DD is decisive in both bases.** At R = 2, P_L falls from 0.490 to 0.158 (Z) and from 0.416 to
0.142 (X), z +22.5 and +19.3; at R = 3, from 0.415 to 0.299 (Z) and 0.449 to 0.263 (X). Without DD, the
undecoded flip rate at R = 2 is 0.577 (Z) and 0.553 (X): **above ½**, which independent stochastic flips
cannot give. It is a coherent or correlated error across the 5 µs idle, which DD removes (0.252, 0.285).
At R ≥ 5 the DD-off arm is at chance, so its fitted ε (0.285 X, 0.353 Z at the bound) is a floor, not a rate;
with DD, ε is 0.115 (X) and 0.129 (Z).

**What limits the LER, as far as these data say.** On eA, a localized systematic error on QB31 during
the readout windows, which an echo refocuses, limits the Z basis. It is a standing feature: the
batches' DD-off runs, 38 to 55 minutes earlier, show it at the same size. For long idles, coherent
dephasing on every data qubit dominates, and DD removes most of it. The idealized echo model (DD pulses free,
echo T2 everywhere) predicted gains in both bases at R = 3 (0.073 → 0.043 Z, 0.073 → 0.047 X). The hardware
gave the Z basis a smaller one (0.1085 → 0.0862) and the X basis's decoded P_L none. Ramsey decay spread
evenly over the data qubits is not what limits the X basis's logical rate, and the DD pulses are not free
(QB19).

**Noise notes.** Of the twelve drift cells (A1 − A2 and B1 − B2, both bases, three R), one has |z| > 2: X at
R = 15, A1 − A2 = −0.035 (z −2.3). That is consistent with chance at 5% per cell. The pilot DDP (DD on,
X R = 3, 1,000 shots: P_L 0.081 [0.066, 0.100]) is unpaired and is not in the pooled arms.
<!-- conclusion:end -->

## The question

The challenge names DD explicitly, and its bonus is "experimentally finding what limits the LER". Our
hypothesis was that idle dephasing during the ancilla readouts limits the X basis: the data qubits wait
through every mid-circuit readout (360 ns of probe per round on Emerald, more with the acquisition and
dead time), and eA's Ramsey T2 (7–48 µs, median 14 µs) is well below its echo T2 (20–68 µs, median
34 µs). DD is the experiment that tests it: it refocuses quasi-static dephasing (T2* toward T2 echo)
and does nothing for T1.

## What DD is here

`CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=STANDARD_DD_STRATEGY)` (iqm-client 35.0.3).
The server's compiler fills each qubit's idle windows with a DD sequence, chosen by the window's length
in PRX durations (24 ns on Emerald): EDD `XYXYYXYX` from 9 (216 ns), XY4 `YXYX` from 5 (120 ns), CPMG
`XX` from 2 (48 ns), none below that; leading and trailing waits are skipped. Every sequence is the
identity, so the circuit's meaning, its detectors and its decoders are unchanged: the A/B runs the very
same frozen circuits, with the same `native_sha256`, and only the job's option differs. The strategy is
named rather than left to the server default, so `meta.json` records the sequences that ran. In our
circuits the data qubits' wait through each ancilla readout layer and, in E4, every qubit's 5 µs delay
after each readout certainly qualify (EDD). A single CZ layer's 40 ns gap is below the CPMG threshold, but
the strategy merges contiguous waits separated only by blocks (`merge_contiguous_waits=True`), so the
longer gaps a qubit sits out across several CZ and PRX layers (a weight-2 boundary ancilla skips two of the
four CZ layers of a round) may get XY4 or CPMG too: not inspected. The pulse schedule the server compiled was not inspected (the brief rules out Pulla); the
server's record of each job says which mode it ran with.

## The design

| segment | task | job | DD | circuits | shots | what |
|---|---|---|---|---|---|---|
| 1 | `emerald-dd-pilot` | DDP | on | `E1_d3_X_r03_eA` | 1 x 1,000 | the pilot: does the server take the option, and is the record sane |
| 2 | `emerald-dd` | A1 | on | E1 Z and X, R ∈ {3, 8, 15}, eA | 6 x 2,000 | E1's ABBA, back to back |
| 2 | `emerald-dd` | B1 | off | the same 6 | 6 x 2,000 | |
| 2 | `emerald-dd` | B2 | off | the same 6 | 6 x 2,000 | |
| 2 | `emerald-dd` | A2 | on | the same 6 | 6 x 2,000 | |
| 2 | `emerald-dd` | E4A | on | E4 Z and X, R ∈ {2, 3, 5, 8}, 5 µs, eA | 8 x 2,000 | E4 at 5 µs, where DD should matter most |
| 2 | `emerald-dd` | E4B | off | the same 8 | 8 x 2,000 | |

- **Why ABBA.** Compilation options belong to a job, so DD on and off cannot share one. Four jobs in the
  order on, off, off, on, back to back (59 s from A1's ledger row to A2's), make a linear drift cancel in
  A1 + A2 against B1 + B2, and A1 − A2 and B1 − B2 measure the drift and the shot noise that are left.
  Every job runs its circuits in one fixed order (Z r03, X r03, Z r08, X r08, Z r15, X r15), so each
  circuit sits at the same place in all four jobs.
- **E4.** The frozen E4 has 8 circuits at 5 µs (Z and X at R ∈ {2, 3, 5, 8}), so its two jobs are 8 x
  2,000, not the 12 the brief allowed. Segment 2 is 80,000 shots, armed exactly
  (`DQ_QPU_MAX_SHOTS=80000`; the brief allowed up to 96,000). E4 is one pair, with no drift pair of its
  own: E1's drift scale applies.
- **The circuits are frozen.** Every circuit was rebuilt from the committed snapshot the plan was made from
  (`emerald.build(emerald.live_calibration(emerald.DEFAULT_SNAPSHOT))`, calibration set 23e12f6b) and its
  `native_sha256`, `stim_sha256` and `qasm_sha256` checked against `InfoRecord/emerald/circuits/<exp>/placement.json`.
  `hw.submit` also got the `native_sha256` per circuit, so it would have refused a request that differed.
  Nothing under `InfoRecord/emerald/circuits/` was regenerated.
- **Not in the ML split.** These shots' `split` is `dd_ab`, never finetune, validation or test: the
  batches' split is untouched.

## What ran

The job table (ids, ledger times, the timeline's queue and execution spans, the server's `dd_mode`) is the
first table of "The results".

- **One path.** Every job went through `hw.submit` only, armed on its own command:
  `DQ_QPU=run DQ_QPU_TASK=emerald-dd-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000` for the pilot, then
  `DQ_QPU=run DQ_QPU_TASK=emerald-dd DQ_QPU_MAX_JOBS=6 DQ_QPU_MAX_SHOTS=80000` for segment 2. The driver
  refuses any other arming, and refused segment 2 until the pilot was in the ledger with its checks passed.
- **The ledger.** Exactly one row per job, written by `hw.submit` the moment the job id came back. The
  DD-on rows' purpose ends `[dd=ENABLED,dd_strategy=879fe63c]` (the first 8 hex digits of the sha256 of
  the strategy's JSON); the DD-off rows carry no tag.
- **The calibration set** was `23e12f6b-c23c-4736-8736-b0142989c0e6` at the before snapshot (01:50 UTC), at
  every job (re-read before each), and at the after snapshot (01:57 UTC): the design's own set. Nothing was
  re-ranked and nothing lost a qubit or coupler.
- **The server took the option.** After each job, the server's own record of it (the job's payload,
  re-pulled read-only with `hw.retrieve_native(job_id, "emerald").parameters`) gave `dd_mode` `enabled`
  with the standard sequences on DDP, A1, A2 and E4A, and `disabled` with no strategy on B1, B2 and E4B,
  and the circuits it holds hash to the frozen `native_sha256`. `meta.json`'s `compilation_options` says
  the same. That is the record of which arm a job is; it is checked for every job, not assumed.
- **No failure, no retry.** No submit raised, no `UNKNOWN-…` row was written, and `recover` was never used.
- **Re-pulled.** Every job was re-pulled read-only by id (`hw.retrieve`) into scratch: the memory, the
  calibration set, the circuit count and the timeline are identical to the committed raw data.

## The checks

**The pilot (DDP), before segment 2**, all PASS:

| check | value | band |
|---|---|---|
| the server's `dd_mode` (payload, read-only) | `enabled`, standard sequences | `enabled` |
| the server's circuit = the frozen `native_sha256` | `6bd0d5bc…` | equal |
| `meta.json` `compilation_options` | `dd_mode: enabled`, strategy recorded | the option sent |
| ledger row | eA's 17, `1 x 1000`, `[dd=ENABLED,dd_strategy=879fe63c]` | as planned |
| timeline | `execution_started` 01:52:35.31, `execution_ended` 01:52:36.62 (1.31 s; EP0, DD off: 1.36 s) | present |
| memory shape | (1000, 33) | (1000, 33) |
| all-zero rows | 0 | ≤ 20.6 |
| Z-type ancillas' first readout (random in the X basis) | 0.500 | [0.4, 0.6] |
| X-type ancillas' first readout misread (deterministic in the X basis) | 0.054 | < 0.3 |
| detection fraction | 0.1196 | < 0.45, and [0.5, 2] × the priors' own 0.1057 |
| decoded P_L (PyMatching, `hardware_noise`) | 81/1000 = 0.081 [0.066, 0.100] | ≤ 2 × the batch's DD-off 0.0995 |

The batch's DD-off runs of the same circuit gave detection fractions 0.134, 0.137, 0.125 and P_L
167, 199, 167 per 2,000 (jobs `01a17093-aba4…`, `01a17097-6a9d…`, `01a170a1-17bf…`).

**Every job of segment 2** passed the batch's sanity checks, circuit by circuit (shape, all-zero rows,
the first readouts' random/deterministic pattern per basis, the detection fraction, `physical_qubits`),
plus the ledger row, `meta.json` (`qubit_mapping`, `physical_qubits`, `measured_qubits` = Stim's
measurement j on its device qubit, `circuit_order` with names and hashes, `compilation_options`,
calibration set, shots) and the server's record. The per-job verdicts are in the manifests
(`InfoRecord/emerald/raw/20261025-emerald-dd/<label>.json`: `ok`, `sanity`, `server`, `server_check`).

## The results

The decoder of the record (`pm_fail` in `shots.csv`) and of the headline tables is the batch's:
PyMatching on the noisy copy of the circuit that ran, priors `device.hardware_noise(cal, spec)` (Ramsey
T2, the symmetric idle over the 360 ns readout window plus E4's delay, `measure_fidelity` at the final
readout), on **both arms**. The 360 ns is the `measure_ns` fallback: the records predate main's compiled
1,036 ns window (`53b2a19`), and "How to run it" says which window reproduces which column. The design names an echo-T2 variant for the DD shots
(`hardware_noise(..., t2="echo")`); it is reported for both arms too, so each comparison uses one decoder.
Neither decoder was chosen on these shots. The undecoded observable flips and the detection fractions
use no decoder at all.

Intervals: P_L, Wilson 95%; differences, Newcombe's hybrid score 95% (from the two Wilson intervals), with
the pooled z; ε_L, maximum likelihood (`stats.fit_eps`) with a bootstrap 95% interval (1,000, seed 2026),
and its difference by a seeded parametric bootstrap of both arms.

<!-- dd:begin -->
### The jobs (HARDWARE)

| job | DD | job id | calibration set | circuits x shots | ledger (UTC) | queue (s) | execution (s) | per shot (µs) | server dd_mode | checks |
|---|---|---|---|---|---|---|---|---|---|---|
| DDP | on | `01a170c2-72b7-74bd-a439-40a869cc85fc` | 23e12f6b | 1 x 1000 | 2026-10-25T01:52:34Z | 0.7 | 1.3 | 1309 | enabled | OK |
| A1 | on | `01a170c4-f8aa-73e5-9c64-28bf260c1510` | 23e12f6b | 6 x 2000 | 2026-10-25T01:55:19Z | 1.5 | 7.6 | 630 | enabled | OK |
| B1 | off | `01a170c5-4c8e-71fc-b805-5b902192460b` | 23e12f6b | 6 x 2000 | 2026-10-25T01:55:41Z | 0.9 | 7.8 | 654 | disabled | OK |
| B2 | off | `01a170c5-9852-70a7-ac60-456070979378` | 23e12f6b | 6 x 2000 | 2026-10-25T01:56:00Z | 1.2 | 7.6 | 632 | disabled | OK |
| A2 | on | `01a170c5-de38-72b4-afd1-e4002ad09786` | 23e12f6b | 6 x 2000 | 2026-10-25T01:56:18Z | 1.3 | 7.6 | 633 | enabled | OK |
| E4A | on | `01a170c6-2526-7766-b6ae-2900a4dfbc85` | 23e12f6b | 8 x 2000 | 2026-10-25T01:56:36Z | 1.2 | 8.6 | 539 | enabled | OK |
| E4B | off | `01a170c6-696a-779b-9425-ef353504b1f8` | 23e12f6b | 8 x 2000 | 2026-10-25T01:56:55Z | 0.9 | 8.6 | 535 | disabled | OK |

### E1, P_L, decoder hardware_noise (Ramsey T2), both arms: the headline

| basis | R | DD on (A1+A2) | DD off (B1+B2) | off − on [Newcombe 95%] | drift on A1 − A2 | drift off B1 − B2 |
|---|---|---|---|---|---|---|
| Z | 3 | 345/4000 = 0.0862 [0.0779, 0.0954] | 434/4000 = 0.1085 [0.0992, 0.1185] | +0.0223 [0.0093, 0.0353] (z +3.4) | +0.0065 [-0.0109, 0.0240] (z +0.7) | -0.0020 [-0.0213, 0.0173] (z -0.2) |
| Z | 8 | 1012/4000 = 0.2530 [0.2398, 0.2667] | 1141/4000 = 0.2853 [0.2715, 0.2994] | +0.0323 [0.0128, 0.0517] (z +3.3) | +0.0010 [-0.0259, 0.0279] (z +0.1) | -0.0115 [-0.0395, 0.0165] (z -0.8) |
| Z | 15 | 1486/4000 = 0.3715 [0.3567, 0.3866] | 1592/4000 = 0.3980 [0.3829, 0.4133] | +0.0265 [0.0052, 0.0478] (z +2.4) | -0.0170 [-0.0469, 0.0129] (z -1.1) | +0.0030 [-0.0273, 0.0333] (z +0.2) |
| X | 3 | 316/4000 = 0.0790 [0.0710, 0.0878] | 302/4000 = 0.0755 [0.0677, 0.0841] | -0.0035 [-0.0152, 0.0082] (z -0.6) | +0.0010 [-0.0158, 0.0178] (z +0.1) | +0.0100 [-0.0064, 0.0265] (z +1.2) |
| X | 8 | 926/4000 = 0.2315 [0.2187, 0.2448] | 955/4000 = 0.2387 [0.2258, 0.2522] | +0.0072 [-0.0113, 0.0258] (z +0.8) | +0.0030 [-0.0231, 0.0291] (z +0.2) | -0.0005 [-0.0269, 0.0259] (z -0.0) |
| X | 15 | 1476/4000 = 0.3690 [0.3542, 0.3841] | 1489/4000 = 0.3723 [0.3574, 0.3873] | +0.0033 [-0.0179, 0.0244] (z +0.3) | -0.0350 [-0.0648, -0.0051] (z -2.3) | -0.0005 [-0.0304, 0.0294] (z -0.0) |

### E1, P_L, decoder hardware_noise t2='echo', both arms

| basis | R | DD on (A1+A2) | DD off (B1+B2) | off − on [Newcombe 95%] | drift on A1 − A2 | drift off B1 − B2 |
|---|---|---|---|---|---|---|
| Z | 3 | 336/4000 = 0.0840 [0.0758, 0.0930] | 374/4000 = 0.0935 [0.0849, 0.1029] | +0.0095 [-0.0030, 0.0220] (z +1.5) | -0.0020 [-0.0193, 0.0152] (z -0.2) | +0.0040 [-0.0141, 0.0221] (z +0.4) |
| Z | 8 | 926/4000 = 0.2315 [0.2187, 0.2448] | 1119/4000 = 0.2797 [0.2661, 0.2939] | +0.0482 [0.0291, 0.0673] (z +4.9) | -0.0030 [-0.0291, 0.0231] (z -0.2) | +0.0035 [-0.0243, 0.0313] (z +0.2) |
| Z | 15 | 1411/4000 = 0.3528 [0.3381, 0.3677] | 1640/4000 = 0.4100 [0.3949, 0.4253] | +0.0572 [0.0360, 0.0785] (z +5.3) | -0.0215 [-0.0511, 0.0081] (z -1.4) | +0.0100 [-0.0205, 0.0404] (z +0.6) |
| X | 3 | 307/4000 = 0.0767 [0.0689, 0.0854] | 301/4000 = 0.0752 [0.0675, 0.0838] | -0.0015 [-0.0131, 0.0101] (z -0.3) | +0.0015 [-0.0151, 0.0181] (z +0.2) | +0.0065 [-0.0099, 0.0229] (z +0.8) |
| X | 8 | 928/4000 = 0.2320 [0.2192, 0.2453] | 941/4000 = 0.2352 [0.2224, 0.2486] | +0.0032 [-0.0153, 0.0218] (z +0.3) | +0.0010 [-0.0252, 0.0272] (z +0.1) | -0.0045 [-0.0308, 0.0218] (z -0.3) |
| X | 15 | 1470/4000 = 0.3675 [0.3527, 0.3826] | 1472/4000 = 0.3680 [0.3532, 0.3831] | +0.0005 [-0.0206, 0.0216] (z +0.0) | -0.0290 [-0.0588, 0.0009] (z -1.9) | -0.0020 [-0.0319, 0.0279] (z -0.1) |

### E1, P_L, undecoded: the observable's raw flip rate

| basis | R | DD on (A1+A2) | DD off (B1+B2) | off − on [Newcombe 95%] | drift on A1 − A2 | drift off B1 − B2 |
|---|---|---|---|---|---|---|
| Z | 3 | 794/4000 = 0.1985 [0.1864, 0.2111] | 997/4000 = 0.2492 [0.2361, 0.2629] | +0.0507 [0.0325, 0.0690] (z +5.4) | +0.0200 [-0.0047, 0.0447] (z +1.6) | +0.0275 [0.0007, 0.0543] (z +2.0) |
| Z | 8 | 1531/4000 = 0.3827 [0.3678, 0.3979] | 1770/4000 = 0.4425 [0.4272, 0.4579] | +0.0598 [0.0382, 0.0812] (z +5.4) | +0.0085 [-0.0216, 0.0386] (z +0.6) | -0.0030 [-0.0338, 0.0278] (z -0.2) |
| Z | 15 | 1866/4000 = 0.4665 [0.4511, 0.4820] | 1941/4000 = 0.4853 [0.4698, 0.5007] | +0.0187 [-0.0031, 0.0406] (z +1.7) | -0.0090 [-0.0399, 0.0219] (z -0.6) | +0.0115 [-0.0195, 0.0424] (z +0.7) |
| X | 3 | 850/4000 = 0.2125 [0.2001, 0.2254] | 934/4000 = 0.2335 [0.2206, 0.2469] | +0.0210 [0.0028, 0.0392] (z +2.3) | +0.0100 [-0.0154, 0.0353] (z +0.8) | -0.0030 [-0.0292, 0.0232] (z -0.2) |
| X | 8 | 1563/4000 = 0.3907 [0.3757, 0.4060] | 1677/4000 = 0.4193 [0.4040, 0.4346] | +0.0285 [0.0070, 0.0500] (z +2.6) | -0.0235 [-0.0537, 0.0067] (z -1.5) | +0.0095 [-0.0211, 0.0400] (z +0.6) |
| X | 15 | 1875/4000 = 0.4688 [0.4533, 0.4842] | 1954/4000 = 0.4885 [0.4730, 0.5040] | +0.0197 [-0.0021, 0.0416] (z +1.8) | -0.0265 [-0.0574, 0.0044] (z -1.7) | -0.0050 [-0.0359, 0.0260] (z -0.3) |

### E1, detection fraction (decoder-free)

| basis | R | on | off | off − on | per job | per round, on | per round, off |
|---|---|---|---|---|---|---|---|
| Z | 3 | 0.1358 | 0.1345 | -0.0013 | A1 0.1281, A2 0.1434, B1 0.1366, B2 0.1324 | 0.105, 0.158, 0.159, 0.076 | 0.086, 0.162, 0.165, 0.067 |
| Z | 8 | 0.1637 | 0.1658 | +0.0020 | A1 0.1564, A2 0.1710, B1 0.1655, B2 0.1660 | 0.101, 0.156, 0.160, 0.165, 0.171, 0.179, 0.182, 0.193, 0.107 | 0.087, 0.163, 0.168, 0.165, 0.179, 0.184, 0.186, 0.192, 0.092 |
| Z | 15 | 0.1859 | 0.1868 | +0.0008 | A1 0.1783, A2 0.1936, B1 0.1879, B2 0.1857 | 0.103, 0.156, 0.164, 0.167, 0.171, 0.177, 0.185, 0.193, 0.199, 0.192, 0.204, 0.210, 0.213, 0.213, 0.221, 0.145 | 0.090, 0.163, 0.162, 0.172, 0.179, 0.183, 0.187, 0.193, 0.199, 0.200, 0.203, 0.208, 0.212, 0.218, 0.218, 0.119 |
| X | 3 | 0.1170 | 0.1218 | +0.0049 | A1 0.1137, A2 0.1202, B1 0.1217, B2 0.1219 | 0.055, 0.142, 0.149, 0.064 | 0.054, 0.150, 0.158, 0.059 |
| X | 8 | 0.1536 | 0.1610 | +0.0074 | A1 0.1518, A2 0.1554, B1 0.1594, B2 0.1627 | 0.056, 0.142, 0.151, 0.159, 0.165, 0.172, 0.181, 0.183, 0.096 | 0.052, 0.151, 0.159, 0.170, 0.176, 0.180, 0.186, 0.196, 0.089 |
| X | 15 | 0.1775 | 0.1852 | +0.0077 | A1 0.1717, A2 0.1833, B1 0.1854, B2 0.1850 | 0.049, 0.145, 0.152, 0.160, 0.163, 0.170, 0.177, 0.184, 0.193, 0.195, 0.200, 0.205, 0.206, 0.213, 0.211, 0.128 | 0.055, 0.154, 0.162, 0.172, 0.174, 0.179, 0.190, 0.189, 0.195, 0.201, 0.205, 0.211, 0.213, 0.221, 0.223, 0.124 |

### E1, ε_L per round (2P = 1 − A(1 − 2ε)^R, maximum likelihood, bootstrap 95%)

| basis | decoder | ε on (A=1) | ε off (A=1) | ε off − on [bootstrap 95%] | ε on (free A), A | ε off (free A), A |
|---|---|---|---|---|---|---|
| Z | ramsey | 0.0396 [0.0377, 0.0415] | 0.0470 [0.0448, 0.0493] | +0.0074 [0.0045, 0.0104] | 0.0396, 1.000 | 0.0470, 1.000 |
| Z | echo | 0.0362 [0.0345, 0.0380] | 0.0458 [0.0436, 0.0479] | +0.0095 [0.0069, 0.0121] | 0.0362, 1.000 | 0.0458, 1.000 |
| Z | ramsey, per job | A1 0.0394, A2 0.0399, B1 0.0464, B2 0.0476 | | drift on -0.0006 [-0.0045, 0.0030]; drift off -0.0012 [-0.0056, 0.0033] | | |
| X | ramsey | 0.0370 [0.0351, 0.0389] | 0.0375 [0.0357, 0.0394] | +0.0005 [-0.0021, 0.0030] | 0.0370, 1.000 | 0.0375, 1.000 |
| X | echo | 0.0367 [0.0349, 0.0385] | 0.0369 [0.0351, 0.0387] | +0.0002 [-0.0023, 0.0027] | 0.0367, 1.000 | 0.0369, 1.000 |
| X | ramsey, per job | A1 0.0359, A2 0.0380, B1 0.0379, B2 0.0371 | | drift on -0.0022 [-0.0055, 0.0012]; drift off +0.0008 [-0.0026, 0.0043] | | |

### E4, P_L, decoder hardware_noise (Ramsey T2), both arms: the headline

| basis | R | DD on (E4A) | DD off (E4B) | off − on [Newcombe 95%] | 
|---|---|---|---|---|
| Z | 2 | 315/2000 = 0.1575 [0.1422, 0.1741] | 980/2000 = 0.4900 [0.4681, 0.5119] | +0.3325 [0.3050, 0.3592] (z +22.5) |
| Z | 3 | 598/2000 = 0.2990 [0.2793, 0.3194] | 830/2000 = 0.4150 [0.3936, 0.4367] | +0.1160 [0.0864, 0.1453] (z +7.7) |
| Z | 5 | 885/2000 = 0.4425 [0.4209, 0.4644] | 948/2000 = 0.4740 [0.4522, 0.4959] | +0.0315 [0.0006, 0.0623] (z +2.0) |
| Z | 8 | 1008/2000 = 0.5040 [0.4821, 0.5259] | 990/2000 = 0.4950 [0.4731, 0.5169] | -0.0090 [-0.0399, 0.0220] (z -0.6) |
| X | 2 | 284/2000 = 0.1420 [0.1274, 0.1580] | 832/2000 = 0.4160 [0.3946, 0.4377] | +0.2740 [0.2473, 0.3002] (z +19.3) |
| X | 3 | 525/2000 = 0.2625 [0.2437, 0.2822] | 898/2000 = 0.4490 [0.4273, 0.4709] | +0.1865 [0.1572, 0.2154] (z +12.3) |
| X | 5 | 828/2000 = 0.4140 [0.3926, 0.4357] | 973/2000 = 0.4865 [0.4646, 0.5084] | +0.0725 [0.0417, 0.1031] (z +4.6) |
| X | 8 | 988/2000 = 0.4940 [0.4721, 0.5159] | 982/2000 = 0.4910 [0.4691, 0.5129] | -0.0030 [-0.0340, 0.0280] (z -0.2) |

### E4, P_L, decoder hardware_noise t2='echo', both arms

| basis | R | DD on (E4A) | DD off (E4B) | off − on [Newcombe 95%] | 
|---|---|---|---|---|
| Z | 2 | 252/2000 = 0.1260 [0.1122, 0.1413] | 967/2000 = 0.4835 [0.4617, 0.5054] | +0.3575 [0.3308, 0.3834] (z +24.6) |
| Z | 3 | 531/2000 = 0.2655 [0.2466, 0.2853] | 846/2000 = 0.4230 [0.4015, 0.4448] | +0.1575 [0.1283, 0.1863] (z +10.5) |
| Z | 5 | 835/2000 = 0.4175 [0.3961, 0.4393] | 969/2000 = 0.4845 [0.4626, 0.5064] | +0.0670 [0.0362, 0.0977] (z +4.3) |
| Z | 8 | 1021/2000 = 0.5105 [0.4886, 0.5324] | 1000/2000 = 0.5000 [0.4781, 0.5219] | -0.0105 [-0.0414, 0.0205] (z -0.7) |
| X | 2 | 273/2000 = 0.1365 [0.1221, 0.1522] | 837/2000 = 0.4185 [0.3971, 0.4403] | +0.2820 [0.2554, 0.3081] (z +19.9) |
| X | 3 | 527/2000 = 0.2635 [0.2447, 0.2832] | 896/2000 = 0.4480 [0.4263, 0.4699] | +0.1845 [0.1552, 0.2134] (z +12.2) |
| X | 5 | 807/2000 = 0.4035 [0.3822, 0.4252] | 1013/2000 = 0.5065 [0.4846, 0.5284] | +0.1030 [0.0722, 0.1335] (z +6.5) |
| X | 8 | 954/2000 = 0.4770 [0.4552, 0.4989] | 979/2000 = 0.4895 [0.4676, 0.5114] | +0.0125 [-0.0185, 0.0434] (z +0.8) |

### E4, P_L, undecoded: the observable's raw flip rate

| basis | R | DD on (E4A) | DD off (E4B) | off − on [Newcombe 95%] | 
|---|---|---|---|---|
| Z | 2 | 504/2000 = 0.2520 [0.2335, 0.2715] | 1154/2000 = 0.5770 [0.5552, 0.5985] | +0.3250 [0.2958, 0.3534] (z +20.9) |
| Z | 3 | 743/2000 = 0.3715 [0.3506, 0.3929] | 1017/2000 = 0.5085 [0.4866, 0.5304] | +0.1370 [0.1064, 0.1673] (z +8.7) |
| Z | 5 | 885/2000 = 0.4425 [0.4209, 0.4644] | 999/2000 = 0.4995 [0.4776, 0.5214] | +0.0570 [0.0261, 0.0878] (z +3.6) |
| Z | 8 | 1011/2000 = 0.5055 [0.4836, 0.5274] | 1011/2000 = 0.5055 [0.4836, 0.5274] | +0.0000 [-0.0310, 0.0310] (z +0.0) |
| X | 2 | 570/2000 = 0.2850 [0.2656, 0.3052] | 1106/2000 = 0.5530 [0.5311, 0.5747] | +0.2680 [0.2382, 0.2971] (z +17.2) |
| X | 3 | 795/2000 = 0.3975 [0.3763, 0.4191] | 921/2000 = 0.4605 [0.4388, 0.4824] | +0.0630 [0.0323, 0.0935] (z +4.0) |
| X | 5 | 948/2000 = 0.4740 [0.4522, 0.4959] | 986/2000 = 0.4930 [0.4711, 0.5149] | +0.0190 [-0.0120, 0.0499] (z +1.2) |
| X | 8 | 988/2000 = 0.4940 [0.4721, 0.5159] | 982/2000 = 0.4910 [0.4691, 0.5129] | -0.0030 [-0.0340, 0.0280] (z -0.2) |

### E4, detection fraction (decoder-free)

| basis | R | on | off | off − on | per job | per round, on | per round, off |
|---|---|---|---|---|---|---|---|
| Z | 2 | 0.2102 | 0.2488 | +0.0386 | E4A 0.2102, E4B 0.2488 | 0.102, 0.296, 0.147 | 0.092, 0.418, 0.068 |
| Z | 3 | 0.2486 | 0.3067 | +0.0581 | E4A 0.2486, E4B 0.3067 | 0.101, 0.291, 0.326, 0.158 | 0.090, 0.410, 0.407, 0.117 |
| Z | 5 | 0.2801 | 0.3554 | +0.0754 | E4A 0.2801, E4B 0.3554 | 0.107, 0.297, 0.321, 0.318, 0.330, 0.161 | 0.088, 0.407, 0.419, 0.422, 0.424, 0.123 |
| Z | 8 | 0.3008 | 0.3825 | +0.0817 | E4A 0.3008, E4B 0.3825 | 0.098, 0.296, 0.332, 0.328, 0.325, 0.326, 0.327, 0.341, 0.167 | 0.095, 0.410, 0.413, 0.419, 0.421, 0.422, 0.427, 0.432, 0.138 |
| X | 2 | 0.1904 | 0.2286 | +0.0382 | E4A 0.1904, E4B 0.2286 | 0.053, 0.296, 0.117 | 0.056, 0.398, 0.062 |
| X | 3 | 0.2352 | 0.2973 | +0.0621 | E4A 0.2352, E4B 0.2973 | 0.056, 0.293, 0.320, 0.131 | 0.058, 0.401, 0.408, 0.108 |
| X | 5 | 0.2715 | 0.3427 | +0.0712 | E4A 0.2715, E4B 0.3427 | 0.058, 0.285, 0.316, 0.328, 0.330, 0.138 | 0.055, 0.395, 0.404, 0.406, 0.420, 0.121 |
| X | 8 | 0.2965 | 0.3778 | +0.0812 | E4A 0.2965, E4B 0.3778 | 0.060, 0.297, 0.323, 0.319, 0.327, 0.337, 0.326, 0.340, 0.147 | 0.057, 0.395, 0.408, 0.425, 0.426, 0.420, 0.420, 0.430, 0.140 |

### E4, ε_L per round (2P = 1 − A(1 − 2ε)^R, maximum likelihood, bootstrap 95%)

| basis | decoder | ε on (A=1) | ε off (A=1) | ε off − on [bootstrap 95%] | ε on (free A), A | ε off (free A), A |
|---|---|---|---|---|---|---|
| Z | ramsey | 0.1291 [0.1228, 0.1355] | 0.3529 [0.3135, 0.5000] | +0.2239 [0.1844, 0.3748] | 0.1291, 1.000 | 0.5000, 0.000 |
| Z | echo | 0.1147 [0.1094, 0.1202] | 0.3484 [0.3112, 0.5000] | +0.2337 [0.1961, 0.3867] | 0.1147, 1.000 | 0.0928, 0.129 |
| X | ramsey | 0.1150 [0.1096, 0.1208] | 0.2850 [0.2655, 0.3078] | +0.1701 [0.1486, 0.1945] | 0.1150, 1.000 | 0.2061, 0.488 |
| X | echo | 0.1111 [0.1056, 0.1167] | 0.2897 [0.2695, 0.3118] | +0.1786 [0.1574, 0.2001] | 0.1111, 1.000 | 0.2541, 0.700 |

### E1, per stabilizer: `E1_d3_Z_r15_eA` (decoder-free; data frames through the readout window)

| stabilizer | ancilla | data (frame) | on (A1+A2) | off (B1+B2) | off − on | batch, DD off | model Ramsey, 360 ns fallback (as run) | model echo, 360 ns fallback (as run) |
|---|---|---|---|---|---|---|---|---|
| Z(0,4) | QB19 | QB20 I, QB27 I | 0.195 | 0.124 | -0.071 | 0.124 | 0.132 | 0.117 |
| Z(2,2) | QB21 | QB13 H, QB22 H, QB20 I, QB29 I | 0.196 | 0.177 | -0.018 | 0.188 | 0.136 | 0.109 |
| Z(4,4) | QB37 | QB29 I, QB38 H, QB36 H, QB45 I | 0.238 | 0.236 | -0.002 | 0.245 | 0.172 | 0.144 |
| Z(6,2) | QB39 | QB31 H, QB38 H | 0.161 | 0.273 | +0.113 | 0.281 | 0.086 | 0.072 |
| X(2,0) | QB14 | QB13 H, QB22 H | 0.138 | 0.127 | -0.011 | 0.142 | 0.085 | 0.068 |
| X(4,2) | QB30 | QB22 H, QB31 H, QB29 I, QB38 H | 0.227 | 0.225 | -0.002 | 0.244 | 0.145 | 0.120 |
| X(2,4) | QB28 | QB20 I, QB29 I, QB27 I, QB36 H | 0.177 | 0.184 | +0.007 | 0.197 | 0.167 | 0.129 |
| X(4,6) | QB44 | QB36 H, QB45 I | 0.150 | 0.139 | -0.012 | 0.170 | 0.105 | 0.081 |

Batch jobs: b1 `01a17093-32c7-7062-9e4a-bf29b6b41113`, b2 `01a17096-f4c6-74d9-8807-4c75dec770c4`, b3 `01a170a0-97f8-74c6-ad4a-b93ff353b123`. Frame I: the identity; H: a Hadamard frame, where a physical Z acts as Stim's X.
On Emerald's compiled 1,036 ns window (main `53b2a19`), the model gives Z(6,2) 0.131 Ramsey and 0.105 echo here (the emerald-dd review's re-run, 20,000 shots, seed 2026); the other rows were not re-run.

### E1, per stabilizer: `E1_d3_X_r15_eA` (decoder-free; data frames through the readout window)

| stabilizer | ancilla | data (frame) | on (A1+A2) | off (B1+B2) | off − on | batch, DD off | model Ramsey, 360 ns fallback (as run) | model echo, 360 ns fallback (as run) |
|---|---|---|---|---|---|---|---|---|
| X(2,0) | QB14 | QB13 H, QB22 H | 0.121 | 0.117 | -0.004 | 0.115 | 0.079 | 0.064 |
| X(2,4) | QB28 | QB20 I, QB29 I, QB27 I, QB36 H | 0.165 | 0.180 | +0.015 | 0.184 | 0.157 | 0.123 |
| X(4,2) | QB30 | QB22 H, QB31 H, QB29 I, QB38 H | 0.206 | 0.204 | -0.002 | 0.213 | 0.136 | 0.113 |
| X(4,6) | QB44 | QB36 H, QB45 I | 0.111 | 0.110 | -0.000 | 0.111 | 0.099 | 0.078 |
| Z(2,2) | QB21 | QB13 H, QB22 H, QB20 I, QB29 I | 0.208 | 0.196 | -0.012 | 0.192 | 0.145 | 0.115 |
| Z(6,2) | QB39 | QB31 H, QB38 H | 0.172 | 0.302 | +0.130 | 0.311 | 0.092 | 0.077 |
| Z(0,4) | QB19 | QB20 I, QB27 I | 0.196 | 0.130 | -0.066 | 0.136 | 0.140 | 0.123 |
| Z(4,4) | QB37 | QB29 I, QB38 H, QB36 H, QB45 I | 0.256 | 0.261 | +0.004 | 0.263 | 0.184 | 0.152 |

Batch jobs: b1 `01a17093-aba4-773f-8755-4727f0c943ee`, b2 `01a17097-6a9d-70e2-9ec4-6b8669dce3d4`, b3 `01a170a1-17bf-77ad-a986-fadac8637f39`. Frame I: the identity; H: a Hadamard frame, where a physical Z acts as Stim's X.
On Emerald's compiled 1,036 ns window (main `53b2a19`), the model gives Z(6,2) 0.142 Ramsey and 0.113 echo here (the emerald-dd review's re-run, 20,000 shots, seed 2026); the other rows were not re-run.

### E1, deterministic first readouts: misread rate, off − on [Newcombe 95%]

| circuit | ancilla: on / off / off − on | ancilla: on / off / off − on | ancilla: on / off / off − on | ancilla: on / off / off − on |
|---|---|---|---|---|
| E1_d3_Z_r03_eA | QB21: 0.063 / 0.054 / -0.009 [-0.019, 0.002] | QB39: 0.103 / 0.107 / +0.004 [-0.010, 0.017] | QB19: 0.140 / 0.070 / -0.070 [-0.083, -0.056] | QB37: 0.116 / 0.114 / -0.003 [-0.017, 0.011] |
| E1_d3_X_r03_eA | QB14: 0.035 / 0.030 / -0.004 [-0.012, 0.004] | QB30: 0.050 / 0.052 / +0.002 [-0.007, 0.012] | QB28: 0.078 / 0.077 / -0.001 [-0.013, 0.011] | QB44: 0.058 / 0.058 / +0.000 [-0.010, 0.011] |
| E1_d3_Z_r08_eA | QB21: 0.063 / 0.058 / -0.005 [-0.015, 0.006] | QB39: 0.100 / 0.104 / +0.004 [-0.009, 0.018] | QB19: 0.132 / 0.077 / -0.055 [-0.069, -0.042] | QB37: 0.110 / 0.110 / -0.000 [-0.014, 0.013] |
| E1_d3_X_r08_eA | QB14: 0.032 / 0.021 / -0.011 [-0.018, -0.004] | QB30: 0.049 / 0.052 / +0.003 [-0.006, 0.013] | QB28: 0.085 / 0.077 / -0.008 [-0.019, 0.004] | QB44: 0.057 / 0.057 / +0.000 [-0.010, 0.010] |
| E1_d3_Z_r15_eA | QB21: 0.054 / 0.061 / +0.007 [-0.003, 0.018] | QB39: 0.112 / 0.108 / -0.003 [-0.017, 0.010] | QB19: 0.134 / 0.077 / -0.056 [-0.069, -0.043] | QB37: 0.114 / 0.115 / +0.001 [-0.013, 0.015] |
| E1_d3_X_r15_eA | QB14: 0.029 / 0.030 / +0.001 [-0.006, 0.009] | QB30: 0.044 / 0.049 / +0.005 [-0.004, 0.015] | QB28: 0.077 / 0.085 / +0.007 [-0.005, 0.019] | QB44: 0.048 / 0.058 / +0.010 [0.000, 0.020] |

### E4, per stabilizer: `E4_d3_Z_r03_eA_t05000ns` (decoder-free; data frames through the readout window)

| stabilizer | ancilla | data (frame) | on (E4A) | off (E4B) | off − on | batch, DD off | model Ramsey, 360 ns fallback (as run) | model echo, 360 ns fallback (as run) |
|---|---|---|---|---|---|---|---|---|
| Z(0,4) | QB19 | QB20 I, QB27 I | 0.246 | 0.136 | -0.110 | 0.122 | 0.215 | 0.179 |
| Z(2,2) | QB21 | QB13 H, QB22 H, QB20 I, QB29 I | 0.214 | 0.256 | +0.041 | 0.254 | 0.248 | 0.206 |
| Z(4,4) | QB37 | QB29 I, QB38 H, QB36 H, QB45 I | 0.282 | 0.310 | +0.028 | 0.303 | 0.309 | 0.277 |
| Z(6,2) | QB39 | QB31 H, QB38 H | 0.161 | 0.346 | +0.185 | 0.351 | 0.205 | 0.170 |
| X(2,0) | QB14 | QB13 H, QB22 H | 0.314 | 0.254 | -0.060 | 0.257 | 0.287 | 0.207 |
| X(4,2) | QB30 | QB22 H, QB31 H, QB29 I, QB38 H | 0.359 | 0.455 | +0.096 | 0.393 | 0.414 | 0.331 |
| X(2,4) | QB28 | QB20 I, QB29 I, QB27 I, QB36 H | 0.280 | 0.491 | +0.211 | 0.604 | 0.421 | 0.336 |
| X(4,6) | QB44 | QB36 H, QB45 I | 0.224 | 0.384 | +0.161 | 0.290 | 0.337 | 0.236 |

Batch jobs: b2 `01a17098-a0a2-7434-974d-2dc5b7093dc1`. Frame I: the identity; H: a Hadamard frame, where a physical Z acts as Stim's X.

### E4, per stabilizer: `E4_d3_X_r03_eA_t05000ns` (decoder-free; data frames through the readout window)

| stabilizer | ancilla | data (frame) | on (E4A) | off (E4B) | off − on | batch, DD off | model Ramsey, 360 ns fallback (as run) | model echo, 360 ns fallback (as run) |
|---|---|---|---|---|---|---|---|---|
| X(2,0) | QB14 | QB13 H, QB22 H | 0.196 | 0.133 | -0.063 | 0.129 | 0.172 | 0.130 |
| X(2,4) | QB28 | QB20 I, QB29 I, QB27 I, QB36 H | 0.182 | 0.295 | +0.113 | 0.342 | 0.262 | 0.213 |
| X(4,2) | QB30 | QB22 H, QB31 H, QB29 I, QB38 H | 0.236 | 0.288 | +0.052 | 0.230 | 0.259 | 0.212 |
| X(4,6) | QB44 | QB36 H, QB45 I | 0.143 | 0.217 | +0.074 | 0.147 | 0.206 | 0.155 |
| Z(2,2) | QB21 | QB13 H, QB22 H, QB20 I, QB29 I | 0.339 | 0.389 | +0.050 | 0.400 | 0.400 | 0.324 |
| Z(6,2) | QB39 | QB31 H, QB38 H | 0.246 | 0.595 | +0.349 | 0.596 | 0.327 | 0.259 |
| Z(0,4) | QB19 | QB20 I, QB27 I | 0.333 | 0.222 | -0.110 | 0.223 | 0.335 | 0.268 |
| Z(4,4) | QB37 | QB29 I, QB38 H, QB36 H, QB45 I | 0.391 | 0.493 | +0.102 | 0.471 | 0.454 | 0.400 |

Batch jobs: b2 `01a17098-fa08-7601-8144-ccab065c1733`. Frame I: the identity; H: a Hadamard frame, where a physical Z acts as Stim's X.

### E4, deterministic first readouts: misread rate, off − on [Newcombe 95%]

| circuit | ancilla: on / off / off − on | ancilla: on / off / off − on | ancilla: on / off / off − on | ancilla: on / off / off − on |
|---|---|---|---|---|
| E4_d3_Z_r02_eA_t05000ns | QB21: 0.050 / 0.066 / +0.015 [0.001, 0.030] | QB39: 0.112 / 0.106 / -0.006 [-0.025, 0.014] | QB19: 0.131 / 0.078 / -0.053 [-0.071, -0.034] | QB37: 0.118 / 0.118 / +0.001 [-0.020, 0.021] |
| E4_d3_X_r02_eA_t05000ns | QB14: 0.028 / 0.022 / -0.005 [-0.015, 0.005] | QB30: 0.042 / 0.056 / +0.014 [0.001, 0.027] | QB28: 0.082 / 0.084 / +0.002 [-0.015, 0.019] | QB44: 0.061 / 0.062 / +0.001 [-0.014, 0.016] |
| E4_d3_Z_r03_eA_t05000ns | QB21: 0.050 / 0.066 / +0.016 [0.001, 0.031] | QB39: 0.105 / 0.110 / +0.005 [-0.015, 0.024] | QB19: 0.134 / 0.074 / -0.060 [-0.079, -0.041] | QB37: 0.114 / 0.110 / -0.004 [-0.024, 0.016] |
| E4_d3_X_r03_eA_t05000ns | QB14: 0.029 / 0.021 / -0.008 [-0.017, 0.002] | QB30: 0.058 / 0.053 / -0.005 [-0.019, 0.010] | QB28: 0.085 / 0.086 / +0.002 [-0.015, 0.019] | QB44: 0.051 / 0.070 / +0.019 [0.004, 0.033] |
| E4_d3_Z_r05_eA_t05000ns | QB21: 0.066 / 0.061 / -0.005 [-0.020, 0.010] | QB39: 0.107 / 0.103 / -0.004 [-0.023, 0.016] | QB19: 0.141 / 0.076 / -0.065 [-0.085, -0.046] | QB37: 0.116 / 0.111 / -0.005 [-0.024, 0.015] |
| E4_d3_X_r05_eA_t05000ns | QB14: 0.035 / 0.024 / -0.011 [-0.022, -0.000] | QB30: 0.051 / 0.058 / +0.007 [-0.007, 0.021] | QB28: 0.083 / 0.082 / -0.001 [-0.018, 0.016] | QB44: 0.061 / 0.056 / -0.006 [-0.021, 0.009] |
| E4_d3_Z_r08_eA_t05000ns | QB21: 0.051 / 0.070 / +0.020 [0.005, 0.034] | QB39: 0.102 / 0.117 / +0.015 [-0.004, 0.034] | QB19: 0.130 / 0.068 / -0.062 [-0.080, -0.044] | QB37: 0.111 / 0.126 / +0.014 [-0.006, 0.035] |
| E4_d3_X_r08_eA_t05000ns | QB14: 0.029 / 0.026 / -0.003 [-0.014, 0.007] | QB30: 0.052 / 0.056 / +0.004 [-0.011, 0.018] | QB28: 0.082 / 0.085 / +0.003 [-0.014, 0.020] | QB44: 0.075 / 0.062 / -0.013 [-0.029, 0.002] |

### E1: what the priors' own model predicts (SIMULATED, 20,000 shots per circuit, seed 2026)

| circuit | noiseless events / flips | DEM errors | distance | P_L Ramsey (DD off) | P_L echo (DD on, ideal) | detection Ramsey / echo |
|---|---|---|---|---|---|---|
| E1_d3_Z_r03_eA | 0 / 0 | 294 | 3 | 1458/20000 = 0.0729 [0.0694, 0.0766] | 863/20000 = 0.0432 [0.0404, 0.0461] | 0.1084 / 0.0885 |
| E1_d3_X_r03_eA | 0 / 0 | 299 | 3 | 1466/20000 = 0.0733 [0.0698, 0.0770] | 933/20000 = 0.0466 [0.0438, 0.0497] | 0.1057 / 0.0860 |
| E1_d3_Z_r08_eA | 0 / 0 | 1044 | 3 | 4220/20000 = 0.2110 [0.2054, 0.2167] | 2692/20000 = 0.1346 [0.1299, 0.1394] | 0.1240 / 0.1016 |
| E1_d3_X_r08_eA | 0 / 0 | 1049 | 3 | 4082/20000 = 0.2041 [0.1986, 0.2097] | 2869/20000 = 0.1434 [0.1387, 0.1484] | 0.1231 / 0.1007 |
| E1_d3_Z_r15_eA | 0 / 0 | 2094 | 3 | 6727/20000 = 0.3363 [0.3298, 0.3429] | 4613/20000 = 0.2306 [0.2249, 0.2365] | 0.1286 / 0.1053 |
| E1_d3_X_r15_eA | 0 / 0 | 2099 | 3 | 6606/20000 = 0.3303 [0.3238, 0.3369] | 4833/20000 = 0.2417 [0.2358, 0.2476] | 0.1282 / 0.1048 |

### E4: what the priors' own model predicts (SIMULATED, 20,000 shots per circuit, seed 2026)

| circuit | noiseless events / flips | DEM errors | distance | P_L Ramsey (DD off) | P_L echo (DD on, ideal) | detection Ramsey / echo |
|---|---|---|---|---|---|---|
| E4_d3_Z_r02_eA_t05000ns | 0 / 0 | 149 | 3 | 5238/20000 = 0.2619 [0.2559, 0.2680] | 2564/20000 = 0.1282 [0.1236, 0.1329] | 0.2359 / 0.1899 |
| E4_d3_X_r02_eA_t05000ns | 0 / 0 | 149 | 3 | 5241/20000 = 0.2621 [0.2560, 0.2682] | 2691/20000 = 0.1346 [0.1299, 0.1393] | 0.2239 / 0.1774 |
| E4_d3_Z_r03_eA_t05000ns | 0 / 0 | 294 | 3 | 8142/20000 = 0.4071 [0.4003, 0.4139] | 5178/20000 = 0.2589 [0.2529, 0.2650] | 0.2844 / 0.2311 |
| E4_d3_X_r03_eA_t05000ns | 0 / 0 | 299 | 3 | 8039/20000 = 0.4019 [0.3952, 0.4088] | 5109/20000 = 0.2555 [0.2495, 0.2615] | 0.2762 / 0.2229 |
| E4_d3_Z_r05_eA_t05000ns | 0 / 0 | 594 | 3 | 9764/20000 = 0.4882 [0.4813, 0.4951] | 8024/20000 = 0.4012 [0.3944, 0.4080] | 0.3236 / 0.2640 |
| E4_d3_X_r05_eA_t05000ns | 0 / 0 | 599 | 3 | 9603/20000 = 0.4802 [0.4732, 0.4871] | 7808/20000 = 0.3904 [0.3837, 0.3972] | 0.3187 / 0.2592 |
| E4_d3_Z_r08_eA_t05000ns | 0 / 0 | 1044 | 3 | 9939/20000 = 0.4970 [0.4900, 0.5039] | 9472/20000 = 0.4736 [0.4667, 0.4805] | 0.3460 / 0.2823 |
| E4_d3_X_r08_eA_t05000ns | 0 / 0 | 1049 | 3 | 9885/20000 = 0.4943 [0.4873, 0.5012] | 9314/20000 = 0.4657 [0.4588, 0.4726] | 0.3431 / 0.2792 |

<!-- dd:end -->

## How to run it

```
uv sync --locked
uv run python scripts/emerald_dd.py plan                                    # offline: the segments against the brief
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
    uv run python scripts/emerald_dd.py predict                             # offline: structure + Stim's prediction
uv run python scripts/emerald_dd.py snapshot                                # live, read-only: an Emerald snapshot
uv run python scripts/emerald_dd.py verify --segment emerald-dd-pilot --snapshot <S> --live   # read-only requests
DQ_QPU=run DQ_QPU_TASK=emerald-dd-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \
    uv run python scripts/emerald_dd.py submit --segment emerald-dd-pilot --snapshot <S>
DQ_QPU=run DQ_QPU_TASK=emerald-dd DQ_QPU_MAX_JOBS=6 DQ_QPU_MAX_SHOTS=80000 \
    uv run python scripts/emerald_dd.py submit --segment emerald-dd --snapshot <S>
uv run python scripts/emerald_dd.py recover --segment <task> --job-id <J>   # read-only; was never needed
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
    uv run python scripts/emerald_dd.py analyze --snapshots <S> <T>         # offline: the records and the A/B
uv run python scripts/emerald_dd.py stabilizers                             # offline (30 s): per stabilizer, on vs off
uv run python scripts/emerald_dd.py doc                                     # offline: this file's tables
uv run pytest tests/test_emerald_dd.py                                      # offline
```

`<S>` = `data/calibration/emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261025T015024Z.json` (before),
`<T>` = `data/calibration/emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261025T015750Z.json` (after).
`snapshot` is `scripts/resonance_check.py`'s fetch and save for Emerald alone: that script's `main`
also writes a Garnet snapshot, under a path this task does not own. A reviewer re-pulls the jobs by id
with `hw.retrieve`, and re-derives the numbers here from the committed raw data with `analyze` into a
scratch `--root` (it never needs the network).

**Which window reproduces which column.** The records and every table here come from branch `emerald-dd`
(approved at `fec6e3e`), made before main had Emerald's compile file (`53b2a19`). So both decoders' P_L and ε_L,
the per-stabilizer model columns and the SIMULATED prediction tables use the 360 ns `measure_ns` fallback
window. On main, `hardware_noise`'s default `windows="auto"` finds that file and uses the compiled
1,036 ns window (E1's idle per readout layer 360 → 1,036 ns, E4's 5,360 → 6,036 ns). So `analyze`,
`predict` and `stabilizers` there decode and model at 1,036 ns, and those columns move. "The answer"
leads with the 1,036 ns Z(6,2) model figures (from the review's re-run) and gives the 360 ns ones beside
them; its R = 3 prediction (0.073 → 0.043 Z, 0.073 → 0.047 X) is at 360 ns. The decoder-free tables (the raw flips, the detection
fractions, the per-stabilizer on/off rates, the first-readout misreads) don't depend on the window and
come back unchanged. The driver has no switch for the window yet (`hardware_noise(..., windows=None)`
forces the fallback), so reproducing the 360 ns columns on main needs one.

## Files

| Path | What |
|---|---|
| `scripts/emerald_dd.py` | the driver: plan, predict, snapshot, verify, submit, recover, analyze, doc |
| `tests/test_emerald_dd.py` | offline: the segments, the options through IQM's own `create_run_request` and `hw.submit`, the checks, the guards, the statistics |
| `data/calibration/emerald/23e12f6b-….20261025T015024Z.json`, `…T015750Z.json` | the before and after snapshots |
| `InfoRecord/emerald/raw/20261025-emerald-dd-pilot/`, `…/20261025-emerald-dd/` | the raw data, exactly as `hw.save_raw` wrote it (`<job id>/meta.json`, `memory.npz`), each job's manifest (`<label>.json`: position → circuit, `native_sha256`, DD state, every check), `preflight.json`, `segment.json` |
| `InfoRecord/emerald/datasets/<exp>/dd_<job>_<on/off>_<circuit>/` | every run's dataset, labelled HARDWARE; the DD state in its manifest's `raw` block (`dd`, `compilation_options`, `job`) |
| `InfoRecord/emerald/derived/<exp>/dd_…/` | derived features, `summary.json` (with `dd`), and `dd_checks.json`: sanity, both decoders, per-round detection rates |
| `InfoRecord/emerald/ler/E1/dd_<basis>_eA_<on/off>/`, `…_<job>_<on/off>/`, `ler/E4/dd_<basis>_eA_t5000ns_<on/off>/` | LER records per arm (pooled) and per job |
| `InfoRecord/emerald/ler/E1/dd_ab/ab.json`, `ler/E4/dd_ab/ab.json` | the A/B: every cell of the tables above |
| `InfoRecord/emerald/ler/E1/dd_ab/by_stabilizer.json`, `ler/E4/dd_ab/by_stabilizer.json` | decoder-free, per circuit and stabilizer: on/off rates (pooled, per round, per job), the data qubits' frames, the batches' DD-off rate of the same circuit (`git show 47eb36a:...`), the priors' own model, and each deterministic first readout's misread rate |
| `InfoRecord/emerald/ler/E1/dd_ab/prediction.json`, `ler/E4/dd_ab/prediction.json` | SIMULATED: the circuits' noiseless and DEM checks, and Stim's DD-off (Ramsey) and DD-on (echo) prediction |

The DD state is in the manifest's `raw` block because `inforecord.MANIFEST` has no `compilation` key
yet (the design proposed one for whoever owns `inforecord.py`) and `check_run` refuses a key outside it.

## What is left

<!-- left:begin -->
- **Price the QB19 cost.** A DD run with `DDStrategy(target_qubits=...)` that leaves out QB19's neighbours
  (QB20, QB27), or QB19 itself, should keep Z(6,2)'s gain and drop Z(0,4)'s cost; if it does, the Z basis
  gains more than 16%.
- **E4 at 1 and 2 µs with DD,** to map the idle-length dependence of the systematic rotation.
- **Look at where the sequences land** in the compiled pulse schedule (a local Pulla compile, read-only),
  which this brief ruled out, and confirm EDD fills the readout windows of QB31 and QB38.
- **Decoder priors for DD shots.** Neither `hardware_noise` variant models Z(6,2)'s excess or QB19's
  readout; a per-qubit idle term fitted on a training split (never on these test-like A/B shots) is the
  next step.
- **`inforecord.MANIFEST`** could carry a `compilation` key (for whoever owns `inforecord.py`); the DD state
  is in each manifest's `raw` block until then.
- **Pin the readout window.** `analyze` should record and pin the window it decodes with (the emerald-dd
  review's MINOR 2), and `doc` should print the model columns' window: `scripts/emerald_dd.py doc` writes
  the plain "model Ramsey | model echo" header and drops the 1,036 ns notes above.
<!-- left:end -->
