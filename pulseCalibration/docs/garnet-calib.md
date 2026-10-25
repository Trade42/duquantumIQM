# garnet-calib: the coherent errors of Garnet's calibrated gates, measured at circuit level

**What this is.** A measurement plan, its analysis and its offline validation for one table: the coherent errors of the exact gates our d=3 surface code uses on IQM Garnet. Per qubit it gives the signed PRX amplitude error ε and the axis (Z) error η. Per coupler it gives the CZ conditional-phase error δ and the residual local Z phases φ. Every value comes with a 95% interval. The pilot patch is the 17 qubits (QB1, QB2, QB4–QB16, QB19, QB20) and 24 couplers of `E1_d3_Z_r03_pA`. Calibration set `a18a947f-0408-4628-81ea-1715637e434f`.

**What has run.** Nothing on hardware. No token, no job, no ledger row. What exists:
- the 761 placed and checked circuits, and the job plan for Sal's approval (`InfoRecord/characterization/`);
- the analysis code (`src/dqiqm/characterization/`);
- a SIMULATED closure test: the analysis recovers known injected errors on the stand-in.

The pulse-level half (Track B) is designed only.

## Run it

```
uv sync --locked
uv run pytest tests/test_calib_circuits.py tests/test_calib_fit.py          # offline, ~10 s
uv run python scripts/garnet_calib_build.py                                  # the circuits, job_plan.json, pulse_plan.json (~20 s)
uv run python scripts/garnet_calib_build.py --check                          # a rebuild reproduces every committed sha256
uv run python scripts/garnet_calib_build.py --plan-md                        # the plan table below
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
  uv run python scripts/garnet_calib_closure.py --clean                      # the closure test, ~20 min, seeded
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
  uv run python scripts/garnet_calib_compensate.py                           # the compensated E1 twin, ~3 min
uv run python scripts/garnet_calib_report.py                                 # the closure's numbers as markdown (read-only)
```

| Measured (will be, on hardware) | Designed (this task) |
|---|---|
| ε, η per qubit (C1); δ, φ per coupler (C2) | the circuits, the model, the fits, the plan, the pilot's stop/go |
| C4 differences (isolated against parallel) | C3 leakage (blocked), Track B pulse sweeps (blocked) |
| — | the closure test: SIMULATED recovery of injected ε, η, δ, φ |

## The access boundary

| What | Can we? | How we know |
|---|---|---|
| Run circuits of native gates (`prx`, `cz`, `measure`, `barrier`, `delay`) | **Yes** | `dqiqm.hw.submit`, guarded: armed per task, within an allowance, a ledger line per job |
| Circuit compilation options (DD, heralding, active reset) | **Yes**, through `hw.submit(..., circuit_compilation_options=...)` | pulla-check; `hw.py` does not yet record them (below) |
| Read pulse-level parameters, compile locally with IQM's compiler | **Yes** | pulla-check, 21:51 UTC: `Pulla(...)` works on garnet and emerald |
| Submit a pulse-level job (`Pulla.submit_playlist`) | **Unknown** | never tried. The challenge says PulLA needs a paid organisation. A question for the mentors |
| Submit `prx_12` / `measure_012` | **No**, through our path | `prx_12` is a native op of iqm-client 35.0.3, but the qiskit serializer has no gate for it. `measure_012` is not an op of 35.0.3 at all |
| Change IQM's calibration | **Never** | no call writes a calibration set; any pulse change would live in one job's run definition |

So this task has two tracks:
- **A, circuit level, executable now:** C1, C2 and C4, measuring the existing calibrated gates.
- **B, pulse level, designed:** sweeps of the saved per-locus settings. They run once submission is allowed and a guarded `hw.submit_playlist` exists.

## The patch

`characterization.patch.pilot_patch` reads the patch from the pilot circuit itself, not by hand. It rebuilds `E1_d3_Z_r03_pA` (`memory.spec` on `InfoRecord/circuits/E1/placement.json`), checks its `native_sha256` (`9c6dba2b…`), and reads the CZ groups between barriers. The round has **4 CZ layers of 6 couplers**, the same 4 in every round: 24 distinct couplers, each a (data, ancilla) pair.

| Layer | Couplers (data–ancilla) |
|---|---|
| 1 | QB4–QB5, QB6–QB7, QB8–QB9, QB10–QB11, QB14–QB15, QB19–QB20 |
| 2 | QB4–QB9, QB6–QB11, QB10–QB5, QB12–QB7, QB16–QB20, QB19–QB15 |
| 3 | QB2–QB5, QB4–QB1, QB8–QB13, QB10–QB15, QB14–QB9, QB16–QB11 |
| 4 | QB2–QB1, QB6–QB5, QB10–QB9, QB12–QB11, QB14–QB13, QB16–QB15 |

Ancillas: QB1, QB5, QB7, QB9, QB11, QB13, QB15, QB20. In IQM's calibration, every one of these CZs has its qubit flux pulse on the **data** qubit (the first qubit of each locus, for example `QB4__QB1`).

## The model

**One `prx(π/2, φ)`.** The modelled device gate is

```
G(φ) = RZ(φ) · exp(−i/2 [(π/2 + ε) X + η Z]) · RZ(−φ)
     = exp(−i/2 [(π/2 + ε)(cos φ X + sin φ Y) + η Z])
```

- **ε** is the signed amplitude error in rad; ε > 0 over-rotates.
- **η** is the Z component of the rotation vector in rad. It is the axis tilt out of the equator, from a detuning, a DRAG error or a miscalibrated `rz_before`/`rz_after`.
- IQM applies φ as a frame change, which is exact.
- **The in-plane axis offset is a gauge.** A constant phase offset on all of one qubit's gates is a frame change. It commutes with CZ and with Z readout, so no circuit that starts in |0⟩ can see it, and the model has no such parameter.
- So the "axis/phase error" of the table is η.

**One CZ on (a, b).** In the basis |ab⟩:

```
diag(1, e^{iφ_b}, e^{iφ_a}, −e^{i(φ_a + φ_b + δ)})
```

- **δ** is the conditional-phase error: the |11⟩ phase is π + δ after the local phases.
- **φ_a, φ_b** are the residual local Z phases left after IQM's own virtual-RZ corrections. A phase φ is what RZ(φ) ∝ diag(1, e^{iφ}) adds to |1⟩.
- Leakage and residual swap terms are not in the model (C3 is the leakage design).

**Why every gate is `prx(π/2)`.** With `drag_crf_sx`, `prx(π/2)` is one 16 ns pulse, but `prx(π)` and every other angle are two pulses (pulla-check). So an X is written as two `prx(π/2)`, and every repetition count is a pulse count.

## The experiments, and why each

Every circuit is layers of gates, with a **barrier on all 17 qubits after every layer**. Then all 17 qubits are measured, clbit j into column j (`column_qubits`). The circuits are placed at optimization level 0.

### C1, PRX error amplification, per qubit (Must; 32 circuits, all 17 qubits in parallel)

| Family | Sequence | N or n | What it isolates |
|---|---|---|---|
| `amp_x` | N × `prx(π/2, 0)` from \|0⟩ | 0, 1, 2, 3, 5, 7, 9, 11, 17, 19, 33, 35, 40, 41, 42, 43 | **ε**. Odd N sit on the equator, P1 ≈ (1 ± sin Nε)/2. N = 4k+1 and 4k+3 flip the sign, so an offset or a drift toward \|0⟩ cannot mimic ε. Even N sit on a pole: N = 0 and 2 anchor the readout; N = 40 and 42 show 40 pulses of decay (the no-merging sentinel) |
| `amp_y` | N × `prx(π/2, π/2)` | 17, 19, 41, 43 | the same at the surface code's own phase: a check that one ε fits both |
| `tilt` | `prx(π/2, π/2)`, then n × [`prx(π/2, 0)`, `prx(π/2, π)`], then `prx(π/2, 0)` (pos) or `prx(π/2, π)` (neg) | n = 0, 2, 4, 8, 12, 16 | **η**. The X, −X pair cancels ε to first order. With η ≠ 0, each pair is a rotation by about 1.8 η about (Y+Z)/√2, and that accumulates linearly. The analysis pulse reads the Y component, sin(α)/√2, with opposite signs for pos and neg |

**Why all 17 in parallel.** The code's PRX layers are parallel. Measuring in that context costs one job, against 17. C4 measures the isolated twins.

**Readout handling.** Every C1/C2 readout is terminal, and IQM compiles a terminal `measure` as **`measure_fidelity`** (pulla-check). The fit therefore keeps the readout errors a0 = P(1|0) and a1 = P(0|1) free per qubit, anchored by N = 0 and N = 2. The priors and the pilot's bands use the unheralded `metrics.ssro.measure_fidelity.constant.<q>.error_0_to_1 / error_1_to_0` (QB20: P(0|1) = 0.090). The stand-in applies `measure`'s errors instead, so the closure tests the fit, not this prior.

**The fit, per qubit.** It is a maximum-likelihood fit (binomial) of the exact Bloch-vector model, `characterization.model`:
- per gate: the rotation of G(φ), then T1/T2 relaxation over 16 ns (fixed from the calibration), then a depolarizing shrink λ;
- then the classical readout.

There are five free parameters (ε, η, λ, a0, a1). A grid of starts over ε and η comes first, so an aliased ε cannot win. The 95% interval is Wald's from the numerical Hessian, widened by √max(1, χ²/dof). A parameter on its bound (λ = 0 in a noiseless run) is held there.

### C2, CZ conditional phase, per coupler (Must; 256 circuits, one CZ layer at a time)

For each of a layer's 6 couplers in parallel, every circuit has these layers:
1. **Prep, layer 1:** the target gets `prx(π/2, π/2)`; the control gets `prx(π/2, 0)`.
2. **Prep, layer 2:** the control gets `prx(π/2, 0)` (s = 1: X90·X90 = X) or `prx(π/2, π)` (s = 0: X90·X−90 = I). Both control states then have the same pulses and the same timing.
3. **n CZs**, n ∈ {0, 1, 3, 9}, a barrier after each.
4. **The target's analysis `prx(π/2, ψ)`**, ψ ∈ {0, π/2, π, 3π/2}.

Both orientations run: the ancilla as target (`tanc`), then the data qubit (`tdat`). That is 4 layers × 2 orientations × 2 control states × 4 n × 4 ψ.

**The analysis, per (s, n).**
- **Phase.** sin Φ = P(π) − P(0) and cos Φ = P(π/2) − P(3π/2). These differences cancel every readout offset, and decay and readout scale only the contrast. So Φ = atan2(sin, cos) needs no readout model. Its variance comes from the delta method.
- **The line in n.** Φ_0(n) = A_0 + n φ_t, and Φ_1(n) − nπ = A_1 + n(φ_t + δ). It is fitted by weighted least squares, on phases unwrapped point by point in increasing n, each onto the branch the fit so far predicts, as robust phase estimation does.
- **Separate intercepts.** Each control state has its own intercept, so nothing constant in n can bias a slope: not the preparation, not the analysis pulse's own η, not crosstalk from the control's pulses.
- **Orientations.** Both measure δ; the table gives their inverse-variance mean, plus a consistency χ². φ of each qubit comes from the orientation where it is the target.
- **Repeated CZ at odd n amplifies.** δ and φ grow n-fold. n = 1 fixes the branch for the rest.

**Context.** In a layer, every other qubit of the layer is a prepared control or target, and the 5 qubits outside it idle in |0⟩. A static ZZ through an idle coupler between a target and a control in |1⟩ adds to that target's δ. That is the spectator phase of the code's own context, and C4 isolates it.

### C3, CZ leakage (Should; designed, blocked)

The design is in `characterization.leakage` and `circuits/C3/design.json`: 15 circuits per coupler, as IQM native instructions.
- **Leaked level.** The flux CZ leaks through the fluxed qubit's |11⟩ ↔ |20⟩ crossing, so the leaked population is the fluxed (data) qubit's |2⟩. That is inferred from the gate family, not measured.
- **The circuits.** Both qubits are prepared in |1⟩, then n ∈ {0, 1, 3, 9} CZs run. The fluxed qubit is read three ways: plain; after `prx_12(π)`, which swaps |1⟩ and |2⟩; and after an X, which swaps |0⟩ and |1⟩. Three references (|0⟩, |1⟩, and |2⟩ = X then `prx_12(π)`) give how the two-state discriminator reads each level.
- **The algebra.** `leakage.populations`, tested, gives (p0, p1, p2). The leakage per CZ is the slope of p2 against n.
- **Cross-check.** It works on all 24 couplers, since `prx_12` is calibrated on all 20 qubits. Where the fluxed qubit has `measure_012` (QB14 on this patch: QB14–QB9, QB14–QB13 and QB14–QB15), a direct three-level readout cross-checks it.
- **Why it is blocked.** `prx_12` cannot be sent through `hw.submit` (below).

### C4, isolated against parallel (Should; 469 circuits)

- **C1 on one qubit at a time** (the other 16 idle in |0⟩): `amp_x` N = 2, 41, 43, and `tilt` n = 16 pos/neg. That is 5 circuits × 17 qubits. Fit ε and η with λ, a0 and a1 held at the parallel fit's values (`fit_c1(..., fixed=...)`).
- **C2 on one coupler at a time** (target = ancilla): n ∈ {1, 9}, both control states, 4 ψ. That is 16 circuits × 24 couplers.
- **What the differences show.** C1 against C4 is the drive crosstalk of the parallel PRX layers. C2 against C4 is the spectator phase of the layer context. The analysis is the same functions with `experiment="C4"`. Not run in the closure.

### C5, drift subset (if time; 70 circuits)

The subset is C1 `amp_x` N = 0, 2, 41, 43 and `tilt` n = 16 ±, plus C2 n ∈ {1, 9}, target = ancilla, on all 4 layers. It is repeated at each batch boundary next to the E6 snapshots, keyed to the calibration set.

### Should, in the doc only: a DD on/off A/B on E1

- **Where the idle is.** pulla-check found the data qubits idling through every ~1.1 µs mid-circuit readout window as plain waits, which is where DD would go.
- **The A/B.** Run E1 Z and X, R = 3 and 8, on pA, in two adjacent jobs at 4,000 shots each: one with the server's default options, one with `CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=DDStrategy(...))` through `hw.submit(..., circuit_compilation_options=...)`.
- **Compare** the detection fraction (Wilson) and the decoded P_L per round.
- **Prerequisite:** `hw.py` must record the options (below).

## No gate merging (critical)

Qiskit merges repeated gates at optimization level ≥ 1: 5 or 21 `r(π/2)` become 1 (pulla-check). IQM's standard compiler merges nothing. Defence in depth:

1. **Barriers.** There is a barrier after every layer, so between any two gates on a qubit. `circuits.check` proves it on the serialized instructions, the ones the server would receive.
2. **Level 0, and `count_ops()` on the exact circuit handed to `hw.submit`,** equal to the design's counts. `placement.json` records both `ops` (the circuit) and `sent` (the serialized instructions). A test shows the check catching a level-1 merge and a barrier-free circuit.
3. **Before submission,** offline. The hw worker rebuilds with `scripts/garnet_calib_build.py --check` (every `native_sha256` equal) and validates against `hw.backend('garnet')`. With read access only, it can also compile the exact circuits with Pulla's standard compiler, as `scripts/pulla_check.py` does, and count N pulses of 16 ns per qubit and 2N flux pulses per CZ chain. That shows what IQM's compiler does with our circuits, not what the server ran.
4. **On the pilot's data: the stop/go.** A circuit job returns no compiled duration, only status timestamps, so the brief's "compiled duration against N × the gate duration" cannot be read back. A correct merge also leaves the ideal state unchanged. What a merge cannot fake is **error accumulation**:
   - **C1:** D = mean over the 17 qubits of P1(N = 2) − P1(N = 42), both ideally |1⟩, so D is the decay of 40 pulses. A merged chain gives D ≈ 0.
   - **C2:** R = the mean contrast ratio after 17 CZs against after 1 (layer 1, target = ancilla, s = 1; a pilot-only n = 17 chain). A merged or cancelled chain gives R ≈ 1.
   - The amplification signal N·ε is reported but is not a stop/go: a well-calibrated gate has ε ≈ 0, and then it looks the same merged or not.

## The closure test (SIMULATED)

Generated by `scripts/garnet_calib_report.py` from `InfoRecord/characterization/sim/` (closure run of 2026-10-24, 22:54–23:15 UTC, under the sim lock, ~21 min).

**Regeneration** with `--clean` reproduces every number; the corrected noiseless check runs as part of it, and `--recheck-noiseless` redoes only that check from the saved records. The bytes differ only in time fields: `timestamp_utc` in the table rows, `saved_utc` in each `meta.json`, `seconds_simulating`, and the log.

**The path.** Each world runs the plan's Must jobs at the plan's shots, the real way: the placed and checked circuits, their twins with the injected errors (exact unitaries, `characterization.inject`), Aer, `hw.save_raw` into `sim/raw/`, and `characterization.analysis` reading `memory.npz` back.

All SIMULATED: stand-in `IQMFakeGarnet+profile(a18a947f)`, calibration set `a18a947f-0408-4628-81ea-1715637e434f`, Aer MPS, seeded (2026 + crc32(job label) % 1e6; errors 1000 + replica). Jobs: C1 32 x 1000, C2-L1 64 x 500, C2-L2 64 x 500, C2-L3 64 x 500, C2-L4 64 x 500. Worlds: noiseless 1, calibrated noise with nothing injected 5, injected 10 (per qubit eps, eta uniform in +-0.03 rad; per coupler delta, phi_a, phi_b in +-0.08 rad, a fresh draw per replica).

**Coverage of the injected values** (95% intervals; if they are right, the covered count falls in the binomial range about 95% of the time):

| parameter | intervals | hold the injected value | fraction | binomial 95% range of the count | median se (rad) | rms of (estimate - injected)/se |
|---|---|---|---|---|---|---|
| eps | 170 | 162 | 0.953 | 156–167 | 0.00039 | 1.00 |
| eta | 170 | 164 | 0.965 | 156–167 | 0.00097 | 0.89 |
| delta | 240 | 229 | 0.954 | 221–235 | 0.0052 | 0.96 |
| phi | 480 | 467 | 0.973 | 447–465 | 0.0052 | 0.90 |

**Nothing injected** (every estimate's truth is 0):

| world | parameter | estimates | intervals holding 0 | largest \|estimate\| (rad) | largest \|estimate\|/se |
|---|---|---|---|---|---|
| noiseless | eps | 17 | 17 | 0.00061 | 1.16 |
| noiseless | eta | 17 | 17 | 0.0019 | 1.62 |
| noiseless | delta | 24 | 23 | 0.012 | 2.44 |
| noiseless | phi | 48 | 46 | 0.011 | 2.41 |
| none | eps | 85 | 83 | 0.001 | 2.47 |
| none | eta | 85 | 81 | 0.0024 | 2.75 |
| none | delta | 120 | 112 | 0.015 | 2.51 |
| none | phi | 240 | 229 | 0.017 | 3.19 |

Per-locus bias check over the 5 nothing-injected noisy replicas: for each (locus, parameter), sqrt(n) x mean(estimate/se) is N(0, 1) if unbiased. Over 106 (locus, parameter) pairs the largest are QB14 eps 2.21, QB1__QB4 phi_QB4 2.15, QB2 eps 2.04.

Noiseless world: all 3652 deterministic columns (ideal P1 = 0 or 1) are exact, and the 1244 random ones scatter as binomial: z = (P_hat - p)/sqrt(p(1 - p)/shots) has mean +0.027, standard deviation 1.000 and max |z| 4.02.

**The pilot, dry** (its 13 circuits on the stand-in, 1,000 shots):

| record | D (C1 no-merge) | R (C2 no-merge) | readout | verdict |
|---|---|---|---|---|
| stand-in | 0.0491 | 0.8119 | ok | **GO** |
| mock merged | 0.0000 | 1.0000 | ok | **STOP** |

**The compensated E1 twin** (noise-free Aer statevector, 2000 shots, the injected world r01's local phases after every CZ; the C2-measured phases differ from the injected by at most 0.0130 rad):

Cells: shots with any detection event, with its Wilson 95% interval as a fraction, and observable flips.

| circuit | ideal | with the phases | compensated (true phases) | compensated (measured phases) |
|---|---|---|---|---|
| E1_d3_Z_r03_pA | 0/2000 shots with events [0.0000, 0.0019], 0 flips | 94/2000 shots with events [0.0386, 0.0572], 14 flips | 0/2000 shots with events [0.0000, 0.0019], 0 flips | 2/2000 shots with events [0.0003, 0.0036], 1 flips |
| E1_d3_X_r03_pA | 0/2000 shots with events [0.0000, 0.0019], 0 flips | 113/2000 shots with events [0.0472, 0.0675], 7 flips | 0/2000 shots with events [0.0000, 0.0019], 0 flips | 0/2000 shots with events [0.0000, 0.0019], 0 flips |

**Reading it:**
- The intervals have their nominal 95% coverage for ε, η and δ. For φ they are slightly conservative (467/480, just above the binomial range; rms z 0.90), from the √(χ²/dof) widening at C2's 4 degrees of freedom.
- **Resolution per locus at the plan's shots:** ε about 0.0004 rad, η about 0.001 rad, δ and φ about 0.005 rad.
- With nothing injected, no locus is biased.
- The pilot's bands pass the stand-in and stop the mock merged record.
- **The compensated twin cancels the measured local phases.** Phase tracking with the injected phases gives back the ideal circuit's zero events exactly. With C2's estimates, which are within 0.013 rad of the truth, it leaves 2 of 2,000 shots with events, against 94 without compensation (Z basis).

## The table

`characterization.table.COLUMNS` holds the CSV schema: the runbook's Section 8 columns, in its order, plus `context`, `parameter`, `se`, `chi2_per_dof`, `simulated` and `injected`.

```
timestamp_utc, device, calibration_set_id, locus, gate, implementation, experiment, context, parameter,
shots, seed, fit_value, fit_unit, fit_ci_low, fit_ci_high, se, chi2_per_dof, leakage, rb_error,
simultaneous_rb_error, t1_us, t2star_us, change_status, simulated, injected, notes
```

The rows:
- **Per qubit:** `eps` and `eta` (`gate` prx, `implementation` drag_crf_sx), with λ, a0 and a1 in `notes`.
- **Per coupler** (`QBa__QBb`): `delta`, plus `phi_<q>` for each qubit.
- **Units:** angles in rad.
- **Copied from the calibration** for comparison: `rb_error` (the PRX RB infidelity, or the CZ IRB infidelity), `t1_us` and `t2star_us`.
- **`leakage`** stays empty until C3 can run.

Hardware rows will come from `characterization.analysis` reading `InfoRecord/characterization/raw/<day>-garnet-calib/<job id>/memory.npz`: the code path the closure uses.

## The hardware plan

`InfoRecord/characterization/job_plan.json`, generated by `scripts/garnet_calib_build.py`. It runs only after Sal's yes, through `dqiqm.hw.submit`, one hardware worker at a time.

- **Estimated QPU time** = shots × (circuit duration + `reset_wait` 298 µs) + circuits × 0.1 s.
- **Circuit duration** = the gate layers (16 ns per PRX layer, 40 ns per CZ layer) + the terminal readout window of 1,036 ns (`measure_fidelity`, compiled, the slowest on the patch).
- **The "pessimistic" column** charges 1.92 ms per shot (Garnet's published CLOPS), as garnet-design's plan does.

| job | phase | circuits | shots / circuit | total shots | circuit (µs), min–max | est. QPU (s) | pessimistic (s) | what |
|---|---|---|---|---|---|---|---|---|
| P0 | pilot | 13 | 1,000 | 13,000 | 1.04–1.76 | **5.2** | 26 | pilot: readout anchors, the C1 and C2 no-merging sentinels, ε·N and the C2 phase; stop/go before C1 |
| C1 | must | 32 | 1,000 | 32,000 | 1.04–1.72 | **12.8** | 65 | PRX amplification, all 17 qubits in parallel |
| C2-L1 | must | 64 | 500 | 32,000 | 1.08–1.44 | **16.0** | 68 | conditional Ramsey, CZ layer 1's 6 couplers in parallel, both orientations |
| C2-L2 | must | 64 | 500 | 32,000 | 1.08–1.44 | **16.0** | 68 | conditional Ramsey, CZ layer 2's 6 couplers in parallel, both orientations |
| C2-L3 | must | 64 | 500 | 32,000 | 1.08–1.44 | **16.0** | 68 | conditional Ramsey, CZ layer 3's 6 couplers in parallel, both orientations |
| C2-L4 | must | 64 | 500 | 32,000 | 1.08–1.44 | **16.0** | 68 | conditional Ramsey, CZ layer 4's 6 couplers in parallel, both orientations |
| C4-C1 | should | 85 | 500 | 42,500 | 1.07–1.72 | **21.2** | 90 | C1 isolated: one qubit driven at a time |
| C4-C2-L1 | should | 96 | 250 | 24,000 | 1.12–1.44 | **16.8** | 56 | C2 isolated: layer 1's couplers one at a time, target = ancilla |
| C4-C2-L2 | should | 96 | 250 | 24,000 | 1.12–1.44 | **16.8** | 56 | C2 isolated: layer 2's couplers one at a time, target = ancilla |
| C4-C2-L3 | should | 96 | 250 | 24,000 | 1.12–1.44 | **16.8** | 56 | C2 isolated: layer 3's couplers one at a time, target = ancilla |
| C4-C2-L4 | should | 96 | 250 | 24,000 | 1.12–1.44 | **16.8** | 56 | C2 isolated: layer 4's couplers one at a time, target = ancilla |
| C5 | if time | 70 | 500 | 35,000 | 1.04–1.72 | **17.5** | 74 | drift subset: repeat at each batch boundary, next to the E6 snapshots |
| **pilot** | | 13 | | **13,000** | | **5.2** | 26 | 1 job(s) |
| **must** | | 288 | | **160,000** | | **76.7** | 337 | 5 job(s) |
| **should** | | 469 | | **138,500** | | **88.4** | 313 | 5 job(s) |
| **if time** | | 70 | | **35,000** | | **17.5** | 74 | 1 job(s) |

**The pilot (P0), before anything else.** It is 1 job: 13 circuits × 1,000 shots, 5.2 s estimated.
- **Circuits:** C1 `amp_x` N = 0, 2, 41, 42, 43, and the C2 sentinel (layer 1, target = ancilla, s = 1, n = 1 and 17, 4 ψ each).
- **Bands:** in `job_plan.json: pilot`, computed from the calibration's priors. **Stop and report if any check fails; never spend more shots chasing it.**

| Check | Band | Prediction | If merged |
|---|---|---|---|
| shape | 13 × (1000, 17) arrays, in the plan's order; `calibration_set_id` recorded | | |
| readout | every qubit: P1(N=0) ≤ 0.10 and P1(N=2) ≥ 0.75 | | |
| C1 no-merge | **D ≥ 0.0108** | 0.0216 from T1/T2 alone, 0.052 with the RB error; σ = 0.0025 pooled | D ≈ 0 |
| C2 no-merge | **R ≤ 0.943** | 0.886 from the IRB error alone; σ = 0.013 pooled | R ≈ 1 |
| info only | ε from N = 41, 43; the C2 phase per CZ; the timeline's per-shot overhead (re-estimate the plan) | | |

Each no-merge band sits about 4σ from both the prediction and the merged value. The closure's pilot dry run checks the bands on the stand-in and on a mock merged record (results above).

**Order after the pilot:** C1, then C2-L1..L4 (Must). Then C4 and C5 if allowed. Rebuild from the calibration set current at submission: a new set changes the priors, and the bands with them.

## What `hw.py` and `layout.py` need (not edited here)

**For C1, C2, C4 and C5: nothing.** `hw.submit(circuits, device="garnet", shots=…, qubit_index_to_name=…, calibration_set_id=…, purpose=…)` takes the placed circuits as they are. `layout.py` is only imported (`sent_instructions`, `Architecture`); the characterization has its own check, because these circuits have no Stim source.

**For the DD A/B (Should):** `hw.submit` already forwards `circuit_compilation_options` to `create_run_request`, but it records them nowhere. It needs to:
1. write the options (`dd_mode`, `dd_strategy`, `heralding_mode`, `active_reset_cycles`, `max_circuit_duration_over_t2`) into `meta.json`;
2. append a short form of them to the ledger row's purpose.

Without that, a DD-on job and a DD-off job are indistinguishable in the record.

**For C3 (leakage):**
1. **A native-circuit path.** `hw.submit_native(circuits: list[iqm.pulse.Circuit], ...)` would build `IQMClient.create_run_request` from `CircuitOperation` lists. iqm-client 35.0.3 accepts `prx_12` (in its `_SUPPORTED_OPERATIONS`; the DQA has `modulated_drag_crf` on all 20 qubits). It needs the same arming, allowance, ledger line and raw saving as `hw.submit`, with `save_raw` reading per-key measurement arrays rather than a qiskit `Result`.
   - **Why not qiskit.** The serializer's `allowed_nonnative_gates` would emit a custom `prx_12` gate with arguments `p0`/`p1`, not `angle`/`phase`, which the client's validation rejects.
2. **`measure_012`** is not an operation of iqm-client 35.0.3 at all. It needs a newer client (a pin move, against the challenge's `requirements.txt`) and a ternary memory format, since `hw._memory_bits` assumes '0'/'1'. Until then, the `prx_12` swap readout is the route.
3. **`layout.py`.** `check_hardware` allows only {prx, cz, measure, barrier}. A native C3 check would allow `prx_12`. `characterization.circuits.check` could be widened instead.

**For Track B:** a guarded `hw.submit_playlist`, below.

## Track B: the pulse level (PulLA), designed only

`InfoRecord/characterization/pulse_plan.json` (`characterization.pulse.plan`) gives, per qubit and per coupler, the saved centre of every setting and its sweep range, from pulla-check's `data/calibration/garnet/pulse/a18a947f-….json`.

**How a sweep runs** (pulla-check's findings):
1. `compiler = pulla.get_standard_compiler()` and `settings = compiler.get_settings(circuits=...)`.
2. Edit one field, for example `settings.gates.prx.drag_crf_sx.QB2.amplitude_i`, or sweep it with `exa.common.control.sweep.sweep.Sweep`.
3. `compiler.compile(circuits, settings=settings)`, then submit through the guarded wrapper.

The probe circuits are the C1/C2 circuits, so every sweep point becomes an ε, η, δ or φ with an interval. **The change lives in that job only.**

| Exp | Setting | Range (centred on the saved value) | Points × shots | Probe | Fit |
|---|---|---|---|---|---|
| B1 coarse | `gates.prx.drag_crf_sx.<q>.amplitude_i` | ±8% (patch: 0.104–0.288) | 17 × 500 | C1 `amp_x` N = 5, 7 | ε(A) = (π/2)(A/A_ref − 1) + ε₀; the zero. N = 5, 7 keep Nε within ±0.9 rad, so nothing aliases |
| B1 fine | the same | ±1% around the coarse zero | 9 × 1000 | C1 `amp_x` N = 41, 43 | the same, about 8× more sensitive |
| B2 DRAG | `….amplitude_q` | ±50% (Q/I stays within 1.5× the patch's −0.16…−0.22) | 11 × 500 | C1 `tilt` n = 16 ± | the zero of η. Leakage is the real DRAG objective, and needs C3's readout |
| B3 detuning | `controllers.<q>.drive.frequency` | ±200 kHz cap | — | Ramsey with virtual detuning, a *circuit* job: `prx(π/2,0)`, `delay(t)`, `prx(π/2, 2π f_v t)`, t up to 4 µs | fringe − f_v = the detuning; set once |
| B4 virtual phase | `….rz_before` = `….rz_after` | ±0.1 rad (virtual: no hardware risk) | 11 × 500 | C1 `tilt` n = 16 ± | the zero of η. Degenerate with B2 in C1, so B2 stays at its saved value meanwhile |
| B5 conditional phase | `gates.cz.slepian_crf.<QBa__QBb>.coupler.amplitude` | ±4% | 9 × 500 | C2 layer, `tanc`, n ∈ {1, 9}, s ∈ {0, 1}, 4 ψ | the zero of δ; the n = 9 contrast and the control's P1 flag leakage |
| B6 qubit flux | `….qubit.amplitude` | ±4% | 9 × 500 | as B5 | as B5, after it |
| B7 local phase | `….rz.<q>`, both qubits | ±0.2 rad (virtual) | 1 verification | C2 both orientations | rz −= φ̂ from the table |
| — | `….duration` | fixed at 40 ns | | | changing it moves every other parameter |

**Acceptance, for our own jobs only:**
1. The probe re-run at the new value moves the coherent error toward 0 by ≥ 3σ.
2. The incoherent part does not get worse: λ (C1), or the n = 9 contrast (C2), within 2σ.
3. The E1 pilot at the new values has a detection fraction no higher than at the saved ones (Wilson).
4. Leaving the range, or failing a pilot band, stops the run.

**What a guarded `hw.submit_playlist` must do:**
1. **Arm** like `hw.submit`: `DQ_QPU=run`, the task, the allowance. Count shots as segments × shots per segment, and refuse `use_timeslot`.
2. **Bound the settings.** Accept only an allowlist of setting paths (the table above), on patch loci only, within `pulse_plan.json`'s ranges. Refuse any readout, reset, or out-of-patch setting.
3. **Record.** Build the playlist and settings once. Write the diff against the calibration set (path, saved value, new value) into `meta.json` and a short form into the ledger purpose. Hash the playlist.
4. **Ledger first.** Submit with `Pulla.submit_playlist` and append the ledger line the moment a job id exists.
5. **Save** the per-shot raw data under `data/hw/<day>-<task>/<job id>/`, with the calibration set, the compiler and stage versions, and the commit.
6. **Dry run.** A mode that compiles locally and prints pulse counts and durations, and submits nothing.
7. **Never write calibration.** No such call exists; keep it that way.

**Questions for the mentors:**
1. Does our organisation have **PulLA submission rights** (`Pulla.submit_playlist`, a playlist to `POST jobs/<device>/run`) on garnet and emerald, on the shared queue, without a timeslot? Is it charged like a circuit job?
2. Which **compiler stage options** does the server use for circuit jobs? For example `subscribe_and_probe` (the `measure_fidelity` conversion), `multiplex_readout`, DD, `prepend_reset`, the timebox neighborhood. Are they `get_standard_compiler()`'s defaults?
3. Are there **server-side bounds** on submitted settings (amplitudes, drive frequency, coupler flux)?
4. Can circuit jobs carry **`prx_12` and `measure_012`**, and with which iqm-client? Does the server honour `CircuitOperation.implementation` (for example `measure` → `fast_constant`)?

## Uses of the table (design)

- **Noise model.** Pauli-twirled, a coherent error becomes small Pauli terms:
  - ε on a `prx(π/2)`: an X/Y-type error with probability ≈ sin²(ε/2);
  - η: ≈ sin²(η/2), mostly Z;
  - φ: a Z error with probability sin²(φ/2);
  - δ: ZI, IZ and ZZ terms of ≈ (δ/4)² each.

  At the closure's resolution (|ε| ~ 10⁻³) these are ~10⁻⁶, against ~10⁻³ of incoherent error. **The twirl is not the whole story, though.** Coherent errors add in amplitude over a circuit's repeated layers, which a Pauli channel cannot represent. That is why the table keeps the signed values: a simulator can apply them exactly, as `characterization.inject` does for Aer.
- **Compensation in our own circuits** (`characterization.compensate`, checked in the closure section). φ can be cancelled at circuit level with no pulse access: after each CZ, add φ_q to the phase of every later `prx` on that qubit. That is RZ(−φ_q) moved forward: a virtual Z commutes through CZ and the Z readout, and RZ(a) then `prx(θ, ψ)` equals `prx(θ, ψ − a)` then RZ(a). No gate is added. ε cannot: any angle other than π/2 is two pulses with `drag_crf_sx`, so changing the angle doubles the pulse count. η and δ need the pulse level (Track B: B4, B5).

## For a standalone `pulseCalibration/` folder

- **Package:** `src/dqiqm/characterization/` (`__init__` and 11 modules).
- **Scripts:** `scripts/garnet_calib_build.py`, `garnet_calib_closure.py`, `garnet_calib_compensate.py` and `garnet_calib_report.py`.
- **Tests:** `tests/test_calib_circuits.py` and `tests/test_calib_fit.py`.
- **Record and doc:** `InfoRecord/characterization/` and this doc.

**What it imports from `dqiqm`:**
- `dqiqm.layout`: `sent_instructions`, `Architecture`, `Placement`, `SEED`;
- `dqiqm.experiments.device`: `live_calibration`, `Calibration`, `pair_key`, `qubit_sort_key`, `t2_ns`, `fake_backend`, `REPO_ROOT`;
- `dqiqm.experiments.memory.spec` and `dqiqm.experiments.circuits.native_sha256`: only in `patch.pilot_patch`, to rebuild the pilot;
- `dqiqm.hw.save_raw`: in the closure script;
- `dqiqm.decode.detection_events`: in `compensate.detection_check`;
- data files: the live snapshot (`data/calibration/garnet/a18a947f-….20261024T191313Z.json` and its InfoRecord copy), pulla-check's `data/calibration/garnet/pulse/` (pulse parameters, compiled readout windows), and `InfoRecord/circuits/E1/placement.json`.

To cut the `memory` dependency, ship the committed `circuits/patch.json` and read the layers from it.

## Limitations

- **The stand-in is not Garnet.**
  - It has no coherent error of its own, no crosstalk, no leakage and no spectator ZZ.
  - Its gate noise is about twice the benchmark (garnet-design review, MINOR 4).
  - Its readout is `measure`'s, not `measure_fidelity`'s.

  The closure shows the analysis recovers what is injected. It cannot show what hardware contains.
- **The C1 model's incoherent part** is one depolarizing λ plus calibrated T1/T2. Leakage, non-Markovian drift or pulse-tail memory would show as χ²/dof > 1. The interval is then widened, but the model may be wrong; the residuals per N are in each fit.
- **η and a static detuning are the same thing in C1,** because the pulses are back to back. The idle precession during readout windows is a separate Ramsey quantity (B3).
- **δ and φ in C2 are measured in the layer context:** they include the spectators' static ZZ. C4 separates the two.
- **C3's leaked level** (the fluxed qubit's |2⟩) is inferred from the gate family, not measured.
