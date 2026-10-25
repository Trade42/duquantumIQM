# Notes for the team (Ethan)

`challenge/` is a read-only library for `mlDecoder/`. Nothing in it was changed. These are the points where a change there would help, plus the conventions `mlDecoder` relies on, so a change on either side doesn't silently break the other.

## Requests (none blocking)

1. **The leakage sampler is the bottleneck.** `challenge.leakage.sample_leakage` walks the flattened circuit in Python, with numpy work per instruction. On this laptop (Ryzen 9 8945HS) it gives about 220k shots/s at d=3, R=1, but only 25k at d=3, R=10, 7k at d=5, R=10 and 3.3k at d=7, R=10. Stim's sampler is 1–20M shots/s on the same circuits.
   - `mlDecoder` works around it by sampling seeded 10k-shot chunks in worker processes, then training leakage models from a cached pool.
   - A faster sampler would allow streamed leakage training and larger pools. Fusing consecutive noise channels, or batching the per-qubit loops, would do it.
   - If you change the sampler, please say so. Frozen test sets stay valid, because they are stored arrays checked by hash. But regenerating one from its seeds would give different shots, and the circuit hash wouldn't notice. We would bump `mlDecoder.data.DATA_VERSION` to publish new dataset IDs.
2. **A per-detector table in `challenge`.** `mlDecoder.data.detector_layout` builds round, check, X/Z type and (x, y) for each detector:
   - the round and check come from the circuit's own measurement records and Stim's detector coordinates;
   - the check types come from your CX-basis template (`make_stim_circuit(..., gate_basis="cx")`: the control is an X check, the target a Z check).

   An authoritative helper in `challenge`, e.g. next to `get_circuit_info`, would remove that inference. It would also cover the dual frame after a logical H, which `detector_layout` does not handle (it is for memory circuits with no logical operations). The hardware record planned for Garnet carries a detector table, and the two should come from one function.
3. **Placement for calibrated models at d ≥ 5.** `rank_placements` fails on the default 8×8 grid for d = 5, as IMPLEMENTATION.md says. So `mlDecoder` supports the calibrated noise models at d = 3 only. Not needed for stage 0.

4. **`fit_ler(fit_spam=True)` bounds A to [0, 1], and every one of our SPAM fits sits on that bound.** That covers MWPM, X and Z, d = 3/5/7, uniform and leakage, p = 0.001 and 0.003, over R = 1–10.
   - The A = 1 fits fit poorly. In the baseline, Pearson χ² p ≪ 0.05 for most groups, and is 0.0036 even for d = 3 uniform X.
   - In the no-reset circuit the first rounds look less noisy than a one-rate model assumes: P_L at small R falls below the curve. That would need A > 1, or a separate rate for the boundary rounds.
   - A per-round ε_L from these fits depends on which R are included. For example, at d = 3 Z uniform it is 1.20e-3 over R = 1–3 and 1.46e-3 over R = 1–10.
   - Worth deciding, before quoting a hardware LER: allow A > 1, drop R = 1 (and 2) from the fit, or use the 2-parameter form on R ≥ 3.

5. **Stage 1, real devices: what the calibrated models leave out.** `mlDecoder/hardware.py` builds the team's `CalibrationSnapshot` from InfoRecord's Garnet package and adds, after `add_noise`, a documented extension (`docs/hardware.md`). Each item below would be better in `challenge/`:
   - **The readout window idles the measured ancillas too.** `add_noise` idles only the qubits not active in a moment, so a measured ancilla never decoheres through its ~1.1 µs window. Without reset, its post-measurement state carries into the next round.
   - **The terminal layer is one simultaneous readout on hardware** (the last round's ancillas and the data, compiled as `measure_fidelity`). The generator emits two measurement moments, so the data qubits idle a full window in between.
   - **An X memory's initial RX (data) and R (ancillas) are two moments,** so any reset duration idles half the qubits. mlDecoder sets the reset duration to 1 ns: the passive reset is between shots.
   - **Readout is asymmetric** (P(0|1) is 2–9% on Garnet, P(1|0) 0.4–2.7%). A symmetric mean overstates the final layer of a |0⟩-heavy Z memory. mlDecoder samples the record without readout flips and flips each bit by the value read.
   - **Readout-induced leakage and QND-ness are measured per qubit** (`metrics.rrc.measure.constant.*`), but the snapshot has only `leakage_per_gate`. mlDecoder spreads an ancilla's per-readout leakage over its CZs, and optionally adds the measured non-QND state flips after each readout.
   - **Even so,** the real Garnet pilot's bulk rounds fire 1.3–1.5× above every calibrated build, with stronger lag-1 correlation (0.063). A fitted model (p_ij from the batches) would be the next step.

## Conventions mlDecoder relies on

- **Circuits** are built exactly as `challenge.experiment.simulate_memory` builds them:
  - `make_logical_algorithm(d, R, (), basis=..., no_reset=True, gate_basis="cz", transversal_gate_set="pauli_h", logical_after_round=1, optimize=False).circuit`, then `add_noise(ideal, model, qubit_map=...)`;
  - Pauli shots: `noisy.compile_detector_sampler(seed).sample(...)`;
  - leakage shots: `sample_leakage(noisy, model, n, seed=..., qubit_map=...)`, then `ideal.compile_m2d_converter().convert(...)`.

  `tests/test_data.py::test_sampling_matches_simulate_memory` reproduces `simulate_memory`'s failure count and detector firing rates exactly, in X and Z, for uniform and leakage. If `simulate_memory` changes its construction, that test fails.
- **The leakage path** is chosen with `simulate_memory`'s own test: `model.leakage` or per-qubit calibrated leakage. So `noise_suite(p=0)["leakage"]` takes the Pauli path, because its leakage rate is p.
- **The baseline decoder** is `make_matcher(noisy, correlated=True)` with `decode_batch(..., enable_correlations=True)`; `correlated=False` gives uncorrelated MWPM. Leakage sets are decoded with the static Pauli DEM, the same model mismatch `simulate_memory` records.

## Observations

- **On the `leakage` model, correlated MWPM's edge over plain MWPM shrinks or reverses** (see `RESULTS.md`). At d = 3, Z, p = 0.001, plain MWPM's ε_L is 0.949× correlated MWPM's [0.939, 0.959]. Under persistent leakage, the decomposed hyperedges that correlated matching reweights are not the dominant errors.
- **Leakage flattens distance scaling.** Λ = ε_L(d)/ε_L(d+2) for correlated MWPM is 7.7–8.9 on `uniform` at p = 0.001, but 1.42–1.48 on `leakage`. At p = 0.003, `leakage` gives Λ = 0.83–0.90: larger codes do worse. This is a property of the phenomenological leakage model (persistent flags, no reset), and is not yet checked against hardware.
- `pyproject.toml` has `testpaths = ["tests"]`, so a bare `python -m pytest -q` runs only the team's tests. Run `python -m pytest -q mlDecoder/tests` for ours. They are offline and take about 10 s.
