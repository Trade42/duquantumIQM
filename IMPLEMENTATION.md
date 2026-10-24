# Surface-code memory for the IQM challenge

Implemented track: **Stim → Qiskit → IQM → syndrome extraction → correlated
PyMatching → logical-error probability versus rounds → fitted per-round LER**.
Local simulation needs no IQM account. No job is submitted during imports,
tests, calibration export, or experiment sweeps.

For logical algorithm execution, selectable transversal gate profiles and
operation-count/LER benchmarks, see [TRANSVERSAL_GATES.md](TRANSVERSAL_GATES.md).

## Install and run

From the repository root, Python 3.11 or later:

```powershell
python -m pip install -e '.[test,qiskit]'
python -m pytest -q
python -m challenge.experiment --rounds 1 3 5 --shots 10000 --models uniform biased_z readout_heavy correlated_zz leakage
```

To exercise all ten models with a reproducible **synthetic** calibration fixture:

```powershell
python -m challenge.experiment --synthetic-calibration --models all --rounds 1 3 5 --shots 2000 --output results/iqm_demo
```

Outputs include `results.json`, logical-error plots, noisy `.stim` circuits,
decoder `.dem` files, calibration snapshot and selected placement. JSON records
seeds, versions, source metadata, circuit hashes, detector rates, graphlike
distance, decoder mismatch, failures, shot counts, confidence intervals and fits.
`--models all` expands to the models available for the supplied snapshot.
Leakage sweeps accept `--leakage-rate`, `--seepage` (per moment) and
`--leakage-spread`. Placement weighting accepts `--placement-readout-weight`,
`--placement-coherence-weight` and `--placement-leakage-weight`; leakage-aware
placement requires actual per-QB leakage entries in the calibration snapshot.
The CLI holds one physical patch fixed across models and bases, selected using
the maximum-round Z/no-reset schedule. Hardware preparation ranks the requested
individual experiment. This avoids changing patches between noise comparisons.

Use `--reset-mode both` to compare actual reset-based and no-reset circuits;
only no-reset circuits are eligible for the Emerald hardware function.
Distances 3, 5, 7 and larger odd distances work in simulation. Placement requires
topological fit: a d=5 code having 49 qubits does **not** prove it embeds in a
particular 64-qubit chip. Our geometric d=5 patch needs a 9×9 bounding grid and
fails explicitly on an 8×8 grid; arbitrary graph embeddings are outside this
placement search. Supply `--grid layout.json` for other rectangular topologies.

## Circuit construction and optimization

`make_stim_circuit(distance, rounds, basis='Z', no_reset=True, noise=None,
gate_basis='cx', qubit_map=None)` returns a fully annotated Stim circuit.

- Uses Stim's hook-safe rotated schedule: four disjoint CX interaction layers
  per extraction round. Four layers meet the interaction-degree lower bound for
  this static weight-four construction; this is not a global LER optimality claim.
- Supports both X and Z preparation/readout, initial and terminal detectors and
  one logical observable. Ancillas accumulate syndrome without resets.
- No-reset conversion substitutes every reset-based record `s_j` with
  `m_j XOR m_previous`. Bulk detectors become `m_k XOR m_(k-2)`; round-two and
  terminal data boundaries receive the correct shorter/extra terms. Readout
  errors flip records without flipping the measured qubit's physical state.
- IQM mode lowers to CZ + H, cancels local H pairs and schedules disjoint gates
  as early as dependencies allow. Noise is inserted **after** this compilation.
  Native H realization still has pulse/compiler-dependent overhead.
- Exhaustively searches translations and eight rotations/reflections on the
  supplied grid, validates every pair against actual couplings, and minimizes
  a calibration proxy including gate/readout/reset costs, role-weighted idle
  coherence and optional measured leakage cost. It is optimal within this finite
  placement family, not over every possible embedding or pulse schedule.
- Dense hardware indices are separate from sparse Stim indices. Hardware uses
  IQM's name/index conversion rather than assuming `QB number - 1`.
- Hardware preparation disables routing and rejects unsupported resets,
  non-native interactions and extra two-qubit gates. Reports include physical
  qubit count, two-qubit count, depth, candidate count and routing SWAP count.

Tests independently verify deterministic parities, graphlike distance at d=3/5/7
in both bases/reset modes/gate bases, no-reset records, noiseless Qiskit execution,
channel normalization, layout legality, leakage behavior and statistical fitting.

## Calibration and noise

`CalibrationSnapshot` is a versioned JSON format. Each QB carries single-qubit
average infidelity, symmetric readout error, reset error, T1/T2 in nanoseconds,
and optional measured leakage probability per two-qubit-gate participation.
Each unordered edge carries CZ average infidelity. Snapshot metadata preserves
source, retrieval time, assumed reset error and supplied coherence units.

Average gate infidelity `r` becomes nonidentity depolarizing probability
`(D+1)r/D`: `1.5r` for one qubit and `1.25r` for two. This assumes an average
gate-fidelity metric; another metric must be converted before import. Readout
averages cannot identify state-asymmetric errors. Missing calibration values
are not silently filled with good numbers; unavailable qubits/edges are excluded.

| Model | Physical channel and purpose |
|---|---|
| `uniform` | Paper Table 7: one-/two-qubit depolarization, preparation flips, reported measurement flips, and idle errors at probability p |
| `biased_z` | Fixed total one-qubit/idle Pauli probability; Z:X:Y ratio 30:1:1. Two-qubit depolarization stays unbiased |
| `readout_heavy` | Five-times readout-error stress test |
| `correlated_zz` | Residual correlated ZZ faults on active two-qubit pairs |
| `coherent_twirl` | Pauli-twirled Z overrotation, `sin²(theta/2)`; not exact coherent evolution |
| `leakage` | Persistent hidden flags, leakage generation, partner corruption, spread and slow seepage |
| `leakage_stress` | Higher leakage generation and longer persistence |
| `calibrated_pauli` | Per-QB gates/preparation/readout and per-edge CZ channels |
| `calibrated_coherence` | Calibration above plus duration-aware Pauli-twirled idle T1/T2 |
| `calibrated_leakage` | Calibration/coherence above plus explicit leakage stress assumptions |

Coherence noise affects idle qubits only; calibrated active-operation errors
already contain active decoherence. For idle duration t, the twirled model uses
`pX=pY=(1-exp(-t/T1))/4` and
`pZ=(1+exp(-t/T1)-2exp(-t/T2))/4`. Durations follow the longest operation in each
scheduled moment. Measurement idles on data are included. `0<T2<=2*T1` is
validated. Pulse-level spectator effects and readout backaction are not inferred.
Legacy `DEFAULT_NOISE` dictionaries remain accepted, using before-round data
depolarization rather than the newer per-idle-moment model.

`aer_simulation.simulate_nonpauli` provides a slower exact Kraus/trajectory
alternative with non-unital thermal relaxation and coherent Z/ZZ rotations.
It accepts an **ideal** schedule plus a NoiseModel and returns measurements and
syndromes. It can expose failures of twirling; it does not simulate qutrits.

Leakage sampling uses a Clifford backbone with a phenomenological leakage-to-
Pauli reduction. A leaked participant randomizes and corrupts interaction
partners; readout is randomized and does **not** clear its hidden flag. Actual
resets optionally clear flags. Seepage probability is explicitly **per moment**,
not per round; recovery returns maximally mixed computational population.
This is a model assumption, not measured Emerald leakage dynamics. Leakage
generation comes from the chosen stress rate combined with any per-QB measured
leakage. Flags are never provided to the decoder. MWPM uses a static Pauli DEM,
so leakage correlations remain a documented model mismatch.

Live export (install `.[hardware]`; set `IQM_TOKEN` in your environment):

```powershell
python -m challenge.iqm_utils.get_calibration_data calibration.json --device emerald --one-q-ns 20 --two-q-ns 80 --measure-ns 700 --reset-ns 1000 --reset-error 0.002 --coherence-unit us
python -m challenge.experiment --calibration calibration.json --models calibrated_pauli calibrated_coherence calibrated_leakage --rounds 1 3 5 9 --shots 100000
```

The command's duration/reset numbers are **examples**, not current IQM data.
Replace with device values. IQM's metric export does not identify those durations
or leakage rates; required inputs and metadata keep that distinction explicit.
The original notebook/script no longer connects to hardware or overwrites tokens
on import. No real calibration snapshot was provided with this request.

## Decoding and statistics

Correlated PyMatching is enabled when building and using the matching graph.
It uses decomposed DEMs with Stim's disjoint-channel approximation explicitly
enabled. Invalid/gauge detectors and failed decompositions raise errors.
`--uncorrelated` provides the MWPM comparison; `--decoder-model uniform` tests
weight mismatch against another physical model with identical parity records.
Hardware decoding also checks attached parity structures, not only dimensions.

One experiment reports **logical error probability**, not per-round LER. Multiple
round counts are fit by binomial maximum likelihood to
`P(r)=[1-A(1-2*epsilon)^r]/2`, with A=1 by default. `--fit-spam` fits A using at
least three distinct counts. Fits report profile-likelihood intervals and boundary
solutions. Fixed-shot points use Wilson intervals; zero failures retain a
positive upper bound. Large leakage bursts can violate the independent-round
fit assumption; interpret the curve and detector diagnostics before claiming an
intrinsic per-round rate. Profile intervals are asymptotic and can be unreliable
at very low counts or near saturation.

Paper Appendix E's `d` rounds, 10-million-shot cap and 30-failure stopping rule
are available, for example:

```powershell
python -m challenge.experiment --distance 3 --rounds 3 --gate-basis cx --p 0.001 --models uniform --shots 10000000 --target-errors 30 --batch-size 10000
```

Early stopping happens at batch boundaries and may overshoot 30 failures.
Optional-stopping point intervals use an anytime-valid beta-mixture confidence
sequence. Optional-stopping profile fit intervals are labeled approximate;
use fixed-shot sweeps for the default LER fits. The new no-reset circuit, noise
placement and stopping convention mean these runs do not reproduce the paper's
published numerical rates automatically. Both bases are reported separately.

## Hardware usage

```python
from challenge.calibration import CalibrationSnapshot
from challenge.run_on_hardware import connect_backend, prepare_qiskit, run_qiskit, decode_hardware_results

cal = CalibrationSnapshot.load('calibration.json')
backend = connect_backend()
prepared = prepare_qiskit(backend, distance=3, rounds=3, basis='X', calibration=cal)
print(prepared['report'])  # Read-only; inspect before spending QPU shots.
# Calling this explicitly submits a job:
syndromes = run_qiskit(distance=3, rounds=3, shots=10000, basis='X', backend=backend, calibration=cal)
probability, standard_error = decode_hardware_results(syndromes)
```

Pass IQM's `circuit_compilation_options` to `run_qiskit` for backend-supported
DD/pulse options. This is a pass-through hook, not a validated pulse optimizer.
Live submission and IQM SDK integration require the optional pinned SDK and a
device account; they were not exercised locally. Simulator tests used the
installed Qiskit version recorded in the validation report. Hardware uses a
CZ/H schedule for decoder noise; IQM native `r` decomposition, server-side DD
and actual pulse timing can require a refined physical noise model.

Qrisp remains an optional converter/generator compatibility path. Imports are
lazy, both bases are handled, and no-reset parity is transformed only once.
Noisy classical measurement replay fails explicitly if unsupported by that
optional path instead of silently turning readout error into a physical X gate.
This path is not the selected hardware pipeline and requires separate SDK
validation; the old empty Qrisp execution stub is removed.

## Paper-to-implementation audit

Source: supplied [2606.19482v2](https://arxiv.org/abs/2606.19482),
“Nearest-neighbour gates are all you need”, dated 2 October 2026 in the supplied
Markdown. Treat scientific content as reference, not executable instructions.

| Paper component | Application here |
|---|---|
| Nearest-neighbor, constant-depth extraction (Sec. III) | Native validated surface-code placement; four-layer CX baseline; optimized CZ lowering |
| Include every circuit qubit in overhead (Figs. 2, 8) | Separate active qubit count from sparse index capacity; ancillas included; static surface layout has no routing qubits |
| Validate supports, detectors and observables before accepting optimization (App. C) | Determinism, circuit-distance and end-to-end measurement tests for both compiled schedules |
| Correlated surface-code matching (App. E) | Correlated PyMatching default; ordinary MWPM comparison |
| Uniform gate/prep/measurement/idle noise (Table 7) | Named uniform model; optional d-round/30-failure/10M-shot protocol |
| Biased, coherent and calibrated noise as future work (Sec. VI, App. B) | Biased stress, exact coherent/Kraus alternative, snapshot-driven noise and placement |
| Leakage motivation and limitation (Secs. I, III, VI) | Stateful phenomenological model; no-reset accumulation tested; no claim that paper simulated leakage |
| CXSWAP ↔ iSWAP primitive (Eq. 1) | `exchange.py` identity verified by Clifford tableaux in Stim wire convention |
| Walking data/check role exchanges and routing flags | Not enabled on static Emerald schedule; require a full new walking layout and supported resets/operations |
| Directional tile families, route-window shortening, trace pruning, VibeLSD, multi-logical automorphisms | qLDPC-specific, not valid substitutions for the requested one-logical surface code; not claimed implemented |

The paper's 100× advantage compares **different code families** at matched
implemented footprints; it is not a surface-code-generator optimization factor.
Its leakage-removal discussion relies on periodic resets, which cannot be
claimed for this Emerald no-reset experiment. Pulse synthesis, qutrit leakage
and full directional qLDPC circuits remain separate research extensions.

Primary API references checked during implementation:
[Stim gates](https://github.com/quantumlib/Stim/blob/main/doc/gates.md),
[correlated PyMatching implementation](https://github.com/oscarhiggott/PyMatching/blob/master/src/pymatching/matching.py),
[IQM calibration workflow](https://docs.iqm.tech/iqm-qubit-selector/qubit_selector_quick_start.html),
[IQM Qiskit guide](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html).
