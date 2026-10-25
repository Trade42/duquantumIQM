# Configurable transversal gates and smaller logical workloads

The surface-code API now accepts `transversal_gate_set`, separately from the
physical entangling basis `gate_basis='cx'/'cz'`. This is a capability constraint,
not a way to turn an unsupported gate into a transversal one.

| Profile | Available operations | Constraint |
|---|---|---|
| `none` | Memory only | Rejects logical operations |
| `pauli` | X, Y, Z | Logical strings, at most one gate per participating data qubit |
| `pauli_h` | X, Y, Z, H | H also changes the X/Z boundary orientation; later checks/readout follow it |
| `css` | Above, plus a separate CX primitive | CX requires two matching CSS blocks, actual cross-block couplings and a joint decoding protocol |
| Explicit set, e.g. `{'X','H'}` | Restricted subset | Every requested operation is checked |

`H` is transversal as a physical operation but changes the code frame. It is
not claimed to preserve the original fixed boundary geometry. `S` and `T`
requests fail explicitly: neither is supplied as a transversal gate of this
static rotated patch. Fold-S is a separate protocol, not `S` on all data qubits.

## Use

```python
from iqm_surface_code import make_stim_circuit, make_logical_algorithm

# A physical logical block after the first syndrome round:
circuit = make_stim_circuit(
    3, 5, basis='Z', gate_basis='cz',
    transversal_gate_set='pauli_h',
    logical_operations=['H', 'X', 'H', 'Z', 'H'],
)

# Return the classical frame together with an optimized physical circuit:
algorithm = make_logical_algorithm(
    3, 5, ['H', 'X', 'H', 'Z', 'H'],
    basis='Z', transversal_gate_set={'X', 'Z', 'H'}, optimize=True,
)
print(algorithm.metadata())
raw = algorithm.circuit.compile_sampler(seed=1).sample(100)
logical_bits = algorithm.corrected_logical_bits(raw)
assert (logical_bits == algorithm.expected_logical_bit).all()

# Gates at separate round boundaries: every QEC round remains present.
scheduled = make_logical_algorithm(
    3, 5, {1: ['H', 'X'], 2: ['Y'], 4: ['H', 'Z']},
    basis='X', optimize=True,
)
```

For a sequence, `logical_after_round` selects its insertion boundary; for a
mapping, integer keys give the rounds after which each block runs. Require
`1 <= boundary < rounds`. Multiple H gates can cancel inside a single unitary
block; H gates are not moved across scheduled QEC boundaries. Known Pauli
corrections propagate through subsequent H gates and can remain entirely in a
classical frame. The final raw logical parity must be XORed with
`virtual_readout_xor` before interpreting the logical output.

Bare Stim generators return circuits suitable for detector/observable **flip**
sampling; use `make_logical_algorithm` when the algorithm's actual logical bits
or outgoing frame matter. The default `optimize_logical_gates=False` preserves
materialized operations in the basic generator. An explicit logical-step
barrier keeps that baseline separate from logical compiler optimizations.

All subsequent syndrome interactions are conjugated after an odd number of
transversal H gates. The final data measurement basis changes, while logical
strings are interpreted in the evolving code frame. Detector measurements and
no-reset record identities remain aligned. Both bases, several scheduled gates,
actual final logical bits, noiseless Qiskit/Aer execution and d=3/5/7 graphlike
circuit distance are tested. Noise is added only after the physical schedule is
constructed and lowered to CZ.

The same keyword parameters pass through `make_qiskit_circuit`,
`prepare_qiskit`, `run_qiskit` and `simulate_memory`. Hardware results carry
`logical_algorithm` metadata. The optional Qrisp generator also accepts these
keywords, but its SDK path remains unvalidated and may reject noise annotations.

CLI example:

```powershell
python -m iqm_surface_code.experiment --rounds 3 5 7 --models uniform --shots 100000 --transversal-gate-set pauli_h --logical-operations H X H Z H --optimize-logical-gates
python -m iqm_surface_code.benchmark_logical --shots 100000 --models uniform leakage calibrated_coherence --synthetic-calibration --fit-cycle-ler
```

The first command sweeps one algorithm block, not a growing algorithm. Its
round-count fit must not be interpreted as a universal gate error rate.

## What reduces operation count

For this supported gate family, the compiler calculates the exact logical
normal form `U = X^x Z^z H^h`, up to irrelevant global phase. A block therefore
needs at most one physical transversal H layer and a tracked Pauli. H propagates
the frame by swapping x and z; X/Z/Y update its two classical bits. The complete
one-qubit Clifford action is checked for every length-zero-through-four word
over X/Y/Z/H, rather than comparing only its effect on the prepared |0> state.

This is particularly useful for correction streams inside a larger algorithm:
materializing a known correction adds noisy pulses without gaining information.
The scheduled benchmark leaves syndrome extraction between each correction,
so its gain does not depend on deleting compulsory error-correction rounds.
Adaptive corrections from a future online decoder require integrating that
decoder's frame updates; current experiments use explicitly supplied corrections.

For general algorithms this is a compilation tool, not a claim that every
Clifford block or universal algorithm shrinks by the same fraction. A good
existing physical Clifford compiler can already remove some within-block
redundancy. Non-Clifford gates, unknown/adaptive measurement branches, external
observable requirements and unsupported operations are barriers to this simple
normal form. No T-count improvement or Shor/chemistry runtime gain is claimed.

## Concrete measured numbers

Historical full data in the original repository's local archive:
`results/transversal/results.json`; readable report:
`results/transversal/report.md`; plot: `results/transversal/comparison.png`.
These are local d=3 no-reset simulations, not IQM measurements. Every point uses
100,000 shots, correlated PyMatching and explicitly identified noise assumptions.
The calibration example is synthetic. Both variants share the patch, round
count and physical noise model.

| Workload | Algorithm one-qubit gates | Total circuit Clifford gates | QEC rounds |
|---|---:|---:|---:|
| Synthetic 33-gate Clifford block, net H | 201 → 9 (95.5% fewer) | 503 → 311 (38.2% fewer) | 5 → 5 |
| 13 corrections separated by QEC | 47 → 0 | 915 → 868 (5.1% fewer) | 14 → 14 |
| Already minimal H control | 9 → 9 | 311 → 311 | 5 → 5 |

Counts include physical one-/two-qubit Clifford instructions. Preparation,
measurements and their basis rotations are not included in that gate count;
their number is unchanged. These are not calibrated IQM pulse counts, and some
physical Z rotations may already be free frame updates in a native compiler.

At assumed uniform circuit noise p=0.001:

| Workload/basis | Logical failure before | After | Relative reduction (95% interval) |
|---|---:|---:|---:|
| Clifford block / X | 2.351% | 0.748% | 68.2% [63.9%, 72.0%] |
| Clifford block / Z | 2.244% | 0.781% | 65.2% [60.5%, 69.3%] |
| Scheduled corrections / X | 2.416% | 1.957% | 19.0% [10.9%, 26.4%] |
| Scheduled corrections / Z | 2.540% | 2.171% | 14.5% [6.3%, 22.0%] |

The correction stream has **no resolved improvement under the leakage model**:
X 13.344% → 13.369%, Z 14.053% → 14.175%, with reduction intervals crossing
zero. Removing single-qubit corrections does not remove two-qubit leakage
generation or replace a leakage-removal protocol. Synthetic calibrated-coherence
results for the scheduled stream also have intervals crossing zero. These null
results are retained; improvement is not guaranteed across noise models.

For an actual **effective computational-cycle LER** estimate, separate sweeps
use 2/5/8/11/14 QEC rounds, of which the first is fixed preparation overhead.
Fit 1/4/7/10/13 correction-bearing cycles using
`P(n)=[1-A(1-2*epsilon)^n]/2` and binomial likelihood, with A fitted.

Z-basis effective cycle rate is **0.0018629 → 0.0016032**, a **13.9% reduction**.
The respective 95% profile intervals are [0.0017963, 0.0019295] and
[0.0015406, 0.0016657]. Pearson fit diagnostics give p=0.773 and p=0.201.
X estimates decrease 0.0018325 → 0.0014919, but the baseline exponential-fit
diagnostic is marginal (p=0.039); treat that rate cautiously. This is an
effective rate of the changed computational cycle. The unchanged memory
circuit's intrinsic LER is not claimed to improve.

The single-H control has exactly identical physical circuits; samples are
reused and improvement is exactly zero. Distinct-circuit comparisons use
independent seeds and a Bonferroni combination of Wilson intervals. The cycle
fits use additional independent fixed-shot data and asymptotic profile intervals.

## Papers and hardware limits

- [Ruh and Devitt, Pauli tracking (2024)](https://arxiv.org/abs/2405.03970):
  concrete basis for eliminating known physical Pauli corrections through
  Clifford propagation. The implemented compiler uses this principle.
- [Chen et al., transversal Clifford gates on rotated patches (2024)](https://arxiv.org/abs/2412.01391):
  supports transversal H with orientation management and a distinct mid-cycle
  fold-S construction. The latter uses connectivity/reconfiguration not supplied
  by our static Emerald patch; it is not enabled here.
- [Sahay et al., transversal CNOT correction (2025)](https://arxiv.org/abs/2408.01393):
  motivates a separate two-block primitive and joint decoding. Our primitive
  validates every cross-block pair and tests the Clifford action; it does not
  claim to implement that paper's correlated two-block decoder or its threshold.
- [Serra-Peralta et al., decoding across transversal Cliffords (2025)](https://arxiv.org/abs/2505.13599):
  explains why fast logical gates also need gate-boundary decoding and distance
  checks. Our supported single-block path is tested explicitly; these results
  are not a reproduction of that paper's unrotated-code decoder.
- [Eastin and Knill (2009)](https://arxiv.org/abs/0811.4262):
  rules out a universal transversal unitary gate set for a fixed code with the
  required error-detection properties. Gate-set selection respects that limit.

`transversal_cnot(control_data, target_data, available_pairs=...)` emits one
parallel CX layer only when all supplied pairs are physically available. It
does not route missing couplings, add syndrome extraction or silently apply
independent memory decoding to correlated two-block errors. No CNOT LER gain
is claimed from the primitive alone. Future fold-S/CNOT work needs new layouts,
joint detectors, connectivity/pulse validation and noise benchmarks first.
