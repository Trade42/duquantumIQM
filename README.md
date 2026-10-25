# From better quantum memory to useful quantum computing

This repository builds a surface-code memory on IQM Garnet, improves how it runs
and decodes, and measures the result on hardware. The improved pipeline lowers
the fitted logical error rate by **68.6% in X memory and 36.7% in Z memory**, using
the same **17 physical qubits**.

The next step is making reliable decisions quickly enough for a computation.
The included ML telescoping decoder tackles lookup-table growth with a recurrent
model for real-time decoding. On Emerald, its pipeline lowers fitted LER by
**82.4% in X and 30.1% in Z** relative to vanilla memory.

## The vanilla pipeline

A distance-three rotated surface code stores one logical qubit in nine data
qubits, with eight ancillas measuring parity checks. Z memory stores logical
zero and tests bit preservation; X memory stores logical plus and tests phase
preservation.

1. **Prepare the memory.** Stim defines the encoded state, four X checks, four Z
   checks, and the logical observable to read at the end.
2. **Repeat the checks.** Four interaction layers measure the stabilizers, with
   gate ordering chosen to control how ancilla faults spread. Qiskit and IQM
   translate each CX into H–CZ–H, giving **24 CZ gates per round**. A legal native
   placement connects the patch directly. Vanilla uses literal compilation and
   leaves dynamical decoupling off.
3. **Collect measurements.** Ancillas are measured and reused without reset.
   Their recorded bits carry the previous state, so bulk detection events compare
   measurements two rounds apart: `m[t] XOR m[t-2]`. Generated equations handle
   initialization and the final data readout.
4. **Decode the history.** Stim converts the measurements into detection events.
   A logical maximum-likelihood lookup uses a uniform Pauli error prior
   `p=0.003` to predict whether the logical observable flipped.
5. **Score the memory.** Compare the prediction with the measured logical flip,
   count failed shots, and repeat at different round counts. The resulting curve
   gives a fitted logical error rate per round, or LER.

## What the improved pipeline changes

The standout changes connect native control, readout and logical inference:

- **Compile gates and readout together.** Cancel redundant native rotations.
  X memory moves eligible final basis rotations so ancilla and data readout form
  one terminal group; Z memory preserves measurement boundaries while optimizing
  between them. Both expose terminal readout to the hardware compiler's
  fidelity-readout treatment.
- **Correct Z model bias with hardware data.** Use 6,000 independent training
  shots per round count to update modeled logical-class probabilities.
  Bayesian shrinkage lets common histories correct the model while keeping rare
  histories close to its prior and retaining its decisions for unseen histories.
  Correlation-weight fitting is used only when it wins a training-only check;
  the decoder is frozen before evaluation.
- **Decode X with calibrated fault correlations.** Build the prior for the
  executed native gates and readout, including a proxy for disturbance after
  intermediate ancilla measurements. Correlated minimum-weight perfect matching
  (MWPM) uses those probabilities and fault relationships, with no labeled
  hardware training.
- **Use idle time to protect data.** Insert XY4 dynamical-decoupling pulses into
  eligible data-qubit waits while ancillas are measured, preserving wait duration.
  Pulse audits verify actual insertion: none at one round, then 72 and 144 added
  compiled pulse instructions at two and three rounds.
- **Choose the physical patch by calibration.** Rank legal native placements
  using gate and readout quality to select a favorable set of qubits and couplers.

Both versions use **17 qubits, 24 CZ gates per round and zero routing SWAPs**;
placement improves which native patch runs the code. Vanilla already uses a
complete modeled logical-class lookup; Z adds a native prior and hardware
training. Explicit measurement maps preserve the detector and logical frames
through compilation. The headline LER reductions measure the combined pipeline;
they do not isolate each technique's contribution.

See [every retained improvement](improvements.md), the [methods](docs/METHODS.md)
and the [native routing audit](docs/GARNET_ROUTING_PROOF.md).

## Results on Garnet

The comparison uses **54,000 evaluation shots** across one, two and three rounds:
18,000 vanilla and 36,000 improved, collected under the same calibration.

| Memory | Vanilla fitted LER | Improved fitted LER | Relative reduction |
|---|---:|---:|---:|
| X | 18.517% (17.748–19.314%) | **5.808% (5.492–6.136%)** | **68.6%** |
| Z | 6.722% (6.326–7.135%) | **4.255% (4.066–4.449%)** | **36.7%** |

Parentheses give 95% profile intervals. The fitted rates summarize the three
round counts; the plot shows the measured failure probabilities.

![Garnet memory failures across round counts, with 95% Wilson intervals](docs/assets/garnet_comparison.png)

At three rounds, failure probability falls from **40.667% to 18.200% for X** and
**19.433% to 14.3375% for Z**. Placement, compilation and protected waits improve
physical execution. Calibrated priors and better inference improve the logical
decision. The measured gain belongs to that combined pipeline.

**LUT versus MWPM.** On improved three-round X memory, MWPM has 728 failures
versus 789 for the original model-only LUT in 4,000 shots. The [decoder audit](docs/LUT_MWPM_AUDIT.md)
reproduced all 16.8 million LUT decisions; the difference comes from the noise
prior's conditional weights. With identical vanilla records and priors, the LUT
wins 10 of 12 cases.

The new [hardware-calibrated LUT](docs/LUT_IMPROVEMENT.md) learns corrections from
independent development records. On the same benchmark shots, fitted LUT error
falls from **6.039% to 5.620% in X** and **4.283% to 4.162% in Z**. The corrected
policies are included and still decode with one table lookup.

The [evaluation](docs/EVALUATION.md) contains the counts and fit diagnostics;
[error analysis](docs/ERROR_ANALYSIS.md) follows the remaining error pathways.

## From better memory to useful computation

A useful quantum computer needs **low logical error and low decoding latency**.
Errors accumulate as a computation grows, so each logical operation needs to be
reliable. The classical decoder also has to keep up with incoming syndromes and
deliver decisions in time for adaptive operations.

Throughput tells us how much work a decoder can handle. Latency tells us how long
a computation waits for its next decision. A practical decoder needs both,
including predictable response times on difficult cases.

## Choosing surface codes for their logical gates

Code choice can target the computation as well as memory performance. A
**transversal gate** acts independently on data positions within a block, or
between corresponding positions in separate blocks, limiting fault propagation
within each block. Which logical gates have such implementations depends on the
code's stabilizers, boundaries and layout.

| Surface-code construction | Logical-gate options | What the hardware and decoder must support |
|---|---|---|
| Standard rotated CSS patch | Logical Pauli strings; a physical H layer exchanges the X/Z checks and boundary frame. Compatible CSS blocks admit transversal CNOT. | Track the changed frame or reorient the patch after H. CNOT needs pairwise inter-block connectivity and decoding that accounts for propagated errors. |
| Folded surface-code constructions | Enable logical H and S through fold-based operations, extending the available Clifford gates. | Interactions between paired sites across the fold and the corresponding syndrome schedule. These are distinct from independent single-qubit gates on the original planar patch. |
| Rotated patches with mid-cycle protocols | A fold-transversal S inside syndrome extraction can complement H and CNOT to supply a full logical Clifford generating set. | The specified intermediate code state, extra interactions or qubit rearrangement, and detectors spanning the gate boundary. |

See [Moussa's folded constructions](https://arxiv.org/abs/1603.02286) and
[Chen et al.'s rotated-patch protocols](https://arxiv.org/abs/2412.01391) for the
constructions and their connectivity requirements.

[Clifford-deformed variants such as XY and XZZX](https://www.nist.gov/publications/clifford-deformed-surface-codes)
also let code design follow the hardware's noise bias. A local basis change
conjugates both stabilizers and gate implementations; it preserves the abstract
logical gate capabilities while changing their physical realization. Choosing
a variant therefore means assessing gate cost, syndrome extraction and decoding
together. The useful target is lower qubit-time cost for the whole computation.

**Current scope:** the retained runner implements X/Z identity memory. Selecting
gate-capable variants and validating their logical-gate circuits and decoders
is an extension beyond the measured pipeline. Universal computation also needs
an additional resource, such as magic states or code switching: a fixed code
cannot supply a universal transversal unitary gate set
([Eastin–Knill](https://arxiv.org/abs/0811.4262)).

## ML telescoping for real-time decoding

A lookup decoder needs one entry for every detector history: **2^D entries for D
detector bits**. At three rounds, the distance-three patch already needs a 16 MiB
decision table. Longer histories quickly outgrow that approach.

The ML telescope replaces the table with a two-layer LSTM: **243,714 parameters,
about 488 KiB of stored weights**. It compresses successive syndrome rounds into
a learned state and combines that state with final boundary events to predict
the logical flip. The same weights serve all tested round counts from one to
twenty. Model storage stays fixed as the history grows.

For real-time control, the telescope allocates work by confidence. The LSTM makes
the first prediction; uncertain shots go to Relay-BP, with correlated MWPM
handling nonconvergence. Training combines calibrated synthetic data with
hardware fine-tuning. Validation selects how often to use the extra stages.

Using the **one-to-three-round fit** from the vanilla Emerald benchmark:

| Memory | Vanilla Emerald LER | ML telescope LER | Relative reduction |
|---|---:|---:|---:|
| X | 12.352% | **2.179%** | **82.4%** |
| Z | 4.749% | **3.318%** | **30.1%** |

At three rounds, X failure probability falls from **28.633% to 8.050%**, and Z
from **15.200% to 10.100%**. The comparison uses 18,000 vanilla shots and 12,000
ML test shots. The full included ML evaluation spans **154,000 held-out shots**
on Garnet and Emerald.

### Decoding time per round

The main timing measure is **mean decoding cost per syndrome round**: complete
history decoding time divided by its round count. These recorded timings use a
Ryzen 9 8945HS, one CPU thread and one shot at a time.

| Emerald memory | Rounds | LSTM time per round | Full telescope time per round |
|---|---:|---:|---:|
| X | 3 | 164.1 µs | 204.8 µs |
| Z | 3 | 245.0 µs | 242.3 µs |
| X | 20 | 38.2 µs | 710.2 µs |
| Z | 20 | 39.9 µs | 38.9 µs |

The recurrent stage reaches roughly **38–40 µs per round** at twenty rounds.
Emerald X routes 23% of its test shots through Relay-BP; its longer histories
show the extra work that confidence-based refinement adds. This gives a concrete
tuning target: preserve the LER gain while bringing each stage within the control
loop's round budget.

[ML evidence](docs/ML_DECODER.md) contains the counts, intervals and timing
calculations. [ML setup and usage](MLfolders/mlDecoder/README.md) covers the
models, raw measurements, training records and decoder API.

## Scaling to larger quantum computers

The work points to five priorities:

- **Design around the device.** Connectivity, readout behavior and calibration
  determine how well a patch runs. Native scheduling and physical error
  characterization become more valuable as circuits grow.
- **Choose codes for the workload.** Compare surface-code constructions by
  their logical-gate implementations as well as memory LER. Include the cost of
  connectivity, frame changes and decoding across gates in that comparison.
- **Make inference scale.** Full lookup grows exponentially with detector count.
  The telescope replaces history-indexed tables with shared recurrent weights.
  Carry that approach into larger spatial layouts and train across code distances.
- **Bring decoding into the control loop.** Stream syndromes, measure transfers
  and queueing, and optimize time per round alongside LER. Tune confidence and
  fallback work to the controller's deadline.
- **Measure across distances and operations.** Extend to larger code distances,
  longer memories and logical gates, using common held-out records to compare
  decoders. That connects the improvements here to sustained error suppression
  and useful computation.

## Reproduce the results

Run the core comparison locally with Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]" -e ./ethan
python -m examples.reproduce
python -m challenge.audit_garnet_routing
```

Audit the ML evidence with `python -m MLfolders.mlDecoder.evidence`. The
[optional ML environment](MLfolders/mlDecoder/README.md) also supports
`python -m MLfolders.mlDecoder.evidence --decode` to replay the frozen models.
Recompute the Emerald comparison with
`python -m MLfolders.mlDecoder.compare_emerald`.

[Reproduction details](docs/REPRODUCIBILITY.md) ·
[Emerald baseline](docs/EMERALD_BASELINE.md) · [Validation](VALIDATION.md)
