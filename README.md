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

The improvements work across physical execution and classical inference:

- **Choose a better patch.** Rank legal placements using calibrated gate and
  readout quality.
- **Simplify native execution.** Cancel redundant rotations and schedule gates
  around their dependencies. X memory fuses terminal readout; Z memory preserves
  measurement boundaries while simplifying the gates between them.
- **Protect idle data qubits.** Apply XY4 dynamical decoupling during eligible
  waits. One, two and three rounds add 0, 72 and 144 compiled pulse instructions.
- **Model the hardware.** Build decoder priors from the executed native gates
  and readout calibration. X also models disturbance following ancilla readout.
- **Decode X with correlations.** Correlated minimum-weight perfect matching
  (MWPM) uses the calibrated fault probabilities and relationships between faults.
- **Learn Z from hardware.** Use 6,000 independent training shots per round count
  to select correlation weights and update modeled logical probabilities with
  observed counts. Model probabilities support rare and unseen histories.

Both versions use 17 qubits and 24 CZ gates per round, with zero routing SWAPs.
The savings come from simpler control operations and better use of the hardware;
decoupling adds pulses to protect the remaining waits. Measurement order and
logical frames stay aligned through compilation and decoding.

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

The work points to four priorities:

- **Design around the device.** Connectivity, readout behavior and calibration
  determine how well a patch runs. Native scheduling and physical error
  characterization become more valuable as circuits grow.
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
