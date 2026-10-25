# The IQM challenge submission

The experiment connects native execution, syndrome inference and the requested
logical-error metric into one surface-code memory pipeline on Garnet.

| Document | Contents |
|---|---|
| [Retained improvements](../improvements.md) | Complete inventory of the final Garnet improvements and settings |
| [ML decoder evidence](ML_DECODER.md) | Learned cascade, held-out records, accuracy and latency |
| [Methods](METHODS.md) | Every technique used, its mechanism, implementation and X/Z role |
| [Evaluation](EVALUATION.md) | Vanilla baseline, raw failures, LER fits and comparison |
| [Error analysis](ERROR_ANALYSIS.md) | Remaining physical and model error pathways |
| [Reproduction](REPRODUCIBILITY.md) | Installation, offline evaluation and inference interface |
| [References](REFERENCES.md) | Primary literature and software documentation |
| [Emerald vanilla baseline](EMERALD_BASELINE.md) | Native d3 X/Z reference, 18,000 measured shots and offline replay |

The [challenge brief](../challenge/README.md) specifies the memory experiment,
compilation and decoding task. Our implementation uses Garnet's native
connectivity for the seventeen-qubit distance-three patch.
