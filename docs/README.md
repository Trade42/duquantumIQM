# Documentation

The current project is organized around the retained Garnet memory policies:
X uses native correlated MWPM; Z uses a complete modeled posterior corrected
by independent training counts. Both retain the native compiler and XY4.
Acquisition is complete; all normal getting-started paths are offline.

| Read this | To understand |
|---|---|
| [Winning pipeline](WINNING_PIPELINE.md) | The end-to-end implementation, exact configuration and why X/Z differ |
| [Techniques](TECHNIQUES.md) | What every technique does, why it can help and its current status |
| [Remaining error sources](ERROR_ANALYSIS.md) | Measured residual failures, plausible physical channels and model limitations |
| [Results](RESULTS.md) | Counts, fits, confidence intervals, comparisons and what is established |
| [Reproducibility](REPRODUCIBILITY.md) | Offline installation, winner replay, full audit and test commands |
| [References](REFERENCES.md) | Primary literature, SDK references and source-code provenance |

## Completed studies

- [Headline selection](GARNET_HEADLINE.md): 300,000-shot comparison of the independent implementations and selected combinations.
- [Bounded follow-up](GARNET_ROUNDS.md): 96,000 new shots testing readout, DD and history-decoder proposals; no new promotion.
- [Initial native integration](../ethan/docs/INTEGRATED_GARNET.md): why native measurement scheduling and frame matching mattered.
- [Initial DD work](../ethan/docs/DYNAMICAL_DECOUPLING.md): simulator/native implementation background.

These reports preserve their original protocols. Current summary numbers are in
[Results](RESULTS.md); raw scientific artifacts remain in their original run
directories. Source files and saved states used by frozen inference are not
renamed or reformatted for presentation.

## Extensions and engineering

- [Original challenge brief](../challenge/README.md): requested endpoint and initial Emerald pipeline.
- [General implementation](../IMPLEMENTATION.md): circuit generator, model catalog and simulator options; historical setup notes are marked.
- [Transversal gates](../TRANSVERSAL_GATES.md): logical Pauli/H extensions and limitations; separate from the winning identity-memory experiment.
- [Validation history](../VALIDATION.md): earlier test and simulation milestones.
- [Contributing](../CONTRIBUTING.md) and [agent coordination](../AGENTS.md): ownership, preservation and validation rules.

Here, **measured** means acquired hardware shots, **modeled** means a declared
noise approximation, and **synthetic** means simulated data. They are never
interchanged when reporting a headline result.
