# Native compiler implementation

The package turns annotated surface-code memory into a native Garnet circuit,
audits the physical pulse schedule and supplies a matching detector prior.
Its exact lookup implementation computes logical-class probabilities by XOR
convolution over joint DEM fault masks.

The complete [methods](../docs/METHODS.md) describe circuit construction,
embedding, cancellation, readout fences/fusion, XY4, priors and lookup.
[Implementation source map](../IMPLEMENTATION.md) links the corresponding files.
