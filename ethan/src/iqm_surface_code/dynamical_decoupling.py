"""DD-off compilation option for the preserved vanilla comparator.

The final optimized policy inserts XY4 through hardware_optimization.
"""
from dataclasses import replace


def iqm_compilation_options(sequence, *, options=None, target_names=()):
    if sequence != "off":
        raise ValueError("The vanilla comparator supports DD off only; optimized memory uses audited XY4.")
    from iqm.iqm_client import CircuitCompilationOptions, DDMode
    if options is None:
        options = CircuitCompilationOptions()
    return replace(options, dd_mode=DDMode.DISABLED, dd_strategy=None)
