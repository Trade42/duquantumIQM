"""Backends, one module per decoder: mlp, mwpm (reference), and later stages' models.

There is no registry: ``mlDecoder.run`` imports ``mlDecoder.models.<name>`` and calls its
``build(circuit, config)``. The backend API is in CONTRACT.md.
"""
