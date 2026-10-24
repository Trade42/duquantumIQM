# torch and qiskit-aer each ship an OpenMP runtime; on macOS, loading Aer's
# first makes torch's CPU kernels segfault. Load torch first when installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass
