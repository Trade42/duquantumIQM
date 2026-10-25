# Import torch before qiskit/qiskit-aer: on macOS, loading their OpenMP runtimes in the other
# order segfaults torch's CPU kernels later in the same process (test_circuits then test_conv_decoders).
try:
    import torch  # noqa: F401
except ImportError:
    pass
