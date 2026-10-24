"""Paper Eq. (1) primitive for an iSWAP-capable backend.

This does not silently replace CX by CXSWAP: exchanges move quantum states.
A walking-code compiler must also update physical roles and parity records.
The Emerald path deliberately stays on the verified static CZ schedule.
"""
import stim


def cxswap_via_iswap(control=0, target=1):
    circuit = stim.Circuit()
    # Stim/Qiskit wire convention: pre-H target, post-H control. Validate
    # explicitly instead of assuming tensor ordering in the paper's Eq. (1).
    circuit.append("H", [target])
    circuit.append("ISWAP", [control, target])
    circuit.append("S_DAG", [control, target])
    circuit.append("H", [control])
    return circuit


def verify_exchange_identity():
    expected = stim.Circuit("CX 0 1\nSWAP 0 1")
    return expected.to_tableau() == cxswap_via_iswap().to_tableau()
