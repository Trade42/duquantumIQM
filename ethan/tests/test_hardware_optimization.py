import math
import numpy as np
import pytest
import stim

from iqm_surface_code.hardware_optimization import r_clifford


@pytest.mark.parametrize('angle',[-math.pi,-math.pi/2,0,math.pi/2,math.pi,2*math.pi])
@pytest.mark.parametrize('phase',[0,math.pi/2,math.pi,3*math.pi/2,5*math.pi/2])
def test_native_r_clifford_matrix(angle,phase):
    qiskit=pytest.importorskip('qiskit')
    from qiskit.quantum_info import Operator
    qc=qiskit.QuantumCircuit(1);qc.r(angle,phase,0)
    actual=Operator(qc).data
    expected=stim.gate_data(r_clifford(angle,phase)).unitary_matrix
    phase_factor=np.vdot(expected,actual)/2
    assert np.allclose(actual,phase_factor*expected,atol=1e-7)


@pytest.mark.parametrize('phase',[math.pi/4,3*math.pi/4,5*math.pi/4,7*math.pi/4])
def test_native_diagonal_half_turn(phase):
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Operator
    from qiskit import QuantumCircuit
    q=QuantumCircuit(1);q.r(math.pi,phase,0)
    u=Operator(q).data;v=stim.gate_data(r_clifford(math.pi,phase)).unitary_matrix
    assert np.allclose(u,np.vdot(v,u)/2*v,atol=1e-7)


def test_nonclifford_rejected_and_identity_axis_irrelevant():
    assert r_clifford(0,.123)=='I'
    with pytest.raises(ValueError):r_clifford(math.pi/4,0)
