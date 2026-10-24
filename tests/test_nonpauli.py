import numpy as np
import pytest
import stim

pytest.importorskip("qiskit_aer")
from challenge.aer_simulation import simulate_nonpauli
from challenge.noise_models import NoiseModel
from challenge.calibration import CalibrationSnapshot, QubitCalibration


def test_coherent_rotation_survives_exact_simulation():
    ideal = stim.Circuit("QUBIT_COORDS(1,1) 0\nR 0\nTICK\nH 0\nTICK\nH 0\nTICK\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    result = simulate_nonpauli(ideal,NoiseModel.uniform(0),shots=6000,coherent_z_angle=.4,seed=3)
    assert abs(result["obs_flips"].mean()-np.sin(.2)**2) < .01


def test_nonunital_t1_relaxes_towards_zero():
    ideal = stim.Circuit("QUBIT_COORDS(1,1) 0\nQUBIT_COORDS(2,2) 1\nR 0 1\nTICK\nX 0\nTICK\nH 1\nTICK\nH 1\nTICK\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    qubit = QubitCalibration(0,0,0,100,200)
    snapshot = CalibrationSnapshot({"QB1":qubit,"QB2":qubit},{"QB1|QB2":0},
                                    {"1q":50,"2q":50,"measure":50,"reset":50})
    model = NoiseModel.uniform(0).with_changes(calibration=snapshot,coherence=True)
    result = simulate_nonpauli(ideal,model,shots=6000,qubit_map={0:"QB1",1:"QB2"},seed=4)
    # Two 50ns idles at T1=100ns; observable flips are relative to ideal |1>.
    assert abs(result["raw_meas"].mean()-np.exp(-1)) < .025
