import hashlib

import numpy as np
import pytest
import stim

from challenge.benchmark_rounds import digest
from challenge.replay_rounds import verify_case


def evidence(tmp_path):
    c = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR(0,4,0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    ideal = c.without_noise()
    for kind, circuit in (("source", ideal), ("native", ideal), ("decoder", c)):
        circuit.to_file(tmp_path/f"sample.{kind}.stim")
    raw = np.array([[0], [1], [0], [1]], bool)
    np.savez_compressed(tmp_path/"sample.shots.npz", raw_meas=raw, native_raw_meas=raw,
                        det_events=raw, obs_flips=raw)
    return dict(id="sample", source_sha256=hashlib.sha256(str(ideal).encode()).hexdigest(),
                native_sha256=hashlib.sha256(str(ideal).encode()).hexdigest(),
                model_sha256=digest(tmp_path/"sample.decoder.stim"), data_file="sample.shots.npz",
                dataset_sha256=digest(tmp_path/"sample.shots.npz"), shots=4, native_measurement_columns=[0])


def test_replay_reconstructs_every_raw_shot_in_both_frames(tmp_path):
    case = evidence(tmp_path)
    dets, obs = verify_case(tmp_path, case)
    assert dets.shape == obs.shape == (4, 1)
    assert np.array_equal(dets[:, 0], [0, 1, 0, 1])


def test_replay_rejects_same_width_wrong_observables_even_with_updated_dataset_hash(tmp_path):
    case = evidence(tmp_path)
    with np.load(tmp_path/"sample.shots.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["obs_flips"] ^= True
    np.savez_compressed(tmp_path/"sample.shots.npz", **arrays)
    case["dataset_sha256"] = digest(tmp_path/"sample.shots.npz")
    with pytest.raises(ValueError, match="observable frame"):
        verify_case(tmp_path, case)
