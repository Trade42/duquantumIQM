import json
import numpy as np
import pytest

from challenge.decoding import make_matcher
from challenge.experiment import simulate_memory
from mlDecoder import data
from mlDecoder.data import DatasetSpec

SMALL = DatasetSpec(distance=3, rounds=2, basis="Z", noise="uniform", p=0.003)


def test_shapes_and_shot_order():
    ds = data.make_dataset(SMALL, "val", 2500, seed=3, chunk_shots=1000)
    _, noisy = data.make_circuits(SMALL)
    assert ds.dets.shape == (2500, noisy.num_detectors) and ds.dets.dtype == np.uint8
    assert ds.obs.shape == (2500, noisy.num_observables) and ds.obs.dtype == np.uint8
    assert ds.detector_coords.shape == (noisy.num_detectors, 3) and ds.shots == 2500
    # Chunks are concatenated in seed order: shot i of the dataset is shot i of its chunk.
    parts = [data.sample_shots(SMALL, n, data.chunk_seed(SMALL, "val", 3, k), packed=False)
             for k, n in enumerate((1000, 1000, 500))]
    assert np.array_equal(ds.dets, np.concatenate([p[0] for p in parts]))
    assert np.array_equal(ds.obs, np.concatenate([p[1] for p in parts]))
    rng = np.random.default_rng(0)
    dets, obs = data.sample_batch(SMALL, 7, rng)
    assert dets.shape == (7, noisy.num_detectors) and obs.shape == (7, 1)
    packed, obs = data.Stream(SMALL, seed=1).next(9)
    assert packed.shape == (9, (noisy.num_detectors + 7) // 8) and obs.shape == (9, 1)


@pytest.mark.parametrize("noise", ["uniform", "leakage"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_circuits_give_no_detection_events(noise, basis, monkeypatch):
    spec = DatasetSpec(distance=3, rounds=3, basis=basis, noise=noise, p=0.0)
    ideal, noisy = data.make_circuits(spec)
    dets, obs = noisy.compile_detector_sampler(seed=2).sample(200, separate_observables=True)
    assert not dets.any() and not obs.any()
    dets, obs = ideal.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert not dets.any() and not obs.any()
    ds = data.make_dataset(spec, "val", 500)
    assert not ds.dets.any() and not ds.obs.any()
    # Force the leakage sampler's path with zero rates: still no events.
    monkeypatch.setattr(data, "uses_leakage_sampler", lambda s: True)
    dets, obs, diag = data.sample_shots(spec, 500, 11, packed=False)
    assert not dets.any() and not obs.any() and diag["leakage_entries"] == 0


def test_split_streams_never_overlap():
    seeds = {}
    for split, run_seed in (("train", 0), ("train", 1), ("val", 0), ("val", 1), ("test", 0), ("pilot", 0)):
        for chunk in list(range(50)) + [data.STREAM_CHUNK]:
            seeds.setdefault(data.chunk_seed(SMALL, split, run_seed, chunk), []).append((split, run_seed, chunk))
    assert all(len(v) == 1 for v in seeds.values())
    other = DatasetSpec(distance=3, rounds=3)
    assert data.chunk_seed(other, "test") != data.chunk_seed(SMALL, "test")
    arrays = {split: data.make_dataset(SMALL, split, 3000).dets_packed for split in ("train", "val", "test")}
    assert not np.array_equal(arrays["train"], arrays["val"]) and not np.array_equal(arrays["val"], arrays["test"])
    assert np.array_equal(arrays["test"], data.make_dataset(SMALL, "test", 3000).dets_packed)  # Seeded.
    with pytest.raises(ValueError):
        data.make_dataset(SMALL, "test", 10, seed=5)  # Test shots are canonical.


def test_frozen_test_set_reloads_with_same_hashes(isolated_results):
    first = data.frozen_test_set(SMALL, 2000)
    again = data.frozen_test_set(SMALL, 2000)
    for key in ("dataset_id", "dets_sha256", "obs_sha256", "circuit_sha256"):
        assert first.manifest[key] == again.manifest[key]
    assert np.array_equal(first.dets, again.dets) and np.array_equal(first.obs, again.obs)
    directory = data.dataset_dir(SMALL, "test", 2000)
    assert (directory / "data.npz").exists() and json.loads((directory / "manifest.json").read_text())["shots"] == 2000
    with pytest.raises(ValueError):
        data.load_dataset(directory, DatasetSpec(distance=3, rounds=2, basis="X", p=0.003))
    with pytest.raises(FileExistsError):
        data.save_dataset(first, directory)  # Frozen: never overwritten.
    np.savez_compressed(directory / "data.npz", dets_packed=first.dets_packed, obs=1 - first.obs)
    with pytest.raises(ValueError, match="obs_sha256"):
        data.frozen_test_set(SMALL, 2000)


@pytest.mark.parametrize("noise", ["leakage", "uniform"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_sampling_matches_simulate_memory(noise, basis):
    spec = DatasetSpec(distance=3, rounds=2, basis=basis, noise=noise, p=0.003)
    shots, seed = 3000, 17
    result, circuit, decoder = simulate_memory(3, 2, basis=basis, model=data.noise_model(spec), shots=shots,
                                               batch_size=shots, seed=seed, gate_basis="cz")
    _, noisy = data.make_circuits(spec)
    assert str(circuit) == str(noisy) and result["circuit_sha256"] == data.circuit_sha256(noisy)
    assert result["leakage_decoder_mismatch"] == (noise == "leakage") == data.uses_leakage_sampler(spec)
    # simulate_memory seeds a leakage batch from its rng, a Pauli sampler directly from seed.
    batch_seed = int(np.random.default_rng(seed).integers(0, 2**63)) if noise == "leakage" else seed
    dets, obs, _ = data.sample_shots(spec, shots, batch_seed, packed=False)
    pred = make_matcher(noisy, correlated=True).decode_batch(dets, enable_correlations=True)
    assert int(np.any(pred != obs, axis=1).sum()) == result["errors"]
    assert np.allclose(dets.mean(axis=0), result["detector_firing_rates"])


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_detector_layout_covers_every_detector_once(distance, rounds, basis):
    spec = DatasetSpec(distance=distance, rounds=rounds, basis=basis)
    _, noisy = data.make_circuits(spec)
    layout = data.detector_layout(noisy)
    n = noisy.num_detectors
    grid = layout["grid"]
    assert sorted(grid[grid >= 0].tolist()) == list(range(n))
    assert grid.shape == (rounds + 1, distance**2 - 1)
    assert sum(s["type"] == "X" for s in layout["stabilizers"]) == (distance**2 - 1) // 2
    # Only memory-basis checks are deterministic at the first round and at the final data readout.
    assert set(layout["type"][layout["round"] == 0]) == {basis}
    assert set(layout["type"][layout["is_final"]]) == {basis}
    assert np.all(layout["round"][layout["is_final"]] == rounds)
    coords = noisy.get_detector_coordinates()
    assert all(tuple(coords[k][:2]) == (layout["x"][k], layout["y"][k]) for k in range(n))
    assert all(coords[k][2] == layout["round"][k] for k in range(n))


def test_choose_test_shots():
    sized = data.choose_test_shots(dict(p_hat=0.004))
    assert sized["expected_failures"] >= 100 and not sized["cap_binds"]
    assert data.choose_test_shots(dict(p_hat=0.0))["cap_binds"]
    assert data.choose_test_shots(dict(p_hat=1e-6))["shots"] == 10_000_000


def test_external_dataset_checks_widths():
    ds = data.make_dataset(SMALL, "val", 100)
    layout = data.detector_layout(ds.circuit)
    ext = data.external_dataset(SMALL, ds.dets, ds.obs, dict(device="test"), detector_table=layout)
    assert ext.manifest["source"] == "external" and np.array_equal(ext.dets, ds.dets)
    with pytest.raises(ValueError):
        data.external_dataset(SMALL, ds.dets[:, 1:], ds.obs, {})
    with pytest.raises(ValueError):
        data.external_dataset(SMALL, ds.dets, ds.obs, {}, split="train")


@pytest.mark.parametrize("noise", ["calibrated_pauli", "calibrated_leakage"])
def test_calibrated_sampling_matches_simulate_memory(noise):
    from challenge.build_emerald_qubit_rotated import rank_placements
    from challenge.calibration import synthetic_snapshot
    from challenge.surface_code import make_stim_circuit
    spec = DatasetSpec(distance=3, rounds=2, basis="Z", noise=noise, p=0.001, calibration="synthetic:7")
    # The placement experiment.main(--synthetic-calibration) picks, with a 10-round template.
    expected = rank_placements(make_stim_circuit(3, 10, gate_basis="cz"), calibration=synthetic_snapshot(8, 8, 7),
                               rows=8, cols=8)[0].stim_to_physical
    assert data._qubit_map(spec) == expected
    shots, seed = 2000, 5
    result, circuit, _ = simulate_memory(3, 2, basis="Z", model=data.noise_model(spec), shots=shots, batch_size=shots,
                                         seed=seed, gate_basis="cz", qubit_map=expected)
    _, noisy = data.make_circuits(spec)
    assert str(circuit) == str(noisy)
    batch_seed = int(np.random.default_rng(seed).integers(0, 2**63)) if data.uses_leakage_sampler(spec) else seed
    dets, obs, _ = data.sample_shots(spec, shots, batch_seed, packed=False)
    pred = make_matcher(noisy, correlated=True).decode_batch(dets, enable_correlations=True)
    assert int(np.any(pred != obs, axis=1).sum()) == result["errors"]
