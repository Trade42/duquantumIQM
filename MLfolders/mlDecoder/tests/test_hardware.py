import json
import shutil
import numpy as np
import pytest

from mlDecoder import data, hardware
from mlDecoder.data import DatasetSpec

CAL = hardware.device_calibration("garnet", "a18a947f", "pA", window_ns=1096)
STANDIN = hardware.inforecord_root() / "sim" / "datasets" / "E1" / "p0_E1_d3_Z_r03_pA"


def spec(noise="calibrated_coherence", rounds=3, basis="Z", **options):
    return DatasetSpec(distance=3, rounds=rounds, basis=basis, noise=noise, p=0.001,
                       calibration=hardware.device_calibration("garnet", "a18a947f", options.pop("patch", "pA"),
                                                               window_ns=1096, **options))


def test_calibration_names_are_canonical():
    noise = hardware.DeviceNoise.parse(CAL)
    assert noise.text() == CAL and (noise.patch, noise.t2, noise.window_ns) == ("pA", "ramsey", 1096.0)
    for bad in (CAL.replace("t2=ramsey", "t2=hahn"), CAL.replace(":qnd=none", ""), CAL.replace("window=1096", "window=1096.0"),
                CAL.replace("a18a947f", "a18a947"), "synthetic:7"):
        with pytest.raises(ValueError):
            hardware.DeviceNoise.parse(bad)
    assert hardware.is_device(CAL) and not hardware.is_device("synthetic:7") and not hardware.is_device(None)
    with pytest.raises(ValueError):            # A device patch goes with a calibrated model only.
        data.make_circuits(DatasetSpec(noise="uniform", calibration=CAL))
    with pytest.raises(ValueError):            # leak=readout goes with calibrated_leakage only.
        data.make_circuits(spec(leak="readout"))


def test_snapshot_from_the_vendored_package():
    snap = hardware.snapshot(CAL)
    pkg = hardware.package("garnet", "a18a947f")
    assert len(snap.qubits) == 20 and len(snap.edges) == 30
    assert snap.durations_ns == {"1q": 16.0, "2q": 40.0, "measure": 1096.0, "reset": hardware.RESET_NS}
    row = pkg["qubits"]["QB20"]
    q = snap.qubits["QB20"]
    assert q.readout_error == pytest.approx((float(row["p1_given_0"]) + float(row["p0_given_1"])) / 2)
    assert q.t2_ns == pytest.approx(float(row["t2_ramsey_ns"])) and q.leakage_per_gate == 0
    echo = hardware.snapshot(CAL.replace("t2=ramsey", "t2=echo")).qubits["QB2"]
    mid = hardware.snapshot(CAL.replace("t2=ramsey", "t2=mid")).qubits["QB2"]
    assert snap.qubits["QB2"].t2_ns < mid.t2_ns < echo.t2_ns <= 2 * echo.t1_ns
    leaky = hardware.snapshot(hardware.device_calibration("garnet", "a18a947f", "pA", window_ns=1096, leak="readout"))
    assert leaky.qubits["QB11"].leakage_per_gate > 0 and leaky.qubits["QB2"].leakage_per_gate == 0  # Ancilla vs data.
    assert {i["qb"] for i in snap.metadata["imputed"]} == {"QB8", "QB19"}   # No measure.constant init error reported.


@pytest.mark.parametrize("patch", ["pA", "pB"])
def test_qubit_map_by_coordinates(patch):
    s = spec(patch=patch)
    mapping = data._qubit_map(s)
    placed = hardware.package("garnet", "a18a947f")["placement"]["E1"][patch]["qubits"]
    assert mapping == {int(k): v["qb"] for k, v in placed.items()} and len(mapping) == 17
    ideal, _ = data.make_circuits(s)
    coords = ideal.get_final_qubit_coordinates()
    shifted = tuple(sorted((q, (xy[0] + 2, xy[1])) for q, xy in coords.items()))
    with pytest.raises(ValueError):
        hardware._qubit_map(s.calibration, shifted)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_readout_window_extension(basis):
    rounds = 4
    s = spec(rounds=rounds, basis=basis)
    ideal, noisy = data.make_circuits(s)
    assert noisy.without_noise() == ideal                       # The team's ideal circuit, untouched.
    moments = hardware._moments(noisy)
    meas = [i for i, m in enumerate(moments) if hardware._measured(m)]
    mapping = data._qubit_map(s)
    anc = hardware._measured(moments[meas[0]])
    for i in meas[:-2]:                                          # Mid-circuit layers: ancillas idle too.
        idled = {t.value for inst in moments[i] if inst.name == "PAULI_CHANNEL_1" for t in inst.targets_copy()}
        assert anc <= idled and len(idled) == 17
    term = hardware.terminal_readout(s.calibration, mapping)
    for i in meas[-2:]:                                          # Terminal layer: no idle, measure_fidelity readout.
        assert not any(inst.name == "PAULI_CHANNEL_1" for inst in moments[i])
        for inst in moments[i]:
            if inst.name == "M":
                (t,) = inst.targets_copy()
                assert inst.gate_args_copy()[0] == pytest.approx(term[t.value])
    pauli = data.make_circuits(spec(noise="calibrated_pauli", rounds=rounds, basis=basis))[1]
    assert not any(inst.name == "PAULI_CHANNEL_1" and not inst.gate_args_copy()[0] == inst.gate_args_copy()[2]
                   for inst in pauli.flattened())               # No idle channels in calibrated_pauli.
    qnd = data.make_circuits(spec(rounds=rounds, basis=basis, qnd="excess"))[1]
    flips = [i for i in qnd.flattened() if i.name == "X_ERROR"]
    assert len(flips) - sum(i.name == "X_ERROR" for i in noisy.flattened()) == 8 * (rounds - 1)
    noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


def test_asymmetric_readout():
    rng = np.random.default_rng(0)
    raw = np.zeros((200_000, 2), dtype=bool)
    raw[:, 1] = True
    hardware.apply_readout_flips(raw, np.array([0.01, 0.01]), np.array([0.2, 0.2]), rng)
    assert raw[:, 0].mean() == pytest.approx(0.01, abs=0.002) and 1 - raw[:, 1].mean() == pytest.approx(0.2, abs=0.005)
    s = spec(readout="asym")
    ideal, _ = data.make_circuits(s)
    p10, p01 = hardware.readout_flip_table(s.calibration, data._qubit_map(s), ideal)
    assert len(p10) == ideal.num_measurements and p01.mean() > 1.5 * p10.mean()   # P(0|1) dominates on Garnet.
    a, b = data.sample_shots(s, 3000, 5, packed=False), data.sample_shots(s, 3000, 5, packed=False)
    assert np.array_equal(a[0], b[0]) and a[0].shape == (3000, ideal.num_detectors)    # Seeded.
    packed, obs = data.Stream(s, seed=2).next(1000)
    assert packed.shape == (1000, 3) and obs.shape == (1000, 1)
    sym = data.sample_shots(spec(), 40_000, 5, packed=False)[0].mean(0)
    asym = data.sample_shots(s, 40_000, 5, packed=False)[0].mean(0)
    layout = data.detector_layout(ideal)
    assert asym[layout["is_final"]].mean() < sym[layout["is_final"]].mean()     # |0>-heavy final layer reads cleaner.


needs_standin = pytest.mark.skipif(not STANDIN.exists(), reason="InfoRecord's stand-in run is not on this machine")


@needs_standin
def test_load_standin_run_and_split():
    run = hardware.load_run(STANDIN)
    assert run.label == "SIMULATED" and run.dets.shape == (1000, 24) and run.raw_meas.shape == (1000, 33)
    s = spec()
    val = data.inforecord_dataset(s, run, "validation")
    fine = data.inforecord_dataset(s, run, "finetune")
    assert (val.shots, fine.shots) == (200, 800) and val.manifest["label"] == "SIMULATED"
    assert np.array_equal(val.dets, run.dets[run.split == "validation"]) and val.raw.shape == (200, 33)
    with pytest.raises(ValueError):
        data.inforecord_dataset(s, run, "test")                    # The pilot has no test shots.
    with pytest.raises(ValueError):
        data.inforecord_dataset(spec(rounds=4), run, "finetune")   # Wrong rounds for the run.
    with pytest.raises(ValueError):
        data.inforecord_dataset(spec(patch="pB"), run, "finetune")  # Wrong patch.


@needs_standin
def test_loader_refuses_tampering(tmp_path):
    root = tmp_path / "IR"
    run_dir = root / "datasets" / "E1" / STANDIN.name
    shutil.copytree(STANDIN, run_dir)
    (root / "circuits" / "E1").mkdir(parents=True)
    for name in ("E1_d3_Z_r03_pA.stim", "E1_d3_Z_r03_pA.qasm"):
        shutil.copy(hardware.inforecord_root() / "circuits" / "E1" / name, root / "circuits" / "E1" / name)
    hardware.load_run(run_dir)
    dets = np.load(run_dir / "det_events.npy")
    dets[0, 0] ^= 1
    np.save(run_dir / "det_events.npy", dets)
    with pytest.raises(ValueError, match="sha256"):
        hardware.load_run(run_dir)
    shutil.copy(STANDIN / "det_events.npy", run_dir / "det_events.npy")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["spec"]["rounds"] = 4
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rounds"):
        hardware.load_run(run_dir)
