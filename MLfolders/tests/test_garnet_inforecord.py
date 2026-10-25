"""The InfoRecord writer: the files match the column dictionary, shapes and order, packing, the LER record."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from dqiqm import inforecord as IR
from dqiqm import surface as S
from dqiqm.experiments import device as dev
from dqiqm.experiments import suite as SU


@pytest.fixture(scope="module")
def tiny():
    cal = dev.live_calibration()
    return SU.build(cal, experiments=("E1", "E3"), e1_rounds=(2, 3), e3_random=1, e3_repeats=3, rank_shots=1000)


def _write(tiny, root, name, raw, run_id, **kw):
    s = tiny[name]
    files = IR.write_circuits(root, [s], tiny.durations)
    dec = s.noisy(S.NoiseModel.uniform(0.003)) if s.noisy else None
    return IR.write_dataset(root, s, raw, run_id=run_id, batch=1, job_id="sim-x", position_in_job=0, arch=tiny.arch,
                            cal=tiny.cal, durations=tiny.durations, circuit_files=files[name], raw_ref={"path": "x", "key": "c0"},
                            calibration_paths={"before": "a", "after": "b"}, decoder_circuit=dec, decoder_note="uniform 0.003",
                            split_labels=SU.split("B1-2-E1", 1, len(raw)), split_info=SU.split_rule("B1-2-E1", 1, SU.split("B1-2-E1", 1, len(raw))),
                            simulated=True, **kw)


def test_dataset_matches_the_dictionary(tiny, tmp_path):
    name = "E1_d3_Z_r03_pA"
    s = tiny[name]
    noisy = S.memory_circuit(3, 3, "Z", noise=S.NoiseModel.uniform(0.01))
    raw = noisy.compile_sampler(seed=1).sample(300).astype(np.uint8)
    ds, numbers = _write(tiny, tmp_path, name, raw, "b1_" + name)
    der, summary = IR.write_derived(tmp_path, s, "b1_" + name, ds)
    assert IR.check_run(ds, der) == []
    m = json.loads((ds / "manifest.json").read_text())
    assert m["arrays"] == {"dets": "det_events.npy", "obs": "obs_flips.npy", "raw_meas": "raw_meas.npy"}
    assert m["spec"] | {} == m["spec"] and m["spec"]["no_reset"] is True and m["spec"]["d"] == 3 and m["spec"]["rounds"] == 3
    assert m["calibration_set_id"] == tiny.cal.calibration_set_id
    assert IR.load_array(ds, "raw_meas").tolist() == raw.tolist()
    dets = IR.load_array(ds, "dets")
    assert dets.shape == (300, 24) and IR.load_array(ds, "obs").shape == (300, 1)
    # detectors.csv: Stim's order and coordinates; round = t
    with open(ds / "detectors.csv") as f:
        rows = list(csv.DictReader(f))
    coords = s.stim.get_detector_coordinates()
    for r in rows:
        x, y, t = coords[int(r["index"])]
        assert (float(r["x"]), float(r["y"]), int(r["round"])) == (x, y, t)
    assert [int(r["index"]) for r in rows] == list(range(24))
    assert {r["boundary"] for r in rows} == {"first", "second", "bulk", "final"}
    finals = [r for r in rows if r["boundary"] == "final"]
    assert len(finals) == 4 and all(len(r["data_qbs"].split(";")) in (2, 4) for r in finals)
    # the record indices are the no-reset ones: m_k and m_(k-2)
    bulk = [r for r in rows if r["round"] == "2"]
    assert all(len(r["record"].split(";")) == 2 and int(r["record"].split(";")[0]) - int(r["record"].split(";")[1]) == 16
               for r in bulk)
    with open(ds / "measurements.csv") as f:
        meas = list(csv.DictReader(f))
    assert [int(r["round"]) for r in meas] == [0] * 8 + [1] * 8 + [2] * 8 + [3] * 9
    assert [r["qb"] for r in meas] == [s.physical[q] for q in IR.C.measured_qubits(s.stim)]
    with open(ds / "shots.csv") as f:
        shots = list(csv.DictReader(f))
    assert len(shots) == 300 and [int(r["shot"]) for r in shots] == list(range(300))
    assert sum(int(r["total_events"]) for r in shots) == int(dets.sum())
    assert all(int(r["sw_t0"]) + int(r["sw_t1"]) + int(r["sw_t2"]) + int(r["sw_t3"]) == int(r["total_events"]) for r in shots)
    assert numbers["pm_failures"] == sum(int(r["pm_fail"]) for r in shots)
    assert sum(r["split"] == "validation" for r in shots) == 60 and m["split"]["role"] == "finetune+validation"
    assert np.load(der / "pij.npy").shape == (24, 24)


def test_split_is_fixed_and_holds_out_batch_3():
    a, b = SU.split("B1-2-E1", 1, 5000), SU.split("B1-2-E1", 1, 5000)
    assert (a == b).all() and (a == "validation").sum() == 1000 and set(a) == {"finetune", "validation"}
    assert (SU.split("B1-3-E1", 1, 5000) != a).any()  # another job, another draw
    assert set(SU.split("B3-2-E1", 3, 100)) == {"test"}


def test_calibration_package(tiny, tmp_path):
    files = IR.write_calibration_package(tmp_path, tiny.cal, tiny)
    assert all(f.exists() for f in files)
    assert IR.check_csv(tmp_path / "qubits.csv", "calibration/qubits.csv") == []
    assert IR.check_csv(tmp_path / "couplers.csv", "calibration/couplers.csv") == []
    pl = json.loads((tmp_path / "placement.json").read_text())
    assert pl["calibration_set_id"] == tiny.cal.calibration_set_id
    for patch in ("pA", "pB"):
        roles = [v["role"] for v in pl["E1"][patch]["qubits"].values()]
        assert sorted(set(roles)) == ["X_ancilla", "Z_ancilla", "data"] and roles.count("data") == 9
    with open(tmp_path / "qubits.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 and rows[1]["qb"] == "QB2" and float(rows[1]["p0_given_1"]) == pytest.approx(0.0912)


def test_unknown_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not in the column dictionary"):
        IR.write_csv(tmp_path / "x.csv", "derived/streaks.csv", [], ["stabilizer", "length", "count", "extra"])
    with pytest.raises(ValueError, match="required column"):
        IR.write_csv(tmp_path / "x.csv", "derived/streaks.csv", [], ["stabilizer", "length"])


def test_packed_arrays_round_trip(tiny, tmp_path):
    name = "E1_d3_X_r02_pB"
    raw = tiny[name].stim.compile_sampler(seed=2).sample(37).astype(np.uint8)
    ds, _ = _write(tiny, tmp_path, name, raw, "b1_" + name, pack_threshold=0)
    m = json.loads((ds / "manifest.json").read_text())
    assert m["array_info"]["raw_meas.npy"]["packed"] and m["array_info"]["raw_meas.npy"]["shape"] == [37, 25]
    assert np.load(ds / "raw_meas.npy").shape == (37, 4)
    assert (IR.load_array(ds, "raw_meas") == raw).all()
    assert IR.check_run(ds) == []


def test_readout_run(tiny, tmp_path):
    name = "E3_repeat_all1_k03"
    s = tiny[name]
    raw = s.stim.compile_sampler(seed=3).sample(50).astype(np.uint8)
    ds, numbers = _write(tiny, tmp_path, name, raw, "b1_" + name)
    der, summary = IR.write_derived(tmp_path, s, "b1_" + name, ds)
    assert IR.check_run(ds, der) == []
    assert numbers["pm_failures"] is None and numbers["det_rate"] == 0.0
    for f in ("assignment.csv", "readout_flips.csv", "readout_corr.npy"):
        assert (der / f).exists()


def test_ler_record(tmp_path):
    points = [{"circuit": f"c{r}", "dataset": "d", "rounds": r, "shots": 1000, "failures": k, "raw_failures": 2 * k}
              for r, k in ((1, 30), (2, 55), (4, 100), (8, 180))]
    out, res = IR.write_ler(tmp_path, "E1", "Z_pA_b1", points, title="t", label="Z pA b1")
    assert IR.check_csv(out / "ler_points.csv", "ler/ler_points.csv") == []
    assert (out / "ler.png").exists()
    assert 0 < res["fixed_A"]["eps"] < 0.5 and 0 < res["free_A"]["amplitude"] <= 1


def test_dictionary_markdown_lists_every_column():
    md = IR.dictionary_markdown()
    for kind, cols in IR.COLUMNS.items():
        assert f"`{kind}`" in md
        for c in cols:
            assert f"`{c.name}`" in md
    for k in IR.MANIFEST:
        assert f"`{k}`" in md
