"""scripts/emerald_fresh.py offline: the allowance against the job plan, the guards, the 360 ns prior, the committed
raw data and the fresh records (split ``fresh``, out of the batch's pools)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dqiqm import inforecord as IR

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("emerald_fresh", REPO / "scripts" / "emerald_fresh.py")
F = importlib.util.module_from_spec(_spec)
sys.modules["emerald_fresh"] = F
_spec.loader.exec_module(F)

RAW = F.RAW_ROOT / "20261025-emerald-fresh"
JOB_IDS = {"F0": "01a17100-951b-773e-827b-8dc870c96307", "F1": "01a17101-70aa-7365-b524-5c9dae448536",
           "F2": "01a17102-85c9-748a-b2b8-38ea6e0a6d5c"}


def test_the_allowance_is_the_briefs():
    jobs = F.fresh_jobs(F.BR.plan_json("emerald"))
    plan = {j["label"]: j for j in F.BR.plan_json("emerald")["jobs"]}
    assert [j.label for j in jobs] == ["F0", "F1", "F2"]
    assert (jobs[0].names, jobs[0].shots) == (("E1_d3_Z_r03_eA",), 1000)
    assert jobs[1].names == tuple(plan["E-B3-1-E1"]["order"]) and jobs[1].shots == 2000
    assert jobs[2].names == tuple(plan["E-B3-2-E1"]["order"]) and jobs[2].shots == 2000
    assert sum(j.total_shots for j in jobs) == 45_000 == F.SEG.shots and len(jobs) == 3 == F.SEG.jobs
    assert F.SEG.arming() == {"DQ_QPU": "run", "DQ_QPU_TASK": "emerald-fresh", "DQ_QPU_MAX_JOBS": "3",
                              "DQ_QPU_MAX_SHOTS": "45000"}


def test_a_changed_plan_is_refused():
    plan = F.BR.plan_json("emerald")
    for j in plan["jobs"]:
        if j["label"] == "E-B3-2-E1":
            j["order"] = j["order"][:-1] + ["E1_d3_Z_r03_eA"]
    with pytest.raises(SystemExit):
        F.fresh_jobs(plan)


def test_the_split_is_fresh_and_the_purpose_parses():
    assert F.SEG.split == "fresh"
    job = F.fresh_jobs(F.BR.plan_json("emerald"))[0]
    p = F.BR.purpose(F.SEG, job)
    assert p == "emerald-fresh F0 E1 (1 circuits x 1000 shots)"
    assert F.BR.label_of(SimpleNamespace(purpose=p, task="emerald-fresh")) == "F0"


def test_arming_must_be_exact():
    with pytest.raises(SystemExit):
        F.BR.check_arming(F.SEG, {**F.SEG.arming(), "DQ_QPU_MAX_SHOTS": "46000"})
    with pytest.raises(SystemExit):
        F.BR.check_arming(F.SEG, {**F.SEG.arming(), "DQ_QPU_TASK": "emerald-b3"})


def test_the_prior_is_the_360ns_fallback(monkeypatch):
    seen = {}

    def fake(cal, spec, **kw):
        seen.update(kw)
        return "nm", {"idle_channel": "depolarize", "idle_ns": 360.0, "readout_window_ns": 360.0,
                      "readout_window_source": "measure_ns", "delay_ns": 0.0, "t2": "ramsey", "final_readout": "x"}

    monkeypatch.setattr(F.dev, "hardware_noise", fake)
    spec = SimpleNamespace(noisy=lambda nm: ("noisy", nm), stim=SimpleNamespace(num_observables=1))
    noisy, info, note = F.decoder_priors(SimpleNamespace(calibration_set_id="cs"), spec)
    assert seen == {"windows": None} and noisy == ("noisy", "nm")
    assert note.startswith(f"priors: {F.PRIORS}") and "360 ns" in F.PRIORS and "windows=None" in note
    # the override sits on this module's private copy of batch_run, never on the shared ``batch_run``
    assert F.BR.decoder_priors is F.decoder_priors
    assert F.BR.__name__ == "emerald_fresh_batch_run"
    shared = sys.modules.get("batch_run")
    assert shared is None or shared.decoder_priors is not F.decoder_priors


def test_the_committed_raw_data():
    seg = json.loads((RAW / "segment.json").read_text())
    assert seg["split"] == "fresh" and seg["allowance"] == {"jobs": 3, "shots": 45_000}
    assert {j["label"]: j["job_id"] for j in seg["jobs"]} == JOB_IDS
    jobs = {j.label: j for j in F.fresh_jobs(F.BR.plan_json("emerald"))}
    for label, job_id in JOB_IDS.items():
        man = json.loads((RAW / f"{label}.json").read_text())
        meta = json.loads((RAW / job_id / "meta.json").read_text())
        assert man["ok"] is True and man["job_id"] == job_id and man["split"] == "fresh"
        assert [o["circuit"] for o in man["order"]] == list(jobs[label].names)
        assert meta["job_id"] == job_id and meta["calibration_set_id"] == F.BR.DEVICES["emerald"].design_calset
        assert meta["shots"] == jobs[label].shots and meta["circuits"] == len(jobs[label].names)
        assert all(tuple(q) == F.PATCH_QUBITS for q in meta["physical_qubits"])
        with np.load(RAW / job_id / "memory.npz") as z:
            assert sorted(z.files) == sorted(f"c{i}" for i in range(len(jobs[label].names)))
            assert all(z[k].shape[0] == jobs[label].shots for k in z.files)
    assert json.loads((RAW / "F0.json").read_text())["pilot_pass"] is True


def _fresh_runs():
    return sorted(p for p in (F.RECORD / "datasets" / "E1").glob("fresh_F*") if p.is_dir())


@pytest.mark.skipif(not (F.SUMMARY_DIR / "fresh_summary.json").exists(), reason="analyze has not written the records")
def test_the_fresh_records():
    runs = _fresh_runs()
    assert len(runs) == 23
    for ds in runs:
        der = F.RECORD / "derived" / "E1" / ds.name
        assert IR.check_run(ds, der) == []
        m = json.loads((ds / "manifest.json").read_text())
        assert m["split"]["role"] == "fresh" and m["split"]["counts"] == {"fresh": m["shots"]}
        assert m["decoder"]["noise"].startswith(f"priors: {F.PRIORS}") and m["batch"] == F.BATCH
        with open(ds / "shots.csv") as f:
            assert {r["split"] for r in csv.DictReader(f)} == {"fresh"}
        checks = json.loads((der / "fresh_checks.json").read_text())
        assert checks["sanity_pass"] and checks["label"] == "HARDWARE" and checks["split"] == "fresh"
        assert checks["prior"]["idle_ns"] == 360.0
    s = json.loads((F.SUMMARY_DIR / "fresh_summary.json").read_text())
    assert s["priors"] == F.PRIORS and not s["record_problems"] and not s["sanity_failures"]
    for basis in ("Z", "X"):
        assert [p["rounds"] for p in s["points"][basis]] == [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
        assert "fixed_A" in s["fits"][basis]["fresh"]["all_R"] and "fixed_A" in s["fits"][basis]["fresh"]["R_le_3"]
    # the batch's pools never read a fresh record
    assert not any(p["run_id"].startswith("fresh") for p in F.BR.read_points(F.RECORD, "E1"))
