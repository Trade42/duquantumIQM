"""emerald-pulse: the readout map, the shot layout, and the one-job guard, offline against a fake Pulla."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from dqiqm import decode as D
from dqiqm import hw
from dqiqm import surface as S

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import test_hw_playlist as THP  # noqa: E402 -- its fake Pulla and run-definition builder

_spec = importlib.util.spec_from_file_location("emerald_pulse", REPO / "scripts" / "emerald_pulse.py")
E = importlib.util.module_from_spec(_spec)
sys.modules["emerald_pulse"] = E
_spec.loader.exec_module(E)

RECORD = E.RAW_ROOT / "20261025-emerald-pulse-pilot" / "compile.json"


@pytest.fixture(scope="module")
def spec():
    """EP0 as its placement.json entry records it: enough for the readout map and the detectors."""
    e = json.loads(E.PLACEMENT.read_text())[E.CIRCUIT]
    return SimpleNamespace(physical={int(k): v for k, v in e["placement"].items()}, stim=S.memory_circuit(3, 3, "Z"))


@pytest.fixture(scope="module")
def ep0_raw(spec):
    raw, _ = D.load_memory(E.PC.EP0_JOB, 0, circuit=spec.stim)
    return raw


def labels(spec):
    return [f"{q}__c_33_0_{j}" for j, q in enumerate(E.expected_qubits(spec))]


def test_the_committed_record_follows_from_its_labels_and_the_placement(spec):
    record = json.loads(RECORD.read_text())
    assert record["calibration_set_id"] == E.CALSET and record["shape"]["circuits"] == 1
    assert record["shape"]["shots"] == record["shape"]["averaging_bins"] == E.SHOTS
    assert record["qpu_runtime_s"] <= E.RUNTIME_CAP_S
    assert record["iqm_post_processing_matches_readout_map"] and record["context_deepcopy_ok"]
    assert E.readout_map(record["return_parameters"], E.expected_qubits(spec)) == record["readout_map"]
    assert sorted(record["readout_map"]) == sorted(labels(spec))
    assert {r["method"] for r in record["requests"]} == {"GET"}


def test_the_readout_map_checks_every_label(spec):
    qs = E.expected_qubits(spec)
    good = labels(spec)
    assert E.readout_map(good + ["QB3__only_probe"], qs) == {lb: j for j, lb in enumerate(good)}
    wrong = good[:]
    wrong[0] = wrong[0].replace(qs[0], "QB54")
    with pytest.raises(ValueError, match="measured on"):
        E.readout_map(wrong, qs)
    with pytest.raises(ValueError, match="cover"):
        E.readout_map(good[1:], qs)
    with pytest.raises(ValueError, match="two labels"):
        E.readout_map(good + [good[3]], qs)


def test_ep0s_shots_laid_out_as_sweep_results_come_back(spec, ep0_raw):
    rmap = E.readout_map(labels(spec), E.expected_qubits(spec))
    results = {lb: ep0_raw[:, j].astype(np.int8) for lb, j in rmap.items()}
    assert np.array_equal(E.to_raw(results, rmap), ep0_raw)
    det, _ = D.detection_events(spec.stim, E.to_raw(results, rmap))
    assert np.array_equal(det, np.load(E.PC.EP0_DATASET / "det_events.npy"))
    first = next(iter(rmap))
    for bad, match in (({first: results[first] + 0.5j}, "thresholded"), ({first: results[first][:10]}, "thresholded"),
                       ({first: results[first] * 2}, "0/1")):
        with pytest.raises(ValueError, match=match):
            E.to_raw(results | bad, rmap)


def test_newcombe_and_the_minimum_detectable_difference():
    lo, hi = E.newcombe(129, 1000, 129, 1000)
    assert lo < 0 < hi and abs(lo + hi) < 1e-12
    assert 0.04 < E.min_detectable(0.13, 1000) < 0.05


def test_the_arming_must_be_exactly_the_briefs():
    E.check_arming(dict(E.ARMING))
    for k, v in (("DQ_QPU_MAX_SHOTS", "2000"), ("DQ_QPU_MAX_JOBS", "2"), ("DQ_QPU_TASK", "emerald-pulse")):
        with pytest.raises(SystemExit):
            E.check_arming(E.ARMING | {k: v})
    with pytest.raises(SystemExit, match="ledger"):
        E.check_arming(E.ARMING | {"DQ_QPU_LEDGER": "/tmp/other.md"})


# ---------------------------------------------------------------------------
# preflight and submit_once, against test_hw_playlist's fake Pulla
# ---------------------------------------------------------------------------


class NoCopy:
    def __deepcopy__(self, memo):
        raise TypeError("cannot copy")


@pytest.fixture
def armed(monkeypatch, tmp_path):
    ledger = THP.write_ledger(tmp_path / "ledger.md")
    for k, v in E.ARMING.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DQ_QPU_LEDGER", str(ledger))
    monkeypatch.setattr(E, "runtime_estimate", lambda rd: 0.409)
    monkeypatch.setattr(E.PC, "guard_installed", lambda: False)
    return ledger


def case(**info):
    rd = THP.run_definition(segments=1, shots=E.SHOTS, qubits=("QB14", "QB21"))
    base = {"playlist_sha256": hw.playlist_shape(rd).playlist_sha256, "shape": {"circuits": 1, "shots": 1000, "averaging_bins": 1000}}
    record = base | {"calibration_set_id": E.CALSET}
    return rd, {"readout_metrics": "rm"}, base | info, record


def test_preflight_passes_and_hands_on_the_slim_context(armed, tmp_path):
    rd, ctx, info, record = case()
    pre = E.preflight(rd, ctx | {"builder": NoCopy()}, info, record, calset_now=E.CALSET, ledger=armed, raw_dir=tmp_path / "raw")
    assert sorted(pre["context"]) == ["mapped_readout_keys", "readout_metrics"]


@pytest.mark.parametrize("what", ["guard", "calset", "row", "job dir", "record", "shape", "runtime", "context"])
def test_preflight_refuses_without_connecting(armed, tmp_path, monkeypatch, what):
    monkeypatch.setattr(hw, "_connect_pulla", lambda *a, **k: (_ for _ in ()).throw(AssertionError("connected")))
    rd, ctx, info, record = case()
    calset, raw_dir = E.CALSET, tmp_path / "raw"
    if what == "guard":
        monkeypatch.setattr(E.PC, "guard_installed", lambda: True)
    elif what == "calset":
        calset = "00000000-0000-4000-8000-000000000000"
    elif what == "row":
        THP.write_ledger(armed, [hw.ledger_line(task=E.TASK, device="emerald", calibration_set_id=E.CALSET, qubits=["QB1"],
                                                circuits=1, shots=1000, job_id="x", purpose="p")])
    elif what == "job dir":
        (raw_dir / "some-job").mkdir(parents=True)
    elif what == "record":
        info = info | {"playlist_sha256": "0" * 64}
    elif what == "shape":
        info = info | {"shape": {"circuits": 1, "shots": 1000, "averaging_bins": 1}}
        record = record | {"shape": info["shape"]}
    elif what == "runtime":
        monkeypatch.setattr(E, "runtime_estimate", lambda rd: 10.5)
    elif what == "context":
        ctx = {"readout_metrics": NoCopy()}
    with pytest.raises((SystemExit, TypeError)):
        E.preflight(rd, ctx, info, record, calset_now=calset, ledger=armed, raw_dir=raw_dir)
    assert not E.EP.ledger_rows(armed, None) or what == "row"


def test_one_job_one_row_and_the_raw_saved(armed, tmp_path, monkeypatch):
    rd, ctx, info, record = case()
    fake, calls = THP.fake_pulla(monkeypatch, armed, {"QB14__c_33_0_0": [np.zeros(1000, np.int8)]})
    pre = E.preflight(rd, ctx, info, record, calset_now=E.CALSET, ledger=armed, raw_dir=tmp_path / "raw")
    res = E.submit_once(rd, pre["context"], ledger=armed, raw_root=tmp_path / "raw", timeout=5)
    assert res["ok"] and res["job_id"] == THP.JOB_ID and calls == [("emerald", False)]
    (row,) = E.EP.ledger_rows(armed, E.TASK)
    assert (row.circuits, row.shots, row.calibration_set_id) == (1, 1000, E.CALSET) and "[pulse,playlist=" in row.purpose
    assert fake.submitted[0][1] == pre["context"] and fake.submitted[0][2] is False
    assert (tmp_path / "raw" / f"{row.time[:10].replace('-', '')}-{E.TASK}" / THP.JOB_ID / "meta.json").is_file()
    with pytest.raises(SystemExit, match="already has"):
        E.preflight(rd, ctx, info, record, calset_now=E.CALSET, ledger=armed, raw_dir=tmp_path / "other")


def test_a_server_refusal_is_recorded_with_one_fail_closed_row(armed, tmp_path, monkeypatch):
    rd, ctx, info, record = case()
    response = requests.Response()
    response.status_code = 403
    refusal = requests.HTTPError("403 Forbidden: pulse-level access requires a paid organisation", response=response)
    THP.fake_pulla(monkeypatch, armed, None, submit_raises=refusal)
    res = E.submit_once(rd, ctx, ledger=armed, raw_root=tmp_path / "raw", timeout=5)
    assert not res["ok"] and res["ledger_job_id"].startswith(hw.UNKNOWN_JOB_PREFIX)
    assert res["cause"]["http_status"] == 403 and res["cause"]["error_class"] == "requests.exceptions.HTTPError"
    assert res["error"]["error_class"] == "dqiqm.hw.JobMayExistError"
    (row,) = E.EP.ledger_rows(armed, E.TASK)
    assert row.job_id == res["ledger_job_id"] and "submit raised HTTPError" in row.purpose
    with pytest.raises(SystemExit, match="already has"):  # no retry
        E.preflight(rd, ctx, info, record, calset_now=E.CALSET, ledger=armed, raw_dir=tmp_path / "raw2")
