import json
import socket

import pytest

from challenge.benchmark_emerald import (
    SHOT_LIMIT, main, reserve_submission, select_baseline_layout)
from iqm_surface_code.benchmark_vanilla_lut import vanilla_circuit

pytestmark = pytest.mark.filterwarnings("ignore:.*iqm.station_control.interface.models.circuit.*:DeprecationWarning")


def test_54_qubit_graph_embedding_is_native_fixed_and_unweighted():
    qubits = [f"QB{i+1}" for i in range(54)]
    edges = [(qubits[9*r+c], qubits[9*rr+cc]) for r in range(6) for c in range(9)
             for rr, cc in ((r+1,c),(r,c+1)) if rr < 6 and cc < 9]
    first, report = select_baseline_layout(vanilla_circuit(3,"Z"), edges, qubits)
    second, _ = select_baseline_layout(vanilla_circuit(3,"Z"), edges[::-1], qubits[::-1])
    assert first == second
    assert len(first) == len(set(first.values())) == 17
    assert report["routing_swaps"] == report["non_native_pairs"] == 0
    assert not report["calibration_weighted"]
    with pytest.raises(ValueError, match="No native"):
        select_baseline_layout(vanilla_circuit(1,"X"), [], qubits)


def test_submission_intent_is_durable_and_cannot_be_retried(tmp_path):
    case=dict(id="X_r1",shots=3000)
    manifest=dict(hardware_eligible=True,charged_shots=0,shot_limit=SHOT_LIMIT,cases=[case])
    reserve_submission(tmp_path,manifest,case)
    disk=json.loads((tmp_path/"manifest.json").read_text())
    assert disk["charged_shots"]==3000 and disk["cases"][0]["submit_intent_utc"]
    with pytest.raises(ValueError,match="already attempted"):
        reserve_submission(tmp_path,manifest,case)
    second=dict(id="Z_r1",shots=3000)
    manifest["cases"].append(second);manifest["shot_limit"]=3000
    with pytest.raises(ValueError,match="budget exceeded"):
        reserve_submission(tmp_path,manifest,second)
    manifest["hardware_eligible"]=False
    with pytest.raises(ValueError,match="offline fixture"):
        reserve_submission(tmp_path,manifest,second)


def test_cli_does_not_read_credentials_or_connect_without_submit(monkeypatch,tmp_path):
    import challenge.benchmark_emerald as m
    def forbidden(*args,**kwargs): raise AssertionError("Credential access before submission flag")
    monkeypatch.setattr(m,"read_token",forbidden)
    with pytest.raises(SystemExit):
        main(["--action","run","--run",str(tmp_path)])


def test_measured_emerald_baseline_replays_offline(monkeypatch):
    import challenge.benchmark_emerald as m
    from examples.replay_emerald import main as replay_main
    def forbidden(*args,**kwargs): raise AssertionError("Hardware/credential access during replay")
    monkeypatch.setattr(m,"read_token",forbidden)
    monkeypatch.setattr(socket.socket,"connect",forbidden)
    result=replay_main(["--json"])
    assert result["physical_shots"]==18000 and result["cases"]==6
    fits={fit["basis"]:fit for fit in result["fits"]}
    assert fits["X"]["errors"]==[341,681,859]
    assert fits["Z"]["errors"]==[46,301,456]
    assert fits["X"]["epsilon_per_round"]==pytest.approx(.1235233467,abs=1e-7)
    assert fits["Z"]["epsilon_per_round"]==pytest.approx(.0474888821,abs=1e-7)
