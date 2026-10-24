import json
import numpy as np
import pytest

from iqm_surface_code.benchmark_garnet import (
    analyze_counts, calibration_from_observations, read_token,
    select_native_layout, server_timing, update_report,
)
from iqm_surface_code.noise_models import NoiseModel, add_noise
from iqm_surface_code.surface_code import make_stim_circuit


@pytest.mark.parametrize("contents", ["local-test-token", '"local-test-token"',
                                     '{"api_key":"local-test-token"}', '{"IQM_TOKEN":"local-test-token"}'])
def test_token_formats(tmp_path, contents):
    path = tmp_path/"credential"
    path.write_text(contents)
    assert read_token(path) == "local-test-token"


@pytest.mark.parametrize("contents", ["", "   ", "{}", "[]", '{"token":5}', "two words"])
def test_token_invalid_never_echoes_content(tmp_path, contents):
    path = tmp_path/"credential"
    path.write_text(contents)
    with pytest.raises(ValueError) as exc:
        read_token(path)
    assert contents.strip() not in str(exc.value) or not contents.strip()


def calibration_fixture():
    arch = dict(qubits=["QB1","QB2"],calibration_set_id="test",gates={
        name:dict(default_implementation=impl,implementations={impl:dict(loci=[["QB1","QB2"]])})
        for name,impl in (("prx","drag"),("cz","flux"),("measure","constant"),("reset_wait","wait"))})
    def obs(key, value, unit="", invalid=False):
        return dict(dut_field=key,value=value,unit=unit,invalid=invalid)
    quality = [obs("metrics.irb.cz.flux.QB2__QB1.fidelity:par=d2",.995)]
    pulse=[]
    for q in arch["qubits"]:
        quality += [obs(f"metrics.rb.prx.drag.{q}.fidelity:par=d2",.999),
                    obs(f"metrics.ssro.measure.constant.{q}.fidelity",.98),
                    obs(f"characterization.model.{q}.t1_time",40,"us"),
                    obs(f"characterization.model.{q}.t2_time",20,"us")]
        for name,impl,value in (("prx","drag",16e-9),("cz","flux",40e-9),
                                ("measure","constant",500e-9),("reset_wait","wait",300e-6)):
            pulse.append(obs(f"gates.{name}.{impl}.{q}.duration",value))
    return arch,dict(observations=quality),dict(observations=pulse)


def test_live_calibration_units_implementation_and_invalid_flags():
    arch,q,p=calibration_fixture()
    q["observations"].append(dict(dut_field="metrics.rb.prx.unused.QB1.fidelity",value=.5))
    cal=calibration_from_observations(arch,q,p)
    assert cal.qubits["QB1"].single_qubit_infidelity == pytest.approx(.001)
    assert cal.qubits["QB1"].t1_ns == 40000
    assert cal.durations_ns["1q"] == pytest.approx(16)
    assert cal.durations_ns["reset"] == pytest.approx(300000)
    assert cal.edge_error("QB1","QB2") == pytest.approx(.005)
    q["observations"][0]["invalid"]=True
    with pytest.raises(ValueError,match="Missing native CZ"):
        calibration_from_observations(arch,q,p)


def test_live_calibration_rejects_unhandled_units_and_locus_overrides():
    arch,q,p=calibration_fixture()
    q["observations"][3]["unit"]="unknown"
    with pytest.raises(ValueError,match="time unit"):
        calibration_from_observations(arch,q,p)
    arch["gates"]["prx"]["override_default_implementation"]={"QB1":"other"}
    with pytest.raises(ValueError,match="overrides"):
        calibration_from_observations(arch,q,p)


def test_hardware_count_scoring_and_order():
    ideal=make_stim_circuit(3,3,basis="X",gate_basis="cz")
    samples=ideal.compile_sampler(seed=5).sample(32)
    from collections import Counter
    counts=Counter("".join(str(int(b)) for b in row[::-1]) for row in samples)
    analysis,raw,syndromes=analyze_counts(counts,ideal,add_noise(ideal,NoiseModel.uniform()),expected_shots=32)
    assert analysis["errors"]==0
    assert raw.shape==(32,ideal.num_measurements)
    assert not syndromes["det_events"].any()
    assert analysis["ci95"][1]>0
    assert analysis["timing"]["decode_shots_per_second"]>0
    memory=["".join(str(int(b)) for b in row[::-1]) for row in samples]
    ordered,raw_ordered,_=analyze_counts(counts,ideal,add_noise(ideal,NoiseModel.uniform()),
        expected_shots=32,memory=memory)
    assert np.array_equal(raw_ordered,samples)
    assert ordered["shot_order"]=="original acquisition order"
    with pytest.raises(ValueError,match="memory"):
        analyze_counts(counts,ideal,add_noise(ideal,NoiseModel.uniform()),expected_shots=32,memory=memory[:-1])
    with pytest.raises(ValueError):
        analyze_counts(counts,ideal,add_noise(ideal,NoiseModel.uniform()),expected_shots=31)


def test_server_phase_timing():
    timeline=[dict(source="iqm-station-control",status=f"execution_{edge}",
                   timestamp=f"2026-10-24T19:00:0{second}+00:00")
              for edge,second in (("started",1),("ended",3))]
    assert server_timing(timeline)=={"execution_s":2.}


@pytest.mark.parametrize("basis", ["X","Z"])
def test_real_iqm_sdk_native_serialization_and_noiseless_records(basis):
    pytest.importorskip("iqm.qiskit_iqm")
    pytest.importorskip("networkx")
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    from iqm_surface_code.run_on_hardware import prepare_qiskit
    from iqm_surface_code.build_emerald_qubit_rotated import validate_layout
    backend=IQMFakeGarnet()
    # Bind the real hardware serializer to the offline architecture. No client
    # or credentials are created. This catches IQM's virtual-register layout.
    backend.serialize_circuit=IQMBackend.serialize_circuit.__get__(backend)
    couplings=[(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b))
               for a,b in backend.coupling_map.get_edges()]
    ideal=make_stim_circuit(3,3,basis=basis,gate_basis="cz")
    mapping,ranking=select_native_layout(ideal,couplings,backend.physical_qubits)
    assert ranking["candidates"]==4
    assert validate_layout(ideal,mapping,couplings)["routing_swaps"]==0
    prepared=prepare_qiskit(backend,rounds=3,basis=basis,qubit_map=mapping,dd="xx")
    serialized=backend.serialize_circuit(prepared["circuit"])
    edges={frozenset(pair) for pair in couplings}
    assert all(frozenset(i.locus) in edges for i in serialized.instructions if i.name=="cz")
    assert prepared["report"]["two_qubit_gate_count"]==72
    assert len(prepared["dd_target_qubits"])==9
    from qiskit import transpile
    aer=pytest.importorskip("qiskit_aer").AerSimulator(method="matrix_product_state")
    circuit=transpile(prepared["circuit"],aer,optimization_level=0)
    counts=aer.run(circuit,shots=32,seed_simulator=2).result().get_counts()
    analysis,_,_=analyze_counts(counts,ideal,prepared["decoder_circuit"],expected_shots=32)
    assert analysis["errors"]==0
    bad=dict(mapping)
    bad[next(iter(bad))]="QB999"
    with pytest.raises(ValueError):
        prepare_qiskit(backend,rounds=3,basis=basis,qubit_map=bad)


def test_no_native_layout_fails_without_routing():
    pytest.importorskip("networkx")
    ideal=make_stim_circuit(3,1,gate_basis="cz")
    with pytest.raises(ValueError,match="No native"):
        select_native_layout(ideal,[("QB1","QB2")],["QB1","QB2"])


def test_report_marks_unresolved_difference(tmp_path):
    manifest=dict(calibration_set_id="test",cases=[])
    for dd,errors in (("off",100),("xx",103)):
        from iqm_surface_code.decoding import wilson_interval
        manifest["cases"].append(dict(id=dd,basis="Z",rounds=3,dd=dd,status="DONE",client_wall_s=1,
            analysis=dict(errors=errors,shots=1000,probability=errors/1000,
                ci95=wilson_interval(errors,1000),timing=dict(decode_shots_per_second=1e6))))
    update_report(tmp_path,manifest)
    low,high=manifest["comparisons"][0]["ci95_absolute_reduction"]
    assert low<0<high
    assert json.loads((tmp_path/"manifest.json").read_text())["fits"]==[]
