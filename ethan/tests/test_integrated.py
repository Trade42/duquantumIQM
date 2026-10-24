import numpy as np
import pytest
from iqm_surface_code.benchmark_integrated import summarize,PROFILES


def test_summary_counts_every_shot_and_retains_nonzero_uncertainty():
    truth=np.array([[0],[1],[1],[0]],dtype=bool)
    result,failed=summarize(np.zeros((4,1),dtype=bool),truth)
    assert result['errors']==2 and result['shots']==4
    assert failed.tolist()==[False,True,True,False]
    result,_=summarize(truth,truth)
    assert result['ci95'][1]>0


def test_prespecified_physical_ablation_profiles():
    assert PROFILES['current']==dict(native=False,dd='xx',fast=False)
    assert PROFILES['vanilla']['dd']=='off'
    assert PROFILES['native_off']['native'] and PROFILES['fast_off']['fast']
    assert PROFILES['protected_xx']['native']=='retain'
    assert PROFILES['fenced_xx']['native']=='fenced'


def test_public_runner_dispatches_frozen_profile_and_logical_options(monkeypatch,tmp_path):
    from iqm_surface_code import run_on_hardware as hardware
    from iqm_surface_code import hardware_optimization as integration
    captured={}
    def fake(backend,**kwargs):captured.update(kwargs);return {'job_id':'offline'}
    monkeypatch.setattr(integration,'run_integrated_qiskit',fake)
    selection=dict(profile='fast_adaptive',decoder=dict(family='lut',parameters=dict(phase_round=.05,backaction_scale=1)))
    result=hardware.run_qiskit(backend=object(),token='offline-token',basis='X',rounds=3,shots=8,
        hardware_profile=selection,logical_operations=['X','X'],optimize_logical_gates=True,artifact_dir=tmp_path/'run')
    assert result['job_id']=='offline'
    assert captured['profile']==selection and captured['source_options']['optimize_logical_gates']
    assert captured['source_options']['logical_operations']==['X','X']
    with pytest.raises(ValueError,match='owns DD'):
        hardware.run_qiskit(backend=object(),hardware_profile=selection,dd='xx')


def test_public_decode_preserves_selected_adapter(monkeypatch):
    from iqm_surface_code import run_on_hardware as hardware
    from types import SimpleNamespace
    chosen=object();captured={}
    def decode(syndromes,circuit,**kwargs):captured.update(kwargs);return SimpleNamespace(probability=.1,standard_error=.02)
    monkeypatch.setattr(hardware,'decode_syndromes',decode)
    assert hardware.decode_hardware_results(dict(decoder=chosen,decoder_circuit=object()))==(.1,.02)
    assert captured['matcher'] is chosen


def test_validated_policy_selects_confirmed_basis_specific_profiles(monkeypatch):
    from iqm_surface_code.hardware_optimization import validated_memory_profile
    from iqm_surface_code import run_on_hardware as hardware
    assert validated_memory_profile('X')['profile']=='fused_xy4'
    assert validated_memory_profile('X')['decoder']['family']=='mwpm'
    assert validated_memory_profile('X')['decoder']['parameters']==dict(phase_round=0.,backaction_scale=.5)
    assert validated_memory_profile('Z')['profile']=='fenced_xy4'
    assert validated_memory_profile('Z')['decoder']['family']=='lut'
    isolated=validated_memory_profile('Z');isolated['decoder']['parameters']['phase_round']=.5
    assert validated_memory_profile('Z')['decoder']['parameters']['phase_round']==0
    captured={}
    def run(*args,**kwargs):captured.update(kwargs);return 'offline'
    monkeypatch.setattr(hardware,'run_qiskit',run)
    assert hardware.run_optimized_memory(token='offline-token',basis='Z')=='offline'
    assert captured['hardware_profile']=='validated' and captured['quantum_computer']=='garnet'


def test_pilot_search_never_forces_a_worse_advanced_profile():
    from iqm_surface_code.benchmark_integrated import choose_pilot_candidate
    cases=[dict(profile=p,basis='X',pilot_score=dict(errors=k),pulse_audit=dict(total_dd_added_iq_pulses=100))
           for p,k in [('current',100),('native_xx',150),('fenced_xy4',120),('vanilla',90)]]
    assert choose_pilot_candidate(cases,'X')['profile']=='current'


def test_release_rejects_confirmed_regression_and_accepts_clear_gain():
    from iqm_surface_code.benchmark_integrated import qualified_release
    manifest=dict(calibration_set_id='offline',selection={},cases=[])
    for basis,previous,selected in [('X',200,300),('Z',200,100)]:
        manifest['selection'][basis]=dict(profile='fenced_xy4',decoder=dict(family='lut',parameters={}))
        for condition,k in [('current',previous),('integrated',selected)]:
            manifest['cases'].append(dict(phase='confirmation',condition=condition,basis=basis,rounds=3,
                analysis=dict(errors=k,shots=1500,probability=k/1500)))
    result=qualified_release(manifest)
    assert not result['X']['promoted'] and result['X']['profile']=='current'
    assert result['Z']['promoted'] and result['Z']['profile']=='fenced_xy4'
