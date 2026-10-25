import numpy as np
import pytest








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
    assert validated_memory_profile('Z')['decoder']['family']=='posterior_10'
    isolated=validated_memory_profile('Z');isolated['decoder']['parameters']['phase_round']=.5
    assert validated_memory_profile('Z')['decoder']['parameters']['phase_round']==0
    captured={}
    def run(*args,**kwargs):captured.update(kwargs);return 'offline'
    monkeypatch.setattr(hardware,'run_qiskit',run)
    assert hardware.run_optimized_memory(token='offline-token',basis='Z')=='offline'
    assert captured['hardware_profile']=='validated' and captured['quantum_computer']=='garnet'
