"""emerald-compile: the compile file's lookup is per device; Garnet's prior is byte for byte what it was."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from dqiqm import surface as S
from dqiqm.experiments import device as dev
from dqiqm.experiments import emerald as EM
from dqiqm.experiments import memory

REPO = Path(__file__).resolve().parents[1]
GARNET_CALSET = "a18a947f-0408-4628-81ea-1715637e434f"
#: str(spec.noisy(nm)) for hardware_noise(<default Garnet snapshot>, spec) on main 79092cf, before the lookup changed
GARNET_PRIOR_SHA256 = {
    "E1_d3_Z_r03_pA": "c91f786a6829dc093d045f584669b3603d0d6b90fe6e6e9c86ddaf25838f28b4",
    "E1_d3_X_r05_pB": "af36e15596c71d90383a2d4827b1276aadda6c878d58b16e78893aa8073e5c0a",
}
#: the same on main for Emerald, which found no compile file there: the measure_ns fallback (360 ns)
EMERALD_MAIN_PRIOR_SHA256 = {
    "E1_d3_Z_r03_eA": "c799318236df0d8240aed66c2dd878b2e6944ee90d8b92e8c761446b90d6ccdf",
    "E1_d3_Z_r03_eB": "89b15b513528964cf3ab7c7e2bb36b2d2ba3d31401941f0aaea66a2081e97dfa",
}


def spec_from(placement: Path, name: str) -> SimpleNamespace:
    """A circuit as its ``placement.json`` entry records it: enough of a CircuitSpec for the priors."""
    e = json.loads(placement.read_text())[name]
    d, r, basis = int(e["params"]["distance"]), int(e["params"]["rounds"]), e["params"]["basis"]
    g = S.geometry(d)
    return SimpleNamespace(
        name=name, physical={int(k): v for k, v in e["placement"].items()}, roles={int(k): v for k, v in e["roles"].items()},
        cx_pairs=memory.cx_pairs(g), x_ancillas=tuple(g.x_ancillas), params=e["params"],
        noisy=partial(memory.memory_noisy, d, basis), stim=S.memory_circuit(d, r, basis))


def garnet_spec(name):
    return spec_from(REPO / "InfoRecord" / "circuits" / "E1" / "placement.json", name)


def emerald_spec(name):
    return spec_from(EM.RECORD / "circuits" / "E1" / "placement.json", name)


def _sha(spec, nm):
    return hashlib.sha256(str(spec.noisy(nm)).encode()).hexdigest()


def _ancillas(spec):
    return [spec.physical[q] for q, r in spec.roles.items() if r == "ancilla"]


def _compile_file(folder: Path, calset: str, device: str | None, windows: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    body = {"calibration_set_id": calset, "readout_windows": {"qubits": windows}} | ({"device": device} if device else {})
    path = folder / f"compile.{calset}.json"
    path.write_text(json.dumps(body))
    return path


@pytest.fixture(scope="module")
def garnet_cal():
    return dev.live_calibration()


@pytest.fixture(scope="module")
def emerald_cal():
    return EM.live_calibration()


# ---------------------------------------------------------------------------
# The lookup.
# ---------------------------------------------------------------------------


def test_compile_dir_is_per_device():
    assert dev.compile_dir("garnet") == dev.COMPILE_DIR == REPO / "data" / "calibration" / "garnet" / "pulse"
    assert dev.compile_dir("emerald") == REPO / "data" / "calibration" / "emerald" / "pulse"


def test_lookup_by_device_and_by_search(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "CALIBRATION_DIR", tmp_path)
    calset = "22222222-2222-2222-2222-222222222222"
    _compile_file(tmp_path / "emerald" / "pulse", calset, "emerald", {"QB1": {"measure_window_ns": 900.0}})
    w = dev.compiled_readout_windows(calset, device="emerald")
    assert w.device == "emerald" and w.of("QB1", "measure_window_ns") == 900.0
    assert dev.compiled_readout_windows(calset, device="garnet") is None  # not in Garnet's folder
    assert dev.compiled_readout_windows(calset).device == "emerald"  # no device: whichever folder holds it
    assert dev.compiled_readout_windows(calset, tmp_path / "emerald" / "pulse").device == "emerald"  # a directory still works
    assert dev.compiled_readout_windows("33333333-3333-3333-3333-333333333333") is None
    # a file in one device's folder that names another device is refused
    _compile_file(tmp_path / "garnet" / "pulse", calset, "emerald", {"QB1": {"measure_window_ns": 1.0}})
    with pytest.raises(ValueError, match="is a emerald compile, not garnet's"):
        dev.compiled_readout_windows(calset, device="garnet")
    with pytest.raises(ValueError, match="more than one device's folder"):
        dev.compiled_readout_windows(calset)


def test_hardware_noise_reads_its_own_devices_compile_file(emerald_cal, tmp_path, monkeypatch):
    spec = emerald_spec("E1_d3_Z_r03_eA")
    windows = {q: {"measure_window_ns": 700.0 + i, "measure_fidelity_window_ns": 650.0} for i, q in enumerate(spec.physical.values())}
    slowest = max(windows[q]["measure_window_ns"] for q in _ancillas(spec))
    monkeypatch.setattr(dev, "CALIBRATION_DIR", tmp_path)
    # only in Garnet's folder: Emerald's prior does not see it
    _compile_file(tmp_path / "garnet" / "pulse", EM.CALSET, None, windows)
    assert dev.hardware_noise(emerald_cal, spec)[1]["readout_window_source"].startswith("measure_ns")
    (tmp_path / "garnet" / "pulse" / f"compile.{EM.CALSET}.json").unlink()
    _compile_file(tmp_path / "emerald" / "pulse", EM.CALSET, "emerald", windows)
    _, info = dev.hardware_noise(emerald_cal, spec)
    assert info["readout_window_ns"] == slowest and info["idle_ns"] == slowest
    assert info["readout_window_source"].startswith("compiled measure_window_ns") and "emerald/pulse" in info["readout_window_source"]


# ---------------------------------------------------------------------------
# Garnet is unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GARNET_PRIOR_SHA256))
def test_garnet_prior_is_byte_for_byte_unchanged(garnet_cal, name):
    spec = garnet_spec(name)
    nm, info = dev.hardware_noise(garnet_cal, spec)
    assert _sha(spec, nm) == GARNET_PRIOR_SHA256[name]
    assert info["readout_window_ns"] == 1096.0
    assert info["readout_window_source"] == f"compiled measure_window_ns (data/calibration/garnet/pulse/compile.{GARNET_CALSET}.json)"
    w = dev.compiled_readout_windows(GARNET_CALSET, device="garnet")
    assert w.device == "garnet" and len(w.qubits) == 17 and w.of("QB5", "measure_window_ns") == 1096.0
    assert dev.compiled_readout_windows(GARNET_CALSET).source == w.source  # found the same way without a device

# ---------------------------------------------------------------------------
# Emerald: the committed compile file (the live run of scripts/pulla_check.py --devices emerald).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EMERALD_MAIN_PRIOR_SHA256))
def test_emerald_prior_reads_the_committed_compile_file(emerald_cal, name):
    spec = emerald_spec(name)
    w = dev.compiled_readout_windows(EM.CALSET, device="emerald")
    assert w is not None and w.device == "emerald" and w.calibration_set_id == EM.CALSET
    assert len(w.qubits) == 30  # every qubit of eA and eB
    assert all(w.of(q, k) is not None for q in spec.physical.values() for k in ("measure_window_ns", "measure_fidelity_window_ns"))
    nm, info = dev.hardware_noise(emerald_cal, spec)
    assert info["readout_window_source"] == f"compiled measure_window_ns (data/calibration/emerald/pulse/compile.{EM.CALSET}.json)"
    assert info["readout_window_ns"] == max(w.of(q, "measure_window_ns") for q in _ancillas(spec)) > 360.0
    assert info["final_readout"] == "measure_fidelity (unheralded)"
    # the fallback is main's prior byte for byte: what the batch decoded with before this task
    fb, finfo = dev.hardware_noise(emerald_cal, spec, windows=None)
    assert _sha(spec, fb) == EMERALD_MAIN_PRIOR_SHA256[name] and finfo["readout_window_ns"] == 360.0
    assert _sha(spec, nm) != EMERALD_MAIN_PRIOR_SHA256[name]
