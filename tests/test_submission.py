import hashlib
import json
from pathlib import Path
import socket

import pytest


def test_submission_reproduces_both_hardware_arms_without_credentials(monkeypatch):
    from examples.reproduce import main
    original_open = Path.open

    def no_credentials(path, *args, **kwargs):
        if path.name in {"apikey.jom", ".env"}:
            raise AssertionError("Reproduction tried to read credentials.")
        return original_open(path, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Reproduction tried to use the network.")

    monkeypatch.setattr(Path, "open", no_credentials)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    result = main(["--json"])
    assert result["physical_shots"] == 54000
    expected = {"vanilla": {"X": [423, 972, 1220], "Z": [115, 409, 583]},
                "optimized": {"X": [159, 404, 728], "Z": [145, 639, 1147]}}
    for arm, bases in expected.items():
        for basis, errors in bases.items():
            assert result["measurements"][arm][basis]["errors"] == errors
    assert socket.create_connection is no_network


def test_bundle_integrity_rejects_changed_inference_state(tmp_path):
    from challenge.submission import verify_bundle
    (tmp_path / "state.dem").write_bytes(b"changed state")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        files={"state.dem": hashlib.sha256(b"published state").hexdigest()}, source_dependencies={})))
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        verify_bundle(tmp_path)
