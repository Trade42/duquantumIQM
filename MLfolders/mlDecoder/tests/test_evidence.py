import hashlib
import json

import pytest

from mlDecoder import evidence


def test_evidence_audit_rejects_altered_imported_artifact(monkeypatch, tmp_path):
    package = tmp_path / "MLfolders/mlDecoder"
    package.mkdir(parents=True)
    artifact = package / "results_record/test/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"altered evidence")
    (package / "upstream_manifest.json").write_text(json.dumps(dict(
        files={"MLfolders/mlDecoder/results_record/test/result.json": hashlib.sha256(b"original evidence").hexdigest()})))
    monkeypatch.setattr(evidence, "HERE", package)
    monkeypatch.setattr(evidence, "REPOSITORY", tmp_path)
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        evidence.verify_artifacts()


def test_offline_audit_refuses_network_access():
    import socket
    with evidence.offline(), pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("example.com", 443))
