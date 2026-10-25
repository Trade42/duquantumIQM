import hashlib
import json

import pytest


def test_bundle_integrity_rejects_changed_inference_state(tmp_path):
    from challenge.submission import verify_bundle
    (tmp_path / "state.dem").write_bytes(b"changed state")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        files={"state.dem": hashlib.sha256(b"published state").hexdigest()}, source_dependencies={})))
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        verify_bundle(tmp_path)


def test_retained_runtime_rejects_changed_source_and_preserves_historical_provenance(monkeypatch, tmp_path):
    import challenge.submission as submission

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps(dict(files={}, source_dependencies={"retired.py": "historical"})))
    source = tmp_path / "retained.py"
    source.write_bytes(b"published retained implementation")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(dict(
        published_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_dependencies={"retained.py": hashlib.sha256(source.read_bytes()).hexdigest()})))
    monkeypatch.setattr(submission, "BUNDLE", bundle)
    monkeypatch.setattr(submission, "REPOSITORY", tmp_path)
    monkeypatch.setattr(submission, "RUNTIME_SOURCES", runtime)
    assert submission.verify_bundle(bundle)["source_dependencies"] == {"retired.py": "historical"}
    source.write_bytes(b"altered implementation")
    with pytest.raises(ValueError, match="Scientific source hash mismatch"):
        submission.verify_bundle(bundle)
