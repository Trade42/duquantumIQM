import json
import numpy as np
import pytest

from mlDecoder import data
from mlDecoder.data import DatasetSpec

torch = pytest.importorskip("torch")
from mlDecoder import final  # noqa: E402
from mlDecoder.models import rnn_lstm  # noqa: E402
from mlDecoder.sequence import Component, Mixture  # noqa: E402


def test_promote_load_and_decode(isolated_results, tmp_path, monkeypatch):
    monkeypatch.setattr(final, "WEIGHTS", tmp_path / "weights")
    monkeypatch.setattr(final, "MANIFEST", tmp_path / "weights" / "manifest.json")
    spec = DatasetSpec(distance=3, rounds=2, basis="Z", noise="uniform", p=0.003)
    _, noisy = data.make_circuits(spec)
    decoder = rnn_lstm.build(noisy, dict(model=dict(hidden=16, head=16),
                                         train=dict(batch_size=128, max_train_shots=128 * 4, eval_every_steps=2,
                                                    eval_every_seconds=1e9, anneal_by_time=False, cpu_threads=1)))
    mix = Mixture(3, "Z", (Component("uniform", 0.003, None, 1.0),), (1, 2), curriculum_fraction=0.0)
    decoder.fit(mix, [data.make_dataset(spec, "val", 300, seed=1)], device="cpu", seed=0, log=lambda *_: None)
    run_dir = isolated_results / "rnn_lstm" / "r1"
    (run_dir / "models").mkdir(parents=True)
    decoder.save(run_dir / "models" / "garnet_pA_Z.pt")
    group = dict(name="garnet_pA_Z", device="garnet", calset="a18a947f", window_ns=1096, patch="pA", basis="Z")
    (run_dir / "result.json").write_text(json.dumps(dict(git=dict(commit="abc"), groups={
        "garnet_pA_Z": dict(group=group, model_file="models/garnet_pA_Z.pt", status="complete")})))
    (added,) = final.promote("rnn_lstm", "r1")
    assert added["stage"] == "pretrained" and added["dtype"] == "float16" and added["default"]
    loaded = final.load("garnet", "pA", "Z")
    test = data.make_dataset(DatasetSpec(distance=3, rounds=5, basis="Z", noise="uniform", p=0.003), "test", 2000)
    assert (loaded.decode_batch(test.dets) == decoder.decode_batch(test.dets)).mean() > 0.99
    with pytest.raises(KeyError):
        final.load("garnet", "pB", "Z")
    path = final.WEIGHTS / added["file"]
    path.write_bytes(path.read_bytes() + b"x")                  # A changed file is refused.
    with pytest.raises(ValueError, match="sha256"):
        final.load("garnet", "pA", "Z")
