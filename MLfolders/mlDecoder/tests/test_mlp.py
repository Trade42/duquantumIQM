import subprocess
import sys
import numpy as np
import pytest

from mlDecoder import data
from mlDecoder.data import DatasetSpec

torch = pytest.importorskip("torch")
from mlDecoder.models import mlp  # noqa: E402

SPEC = DatasetSpec(distance=3, rounds=2, basis="Z", noise="uniform", p=0.003)
TINY_TRAIN = dict(batch_size=256, max_train_shots=256*60, eval_every_steps=20, lr=3e-3, cpu_threads=2,
                  max_wall_seconds=30, patience_evals=100)


def test_import_does_not_need_torch():
    code = ("import sys, mlDecoder, mlDecoder.data, mlDecoder.evaluate, mlDecoder.train, mlDecoder.models.mlp; "
            "assert 'torch' not in sys.modules, 'torch imported eagerly'")
    from pathlib import Path
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[2])


def test_decode_shapes_order_and_validation():
    _, noisy = data.make_circuits(SPEC)
    decoder = mlp.build(noisy, {})
    ds = data.make_dataset(SPEC, "val", 1000)
    x = ds.dets
    out = decoder.decode_batch(x, enable_correlations=True)
    assert out.shape == (1000, 1) and out.dtype == bool
    assert "enable_correlations" in decoder.metadata()["ignored_options"]
    assert decoder.decode_batch(x[:1]).shape == (1, 1)                       # Never squeezed.
    assert decoder.decode_batch(x[:0]).shape == (0, 1)
    perm = np.random.default_rng(0).permutation(1000)
    assert np.array_equal(decoder.decode_batch(x[perm]), out[perm])          # Shot order kept.
    assert np.array_equal(decoder.decode_batch(x, chunk=7), out)             # Batch-independent.
    assert np.array_equal(decoder.decode_packed(ds.dets_packed), out)
    trivial = ~x.any(axis=1)
    assert trivial.any() and not out[trivial].any()                          # Empty syndrome -> no flip.
    with pytest.raises(ValueError):
        decoder.decode_batch(x[:, 1:])
    with pytest.raises(ValueError):
        decoder.decode_batch(x * 2)


def _fit(seed, pool, val):
    _, noisy = data.make_circuits(SPEC)
    decoder = mlp.build(noisy, dict(model=dict(hidden_layers=2, width_factor=4, min_width=32), train=TINY_TRAIN))
    info = decoder.fit(pool, val, device="cpu", seed=seed, log=lambda *_: None)
    return decoder, info


def test_cpu_training_is_reproducible_and_freezes(tmp_path):
    pool = data.make_dataset(SPEC, "train", 6000, seed=1)
    val = data.make_dataset(SPEC, "val", 2000, seed=1)
    a, info = _fit(5, pool, val)
    b, _ = _fit(5, pool, val)
    c, _ = _fit(6, pool, val)
    assert a.weights_sha256() == b.weights_sha256() != c.weights_sha256()
    assert info["steps"] == 60 and info["pool_shots"] == 6000 and info["reproducible"].startswith("bit")
    assert all(not p.requires_grad for p in a.net.parameters())
    path = tmp_path / "m.pt"
    a.save(path)
    _, noisy = data.make_circuits(SPEC)
    loaded = mlp.MLPDecoder.load(path, noisy, "cpu")
    assert loaded.weights_sha256() == a.weights_sha256()
    assert np.array_equal(loaded.decode_batch(val.dets), a.decode_batch(val.dets))
    _, other = data.make_circuits(DatasetSpec(distance=3, rounds=2, basis="X", p=0.003))
    with pytest.raises(ValueError):
        mlp.MLPDecoder.load(path, other, "cpu")


def test_streamed_training_runs():
    val = data.make_dataset(SPEC, "val", 1000, seed=2)
    _, noisy = data.make_circuits(SPEC)
    decoder = mlp.build(noisy, dict(model=dict(hidden_layers=2, width_factor=4, min_width=32),
                                    train=dict(TINY_TRAIN, max_train_shots=256*10)))
    info = decoder.fit(SPEC, val, device="cpu", seed=0, log=lambda *_: None)
    assert info["pool_shots"] is None and info["shots_seen"] == 2560 and info["epochs"] is None


def test_fresh_leakage_stream_never_repeats_a_chunk():
    from mlDecoder.train import LeakageStreamSource
    spec = DatasetSpec(distance=3, rounds=2, basis="Z", noise="leakage", p=0.003)
    source = LeakageStreamSource(spec, 15_000, seed=1, workers=2)
    try:
        a, b = source.next(), source.next()
    finally:
        source.close()
    assert a[0].shape == b[0].shape == (15_000, 2) and not np.array_equal(a[0], b[0])
    first = data.sample_shots(spec, data.LEAKAGE_CHUNK, data.chunk_seed(spec, "train", 1, data.LEAKAGE_STREAM_BASE))
    assert np.array_equal(a[0][:data.LEAKAGE_CHUNK], first[0])       # Seeded, in order, from its own chunk range.
    pool_chunk = data.sample_shots(spec, data.LEAKAGE_CHUNK, data.chunk_seed(spec, "train", 1, 0))
    assert not np.array_equal(first[0], pool_chunk[0])
