import numpy as np
import pytest

from mlDecoder import data, hardware, sequence
from mlDecoder.data import DatasetSpec
from mlDecoder.sequence import Component, Mixture

torch = pytest.importorskip("torch")
from mlDecoder.models import rnn_lstm  # noqa: E402

BACKENDS = {"rnn_lstm": rnn_lstm}
TINY = {"rnn_lstm": dict(hidden=16, head=16)}
TRAIN = dict(batch_size=256, max_train_shots=256 * 12, eval_every_seconds=1e9, eval_every_steps=4, anneal_by_time=False,
             cpu_threads=2, lr=3e-3, workers=1)
UNIFORM = Mixture(3, "Z", (Component("uniform", 0.003, None, 1.0),), (1, 2, 3), curriculum_fraction=0.0)


def spec(rounds, basis="Z", noise="uniform"):
    return DatasetSpec(distance=3, rounds=rounds, basis=basis, noise=noise, p=0.003)


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2, 5])
def test_layout_covers_every_detector_once(rounds, basis):
    lay = sequence.layout_for(3, rounds, basis)
    idx = lay["index"][lay["index"] < lay["num_detectors"]]
    assert sorted(idx.tolist()) == list(range(lay["num_detectors"]))
    assert lay["index"].shape == (rounds + 1, 8) and sequence.rounds_from_width(lay["num_detectors"], 3) == rounds
    assert lay["exists"][0].sum() == 4 and lay["exists"][-1].sum() == 4          # Basis checks only, first and last.
    raw = lay["raw_index"]
    assert raw.shape == (rounds, 8) and len(set(raw.ravel())) == raw.size
    x = torch.zeros((1, lay["num_detectors"]), dtype=torch.uint8)
    x[0, 5] = 1
    r, f = sequence.gather(x, dict(index=torch.from_numpy(lay["index"])))
    assert float(r.sum() + f.sum()) == 1
    with pytest.raises(ValueError):
        sequence.rounds_from_width(lay["num_detectors"] + 1, 3)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_decode_shapes_order_and_validation(name):
    _, noisy = data.make_circuits(spec(3))
    decoder = BACKENDS[name].build(noisy, dict(model=TINY[name]))
    for rounds in (1, 3, 4):                                     # Any R from the width.
        ds = data.make_dataset(spec(rounds), "val", 600)
        out = decoder.decode_batch(ds.dets, enable_correlations=True)
        assert out.shape == (600, 1) and out.dtype == bool
        perm = np.random.default_rng(0).permutation(600)
        assert np.array_equal(decoder.decode_batch(ds.dets[perm]), out[perm])
        assert np.array_equal(decoder.decode_batch(ds.dets, chunk=7), out)
        assert np.array_equal(decoder.decode_dataset(ds), out)
        trivial = ~ds.dets.any(axis=1)
        assert not out[trivial].any()
    assert decoder.decode_batch(ds.dets[:1]).shape == (1, 1) and decoder.decode_batch(ds.dets[:0]).shape == (0, 1)
    assert "enable_correlations" in decoder.metadata()["ignored_options"]
    with pytest.raises(ValueError):
        decoder.decode_batch(ds.dets[:, 1:])
    with pytest.raises(ValueError):
        decoder.decode_batch(ds.dets * 2)


def _fit(name, seed, train=UNIFORM, val=None, cfg=None, decoder=None):
    _, noisy = data.make_circuits(spec(3))
    val = val or [data.make_dataset(spec(r), "val", 800, seed=1) for r in (1, 3)]
    if decoder is None:
        decoder = BACKENDS[name].build(noisy, dict(model=TINY[name], train={**TRAIN, **(cfg or {})}))
    else:
        decoder.config["train"] = {**TRAIN, **(cfg or {})}
    info = decoder.fit(train, val, device="cpu", seed=seed, log=lambda *_: None)
    return decoder, info


@pytest.mark.parametrize("name", list(BACKENDS))
def test_cpu_training_is_reproducible_freezes_and_reloads(name, tmp_path):
    a, info = _fit(name, 5)
    b, _ = _fit(name, 5)
    c, _ = _fit(name, 6)
    assert a.weights_sha256() == b.weights_sha256() != c.weights_sha256()
    assert info["steps"] == 12 and info["fresh_shots"] == 12 * 256 and info["reproducible"].startswith("bit")
    assert all(not p.requires_grad for p in a.net.parameters())
    a.save(tmp_path / "m.pt")
    _, noisy = data.make_circuits(spec(4))
    loaded = BACKENDS[name].build(noisy, {}).__class__.load(tmp_path / "m.pt", noisy, "cpu")
    ds = data.make_dataset(spec(4), "test", 500)
    assert loaded.weights_sha256() == a.weights_sha256()
    assert np.array_equal(loaded.decode_batch(ds.dets), a.decode_batch(ds.dets))
    a.save(tmp_path / "h.pt", half=True)                       # fp16 weights for the committed copies.
    half = a.__class__.load(tmp_path / "h.pt", noisy, "cpu")
    assert (half.decode_batch(ds.dets) == a.decode_batch(ds.dets)).mean() > 0.99
    _, other = data.make_circuits(spec(3, basis="X"))
    with pytest.raises(ValueError):
        a.__class__.load(tmp_path / "m.pt", other, "cpu")


def test_leakage_mixture_uses_fresh_seeded_chunks():
    mix = Mixture(3, "Z", (Component("uniform", 0.003, None, 0.5), Component("leakage", 0.003, None, 0.5)), (2,),
                  curriculum_fraction=0.0)
    decoder, info = _fit("rnn_lstm", 1, train=mix, cfg=dict(workers=2, max_train_shots=256 * 6))
    assert info["fresh_shots"] + info["reused_shots"] == info["shots_seen"] == 256 * 6
    from mlDecoder.train import sample_chunk
    leak = mix.components[1].spec(3, 2, "Z")
    first = sample_chunk(leak, 256, data.chunk_seed(leak, "train", 1, data.LEAKAGE_STREAM_BASE), False)
    again = sample_chunk(leak, 256, data.chunk_seed(leak, "train", 1, data.LEAKAGE_STREAM_BASE), False)
    assert np.array_equal(first[0], again[0])                  # Chunks are seeded by spec, run seed and index.


def test_fine_tuning_on_datasets_stays_near_the_start():
    base, _ = _fit("rnn_lstm", 2)
    start = {k: v.clone() for k, v in base.net.state_dict().items()}
    pools = [data.external_dataset(spec(r), *(lambda d: (d.dets, d.obs))(data.make_dataset(spec(r), "train", 1500, seed=4)),
                                   dict(dataset_id=f"ft{r}"), split="finetune") for r in (2, 3)]
    _, info = _fit("rnn_lstm", 2, train=pools, cfg=dict(l2_to_init=1.0, lr=1e-3), decoder=base)
    assert info["pool_shots"] == 3000 and info["l2_to_init"] == 1.0 and info["epochs"] > 0
    moved = max(float((base.net.state_dict()[k] - v).abs().max()) for k, v in start.items() if v.is_floating_point())
    assert 0 < moved < 0.1


def test_raw_record_variant_needs_the_raw_record():
    s = DatasetSpec(distance=3, rounds=2, basis="Z", noise="calibrated_coherence", p=0.001,
                    calibration=hardware.device_calibration("garnet", "a18a947f", "pA", window_ns=1096, readout="asym"))
    ds = data.make_dataset(s, "val", 500, keep_raw=True)
    assert ds.raw.shape == (500, ds.circuit.num_measurements)
    again = data.make_dataset(s, "val", 500)
    assert np.array_equal(again.dets_packed, ds.dets_packed)   # asym specs sample through the raw record anyway.
    decoder = rnn_lstm.build(ds.circuit, dict(model=dict(TINY["rnn_lstm"], raw_record=True)))
    out = decoder.decode_dataset(ds)
    assert np.array_equal(out, decoder.decode_batch(ds.dets, raw_meas=ds.raw))
    with pytest.raises(ValueError):
        decoder.decode_batch(ds.dets)
    with pytest.raises(ValueError):
        decoder.decode_dataset(again)
