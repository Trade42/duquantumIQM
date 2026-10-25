import numpy as np
import pytest
from scipy.special import expit
import stim

from challenge.decoders.calibrated_lookup import _features, _update, compile_table, fit, load, load_policy
from iqm_surface_code.decoders.lookup_table import build_lookup_table


def model():
    return stim.DetectorErrorModel("error(0.2) D0 L0\nerror(0.1) D1")


def test_independent_training_repairs_an_incorrect_logical_prior(tmp_path):
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, (600, 2), dtype=np.uint8)
    validation = rng.integers(0, 2, (200, 2), dtype=np.uint8)
    test = rng.integers(0, 2, (200, 2), dtype=np.uint8)
    # An unmodeled logical channel reverses the class favored by the DEM.
    truth = lambda x: (1-x[:, :1]).astype(bool)
    decoder = fit(model(), train, truth(train), validation, truth(validation),
                  orders=(0,), penalties=(.01,), alphas=(0.,))
    assert np.count_nonzero(build_lookup_table(model()).decode_batch(test) != truth(test)) == 200
    np.testing.assert_array_equal(decoder.decode_batch(test), truth(test))
    assert decoder.metadata['selection']['order'] == 0
    assert decoder.metadata['train_shots'] == 600
    path = tmp_path/'policy.npz'
    decoder.save(path)
    frozen = load(path)
    np.testing.assert_array_equal(frozen.decode_batch(test[::-1]), truth(test)[::-1])
    assert frozen.decode_batch(test[:0]).shape == (0, 1)
    with pytest.raises(ValueError, match='correlations'):
        frozen.decode_batch(test, enable_correlations=False)


@pytest.mark.parametrize('order', [0, 1, 2])
@pytest.mark.parametrize('block', [3, 17])
def test_compiled_table_matches_classifier_on_every_history(order, block):
    rng = np.random.default_rng(91+order)
    n = 4
    prior = rng.uniform(.01, .99, 1 << n)
    keys = np.arange(1 << n)
    dets = ((keys[:, None] >> np.arange(n)) & 1).astype(bool)
    features = _features(dets, prior, order)
    coefficients = rng.normal(size=features.shape[1]+1)
    expected = expit(coefficients[0]+features @ coefficients[1:]) > .5
    table = compile_table(prior, n, coefficients=coefficients, order=order, block_size=block)
    np.testing.assert_array_equal(table, expected)


def test_validation_can_retain_the_original_rule():
    rng = np.random.default_rng(19)
    train = rng.integers(0, 2, (300, 2), dtype=np.uint8)
    validation = rng.integers(0, 2, (100, 2), dtype=np.uint8)
    decoder = fit(model(), train, train[:, :1], validation, validation[:, :1],
                  orders=(0, 1), penalties=(.01,), alphas=(0., 10.))
    assert decoder.metadata['selection']['order'] == -1
    np.testing.assert_array_equal(decoder.table, build_lookup_table(model()).table)


def test_history_counts_update_only_histories_seen_in_training():
    q = np.array([.1, .8, .2, .6])
    keys = np.array([0, 1, 2, 3], np.uint32)
    empirical = (np.array([1, 3]), np.array([20, 10]), np.array([0., 10.]))
    corrected = _update(q, keys, empirical, 10.)
    np.testing.assert_allclose(corrected, [.1, 8/30, .2, 16/20])
    np.testing.assert_array_equal(q, [.1, .8, .2, .6])


@pytest.mark.parametrize('order', [0, 1, 2])
def test_matching_feature_is_compiled_into_the_table(order):
    rng = np.random.default_rng(73)
    n = 4
    keys = np.arange(1 << n)
    dets = ((keys[:, None] >> np.arange(n)) & 1).astype(bool)
    prior = rng.uniform(.01, .99, 1 << n)
    matching = lambda x: np.logical_xor(x[:, 0], x[:, 3])[:, None]
    features = _features(dets, prior, order, matching(dets))
    coefficients = rng.normal(size=features.shape[1]+1)
    table = compile_table(prior, n, coefficients=coefficients, order=order,
                          matching=matching, block_size=3)
    expected = expit(coefficients[0]+features @ coefficients[1:]) > .5
    np.testing.assert_array_equal(table, expected)


def test_loading_rejects_corrupted_frozen_table(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, (100, 2), dtype=np.uint8)
    decoder = fit(model(), x[:60], x[:60, :1], x[60:], x[60:, :1], orders=(), alphas=(0.,))
    path = tmp_path/'original.npz'
    decoder.save(path)
    with np.load(path) as archive:
        table = archive['table'].copy()
        metadata = archive['metadata'].copy()
    table[0] ^= 1
    np.savez(path, table=table, metadata=metadata)
    with pytest.raises(ValueError, match='checksum'):
        load(path)


@pytest.mark.parametrize('kind', ['missing_validation', 'bad_truth', 'unsupported_observables'])
def test_invalid_fit_records_are_rejected(kind):
    x = np.zeros((8, 2), np.uint8)
    y = np.zeros((8, 1), np.uint8)
    dem, vd, vo = model(), x, y
    if kind == 'missing_validation': vd, vo = x[:0], y[:0]
    if kind == 'bad_truth': vo = y.copy(); vo[0] = 2
    if kind == 'unsupported_observables': dem = stim.DetectorErrorModel('error(.1) D0 L0 L1')
    with pytest.raises(ValueError):
        fit(dem, x, y, vd, vo)


def test_included_policies_reproduce_scores_without_live_matching(monkeypatch):
    import json
    from pathlib import Path
    import pymatching
    from challenge.submission import BUNDLE, _records, verify_bundle
    def forbidden(*args, **kwargs):
        raise AssertionError('A frozen LUT must not construct a matching graph at inference.')
    monkeypatch.setattr(pymatching.Matching, 'from_detector_error_model', forbidden)
    report = json.loads((Path(__file__).resolve().parents[2]/'benchmarks/calibrated_lookup/report.json').read_text())
    manifest = verify_bundle()
    for row in report['scores']:
        decoder = load_policy(row['basis'], row['rounds'])
        cases = [c for c in manifest['cases'] if
                 (c['arm'], c['basis'], c['rounds']) == ('optimized', row['basis'], row['rounds'])]
        errors = 0
        for case in cases:
            dets, truth = _records(BUNDLE, case)
            prediction = decoder.decode_batch(dets)
            assert prediction.shape == truth.shape
            errors += np.count_nonzero(prediction != truth)
        assert errors == row['failures']['calibrated']


@pytest.mark.parametrize('basis,rounds', [('Y', 1), ('X', 4)])
def test_included_policy_scope_is_explicit(basis, rounds):
    with pytest.raises(ValueError):
        load_policy(basis, rounds)
