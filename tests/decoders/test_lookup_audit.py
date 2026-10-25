"""Independent checks of the complete logical-ML lookup calculation."""
from itertools import product

import numpy as np
import pytest
import stim

from challenge.audit_lookup import _xor_fault_inplace, direct_joint, paired_scores
from iqm_surface_code.decoders.lookup_table import (
    LookupTableDecoder, build_lookup_table, fwht_inplace,
)


def enumerate_faults(faults, detector_bits):
    probability = np.zeros(1 << (detector_bits + 1))
    for choices in product((0, 1), repeat=len(faults)):
        outcome, weight = 0, 1.
        for occurred, (p, mask) in zip(choices, faults):
            weight *= p if occurred else 1-p
            if occurred:
                outcome ^= mask
        probability[outcome] += weight
    return probability.reshape(2, -1)


@pytest.mark.parametrize("seed", range(12))
def test_lut_matches_enumeration_of_every_fault_combination(seed):
    rng = np.random.default_rng(seed)
    n = 4
    faults = [(float(rng.choice([0, .01, .13, .37, .5, .8, 1])), int(rng.integers(1, 1 << (n+1))))
              for _ in range(8)]
    lines = [f"detector D{n-1}", "logical_observable L0"]
    for p, mask in faults:
        targets = [f"D{i}" for i in range(n) if mask & (1 << i)]
        if mask & (1 << n):
            targets.append("L0")
        lines.append(f"error({p}) {' '.join(targets)}")
    dem = stim.DetectorErrorModel("\n".join(lines))
    expected = enumerate_faults(faults, n)
    np.testing.assert_allclose(direct_joint(dem), expected, rtol=1e-13, atol=1e-15)
    lut = build_lookup_table(dem)
    keys = np.arange(1 << n)
    histories = ((keys[:, None] >> np.arange(n)) & 1).astype(bool)
    predictions = lut.decode_batch(histories).ravel().astype(int)
    risk = expected[1-predictions, keys].sum()
    optimum = np.minimum(expected[0], expected[1]).sum()
    assert risk == pytest.approx(optimum, abs=1e-13)
    assert lut.metadata["model_bayes_failure"] == pytest.approx(optimum, abs=1e-13)


def test_separators_remain_joint_and_repeat_offsets_are_preserved():
    dem = stim.DetectorErrorModel("""
        detector D2
        error(0.1) D0 L0 ^ D1 L0
        repeat 2 {
            error(0.2) D0 L0
            shift_detectors 1
        }
    """)
    expected = enumerate_faults([(.1, 3), (.2, 1 | 8), (.2, 2 | 8)], 3)
    np.testing.assert_allclose(direct_joint(dem), expected, atol=1e-15)
    lut = build_lookup_table(dem)
    keys = np.arange(8)
    assert expected[1-lut.table, keys].sum() == pytest.approx(np.minimum(*expected).sum())


def test_lut_sums_logical_classes_instead_of_choosing_the_largest_fault_path():
    dem = stim.DetectorErrorModel("error(0.4) D0 L0\nerror(0.3) D0\nerror(0.3) D0")
    joint = direct_joint(dem)
    assert joint[0, 1] == pytest.approx(.252)
    assert joint[1, 1] == pytest.approx(.232)
    assert build_lookup_table(dem).table[1] == 0


def test_fwht_matches_explicit_parity_matrix():
    values = np.random.default_rng(17).normal(size=32)
    matrix = np.array([[(-1)**((i & j).bit_count()) for j in range(32)] for i in range(32)])
    transformed = fwht_inplace(values.copy())
    np.testing.assert_allclose(transformed, matrix @ values, atol=1e-13)
    np.testing.assert_allclose(fwht_inplace(transformed), 32*values, atol=1e-13)


@pytest.mark.parametrize("block_size", [1, 3, 16])
def test_blockwise_convolution_matches_independent_full_array_update(block_size):
    original = np.random.default_rng(19).random(32)
    original /= original.sum()
    for mask in (0, 1, 3, 8, 17, 31):
        for p in (0, .13, .5, 1):
            actual = original.copy()
            _xor_fault_inplace(actual, mask, p, block_size=block_size)
            expected = (1-p)*original + p*original[np.arange(32) ^ mask]
            np.testing.assert_allclose(actual, expected, atol=1e-16)


def test_save_load_and_syndrome_bit_order(tmp_path):
    lut = build_lookup_table(stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.2) D1\nerror(0.15) D2 L0"))
    path = tmp_path / "lut.npz"
    lut.save(path)
    loaded = LookupTableDecoder.load(path)
    keys = np.array([7, 0, 2, 5, 1, 6, 3, 4, 1])
    histories = ((keys[:, None] >> np.arange(3)) & 1).astype(bool)
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], bool)
    np.testing.assert_array_equal(loaded.decode_batch(histories)[:, 0], expected)
    assert not loaded.table.flags.writeable


def test_paired_hardware_score_counts_each_disagreement():
    truth = np.array([0, 0, 1, 1, 0, 1])
    lut = np.array([0, 1, 1, 1, 0, 0])
    mwpm = np.array([1, 0, 0, 1, 0, 0])
    score = paired_scores(lut, mwpm, truth)
    assert score["lut_failures"] == 2
    assert score["mwpm_failures"] == 3
    assert score["lut_only_wrong"] == 1
    assert score["mwpm_only_wrong"] == 2
    assert score["lut_minus_mwpm_probability"] == pytest.approx(-1/6)
