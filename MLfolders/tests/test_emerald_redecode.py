"""scripts/emerald_redecode.py offline: the selection rule, McNemar, the first-round fit, the record names and labels,
and the committed selection and records. Nothing here builds the suite or decodes."""

import fnmatch
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dqiqm import inforecord as IR
from dqiqm import stats

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("emerald_redecode", REPO / "scripts" / "emerald_redecode.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["emerald_redecode"] = R
_spec.loader.exec_module(R)


def test_the_rule_fewer_failures_wins_and_a_tie_is_compiled():
    assert R.choose(100, 99) == "compiled"
    assert R.choose(100, 100) == "compiled"
    assert R.choose(99, 100) == "fallback"


def test_mcnemar_counts_the_discordant_shots():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    b = np.array([1, 0, 1, 0, 0, 0], bool)
    m = R.mcnemar(a, b)
    assert (m["a_only"], m["b_only"]) == (2, 1)
    assert m["p"] == pytest.approx(1.0)
    assert R.mcnemar(a, a) == {"a_only": 0, "b_only": 0, "p": 1.0}


def test_first_round_fit_recovers_its_parameters():
    r = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    eps, eps1, n = 0.05, 0.012, np.full(11, 10**7)
    p = 0.5 * (1 - (1 - 2 * eps1) * (1 - 2 * eps) ** (r - 1))
    f = R.fit_first_round(r, np.round(n * p), n, bootstrap=0)
    assert f["eps"] == pytest.approx(eps, abs=2e-4) and f["eps_1"] == pytest.approx(eps1, abs=2e-4)
    assert f["amplitude"] == pytest.approx((1 - 2 * eps1) / (1 - 2 * eps), rel=1e-3) and f["amplitude"] > 1
    # A = 1 data: the first-round term is the per-round one
    k = np.round(n * stats.p_model(eps, r))
    g = R.fit_first_round(r, k, n, bootstrap=0)
    assert g["eps_1"] == pytest.approx(eps, abs=2e-4)


def test_labels_name_the_window_and_main():
    assert "1036" in R.PRIORS["compiled"] and R.MAIN_SHA in R.PRIORS["compiled"]
    assert R.PRIORS["fallback"] == "hardware_noise (79092cf)"


def test_new_run_ids_stay_out_of_batch_runs_glob():
    for gid in ("X_eA_b1", "Z_eB_b3", "Z_eA_t1000ns_b2"):
        assert fnmatch.fnmatch(gid, "*_b[123]")
        assert not fnmatch.fnmatch(gid + R.SUFFIX, "*_b[123]")


def test_select_never_opens_b3():
    dirs = R.run_dirs((1, 2))
    assert dirs and not any(p.name.startswith("b3_") for p in dirs)
    assert not any("ep0" in p.name for p in R.run_dirs((1, 2, 3)))


def test_committed_selection_follows_its_rule():
    sel = json.loads(R.SELECTION.read_text())
    assert (sel["rule"], sel["fit_rule"], sel["e4_rule"]) == (R.RULE, R.FIT_RULE, R.E4_RULE)
    assert sel["batches_opened"] == [1, 2] and sel["rebuild_exact"]
    for key, v in sel["validation"].items():
        assert sel["choice"][key] == v["choice"] == R.choose(v["fallback"]["failures"], v["compiled"]["failures"])
        assert v["validation_shots"] == 8800
    wins = [v["winner"] for v in sel["fit_model_pools"].values()]
    assert sel["fit_model"] in wins
    for v in sel["fit_model_pools"].values():
        best = min(("A1_R>=2", "first_round"), key=lambda w: v[w]["chi2_per_dof"])
        assert v["winner"] == best


def test_committed_records_are_labelled_and_pass_the_dictionary():
    recs = sorted((R.ROOT / "ler").glob(f"E[14]/*{R.SUFFIX}/ler_fits.json"))
    assert len(recs) == 18  # E1: 2 patches x 2 bases x 3 batches; E4: 2 bases x 3 delays
    for f in recs:
        body = json.loads(f.read_text())
        assert body["priors"] == R.PRIORS["compiled"] and body["label"] == "HARDWARE" and body["device"] == "emerald"
        assert "1036 ns from compiled measure_window_ns" in body["noise"]
        assert (f.parent.parent / body["run_id"].removesuffix(R.SUFFIX)).is_dir(), "the batch's own record is beside it"
        assert IR.check_csv(f.parent / "ler_points.csv", "ler/ler_points.csv") == []
