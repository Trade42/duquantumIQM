import numpy as np
import pytest

from challenge.benchmark_rounds import LIMIT, charge, compare, planned_cases, protocol, verify


def test_budget_charges_ambiguous_and_failed_submissions_and_refuses_retry():
    a = dict(shots=800)
    manifest = dict(cases=[a], shot_limit=LIMIT)
    charge(manifest, a)
    a["status"] = "failed"
    b = dict(shots=96000)
    manifest["cases"].append(b)
    charge(manifest, b)
    assert manifest["charged_shots"] == 96800
    with pytest.raises(ValueError, match="already attempted"):
        charge(manifest, a)
    c = dict(shots=3200)
    manifest["cases"].append(c)
    with pytest.raises(ValueError, match="budget"):
        charge(manifest, c)
    assert not c.get("submit_intent_utc")


@pytest.mark.parametrize("same", [True, False])
def test_fixed_protocol_spends_96000_with_or_without_shared_physical_arm(same):
    selection = dict(candidate={b: dict(profile="baseline" if same else "center") for b in ("X", "Z")})
    cases = planned_cases({}, "pilot")+planned_cases({}, "confirm0", selection)+planned_cases({}, "confirm1", selection)
    assert sum(c["shots"] for c in cases) == protocol()["total_planned_physical_shots"] == 96000
    assert len({c["id"] for c in cases}) == len(cases)


def test_identical_same_shot_decoders_have_exact_zero_gain_interval():
    values = {r: {b: np.arange(100)%9 == 0 for b in (0, 1)} for r in (1, 2, 3)}
    result = compare(values, values, paired=True, draws=50)
    assert result["absolute_ler_reduction"] == 0
    assert result["ci"] == [0, 0]


def test_primary_window_and_equation_are_not_diagnostic_window():
    p = protocol()
    assert p["primary_rounds"] == [1, 2, 3]
    assert p["diagnostic_rounds"] == list(range(1, 8))
    assert "amplitude fixed to one" in p["equation"]


def test_pulse_trials_are_explicit_calibrated_choices():
    pytest.importorskip("iqm_surface_code")
    from challenge.rounds_hardware import strategy
    assert strategy("center", ["QB1"]).gate_sequences == [(5, "YXYX", "center")]
    assert strategy("adaptive", ["QB1"]).gate_sequences[0] == (9, "XYXYYXYX", "center")
    assert strategy("baseline", ["QB1"]).target_qubits == frozenset(["QB1"])


def test_adopted_policy_does_not_silently_extend_qualification(tmp_path):
    from challenge.benchmark_rounds import load_adopted
    with pytest.raises(ValueError, match="R1..3 only"):
        load_adopted(tmp_path, "X", 7)


def test_equal_pilot_failures_prefer_fitted_matching_over_recurrent_complexity():
    from challenge.benchmark_rounds import complexity
    rows = [dict(profile="adaptive", family=f) for f in ("gru_0.6", "mwpm_fit")]
    assert min(rows, key=complexity)["family"] == "mwpm_fit"


def test_combination_requires_gain_over_each_component_and_can_release_simpler_winner():
    from challenge.benchmark_rounds import release_arms, select_release
    selection = release_arms(dict(candidate={"X": dict(profile="adaptive", family="mwpm_fit"),
                                             "Z": dict(profile="baseline", family="released")}))
    assert selection["comparison_count"] == 6
    pairs = {
        "rc": dict(first="reference", second="candidate", ci=[.01, .03]),
        "rp": dict(first="reference", second="physical", ci=[.01, .03]),
        "rh": dict(first="reference", second="history", ci=[-.01, .01]),
        "cp": dict(first="candidate", second="physical", ci=[-.003, .002]),
        "ch": dict(first="candidate", second="history", ci=[-.03, -.01]),
        "ph": dict(first="physical", second="history", ci=[-.03, -.01]),
    }
    arms = selection["release_arms"]["X"]
    rates = dict(reference=.05, candidate=.03, physical=.031, history=.048)
    assert select_release(arms, pairs, rates) == "physical"
    pairs["cp"]["ci"] = [-.003, -.0001]
    assert select_release(arms, pairs, rates) == "candidate"
    pairs["rc"]["ci"] = [-.01, .03]
    pairs["rp"]["ci"] = [-.01, .03]
    assert select_release(arms, pairs, rates) == "reference"


def test_release_rules_cannot_change_after_confirmation_attempt(tmp_path):
    from challenge.benchmark_rounds import finalize
    with pytest.raises(ValueError, match="before every confirmation"):
        finalize(tmp_path, dict(cases=[dict(phase="confirm0")]))
