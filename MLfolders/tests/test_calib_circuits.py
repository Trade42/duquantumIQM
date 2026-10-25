"""The characterization circuits: shapes, native ops on couplers, a barrier between every repeated gate, the record."""

from __future__ import annotations

import json
import math

import pytest
from qiskit import transpile

from dqiqm import layout as L
from dqiqm.characterization import circuits as CC
from dqiqm.characterization import inject as IJ
from dqiqm.characterization import patch as P
from dqiqm.characterization import plan as PL

RECORD = P.REPO_ROOT / "InfoRecord" / "characterization"


@pytest.fixture(scope="module")
def be():
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    return IQMFakeGarnet()


@pytest.fixture(scope="module")
def patch(be):
    return P.pilot_patch(be)


@pytest.fixture(scope="module")
def arch(be):
    return L.Architecture.from_backend(be)


def test_patch_is_the_pilots(patch):
    assert len(patch.qubits) == 17 and not {"QB3", "QB17", "QB18"} & set(patch.qubits)
    assert len(patch.couplers) == 24
    assert [len(layer) for layer in patch.layers] == [6, 6, 6, 6]
    # every coupler is in exactly one layer, and each layer is a matching of data-ancilla pairs
    seen = [frozenset(p) for layer in patch.layers for p in layer]
    assert len(set(seen)) == 24
    for layer in patch.layers:
        assert all(patch.roles[d] == "data" and patch.roles[a] == "ancilla" for d, a in layer)
    assert sorted(q for q, r in patch.roles.items() if r == "ancilla") == sorted(
        ["QB1", "QB5", "QB7", "QB9", "QB11", "QB13", "QB15", "QB20"])
    pri = P.terminal_readout(qubits=patch.qubits)
    assert pri["QB20"]["p0_given_1"] == pytest.approx(0.0898)  # measure_fidelity, unheralded (pulla-check)


def test_design_shapes(patch):
    c1, c2, c4 = CC.c1_designs(patch), CC.c2_designs(patch), CC.c4_designs(patch)
    assert len(c1) == len(CC.C1_AMP_N) + len(CC.C1_AMPY_N) + 2 * len(CC.C1_TILT_PAIRS) == 32
    assert len(c2) == 4 * 2 * 2 * len(CC.C2_N) * len(CC.C2_PSI) == 256
    assert len(c4) == 17 * 5 + 24 * 2 * len(CC.C4_C2_N) * 4 == 469
    names = [d.name for d in c1 + c2 + CC.c2_pilot_designs(patch) + c4]
    assert len(set(names)) == len(names)
    # every gate is prx(pi/2): one pulse each with drag_crf_sx
    assert all(g.angle == pytest.approx(math.pi / 2) for d in c1 + c2 + c4 for layer in d.layers for g in layer if g.op == "r")
    amp41 = next(d for d in c1 if d.name == "C1_amp_x_n41")
    assert amp41.counts(17) == {"r": 41 * 17, "barrier": 41, "measure": 17}
    tilt = next(d for d in c1 if d.name == "C1_tilt_p16_neg")
    assert [ph for ph, _ in tilt.params["sequence"]][:3] == [math.pi / 2, 0.0, math.pi]
    assert tilt.params["pulses"] == 1 + 32 + 1
    c2_9 = next(d for d in c2 if d.name == "C2_L2_tdat_s1_n09_q3")
    assert c2_9.counts(17) == {"r": 24, "cz": 54, "barrier": 12, "measure": 17}  # 6 targets x 2 + 6 controls x 2


def test_placed_circuits_are_native_on_couplers_with_barriers(patch, be, arch):
    """A sample of every family through the full check (the build script checks all 761) and IQM's validator."""
    designs = CC.c1_designs(patch)[::5] + CC.c2_designs(patch)[::37] + CC.c2_pilot_designs(patch)[:1] + CC.c4_designs(patch)[::90]
    for d in designs:
        p = CC.place(d, patch, be)
        sent = CC.check(p, arch, patch.couplers)
        CC.validate(p, be)
        assert set(sent) <= {"prx", "cz", "measure", "barrier"}
        assert p.circuit.count_ops() == d.counts(17)


def test_check_catches_merging_and_missing_barriers(patch, be, arch):
    d = next(x for x in CC.c1_designs(patch) if x.name == "C1_amp_x_n05")
    good = CC.place(d, patch, be)
    # qiskit at level 1 merges the 5 pulses when no barrier separates them (docs/pulla-check.md)
    logical = CC.native(d, patch).remove_final_measurements(inplace=False)
    bare = logical.copy_empty_like()
    for ci in logical.data:
        if ci.operation.name != "barrier":
            bare.append(ci)
    bare.measure_all()
    merged = transpile(bare, backend=be, initial_layout=list(good.initial_layout), optimization_level=1, seed_transpiler=1)
    assert merged.count_ops().get("r", 0) < 5 * 17
    bad = CC.Placed(d, merged, bare, good.qubit_index_to_name, good.initial_layout, good.qubits)
    with pytest.raises(AssertionError):
        CC.check(bad, arch, patch.couplers)
    # level 0 without barriers keeps the gates but breaks the barrier rule
    nobar = transpile(bare, backend=be, initial_layout=list(good.initial_layout), optimization_level=0, seed_transpiler=1)
    bad0 = CC.Placed(d, nobar, bare, good.qubit_index_to_name, good.initial_layout, good.qubits)
    with pytest.raises(AssertionError):
        CC.check(bad0, arch, patch.couplers)


def test_twin_adds_only_the_injected_gates(patch, be):
    d = next(x for x in CC.c2_designs(patch) if x.name == "C2_L1_tanc_s1_n03_q1")
    p = CC.place(d, patch, be)
    phys = IJ.physical(p, be.num_qubits, {q: be.qubit_name_to_index(q) for q in patch.qubits})
    assert {k: v for k, v in phys.count_ops().items()} == d.counts(17)
    err = IJ.random_errors(patch.qubits, patch.couplers, seed=3)
    tw = IJ.twin(p, err, be)
    ops = tw.count_ops()
    assert ops["unitary"] == d.counts(17)["r"]  # one error unitary per prx
    assert ops["rz"] == 2 * ops["cz"] and ops["cp"] == ops["cz"]
    assert IJ.twin(p, IJ.Errors(), be).count_ops() == phys.count_ops()


def test_record_and_plan(patch):
    """The committed record is the rebuild's (spot check; ``garnet_calib_build.py --check`` does all) and the plan is sane."""
    plan = json.loads((RECORD / "job_plan.json").read_text())
    assert plan["device"] == "garnet" and plan["calibration_set_id"] == "a18a947f-0408-4628-81ea-1715637e434f"
    pilot = [j for j in plan["jobs"] if j["phase"] == "pilot"]
    assert len(pilot) == 1 and pilot[0]["shots"] <= 1000 and pilot[0]["circuits"] <= 15
    assert plan["pilot"]["circuits"] == pilot[0]["order"]
    b = plan["pilot"]
    # each no-merging band sits clear of the merged expectation and of the prediction
    assert b["c1_no_merge"]["stop_if_below"] > 3 * b["c1_no_merge"]["sigma_pooled"]
    assert (1 - b["c2_no_merge"]["stop_if_above"]) > 3 * b["c2_no_merge"]["sigma_pooled"]
    must = [j for j in plan["jobs"] if j["phase"] == "must"]
    assert sum(j["circuits"] for j in must) == 32 + 256
    built = {}
    for exp in ("C1", "C2", "C4"):
        rec = json.loads((RECORD / "circuits" / exp / "placement.json").read_text())
        assert rec["device"] == "garnet" and rec["column_qubits"] == list(patch.qubits)
        for name, e in rec["circuits"].items():
            assert e["device"] == "garnet" and e["target"] == "hardware"
            assert (RECORD / "circuits" / exp / f"{name}.qasm").exists()
        built.update(rec["circuits"])
    for j in plan["jobs"]:
        assert set(j["order"]) <= set(built)


def test_pilot_bands_from_priors(patch):
    pri = P.priors(patch)
    b = PL.pilot_bands(patch, pri)
    assert 0 < b["c1_no_merge"]["stop_if_below"] < b["c1_no_merge"]["predicted_t1t2_only"]
    assert b["c2_no_merge"]["predicted_irb"] < b["c2_no_merge"]["stop_if_above"] < 1


def test_pulse_plan_ranges_hold_the_saved_values(patch):
    from dqiqm.characterization import pulse as PU

    p = PU.plan(patch)
    assert set(p["qubits"]) == set(patch.qubits) and len(p["couplers"]) == 24
    for locus in [*p["qubits"].values(), *p["couplers"].values()]:
        for k, v in locus.items():
            if isinstance(v, dict) and "low" in v:
                assert v["low"] < v["centre"] < v["high"], k
    assert p["couplers"]["QB2__QB1"]["fluxed_qubit"] == "QB2"
