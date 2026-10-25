"""garnet-timing: the compiled readout window, the terminal readout, the biased idle (in each data qubit's hardware
frame) and no idle after the last round."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import stim

from dqiqm import layout as L
from dqiqm import surface as S
from dqiqm.experiments import circuits as C
from dqiqm.experiments import device as dev
from dqiqm.experiments import memory, repetition, stability
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
BEFORE = REPO / "data" / "calibration" / "garnet" / "a18a947f-0408-4628-81ea-1715637e434f.20261024T221752Z.json"
#: the pilot decoder's text, sha256 (garnet-pilot, main 811930e); scripts/garnet_reprior.py checks the same
OLD_DECODER_SHA256 = "bec21d8ca17bf9720bc38ba6f12112526a8f084af370218ea7c27519218ca4e6"


@pytest.fixture(scope="module")
def cal():
    return dev.live_calibration()


@pytest.fixture(scope="module")
def pilot():
    """The pilot P0 as ``placement.json`` records it: enough of a CircuitSpec for the priors."""
    e = json.loads((REPO / "InfoRecord" / "circuits" / "E1" / "placement.json").read_text())["E1_d3_Z_r03_pA"]
    g = S.geometry(3)
    return SimpleNamespace(
        physical={int(k): v for k, v in e["placement"].items()}, roles={int(k): v for k, v in e["roles"].items()},
        cx_pairs=memory.cx_pairs(g), x_ancillas=tuple(g.x_ancillas), params=e["params"],
        noisy=partial(memory.memory_noisy, 3, "Z"), stim=S.memory_circuit(3, 3, "Z"))


def _ancillas(spec):
    return [spec.physical[q] for q, r in spec.roles.items() if r == "ancilla"]


def _m_args(c):
    """Each qubit's record-flip argument per M, in measurement order."""
    out = []
    for inst in c.flattened():
        if inst.name == "M":
            p = (inst.gate_args_copy() or [0.0])[0]
            out += [(t.value, p) for t in inst.targets_copy()]
    return out


# ---------------------------------------------------------------------------
# Fix 1: the compiled readout window and the terminal readout.
# ---------------------------------------------------------------------------


def test_readout_window_is_the_compiled_one(cal, pilot, caplog):
    w = dev.compiled_readout_windows(cal.calibration_set_id)
    assert w is not None and w.calibration_set_id == cal.calibration_set_id
    ns, src = dev.readout_layer_ns(cal, _ancillas(pilot), w)
    assert ns == 1096.0 and "measure_window_ns" in src  # QB5 and QB7, pulla-check's figure
    assert dev.readout_layer_ns(cal, _ancillas(pilot), w, terminal=True)[0] == max(
        w.of(q, "measure_fidelity_window_ns") for q in _ancillas(pilot))
    with caplog.at_level(logging.WARNING, logger="dqiqm.experiments.device"):
        ns, src = dev.readout_layer_ns(cal, _ancillas(pilot), None)
    assert ns == 556.0 and src.startswith("measure_ns")  # the old rule, logged
    assert any("falls back" in r.message for r in caplog.records)
    assert dev.compiled_readout_windows("00000000-0000-0000-0000-000000000000") is None


def test_compile_file_must_name_its_calibration_set(tmp_path):
    calset = "11111111-1111-1111-1111-111111111111"
    body = {"readout_windows": {"qubits": {"QB1": {"measure_window_ns": 1000.0}}}}
    (tmp_path / f"compile.{calset}.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="names no calibration set"):
        dev.compiled_readout_windows(calset, tmp_path)
    (tmp_path / f"compile.{calset}.json").write_text(json.dumps(body | {"calibration_set_id": "other"}))
    with pytest.raises(ValueError, match="is for calibration set other"):
        dev.compiled_readout_windows(calset, tmp_path)
    (tmp_path / f"compile.{calset}.json").write_text(json.dumps(body | {"calibration_set_id": calset}))
    assert dev.compiled_readout_windows(calset, tmp_path).of("QB1", "measure_window_ns") == 1000.0


def test_terminal_readout_errors_are_measure_fidelitys_unheralded(cal):
    q = cal.qubits["QB20"]
    assert (q["p1_given_0_terminal"], q["p0_given_1_terminal"]) == (pytest.approx(0.0204), pytest.approx(0.0898))
    assert cal.terminal_readout_error("QB20") == pytest.approx(0.0551)
    assert cal.readout_error("QB20") == pytest.approx(0.5 * (0.0180 + 0.0174))
    assert all(cal.terminal_readout_error(n) is not None for n in cal.qubits)


def test_final_record_flip_lands_on_the_final_layer_only():
    r = 3
    c = S.memory_circuit(3, r, "Z", noise=S.NoiseModel(record_flip_probability=0.01, final_record_flip_probability=0.03))
    args = _m_args(c)
    n_anc = len(S.geometry(3).ancillas)
    assert [p for _, p in args] == [0.01] * (r - 1) * n_anc + [0.03] * (n_anc + 9)  # round R's ancillas and the data
    unset = S.memory_circuit(3, r, "Z", noise=S.NoiseModel(record_flip_probability=0.01))
    assert {p for _, p in _m_args(unset)} == {0.01}
    per_q = S.NoiseModel(record_flip_probability=0.01, qubit_rates={0: {"final_record_flip_probability": 0.2}})
    assert per_q.record_flip(0, final=True) == 0.2 and per_q.record_flip(0) == 0.01 and per_q.record_flip(1, final=True) == 0.01


# ---------------------------------------------------------------------------
# Fixes 2 and 3: the biased idle, and none after the last round.
# ---------------------------------------------------------------------------


def _idle_layers(c, gate: str, qubits: set[int]) -> list[tuple[int, ...]]:
    """After each ancilla readout, the ``gate`` channels on ``qubits`` before the next TICK or M."""
    out, after_m, cur = [], False, []
    for inst in c.flattened():
        if inst.name == "M":
            if after_m and cur:
                out.append(tuple(cur))
            after_m, cur = True, []
            continue
        if inst.name == "TICK" and after_m:
            if cur:
                out.append(tuple(cur))
            after_m, cur = False, []
        elif after_m and inst.name == gate:
            cur += [t.value for t in inst.targets_copy() if t.value in qubits]
    if after_m and cur:
        out.append(tuple(cur))
    return out


@pytest.mark.parametrize("rounds", [1, 2, 3, 5])
def test_no_idle_after_the_last_round(rounds):
    g = S.geometry(3)
    nm = S.NoiseModel(idle_during_measurement=0.01)
    c = S.memory_circuit(3, rounds, "Z", noise=nm)
    assert len(_idle_layers(c, "DEPOLARIZE1", set(g.data))) == rounds - 1
    assert len(_idle_layers(c, "X_ERROR", set(g.ancillas))) == rounds - 1
    legacy = S.memory_circuit(3, rounds, "Z", noise=dataclasses.replace(nm, idle_after_last_round=True))
    assert len(_idle_layers(legacy, "DEPOLARIZE1", set(g.data))) == rounds
    assert len(_idle_layers(legacy, "X_ERROR", set(g.ancillas))) == rounds
    # with reset, the ancillas get their after-reset flip after every round, the last one included (Stim's generated),
    # plus the initial R's two (data, ancillas)
    mr = S.memory_circuit(3, rounds, "Z", no_reset=False, noise=S.NoiseModel(after_reset_flip_probability=0.01))
    assert sum(1 for i in mr.flattened() if i.name == "X_ERROR") == rounds + 2


def test_biased_idle_is_a_pauli_channel_on_the_data():
    g = S.geometry(3)
    nm = S.NoiseModel(idle_pauli_x=0.01, idle_pauli_y=0.01, idle_pauli_z=0.05)
    c = S.memory_circuit(3, 3, "Z", noise=nm)
    layers = _idle_layers(c, "PAULI_CHANNEL_1", set(g.data) | set(g.ancillas))
    assert [sorted(x) for x in layers] == [sorted(g.data)] * 2
    assert {tuple(i.gate_args_copy()) for i in c.flattened() if i.name == "PAULI_CHANNEL_1"} == {(0.01, 0.01, 0.05)}
    assert c.detector_error_model(decompose_errors=True).num_errors > 0
    with pytest.raises(ValueError):
        S.NoiseModel(idle_pauli_x=0.5, idle_pauli_y=0.4, idle_pauli_z=0.2)
    with pytest.raises(ValueError):
        S.NoiseModel(qubit_rates={3: {"idle_pauli_x": 0.6, "idle_pauli_z": 0.6}})
    with pytest.raises(ValueError):
        S.NoiseModel.from_dict({"idle_pauli_w": 0.1})
    assert S.NoiseModel.from_dict({"idle_pauli_z": 0.1, "idle_after_last_round": True}).idle_after_last_round


def test_calibrated_noise_biased_idle_and_echo(cal, pilot):
    args = dict(pairs=pilot.cx_pairs, x_ancillas=pilot.x_ancillas)
    sym = dev.calibrated_noise(cal, pilot.physical, pilot.roles, idle_ns=1096.0, **args)
    pauli = dev.calibrated_noise(cal, pilot.physical, pilot.roles, idle_ns=1096.0, idle_channel="pauli", **args)
    echo = dev.calibrated_noise(cal, pilot.physical, pilot.roles, idle_ns=1096.0, idle_channel="pauli", t2="echo", **args)
    for q, role in pilot.roles.items():
        c = cal.qubits[pilot.physical[q]]
        pxy, pz = dev.relaxation_pauli(1096.0, float(c["t1_ns"]), dev.t2_ns(c, "ramsey"))
        if role == "data":
            assert pauli.idle_pauli(q) == pytest.approx((pxy / 2, pxy / 2, pz))
            assert pauli.rate("idle_during_measurement", q) == 0
            assert sym.rate("idle_during_measurement", q) == pytest.approx(pxy + pz)  # the same total, spread evenly
            assert pz > pxy  # Z-dominated: the reason for the biased channel
        else:
            assert pauli.idle_pauli(q) == (0, 0, 0)
            assert pauli.rate("idle_during_measurement", q) == pytest.approx(pxy)
        e_pxy, e_pz = dev.relaxation_pauli(1096.0, float(c["t1_ns"]), dev.t2_ns(c, "echo"))
        if role == "data":
            assert echo.idle_pauli(q) == pytest.approx((e_pxy / 2, e_pxy / 2, e_pz))
    assert any(echo.idle_pauli(q) != pauli.idle_pauli(q) for q in pilot.physical)
    assert echo.pair_rates != pauli.pair_rates  # T2 enters the CZ twirl too
    with pytest.raises(ValueError):
        dev.calibrated_noise(cal, pilot.physical, pilot.roles, idle_channel="amplitude", **args)


def test_hardware_noise_is_the_compiled_prior(cal, pilot):
    nm, info = dev.hardware_noise(cal, pilot)
    assert info["idle_ns"] == 1096.0 and info["delay_ns"] == 0 and info["idle_channel"] == "depolarize" and info["t2"] == "ramsey"
    assert info["final_readout"].startswith("measure_fidelity") and not info["idle_after_last_round"]
    assert info["family"] == "surface_memory" and info["idle_frame"] is None  # symmetric: the same in every frame
    assert all(nm.idle_pauli(q) == (0, 0, 0) for q in pilot.physical)
    qb20 = next(q for q, n in pilot.physical.items() if n == "QB20")
    assert nm.record_flip(qb20, final=True) == pytest.approx(0.0551) and nm.record_flip(qb20) == pytest.approx(cal.readout_error("QB20"))
    c = pilot.noisy(nm)
    assert c.num_detectors == pilot.stim.num_detectors and len(c.shortest_graphlike_error()) == 3
    delayed, dinfo = dev.hardware_noise(cal, SimpleNamespace(**{**vars(pilot), "params": {**pilot.params, "delay_ns": 2000.0}}))
    assert dinfo["idle_ns"] == 3096.0
    fallback, finfo = dev.hardware_noise(cal, pilot, windows=None)
    assert finfo["idle_ns"] == 556.0 and finfo["readout_window_source"].startswith("measure_ns")


def test_old_priors_rebuild_the_pilots_decoder_exactly(pilot):
    # the pilot decoded with the before snapshot, 556 ns, symmetric idle and the idle after the last round
    old_cal = dev.live_calibration(BEFORE)
    nm = dev.calibrated_noise(old_cal, pilot.physical, pilot.roles, pairs=pilot.cx_pairs, idle_ns=556.0,
                              x_ancillas=pilot.x_ancillas)
    legacy = pilot.noisy(dev.pre_fix_noise(nm))  # what the pilots' drivers should pass for a pre-fix record
    assert hashlib.sha256(str(legacy).encode()).hexdigest() == OLD_DECODER_SHA256
    assert legacy.detector_error_model(decompose_errors=True).num_errors == 294
    assert hashlib.sha256(str(pilot.noisy(nm)).encode()).hexdigest() != OLD_DECODER_SHA256  # fix 3 moved it


def test_hardware_noise_follows_the_family(cal, pilot):
    # the E2 and E5 builders still idle after their last round and don't mark their final readout layer
    for build in (lambda nm: repetition.circuit(5, 3, "bit", noise=nm), lambda nm: stability.circuit(3, "Z", noise=nm)):
        c = build(S.NoiseModel(idle_during_measurement=0.01, record_flip_probability=0.01, final_record_flip_probability=0.2))
        assert sum(1 for i in c.flattened() if i.name == "DEPOLARIZE1") == 3  # one idle per round, the last included
        assert not any(i.name == "M" and i.gate_args_copy() == [0.2] for i in c.flattened())
    for family in ("repetition", "stability"):
        nm, info = dev.hardware_noise(cal, SimpleNamespace(**{**vars(pilot), "params": {**pilot.params, "family": family}}))
        assert info["family"] == family and info["idle_after_last_round"] is True
        assert info["final_readout"].startswith(f"measure (the {family} builder")
        assert all(nm.record_flip(q, final=True) == nm.record_flip(q) for q in pilot.physical)
    with pytest.raises(ValueError, match="families"):
        dev.hardware_noise(cal, SimpleNamespace(**{**vars(pilot), "params": {**pilot.params, "family": "readout"}}))


# ---------------------------------------------------------------------------
# The biased idle's frame (review 1, BLOCKER 1): checked on the committed QASM, apart from device's translation.
# ---------------------------------------------------------------------------

#: IQM's r(theta, phi) in units of pi/2 -> its Stim gate (the reviewer's table, frame_check.py)
_R = {(1, 0): "SQRT_X", (1, 2): "SQRT_X_DAG", (1, 1): "SQRT_Y", (1, 3): "SQRT_Y_DAG", (2, 0): "X", (2, 2): "X", (2, 1): "Y", (2, 3): "Y"}
_PI = 3.141592653589793


def _committed(cal, name):
    """The committed circuit ``name`` rebuilt as a CircuitSpec (its native_sha256 checked), and its placement.json entry."""
    e = json.loads((REPO / "InfoRecord" / "circuits" / name.split("_")[0] / "placement.json").read_text())[name]
    p = e["params"]
    pl = L.Placement(tuple(e["origin"]), e["orientation"], tuple(sorted((int(k), v) for k, v in e["placement"].items())))
    spec = memory.spec(p["rounds"], p["basis"], pl, p["patch"], dev.fake_backend(cal), delay_ns=p["delay_ns"])
    assert C.native_sha256(spec.hardware) == e["native_sha256"]
    return spec, e


def _groups(model):
    """The model's detectors, then its observable, as absolute measurement indices."""
    dets, obs, n = [], [], 0
    for inst in model.flattened():
        if inst.name == "M":
            n += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            dets.append([n + t.value for t in inst.targets_copy()])
        elif inst.name == "OBSERVABLE_INCLUDE":
            obs += [n + t.value for t in inst.targets_copy()]
    return dets, obs


def _sig(c):
    errs = [e for e in c.detector_error_model().flattened() if e.type == "error"]
    if not errs:
        return (), ()
    ts = errs[0].targets_copy()
    return (tuple(sorted(t.val for t in ts if t.is_relative_detector_id())),
            tuple(sorted(t.val for t in ts if t.is_logical_observable_id())))


def _qasm_stim(name, e, model, inject=None, *, after_delay=False):
    """The committed QASM as Stim with the model's detectors; ``inject`` = (Stim qubit, Pauli), a physical error
    right after readout layer 1 or, ``after_delay``, at the end of the qubit's idle (just before its next gate)."""
    txt = (REPO / "InfoRecord" / "circuits" / name.split("_")[0] / f"{name}.qasm").read_text()
    stmts = [x.strip() for x in txt.split(f"bit[{model.num_measurements}] c;")[1].split(";") if x.strip()]
    to_stim = {int(v[2:]) - 1: int(k) for k, v in e["placement"].items()}  # $i is QB(i + 1)
    c, order, layer, in_meas, delays, done = stim.Circuit(), [], 0, False, 0, inject is None
    for st in stmts:
        if (mm := re.match(r"c\[(\d+)\]\s*=\s*measure\s+\$(\d+)", st)):
            c.append("M", [to_stim[int(mm[2])]])
            order.append(int(mm[1]))
            in_meas = True
            continue
        if in_meas:
            layer, in_meas = layer + 1, False
            if layer == 1 and not done and not after_delay:
                c.append(f"{inject[1]}_ERROR", [inject[0]], 0.1)
                done = True
        qs = [to_stim[int(x)] for x in re.findall(r"\$(\d+)", st)]
        if st.startswith("delay") and layer == 1:
            delays += inject is not None and inject[0] in qs
        if not done and after_delay and layer == 1 and inject[0] in qs and not st.startswith(("barrier", "delay")):
            assert delays == 1  # the qubit's delay came first: it is in the same idle
            c.append(f"{inject[1]}_ERROR", [inject[0]], 0.1)
            done = True
        if (m := re.match(r"r\(([^,]+),\s*([^)]+)\)", st)):
            th, ph = (eval(x, {"pi": _PI}) for x in (m[1], m[2]))  # noqa: S307 -- our own committed file
            c.append(_R[(round(th / _PI * 2), round(ph / _PI * 2) % 4)], qs)
        elif st.startswith("cz"):
            c.append("CZ", qs)
    assert done
    pos, n = {k: p for p, k in enumerate(order)}, len(order)
    dets, obs = _groups(model)
    for d in dets:
        c.append("DETECTOR", [stim.target_rec(pos[m] - n) for m in d])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(pos[m] - n) for m in obs], 0)
    return c


def _model_signature(model, q, pauli):
    """Stim's ``pauli`` on ``q`` where the model puts the idle: right after round 1's readout."""
    c, done = stim.Circuit(), False
    for inst in model.flattened():
        c.append(inst)
        if inst.name == "M" and not done:
            c.append(f"{pauli}_ERROR", [q], 0.1)
            done = True
    return _sig(c)


@pytest.mark.parametrize("name", ["E1_d3_Z_r03_pA", "E1_d3_X_r02_pB", "E4_d3_Z_r03_pA_t02000ns"])
def test_biased_idle_is_in_each_data_qubits_hardware_frame(cal, name):
    spec, e = _committed(cal, name)
    model = spec.stim
    noiseless = _qasm_stim(name, e, model)
    noiseless.detector_error_model()  # every detector and the observable deterministic
    signs, signs_m = noiseless.reference_detector_and_observable_signs(), model.reference_detector_and_observable_signs()
    assert all((a == b).all() for a, b in zip(signs, signs_m))  # with the model's noiseless values

    nm, info = dev.hardware_noise(cal, spec, idle_channel="pauli")
    frames = dev.readout_frames(spec)
    # the compiler moves the next round's leading prx before the readout on these five: x <-> z there
    assert info["idle_frame"] == {str(q): "ZYX" for q in (1, 3, 5, 12, 17)}
    data = sorted(q for q, r in spec.roles.items() if r == "data")
    for q in data:
        stim_sig = {s: _model_signature(model, q, s) for s in "XYZ"}
        assert len(set(stim_sig.values())) == 3 and all(v[0] for v in stim_sig.values())  # X, Y, Z tell apart here
        for p in "XYZ":
            for after_delay in (False, True) if spec.params["delay_ns"] else (False,):
                got = _sig(_qasm_stim(name, e, model, (q, p), after_delay=after_delay))
                assert got == stim_sig[frames[q][p]], (q, p, after_delay)
        # the prior: the physical twirl, put on the Stim Paulis the physical ones act as
        c = cal.qubits[spec.physical[q]]
        pxy, pz = dev.relaxation_pauli(info["idle_ns"], float(c["t1_ns"]), dev.t2_ns(c, "ramsey"))
        rates = dict(zip("XYZ", nm.idle_pauli(q)))
        assert rates[frames[q]["Z"]] == pytest.approx(pz) and rates[frames[q]["X"]] == pytest.approx(pxy / 2)
        assert rates["X" if q in (1, 3, 5, 12, 17) else "Z"] == pytest.approx(pz)


def test_frames_refuse_an_unfenced_readout(cal):
    spec, _ = _committed(cal, "E1_d3_Z_r03_pA")
    nobar = memory.spec(3, "Z", spec.placement, "pA", dev.fake_backend(cal), barriers=False)
    with pytest.raises(ValueError):
        dev.hardware_noise(cal, nobar, idle_channel="pauli")
    assert dev.hardware_noise(cal, nobar)[1]["idle_frame"] is None  # the symmetric default needs no frame


# ---------------------------------------------------------------------------
# Fix 4: no P1 in the plan.
# ---------------------------------------------------------------------------


def test_the_committed_plan_has_no_p1():
    plan = json.loads((REPO / "InfoRecord" / "circuits" / "job_plan.json").read_text())
    assert "P1" not in plan["shots"] and "P1" not in SU.SHOTS
    assert [j["label"] for j in plan["jobs"]][0] == "P0" and all(j["label"] != "P1" for j in plan["jobs"])
    assert all(not j["label"].startswith("P") for j in plan["jobs"][1:])
    assert all("_nobar" not in n for j in plan["jobs"] for n in j["order"])
    assert plan["pilot_timeline_span"]["planning"] is False
    assert plan["per_shot_overhead_s"] == pytest.approx(298e-6)
