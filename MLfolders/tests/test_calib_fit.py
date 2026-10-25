"""The characterization analysis: a noiseless closure gives 0, injected errors come back, and the readout algebras invert.

All on exact probabilities (no simulator): the heavy, sampled closure through Aer is
``scripts/garnet_calib_closure.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dqiqm.characterization import analysis as A
from dqiqm.characterization import circuits as CC
from dqiqm.characterization import fit as F
from dqiqm.characterization import leakage as LK
from dqiqm.characterization import model as M
from dqiqm.characterization.patch import Patch, Priors

QUBITS = ("QB4", "QB5", "QB6", "QB7")
PAIRS = (("QB4", "QB5"), ("QB6", "QB7"))  # (data, ancilla), as a CZ layer lists them
PATCH = Patch(qubits=QUBITS, roles={"QB4": "data", "QB5": "ancilla", "QB6": "data", "QB7": "ancilla"},
              layers=(PAIRS,), pilot_native_sha256="", calibration_set_id="test")
PRIORS = Priors("test", {q: {"t1_ns": 40e3, "t2_ns": 25e3, "prx_ns": 16.0, "error_1q": 1e-3, "p1_given_0": 0.01,
                             "p0_given_1": 0.03} for q in QUBITS},
                {p: {"cz_error": 5e-3, "cz_ns": 40.0} for p in (("QB4", "QB5"), ("QB6", "QB7"))}, 1036.0)


def c1_data(theta, shots=1e6, t1=40e3, t2=25e3):
    designs = CC.c1_designs(PATCH)
    seqs = tuple(tuple((float(a), int(b)) for a, b in d.params["sequence"]) for d in designs)
    d0 = F.C1Data("QB4", seqs, np.full(len(seqs), shots), np.zeros(len(seqs)), t1_ns=t1, t2_ns=t2)
    return F.C1Data("QB4", seqs, d0.shots, shots * F.c1_probs(d0, theta), t1_ns=t1, t2_ns=t2)


def test_rotation_matches_the_unitary():
    for phase in (0.0, math.pi / 2, math.pi, 2.1):
        assert np.allclose(F._rot(phase, 0.03, -0.02), M.rotation(M.prx_actual(phase, 0.03, -0.02)), atol=1e-12)
    # the injected error unitary composes with the ideal gate into the modelled one
    e = M.prx_error(0.7, 0.03, -0.02)
    assert np.allclose(M.prx(math.pi / 2, 0.7) @ e, M.prx_actual(0.7, 0.03, -0.02))


def test_c1_noiseless_closure_is_zero():
    f = F.fit_c1(c1_data((0.0, 0.0, 0.0, 0.0, 0.0), t1=math.inf, t2=math.inf))
    assert abs(f.values["eps"]) < 1e-6 and abs(f.values["eta"]) < 1e-6
    assert f.ci["eps"][0] <= 0 <= f.ci["eps"][1] and f.ci["eta"][0] <= 0 <= f.ci["eta"][1]


@pytest.mark.parametrize("eps,eta", [(-0.05, 0.04)])
def test_c1_recovers_injected_errors(eps, eta):
    f = F.fit_c1(c1_data((eps, eta, 0.002, 0.01, 0.03)))
    assert f.values["eps"] == pytest.approx(eps, abs=1e-5)
    assert f.values["eta"] == pytest.approx(eta, abs=1e-5)
    assert f.ci["eps"][0] <= eps <= f.ci["eps"][1]


def _c2_arrays(prx_err, cz_err, shots=4000):
    """Per C2 design, a record with exactly round(P1 * shots) ones in each target and control column."""
    designs = [d for d in CC.c2_designs(PATCH)]
    arrays = []
    for d in designs:
        a = np.zeros((shots, len(QUBITS)), np.uint8)
        for t, c in zip(d.params["targets"], d.params["controls"]):
            p = M.simulate_design(d.layers, (t, c), prx_err, cz_err)
            for q in (t, c):
                a[: int(round(p[q] * shots)), PATCH.index(q)] = 1
        arrays.append(a)
    return designs, arrays


def test_c2_noiseless_closure_is_zero():
    designs, arrays = _c2_arrays({}, {})
    per = A.couplers(A.c2_fits(A.c2_points(designs, arrays, PATCH)))
    assert set(per) == {("QB4", "QB5"), ("QB6", "QB7")}
    for r in per.values():
        assert abs(r["delta"]) < 1e-9 and r["delta_ci"][0] <= 0 <= r["delta_ci"][1]
        assert all(abs(v) < 1e-9 for v in r["phi"].values())


def test_c2_recovers_injected_errors():
    prx_err = {"QB4": (0.02, -0.01), "QB5": (-0.03, 0.02), "QB6": (0.01, 0.03), "QB7": (0.0, -0.02)}
    cz_err = {("QB4", "QB5"): (0.06, -0.04, 0.07), ("QB6", "QB7"): (-0.05, 0.08, -0.03)}
    designs, arrays = _c2_arrays(prx_err, cz_err, shots=20000)
    per = A.couplers(A.c2_fits(A.c2_points(designs, arrays, PATCH)))
    for pair, (delta, pa, pb) in cz_err.items():
        r = per[pair]
        assert r["delta"] == pytest.approx(delta, abs=3e-3)
        assert r["delta_ci"][0] <= delta <= r["delta_ci"][1]
        for q, phi in zip(pair, (pa, pb)):
            assert r["phi"][q] == pytest.approx(phi, abs=3e-3)
            assert r["phi_ci"][q][0] <= phi <= r["phi_ci"][q][1]
    rows = A.c2_rows(per, PRIORS, experiment="C2", context="test", common={}, injected=cz_err)
    cov = A.coverage(rows)
    assert cov["delta"]["covered"] == 2 and cov["phi"]["covered"] == 4


def test_c1_rows_and_analysis_path():
    designs = CC.c1_designs(PATCH)
    seqs = [tuple((float(a), int(b)) for a, b in d.params["sequence"]) for d in designs]
    m = M.QubitModel(eps=0.01, eta=-0.02, lam=0.002, a0=0.01, a1=0.03, t1_ns=40e3, t2_ns=25e3)
    shots = 20000
    arrays = []
    for s in seqs:
        a = np.zeros((shots, len(QUBITS)), np.uint8)
        a[: int(round(M.p_read1(s, m) * shots)), :] = 1
        arrays.append(a)
    data = A.c1_data(designs, arrays, PATCH, PRIORS)
    assert set(data) == set(QUBITS)
    fits = A.c1_fits({"QB4": data["QB4"]}, PRIORS)
    assert fits["QB4"].values["eps"] == pytest.approx(0.01, abs=5e-4)
    assert fits["QB4"].values["eta"] == pytest.approx(-0.02, abs=1e-3)
    rows = A.c1_rows(fits, PRIORS, experiment="C1", context="test", common={}, injected={"QB4": (0.01, -0.02)})
    assert [r["parameter"] for r in rows] == ["eps", "eta"] and all(r["fit_unit"] == "rad" for r in rows)


def test_leakage_readout_algebra_inverts():
    for p in ((0.97, 0.02, 0.01), (0.1, 0.85, 0.05), (0.0, 1.0, 0.0)):
        for r in ((0.02, 0.95, 0.9), (0.01, 0.93, 0.6), (0.02, 0.97, 0.05)):
            assert np.allclose(LK.populations(LK.readouts(p, r), r), p, atol=1e-12)
    slope, se = LK.leakage_per_cz([0, 1, 3, 9], [0.001, 0.003, 0.007, 0.019], [1e-3] * 4)
    assert slope == pytest.approx(0.002, abs=1e-9) and se > 0
    c = LK.circuit("QB14", "QB15", "swap12", 3)
    names = [i["name"] for i in c["instructions"]]
    assert names.count("cz") == 3 and names.count("prx_12") == 1 and names.count("measure") == 2


def test_phase_tracking_cancels_local_cz_phases():
    """compensate + the device's local phases equals the ideal circuit's output distribution, exactly."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RGate
    from qiskit.quantum_info import Statevector

    from dqiqm.characterization import compensate as K

    qc = QuantumCircuit(3)
    for q in range(3):
        qc.append(RGate(math.pi / 2, math.pi / 2), [q])
    qc.cz(0, 1)
    qc.append(RGate(math.pi / 2, -math.pi / 2), [1])
    qc.cz(1, 2)
    for q in range(3):
        qc.append(RGate(math.pi / 2, 0.3 * q), [q])
    names = {0: "QB1", 1: "QB2", 2: "QB5"}
    phis = {("QB1", "QB2"): {"QB1": 0.07, "QB2": -0.05}, ("QB2", "QB5"): {"QB2": 0.04, "QB5": 0.06}}
    ideal = Statevector(qc).probabilities()
    plain = Statevector(K.with_cz_phases(qc, names, phis)).probabilities()
    fixed = Statevector(K.with_cz_phases(K.compensate(qc, names, phis), names, phis)).probabilities()
    assert np.allclose(fixed, ideal, atol=1e-12) and not np.allclose(plain, ideal, atol=1e-4)
    assert np.allclose(Statevector(K.compensate(qc, names, {})).data, Statevector(qc).data)  # no phases: unchanged
