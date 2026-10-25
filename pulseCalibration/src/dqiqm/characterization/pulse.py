"""Track B: the pulse-level experiments, designed only, from the saved per-locus pulse parameters.

Nothing here connects or compiles. Each experiment sweeps one setting of IQM's standard compiler
for **our own jobs only** (Pulla's ``compiler.get_settings(...)``, edited, then
``compiler.compile(circuits, settings=...)``, as ``docs/pulla-check.md`` describes): it never
writes IQM's calibration. The probe circuits are the C1/C2 circuits, whose fits turn each sweep
point into eps, eta, delta or phi. Every range is centred on the saved value of calibration set
a18a947f (``data/calibration/garnet/pulse/a18a947f-....json``, pulla-check) and capped by the
stated margin. **No pulse job can run** until a guarded ``hw.submit_playlist`` exists and the
mentors confirm submission rights.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dqiqm.characterization.patch import PULSE, Patch

#: (setting, relative or absolute half-range, points, shots per point, probe circuits, fit, acceptance)
PRX_EXPERIMENTS = {
    "B1_amplitude_coarse": {
        "setting": "gates.prx.drag_crf_sx.<q>.amplitude_i",
        "half_range": {"relative": 0.08}, "points": 17, "shots": 500,
        "probe": ["C1_amp_x_n05", "C1_amp_x_n07"],
        "fit": "P(n05) - P(n07) ~ C sin(6 eps(A)), eps(A) = (pi/2)(A/A_ref - 1) + eps_0: the zero of eps(A)",
        "why": "Rabi in amplification units: N = 5, 7 keep N eps within +-0.9 rad over +-8%, so no aliasing",
    },
    "B1_amplitude_fine": {
        "setting": "gates.prx.drag_crf_sx.<q>.amplitude_i",
        "half_range": {"relative": 0.01, "around": "the coarse zero"}, "points": 9, "shots": 1000,
        "probe": ["C1_amp_x_n41", "C1_amp_x_n43"],
        "fit": "the same, with N = 41, 43: about 8x the coarse sensitivity",
        "why": "the C1 eps resolution (about 0.0004 rad) at the amplitude that zeroes it",
    },
    "B2_drag": {
        "setting": "gates.prx.drag_crf_sx.<q>.amplitude_q",
        "half_range": {"relative": 0.5}, "points": 11, "shots": 500,
        "probe": ["C1_tilt_p16_pos", "C1_tilt_p16_neg"],
        "fit": "eta(Q) is linear near the optimum: the zero of [P(pos) - P(neg)]; the leakage objective needs C3's readout",
        "why": "DRAG sets the phase error and the |2> leakage of the 16 ns pulse; without a |2> readout only eta is seen, "
               "so the sweep stays within +-50% of the saved Q (Q/I stays within the patch's -0.16..-0.22 x 1.5)",
    },
    "B3_detuning": {
        "setting": "controllers.<q>.drive.frequency",
        "half_range": {"absolute_hz": 200e3}, "points": 0, "shots": 500,
        "probe": ["Ramsey with virtual detuning: prx(pi/2, 0), delay(t), prx(pi/2, 2 pi f_v t), t in 0..4 us "
                  "(a circuit job: no pulse access needed to measure it)"],
        "fit": "the fringe frequency minus f_v is the drive detuning; the correction is drive.frequency - detuning",
        "why": "a detuning tilts every pulse (eta) and precesses idles; set once from the Ramsey fit, capped at +-200 kHz",
    },
    "B4_virtual_phase": {
        "setting": "gates.prx.drag_crf_sx.<q>.rz_before and .rz_after (kept equal, as calibrated)",
        "half_range": {"absolute_rad": 0.1}, "points": 11, "shots": 500,
        "probe": ["C1_tilt_p16_pos", "C1_tilt_p16_neg"],
        "fit": "eta(rz) is linear: the zero; a virtual change, so it costs no hardware risk",
        "why": "the static part of the phase error per pulse; first-order degenerate with B2 in C1, so B2 runs at the "
               "saved Q while B4 runs, and B2 is settled by leakage once C3 can run",
    },
}

CZ_EXPERIMENTS = {
    "B5_coupler_amplitude": {
        "setting": "gates.cz.slepian_crf.<QBa__QBb>.coupler.amplitude",
        "half_range": {"relative": 0.04}, "points": 9, "shots": 500,
        "probe": ["C2 layer circuits, target = ancilla, n in {1, 9}, s in {0, 1}, 4 analysis phases (32 circuits)"],
        "fit": "delta(A) is monotonic near the calibrated point: the zero of delta; the contrast at n = 9 and the "
               "control's P1 (s = 1) flag leakage and decay",
        "why": "the coupler pulse sets the |11>-|20> interaction and so the conditional phase; +-4% stays near the "
               "calibrated trajectory, away from other crossings",
    },
    "B6_qubit_flux_amplitude": {
        "setting": "gates.cz.slepian_crf.<QBa__QBb>.qubit.amplitude",
        "half_range": {"relative": 0.04}, "points": 9, "shots": 500,
        "probe": ["as B5"],
        "fit": "as B5; run after B5, at its zero",
        "why": "the fluxed qubit's detuning trajectory; second knob on delta and on the |20> population",
    },
    "B7_local_phase": {
        "setting": "gates.cz.slepian_crf.<QBa__QBb>.rz.<q> for both qubits",
        "half_range": {"absolute_rad": 0.2}, "points": 0, "shots": 500,
        "probe": ["C2 (both orientations), once, at the corrected value"],
        "fit": "rz.<q> -= phi_q from the table; one verification run",
        "why": "virtual and zero-risk; the same as the circuit-level compensation (phase tracking) but inside the gate",
    },
}

ACCEPTANCE = [
    "a new value is used only in our own jobs, never written to IQM's calibration",
    "accept it only if the probe re-run at the new value moves the coherent error toward 0 by at least 3 sigma "
    "(eps, eta, delta or phi, from the C1/C2 fit)",
    "and the incoherent part does not get worse: lam (C1) or the n = 9 contrast (C2) within 2 sigma of the saved value's",
    "and the E1 pilot (E1_d3_Z_r03_pA, 1,000 shots) at the new values has a detection fraction no higher than at the "
    "saved ones (Wilson intervals)",
    "any change of more than the half-range, or any probe that fails its pilot band, stops the run",
]


def plan(patch: Patch, path: Path = PULSE) -> dict[str, Any]:
    """Per locus: the saved centre and the sweep range of every Track B setting."""
    p = json.loads(Path(path).read_text())
    prx = p["gates"]["prx"]["drag_crf_sx"]
    cz = p["gates"]["cz"]["slepian_crf"]
    qubits: dict[str, Any] = {}
    for q in patch.qubits:
        s = prx[q]
        qubits[q] = {
            "amplitude_i": _span(s["amplitude_i"], PRX_EXPERIMENTS["B1_amplitude_coarse"]["half_range"]),
            "amplitude_q": _span(s["amplitude_q"], PRX_EXPERIMENTS["B2_drag"]["half_range"]),
            "rz_before=rz_after": _span(s["rz_before"], PRX_EXPERIMENTS["B4_virtual_phase"]["half_range"]),
            "drive_frequency_hz": _span(p["drive_frequency_hz"][q], PRX_EXPERIMENTS["B3_detuning"]["half_range"]),
        }
    couplers: dict[str, Any] = {}
    for key, s in sorted(cz.items()):
        a, b = key.split("__")
        couplers[key] = {
            "fluxed_qubit": a,
            "coupler.amplitude": _span(s["coupler.amplitude"], CZ_EXPERIMENTS["B5_coupler_amplitude"]["half_range"]),
            "qubit.amplitude": _span(s["qubit.amplitude"], CZ_EXPERIMENTS["B6_qubit_flux_amplitude"]["half_range"]),
            f"rz.{a}": _span(s[f"rz.{a}"], CZ_EXPERIMENTS["B7_local_phase"]["half_range"]),
            f"rz.{b}": _span(s[f"rz.{b}"], CZ_EXPERIMENTS["B7_local_phase"]["half_range"]),
            "duration_s": {"centre": s["duration"], "sweep": "fixed: changing it moves every other parameter"},
        }
    return {
        "format": "dqiqm.characterization.pulse-plan/1",
        "device": "garnet",
        "calibration_set_id": p["calibration_set_id"],
        "source": str(Path(path).name),
        "status": "designed only: needs a guarded hw.submit_playlist and the mentors' yes on submission rights",
        "prx_experiments": PRX_EXPERIMENTS,
        "cz_experiments": CZ_EXPERIMENTS,
        "acceptance": ACCEPTANCE,
        "qubits": qubits,
        "couplers": couplers,
    }


def _span(centre: float, half: Mapping[str, Any]) -> dict[str, Any]:
    if "relative" in half:
        h = abs(centre) * half["relative"]
    elif "absolute_rad" in half:
        h = half["absolute_rad"]
    else:
        h = half["absolute_hz"]
    return {"centre": centre, "low": centre - h, "high": centre + h,
            "margin": {k: v for k, v in half.items() if k != "around"}}
