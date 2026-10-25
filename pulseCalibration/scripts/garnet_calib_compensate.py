"""The compensated E1 twin, checked: the closure's measured local CZ phases, cancelled by phase tracking.

    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
        uv run python scripts/garnet_calib_compensate.py      # ~3 min; writes InfoRecord/characterization/sim/compensate.json

SIMULATED, noise-free Aer (statevector), seeded. The pilot circuit E1_d3_Z_r03_pA (and X) runs with
the injected local phases of the closure's injected world, replica 1, after every CZ: plain,
compensated with the injected (true) phases, and compensated with the phases C2 *measured* in that
world (``sim/derived/injected/r01/fits.json``). The ideal circuit gives no detection events; so
must the exactly compensated one; the measured compensation leaves only its estimation error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dqiqm.characterization import compensate as K  # noqa: E402
from dqiqm.characterization import patch as P  # noqa: E402
from dqiqm.experiments import device as dev  # noqa: E402
from dqiqm.experiments import memory as M  # noqa: E402

SIM = REPO / "InfoRecord" / "characterization" / "sim"
SHOTS = 2000


def main() -> int:
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    be = IQMFakeGarnet()
    fits = json.loads((SIM / "derived" / "injected" / "r01" / "fits.json").read_text())
    true = {}
    for key, v in fits["injected"]["cz"].items():
        a, b = key.split("-")
        true[dev.pair_key(a, b)] = {a: v[f"phi_{a}_rad"], b: v[f"phi_{b}_rad"]}
    measured: dict = {}
    for key, f in fits["c2"].items():
        _, rest = key.split(":")
        t, c = rest.split("<-")
        measured.setdefault(dev.pair_key(t, c), {})[t] = f["values"]["phi_t"]
    placements = json.loads(P.PILOT_PLACEMENT.read_text())
    out = {"label": "SIMULATED", "device": "garnet", "simulator": "Aer statevector, no noise", "shots": SHOTS,
           "phases": "injected world replica 1 (sim/derived/injected/r01): true and C2-measured local phases",
           "max_abs_phase_error_rad": max(abs(measured[p][q] - true[p][q]) for p in true for q in true[p]),
           "circuits": {}}
    for name in ("E1_d3_Z_r03_pA", "E1_d3_X_r03_pA"):
        e = placements[name]
        spec = M.spec(3, e["params"]["basis"], P._placement(e), "pA", be)
        res = {"true": K.detection_check(spec, true, true, shots=SHOTS),
               "measured": K.detection_check(spec, true, measured, shots=SHOTS)}
        out["circuits"][name] = {"ideal": res["true"]["ideal"], "plain": res["true"]["plain"],
                                 "compensated_true": res["true"]["compensated"],
                                 "compensated_measured": res["measured"]["compensated"]}
        print(name, json.dumps(out["circuits"][name]))
    (SIM / "compensate.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
