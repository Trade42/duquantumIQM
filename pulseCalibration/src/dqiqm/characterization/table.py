"""The coherent-error table as CSV: the runbook's Section 8 schema, adapted.

One row per (locus, parameter, experiment, context). The runbook's columns are kept in its order;
``parameter``, ``context``, ``se``, ``chi2_per_dof``, ``simulated`` and ``injected`` are added, and
``leakage`` holds C3's figure when C3 has run (empty until then). Angles are in rad; the error
columns of the calibration (``rb_error``: the PRX RB infidelity or the CZ IRB infidelity) and T1 / T2*
are copied from the calibration set the circuits were built on, for comparison.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

COLUMNS = (
    "timestamp_utc",
    "device",
    "calibration_set_id",
    "locus",
    "gate",
    "implementation",
    "experiment",
    "context",
    "parameter",
    "shots",
    "seed",
    "fit_value",
    "fit_unit",
    "fit_ci_low",
    "fit_ci_high",
    "se",
    "chi2_per_dof",
    "leakage",
    "rb_error",
    "simultaneous_rb_error",
    "t1_us",
    "t2star_us",
    "change_status",
    "simulated",
    "injected",
    "notes",
)

#: What each parameter means (``docs/garnet-calib.md`` has the model).
PARAMETERS = {
    "eps": "signed amplitude error of one prx(pi/2): angle = pi/2 + eps (rad); > 0 over-rotates",
    "eta": "Z component of the prx(pi/2) rotation vector (rad): the axis tilt (detuning, DRAG, rz_before/after)",
    "delta": "CZ conditional-phase error: the |11> phase is pi + delta after local phases (rad)",
    "phi": "residual local Z phase of the named qubit per CZ, as RZ(phi) ~ diag(1, e^{i phi}) (rad)",
}


def fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.6g}"
    return "" if v is None else v


def write(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            unknown = set(r) - set(COLUMNS)
            if unknown:
                raise ValueError(f"unknown table columns: {sorted(unknown)}")
            w.writerow({k: fmt(r.get(k)) for k in COLUMNS})
    return path


def read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))
