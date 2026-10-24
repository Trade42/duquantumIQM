"""Fetch/export a calibration snapshot explicitly; safe to import offline."""
from __future__ import annotations


def fetch_calibration(backend, *, durations_ns, coherence_unit="us", reset_error=0.002):
    from iqm.qubit_selector.qubit_selector import CalibrationDataManager, CalibrationType
    from iqm_surface_code.calibration import CalibrationSnapshot
    from datetime import datetime, timezone
    metrics = CalibrationDataManager().get_calibration_fidelities(backend)
    # Use enum values directly to avoid depending on their serialized spelling.
    named = {kind.name: metrics[kind.value] for kind in
             (CalibrationType.SQG, CalibrationType.READOUT, CalibrationType.T1,
              CalibrationType.T2, CalibrationType.CZ)}
    return CalibrationSnapshot.from_iqm_metrics(named, durations_ns=durations_ns,
        coherence_unit=coherence_unit, reset_error=reset_error,
        metadata={"source": "IQM CalibrationDataManager", "retrieved_utc": datetime.now(timezone.utc).isoformat()})


def main():
    import argparse
    from iqm_surface_code.run_on_hardware import connect_backend
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--device", default="emerald")
    p.add_argument("--one-q-ns", type=float, required=True)
    p.add_argument("--two-q-ns", type=float, required=True)
    p.add_argument("--measure-ns", type=float, required=True)
    p.add_argument("--reset-ns", type=float, required=True)
    p.add_argument("--reset-error", type=float, required=True)
    p.add_argument("--coherence-unit", choices=["s", "ms", "us", "ns"], required=True)
    a = p.parse_args()
    snapshot = fetch_calibration(connect_backend(quantum_computer=a.device),
        durations_ns={"1q": a.one_q_ns, "2q": a.two_q_ns, "measure": a.measure_ns, "reset": a.reset_ns},
        coherence_unit=a.coherence_unit, reset_error=a.reset_error)
    snapshot.save(a.output)


if __name__ == "__main__":
    main()
