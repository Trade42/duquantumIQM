"""Measurement records to detection events, MWPM decoding, and the logical error probability.

The measurement record is the M1 contract's (``orchestration/briefs/_m1-contract.md``, section
2): a ``uint8`` array, shots x num_measurements, whose column j is Stim measurement j. Layout's
Qiskit circuit writes Stim measurement j into clbit j, and ``memory.npz`` (written by
:func:`dqiqm.hw.save_raw`) stores column j = clbit j, so hardware data arrives in this form.

* :func:`detection_events`: record -> (detection events, observable flips), with Stim's
  ``compile_m2d_converter``. The call is the one in ``challenge/extract_syndromes.py``; the
  shape and value checks are ours.
* :func:`decode`: detection events -> predicted observable flips, with PyMatching built from a
  detector error model: the circuit's own (``decompose_errors=True``) or one passed in.
* :func:`logical_error_probability`: predictions and observed flips -> (k, n, P_L, Wilson 95%).
* :func:`load_memory` and :func:`counts_to_measurements`: hardware results -> the record. The
  counts path follows ``challenge/internal_helpers.py``'s ``counts_to_measurement_array``
  (reverse each Qiskit bitstring so index 0 is clbit 0), and refuses where that one truncates.

The circuit given to :func:`detection_events` must be the circuit that ran (on Emerald, the
no-reset one). The detector error model given to :func:`decode` must have that circuit's
detectors and observables; the circuit that runs on hardware has no noise, so its model comes
from a noisy copy of it or from calibration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pymatching
import stim

from dqiqm.stats import wilson


class LogicalErrorProbability(NamedTuple):
    """Logical failures ``k`` in ``n`` shots, ``p = k / n``, and its 95% Wilson interval."""

    k: int
    n: int
    p: float
    wilson_lo: float
    wilson_hi: float


def _as_record(measurements: np.ndarray, num_measurements: int) -> np.ndarray:
    m = np.asarray(measurements)
    if m.ndim != 2:
        raise ValueError(f"measurements must be 2-D (shots, num_measurements), got shape {m.shape}")
    if m.shape[1] != num_measurements:
        raise ValueError(
            f"measurements have {m.shape[1]} columns but the circuit has {num_measurements} measurements"
        )
    if m.dtype != np.bool_:
        if m.size and (m.min() < 0 or m.max() > 1):
            raise ValueError("measurements must be 0/1 (one column per measurement, not bit-packed)")
        m = m.astype(np.bool_)
    return m


def detection_events(circuit: stim.Circuit, measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn a measurement record into detection events and observable flips.

    ``circuit`` is the circuit that ran; only its measurements, detectors and observables
    matter, so noise in it is ignored. ``measurements`` is (shots, ``circuit.num_measurements``),
    0/1 or bool, column j being Stim measurement j. Stim's converter compares each detector and
    observable with the circuit's noiseless reference sample, so a noiseless record gives no
    events and no flips.

    Returns ``(dets, obs)``: bool arrays of shape (shots, num_detectors) and
    (shots, num_observables).
    """
    record = _as_record(measurements, circuit.num_measurements)
    converter = circuit.compile_m2d_converter()
    dets, obs = converter.convert(measurements=record, separate_observables=True)
    return dets, obs


def detector_error_model(circuit_or_dem: stim.Circuit | stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """The circuit's own model (``decompose_errors=True``), or the model given."""
    if isinstance(circuit_or_dem, stim.DetectorErrorModel):
        return circuit_or_dem
    if isinstance(circuit_or_dem, stim.Circuit):
        return circuit_or_dem.detector_error_model(decompose_errors=True)
    raise TypeError(f"expected a stim.Circuit or stim.DetectorErrorModel, got {type(circuit_or_dem).__name__}")


def _has_errors(dem: stim.DetectorErrorModel) -> bool:
    return any(inst.type == "error" for inst in dem.flattened())


def decode(circuit_or_dem: stim.Circuit | stim.DetectorErrorModel, dets: np.ndarray) -> np.ndarray:
    """Decode detection events with PyMatching (minimum-weight perfect matching).

    ``circuit_or_dem`` is a noisy circuit with the detectors of the circuit that ran (its model is
    built with ``decompose_errors=True``), or a ``stim.DetectorErrorModel`` with those detectors:
    a calibrated or data-learned one, for example. A noiseless circuit is refused, since its
    model has no error mechanisms to match on.

    Returns the predicted observable flips, bool, shape (shots, num_observables).
    """
    dem = detector_error_model(circuit_or_dem)
    dets = np.asarray(dets)
    if dets.ndim != 2:
        raise ValueError(f"dets must be 2-D (shots, num_detectors), got shape {dets.shape}")
    if dets.shape[1] != dem.num_detectors:
        raise ValueError(
            f"dets have {dets.shape[1]} detectors but the detector error model has {dem.num_detectors}: "
            "the model must match the circuit that ran"
        )
    if not _has_errors(dem):
        raise ValueError(
            "the detector error model has no error mechanisms (a noiseless circuit?): pass a noisy circuit "
            "with the same detectors, or a detector error model"
        )
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = matcher.decode_batch(dets.astype(np.uint8))
    return predictions.astype(np.bool_).reshape(len(dets), dem.num_observables)


def logical_error_probability(predictions: np.ndarray, obs: np.ndarray) -> LogicalErrorProbability:
    """Count the shots whose predicted flips differ from the observed ones on any observable."""
    predictions = np.asarray(predictions, dtype=np.bool_)
    obs = np.asarray(obs, dtype=np.bool_)
    if predictions.shape != obs.shape or predictions.ndim != 2:
        raise ValueError(f"predictions {predictions.shape} and obs {obs.shape} must be the same 2-D shape")
    n = len(obs)
    if n == 0:
        raise ValueError("no shots")
    k = int(np.any(predictions != obs, axis=1).sum())
    lo, hi = wilson(k, n)
    return LogicalErrorProbability(k, n, k / n, float(lo), float(hi))


# ---------------------------------------------------------------------------
# Hardware results -> the measurement record.
# ---------------------------------------------------------------------------


def load_memory(
    job_dir: str | Path, index: int = 0, *, circuit: stim.Circuit | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load circuit ``index`` of one saved hardware job as the measurement record.

    ``job_dir`` is ``data/hw/<YYYYMMDD>-<task>/<job id>/``, as :func:`dqiqm.hw.save_raw` writes it.
    ``memory.npz``'s ``c<index>`` already has column j = clbit j, and by the contract clbit j is
    Stim measurement j, so the array is returned as stored. With ``circuit``, its column count is
    checked against ``circuit.num_measurements``.

    Returns ``(measurements, meta)``: uint8 (shots, clbits), and ``meta.json`` as a dict.
    """
    job_dir = Path(job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    if not meta.get("success", False):
        raise ValueError(f"job {meta.get('job_id')} did not succeed: no memory.npz to load")
    n_circuits = meta.get("circuits")
    if n_circuits is not None and not 0 <= index < n_circuits:
        raise IndexError(f"circuit index {index} out of range: job {meta.get('job_id')} has {n_circuits}")
    with np.load(job_dir / "memory.npz") as npz:
        key = f"c{index}"
        if key not in npz.files:
            raise KeyError(f"{job_dir / 'memory.npz'} has no {key} (it has {sorted(npz.files)})")
        bits = npz[key]
    if bits.ndim != 2 or bits.dtype != np.uint8:
        raise ValueError(f"{key} must be a 2-D uint8 array, got {bits.dtype} {bits.shape}")
    if bits.size and bits.max() > 1:
        raise ValueError(f"{key} has values other than 0 and 1")
    if meta.get("shots") is not None and len(bits) != meta["shots"]:
        raise ValueError(f"{key} has {len(bits)} shots but meta.json says {meta['shots']}")
    if circuit is not None and bits.shape[1] != circuit.num_measurements:
        raise ValueError(
            f"{key} has {bits.shape[1]} clbits but the circuit has {circuit.num_measurements} measurements"
        )
    return bits, meta


def counts_to_measurements(
    counts: Mapping[str, int], num_measurements: int, *, shots: int | None = None
) -> np.ndarray:
    """Expand Qiskit counts into the measurement record, one row per shot.

    A Qiskit count key is printed right to left: its rightmost character is clbit 0, and
    registers are separated by spaces, the last register leftmost. Removing the spaces and
    reversing the string puts clbit j at index j. Hexadecimal keys (``0x...``) are read as the
    integer whose bit j is clbit j. Every key must have exactly ``num_measurements`` bits, and
    with ``shots`` the counts must sum to it.

    Counts lose the shot order: rows come grouped by outcome, in the dict's order. That is fine
    for the logical error probability, and for a bootstrap over shots.

    Returns uint8 (total counts, ``num_measurements``).
    """
    rows = []
    reps = []
    for key, count in counts.items():
        if key.startswith("0x"):
            value = int(key, 16)
            if value >> num_measurements:
                raise ValueError(f"count key {key!r} has more than {num_measurements} bits")
            bits = [(value >> j) & 1 for j in range(num_measurements)]
        else:
            s = key.replace(" ", "")
            if len(s) != num_measurements or set(s) - {"0", "1"}:
                raise ValueError(f"count key {key!r} is not {num_measurements} bits of 0 and 1")
            bits = [int(c) for c in reversed(s)]
        if int(count) < 0:
            raise ValueError(f"negative count for {key!r}")
        rows.append(bits)
        reps.append(int(count))
    total = sum(reps)
    if shots is not None and total != shots:
        raise ValueError(f"counts sum to {total}, not {shots} shots")
    if not rows:
        return np.zeros((0, num_measurements), dtype=np.uint8)
    return np.repeat(np.array(rows, dtype=np.uint8).reshape(len(rows), num_measurements), reps, axis=0)
