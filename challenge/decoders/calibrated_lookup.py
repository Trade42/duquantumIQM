"""Hardware-calibrated logical probabilities, compiled into a complete LUT.

Fit only on independent training/validation records. Inference is the existing
binary detector-history lookup and needs neither labels nor a classifier.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import stim

from .hybrid import binary_rows, model_posterior
from iqm_surface_code.decoders.lookup_table import LookupTableDecoder


def _keys(dets):
    return dets.astype(np.uint32) @ (np.uint32(1) << np.arange(dets.shape[1], dtype=np.uint32))


def _logit(probability):
    p = np.clip(probability, 1e-6, 1-1e-6)
    return np.log(p) - np.log1p(-p)


def _features(dets, probability, order, matching=None):
    columns = [_logit(probability)[:, None]]
    if matching is not None:
        columns.append((2*np.asarray(matching, float).reshape(-1)-1)[:, None])
    x = dets.astype(np.float64)
    if order >= 1:
        columns.append(x)
    if order >= 2:
        i, j = np.triu_indices(x.shape[1], 1)
        columns.append(x[:, i]*x[:, j])
    return np.concatenate(columns, axis=1)


def _fit_coefficients(features, truth, penalty):
    center, scale = features.mean(0), features.std(0)
    scale[scale < 1e-8] = 1
    x = (features-center)/scale
    initial = np.zeros(x.shape[1]+1)
    initial[0], initial[1] = center[0], scale[0]

    def objective(theta):
        logits = theta[0] + x @ theta[1:]
        residual = expit(logits)-truth
        value = np.mean(np.logaddexp(0, logits)-truth*logits) + .5*penalty*np.dot(theta[1:], theta[1:])
        gradient = np.r_[residual.mean(), x.T @ residual/len(x) + penalty*theta[1:]]
        return float(value), gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ArithmeticError("Logical-posterior calibration did not converge: " + str(result.message))
    weights = result.x[1:]/scale
    return np.r_[result.x[0]-center @ weights, weights], int(result.nit)


def _probabilities(dets, prior, coefficients, order, matching=None):
    if coefficients is None:
        return prior.copy()
    return expit(coefficients[0] + _features(dets, prior, order, matching) @ coefficients[1:])


def _empirical(keys, truth):
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    flips = np.bincount(inverse, weights=truth, minlength=len(unique))
    return unique, counts, flips


def _update(probabilities, keys, empirical, alpha):
    if not alpha:
        return probabilities
    unique, counts, flips = empirical
    positions = np.searchsorted(unique, keys)
    hit = positions < len(unique)
    hit[hit] &= unique[positions[hit]] == keys[hit]
    i = positions[hit]
    out = probabilities.copy()
    out[hit] = (flips[i]+alpha*out[hit])/(counts[i]+alpha)
    return out


def compile_table(prior, num_detectors, *, coefficients=None, order=0,
                  empirical=None, alpha=0, block_size=65536, matching=None, progress=None):
    """Compile the calibrated decision rule for every possible detector history."""
    table = np.empty(1 << num_detectors, np.uint8)
    if coefficients is not None:
        coefficients = np.asarray(coefficients)
        offset = 3 if matching is not None else 2
        linear = coefficients[offset:offset+num_detectors] if order >= 1 else np.zeros(num_detectors)
        quadratic = np.zeros((num_detectors, num_detectors))
        if order >= 2:
            quadratic[np.triu_indices(num_detectors, 1)] = coefficients[offset+num_detectors:]
    for start in range(0, len(table), block_size):
        if progress and len(table) > 65536 and start % (1 << 22) == 0:
            progress(f"Compiling LUT: {start:,} / {len(table):,} histories")
        end = min(start+block_size, len(table))
        keys = np.arange(start, end, dtype=np.uint32)
        q = prior[start:end]
        if coefficients is not None:
            logits = coefficients[0] + coefficients[1]*_logit(q)
            if order >= 1 or matching is not None:
                x = ((keys[:, None] >> np.arange(num_detectors, dtype=np.uint32)) & 1).astype(np.float64)
                if matching is not None:
                    logits += coefficients[2]*(2*np.asarray(matching(x), float).reshape(-1)-1)
            if order >= 1:
                logits += x @ linear
                if order >= 2:
                    logits += np.einsum("bi,bi->b", x @ quadratic, x)
            q = expit(logits)
        if empirical is not None and alpha:
            q = _update(q, keys, empirical, alpha)
        table[start:end] = q > .5
    return table


def fit(dem, train_dets, train_obs, validation_dets, validation_obs, *,
        orders=(0, 1, 2), penalties=(.1, .01, .001), alphas=(0., 10.), use_matching=True, progress=None):
    """Select a posterior correction on validation and freeze a complete table.

    A regularized logistic correction learns the prior's confidence, detector
    biases, pairwise detector interactions and an optional matching prediction.
    Matching is evaluated during preprocessing; inference uses the frozen table.
    Validation selects complexity and the optional history-count update.
    """
    if not isinstance(dem, stim.DetectorErrorModel) or dem.num_observables != 1 or dem.num_detectors > 24:
        raise ValueError("Calibrated LUT requires one observable and at most 24 detectors.")
    n = dem.num_detectors
    td = binary_rows(train_dets, n)
    to = binary_rows(train_obs, 1, "Training observables")
    vd = binary_rows(validation_dets, n)
    vo = binary_rows(validation_obs, 1, "Validation observables")
    if len(td) != len(to) or len(vd) != len(vo) or min(len(td), len(vd)) < 4:
        raise ValueError("Require aligned independent training and validation records.")
    if any(order not in (0, 1, 2) for order in orders) or any(p <= 0 for p in penalties) or any(a < 0 for a in alphas):
        raise ValueError("Invalid correction order, penalty or pseudo-count strength.")
    start = time.perf_counter()
    prior = model_posterior(str(dem))
    tk, vk = _keys(td), _keys(vd)
    empirical = _empirical(tk, to[:, 0])
    matching = None
    train_matching = validation_matching = None
    if use_matching:
        import pymatching
        matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        matching = lambda x: matcher.decode_batch(np.asarray(x, bool), enable_correlations=True)
        train_matching, validation_matching = matching(td), matching(vd)
    candidates = []
    parameters = []

    def consider(coefficients, order, penalty, iterations):
        prediction = _probabilities(vd, prior[vk], coefficients, order, validation_matching)
        for alpha in alphas:
            q = _update(prediction, vk, empirical, alpha)
            q = np.clip(q, 1e-12, 1-1e-12)
            errors = int(np.count_nonzero((q > .5) != vo[:, 0]))
            loss = float(-np.mean(vo[:, 0]*np.log(q) + (1-vo[:, 0])*np.log1p(-q)))
            candidates.append(dict(order=order if coefficients is not None else -1, penalty=penalty,
                                   alpha=alpha, validation_errors=errors, validation_log_loss=loss,
                                   parameters=0 if coefficients is None else len(coefficients), iterations=iterations))
            parameters.append(coefficients)

    consider(None, 0, 0., 0)
    for order in orders:
        features = _features(td, prior[tk], order, train_matching)
        for penalty in penalties:
            coefficients, iterations = _fit_coefficients(features, to[:, 0].astype(float), penalty)
            consider(coefficients, order, penalty, iterations)
    selected = min(range(len(candidates)), key=lambda i: (
        candidates[i]["validation_errors"], candidates[i]["parameters"],
        bool(candidates[i]["alpha"]), -candidates[i]["penalty"]))
    choice = candidates[selected]
    table = compile_table(prior, n, coefficients=parameters[selected], order=max(0, choice["order"]),
                          empirical=empirical, alpha=choice["alpha"], matching=matching, progress=progress)
    metadata = dict(family="calibrated_lookup", algorithm="validation-selected conditional logical-posterior correction",
        num_observables=1, num_detectors=n, train_shots=len(td), validation_shots=len(vd),
        dem_sha256=hashlib.sha256(str(dem).encode()).hexdigest(), joint_dem_faults=True,
        approximate_disjoint_errors=True, selection=choice, candidates=candidates,
        coefficients=None if parameters[selected] is None else parameters[selected].tolist(),
        matching_feature=bool(use_matching and parameters[selected] is not None),
        table_sha256=hashlib.sha256(table.tobytes()).hexdigest(), table_entries=len(table), table_bytes=table.nbytes,
        estimated_transform_peak_bytes=(1 << (n+1))*16, build_seconds=time.perf_counter()-start,
        inference="binary detection events only; one complete-table lookup; logical 0 on ties")
    decoder = LookupTableDecoder(table, n, metadata)
    if np.count_nonzero(decoder.decode_batch(vd)[:, 0] != vo[:, 0]) != choice["validation_errors"]:
        raise ArithmeticError("Compiled table differs from the selected validation decision rule.")
    return decoder


def load(path):
    """Load a frozen hardware-calibrated LUT using the standard batch interface."""
    decoder = LookupTableDecoder.load(path)
    if decoder.metadata.get("family") != "calibrated_lookup":
        raise ValueError("Expected a calibrated logical lookup policy.")
    if hashlib.sha256(decoder.table.tobytes()).hexdigest() != decoder.metadata["table_sha256"]:
        raise ValueError("Calibrated lookup contents differ from their checksum.")
    return decoder


def load_policy(basis, rounds, *, bundle=None):
    """Load one of the included hardware-calibrated Garnet X/Z policies."""
    import json
    if basis not in {"X", "Z"} or rounds not in {1, 2, 3}:
        raise ValueError("Included calibrated policies support X/Z memory at rounds 1, 2 and 3.")
    bundle = Path(bundle) if bundle is not None else Path(__file__).resolve().parents[2]/"benchmarks/calibrated_lookup"
    selected = json.loads((bundle/"frozen_selection.json").read_text())[f"{basis}{rounds}"]
    path = bundle/selected["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != selected["sha256"]:
        raise ValueError("Included calibrated policy differs from its frozen hash.")
    return load(path)
