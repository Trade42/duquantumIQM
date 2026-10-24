"""Complete full-history logical maximum-likelihood lookup for small DEMs.

For n detection bits the table has 2**n entries. Build sums the probability of
all independent DEM fault combinations, using a Walsh-Hadamard XOR convolution.
No sampling, fitted labels, matching decoder or unseen-syndrome fallback is used.
This is exponential preprocessing, intended for small distance-3 baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
import numpy as np


def fwht_inplace(values):
    """Unnormalized real Walsh-Hadamard transform with vectorized butterflies.

    Consecutive N/2 left-copy allocations can overlap, using up to N scratch
    floats in addition to the N-float transform array.
    """
    if values.ndim != 1 or not len(values) or len(values) & (len(values)-1):
        raise ValueError("FWHT requires a nonempty power-of-two vector.")
    half = 1
    while half < len(values):
        blocks = values.reshape(-1,2*half)
        left,right = blocks[:,:half],blocks[:,half:]
        original = left.copy()
        np.add(original,right,out=left)
        np.subtract(original,right,out=right)
        half *= 2
    return values


@dataclass
class LookupTableDecoder:
    table: np.ndarray
    num_detectors: int
    metadata: dict

    def __post_init__(self):
        if self.table.shape != (1 << self.num_detectors,) or self.table.dtype != np.uint8:
            raise ValueError("Invalid complete lookup table shape or dtype.")
        if np.any(self.table > 1):
            raise ValueError("Lookup predictions must be binary.")
        self.table.flags.writeable = False

    def decode_batch(self, det_events, *, enable_correlations=True):
        if not enable_correlations:
            raise ValueError("Lookup table always retains joint DEM faults; disabling correlations is unsupported.")
        dets = np.asarray(det_events)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError("Detection events must have shape (shots, num_detectors).")
        if not np.all((dets == 0) | (dets == 1)):
            raise ValueError("Detection events must be binary.")
        weights = np.left_shift(np.uint32(1),np.arange(self.num_detectors,dtype=np.uint32))
        keys = dets.astype(np.uint32) @ weights
        return self.table[keys,None].astype(bool)

    def save(self, path):
        np.savez_compressed(path,table=self.table,
            metadata=np.array(json.dumps(dict(self.metadata,num_detectors=self.num_detectors))))

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls(data["table"].copy(),int(metadata["num_detectors"]),metadata)


def build_lookup_table(circuit_or_dem, *, max_detectors=24):
    """Build argmax_L P(L,s) for every full detector record s.

    DEM errors are independent Bernoulli XOR faults. Separator-marked components
    remain one correlated fault. Circuit-to-DEM extraction explicitly permits
    disjoint-channel approximation; that approximation is recorded. Supports one
    logical observable and up to 24 detector bits (16,777,216 table entries).
    Deterministic and p=1/2 faults are supported. Exact ML refers to this DEM,
    not to the physical QPU's unknown distribution.
    """
    start = time.perf_counter()
    import stim
    if isinstance(circuit_or_dem,stim.Circuit):
        dem = circuit_or_dem.detector_error_model(approximate_disjoint_errors=True)
        source_hash = hashlib.sha256(str(circuit_or_dem).encode()).hexdigest()
    elif isinstance(circuit_or_dem,stim.DetectorErrorModel):
        dem,source_hash = circuit_or_dem,None
    else:
        raise TypeError("Expected a Stim circuit or detector error model.")
    n = dem.num_detectors
    if dem.num_observables != 1:
        raise ValueError("Lookup baseline supports exactly one logical observable.")
    if n > max_detectors or n > 24:
        raise ValueError(f"Complete LUT needs 2**{n} entries; supported limit is {min(max_detectors,24)} detector bits.")
    size = 1 << (n+1)
    spectrum = np.zeros(size,dtype=np.float64)
    deterministic,half_faults,faults = 0,set(),0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("DEM error probabilities must lie in [0,1].")
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                mask ^= 1 << (n+target.val)
            elif not target.is_separator():
                raise ValueError("Unsupported DEM target.")
        if not mask or not p:
            continue
        faults += 1
        if p > .5:
            deterministic ^= mask
            p = 1-p
        if p == .5:
            half_faults.add(mask)
        elif p:
            spectrum[mask] += math.log1p(-2*p)
    # log phi(k) = [sum(a) - H(a)(k)]/2, with a[f]=sum log(1-2p_f).
    total = float(spectrum.sum())
    fwht_inplace(spectrum)
    spectrum *= -.5
    spectrum += .5*total
    np.minimum(spectrum,0,out=spectrum)  # remove positive roundoff at phi(0)
    np.exp(spectrum,out=spectrum)
    for mask in half_faults:
        for begin in range(0,size,1 << 20):
            keys=np.arange(begin,min(begin+(1 << 20),size),dtype=np.uint32)
            odd=(np.bitwise_count(keys & mask) & 1).astype(bool)
            spectrum[begin:begin+len(keys)][odd] = 0
    fwht_inplace(spectrum)
    spectrum /= size
    minimum=float(spectrum.min())
    if minimum < -1e-10:
        raise ArithmeticError("Unstable probability convolution.")
    np.maximum(spectrum,0,out=spectrum)
    mass=float(spectrum.sum())
    if abs(mass-1) > 1e-7:
        raise ArithmeticError("Joint probability distribution is not normalized.")
    half = size//2
    table = (spectrum[half:] > spectrum[:half]).astype(np.uint8)
    # Only the posterior decision is needed at decode time; discard joint P.
    model_failure = float(np.minimum(spectrum[:half],spectrum[half:]).sum()/mass)
    if deterministic:
        shift_s=deterministic & (half-1)
        if shift_s:
            table=table[np.arange(half,dtype=np.uint32) ^ shift_s]
        table ^= (deterministic >> n) & 1
    metadata = dict(algorithm="complete logical ML lookup via independent-DEM XOR convolution",
        num_observables=1,detector_bits=n,entries=half,table_bytes=int(table.nbytes),
        transform_array_bytes=size*8,estimated_transform_peak_bytes=size*16,
        build_s=time.perf_counter()-start,fault_terms=faults,
        circuit_sha256=source_hash,dem_sha256=hashlib.sha256(str(dem).encode()).hexdigest(),
        table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        model_bayes_failure=model_failure,probability_mass=mass,minimum_probability_before_clip=minimum,
        approximate_disjoint_errors=True,joint_dem_faults=True,tie_break="logical 0",
        training_shots=0,unseen_syndrome_fallback=None,
        limitation="Exponential preprocessing; exact for supplied independent DEM, with circuit-to-DEM approximation.")
    return LookupTableDecoder(table,n,metadata)
