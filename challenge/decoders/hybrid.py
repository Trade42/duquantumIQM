"""The retained Garnet decoders: correlated MWPM and posterior/count lookup.

Training uses independent labeled records. Inference accepts detection events
only, preserves shot order, and scores every returned shot.
"""
from __future__ import annotations
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np

@lru_cache(maxsize=1)
def model_posterior(dem_text):
    """Exact short-history P(L=1|s), sharing Ethan's XOR/FWHT construction.

    Keep separated pieces of each DEM instruction as ONE Bernoulli event.
    No syndrome or observable observations enter this model calculation.
    """
    import stim
    from iqm_surface_code.decoders.lookup_table import fwht_inplace
    dem=stim.DetectorErrorModel(dem_text)
    n=dem.num_detectors
    if dem.num_observables!=1 or n>24:
        raise ValueError('Posterior lookup requires one observable and <=24 detectors.')
    size=1<<(n+1)
    spectrum=np.zeros(size,np.float64)
    shift,half_faults=0,set()
    for instruction in dem.flattened():
        if instruction.type!='error':continue
        p=instruction.args_copy()[0]
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('Invalid DEM probability.')
        mask=0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():mask^=1<<target.val
            elif target.is_logical_observable_id():mask^=1<<(n+target.val)
            elif not target.is_separator():raise ValueError('Unsupported DEM target.')
        if not mask or not p:continue
        if p>.5:shift^=mask;p=1-p
        if p==.5:half_faults.add(mask)
        elif p:spectrum[mask]+=math.log1p(-2*p)
    total=float(spectrum.sum())
    fwht_inplace(spectrum);spectrum*= -.5;spectrum+=.5*total
    np.minimum(spectrum,0,out=spectrum);np.exp(spectrum,out=spectrum)
    for mask in half_faults:
        for start in range(0,size,1<<20):
            keys=np.arange(start,min(start+(1<<20),size),dtype=np.uint32)
            spectrum[start:start+len(keys)][(np.bitwise_count(keys&mask)&1).astype(bool)]=0
    fwht_inplace(spectrum);spectrum/=size
    if spectrum.min() < -1e-10 or abs(spectrum.sum()-1)>1e-7:
        raise ArithmeticError('Unstable joint DEM probability convolution.')
    np.maximum(spectrum,0,out=spectrum)
    half=size//2
    denominator=spectrum[:half]+spectrum[half:]
    posterior=np.divide(spectrum[half:],denominator,out=np.full(half,.5),where=denominator>0)
    if shift:
        posterior=posterior[np.arange(half,dtype=np.uint32)^(shift&(half-1))]
        if shift>>n:posterior=1-posterior
    posterior.flags.writeable=False
    return posterior


def binary_rows(values, width, name="Detection events"):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (shots, {width}).")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be binary.")
    return array.astype(bool, copy=False)


class FrozenAdapter:
    """Expose a measured, named decoder with the repository's batch contract."""

    def __init__(self, decoder, num_detectors, *, metadata=None):
        self.decoder = decoder
        self.num_detectors = num_detectors
        self.metadata = dict(metadata or {})

    def decode_batch(self, det_events, *, enable_correlations=None):
        if enable_correlations is False:
            raise ValueError("This frozen decoder requires its recorded correlation policy.")
        dets = binary_rows(det_events, self.num_detectors)
        if not len(dets):
            return np.zeros((0, 1), bool)
        if hasattr(self.decoder, "decode"):
            prediction = self.decoder.decode(dets)
        else:
            prediction = self.decoder.decode_batch(dets)
        prediction = binary_rows(prediction, 1, "Predictions")
        if len(prediction) != len(dets):
            raise ValueError("Decoder changed the number of shots.")
        return prediction


def build_decoder(model, family, *, train_dets=None, train_obs=None):
    """Build only the two retained policies; never fit on evaluation records."""
    from darawork.decoders.matching import MatchingDecoder
    from darawork.decoders.dem_fit import choose_dem
    if model.num_observables != 1:
        raise ValueError("Garnet memory requires exactly one logical observable.")
    if family not in {"mwpm", "posterior_10"}:
        raise ValueError("Supported final decoder families are mwpm and posterior_10.")
    meta = dict(family=family, history="offline full measured history",
                approximate_disjoint_errors=True, physical_oracle_inputs=False)
    dem = model.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    if family == "mwpm":
        decoder = MatchingDecoder(dem)
        meta["train_shots"] = 0
    else:
        from iqm_surface_code.decoders.lookup_table import LookupTableDecoder
        if train_dets is None or train_obs is None:
            raise ValueError("This decoder requires independent training shots.")
        td = binary_rows(train_dets, model.num_detectors)
        to = binary_rows(train_obs, 1, "Training observables")
        if len(td) != len(to) or len(td) < 4:
            raise ValueError("Require at least four aligned training shots.")
        dem, chosen, cv = choose_dem(model, td, to)
        meta.update(weight_choice=chosen, training_cv_errors=cv, train_shots=len(td))
        alpha = 10
        posterior = model_posterior(str(dem))
        keys = td.astype(np.uint32) @ (np.uint32(1) << np.arange(model.num_detectors, dtype=np.uint32))
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        flips = np.bincount(inverse, weights=to[:, 0], minlength=len(unique))
        table = (posterior > .5).astype(np.uint8)
        table[unique] = (flips + alpha*posterior[unique]) > .5*(counts + alpha)
        meta.update(algorithm="Dara posterior-plus-data rule on exact native DEM posterior", alpha=alpha,
                    table_entries=len(table), table_bytes=table.nbytes, empirical_entries=len(unique),
                    estimated_transform_peak_bytes=(1 << (model.num_detectors+1))*16,
                    unseen_syndrome_fallback="exact modeled ML", joint_dem_faults=True,
                    table_sha256=hashlib.sha256(table.tobytes()).hexdigest())
        decoder = LookupTableDecoder(table, model.num_detectors, meta)
    meta["dem_sha256"] = hashlib.sha256(str(dem).encode()).hexdigest()
    return FrozenAdapter(decoder, model.num_detectors, metadata=meta), dem


def save_decoder(adapter, dem, directory):
    """Save inference state as DEM, JSON and optional non-pickled lookup."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    dem.to_file(directory / "model.dem")
    (directory / "metadata.json").write_text(json.dumps(adapter.metadata, indent=2, default=str)+"\n")
    if adapter.metadata["family"] == "posterior_10":
        adapter.decoder.save(directory / "table.npz")


def load_decoder(directory):
    """Load a frozen final policy without training or network access."""
    import stim
    from darawork.decoders.matching import MatchingDecoder
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    dem = stim.DetectorErrorModel.from_file(directory / "model.dem")
    if meta["family"] == "mwpm":
        decoder = MatchingDecoder(dem)
    elif meta["family"] == "posterior_10":
        from iqm_surface_code.decoders.lookup_table import LookupTableDecoder
        decoder = LookupTableDecoder.load(directory / "table.npz")
    else:
        raise ValueError("Unknown final decoder family: " + str(meta["family"]))
    return FrozenAdapter(decoder, dem.num_detectors, metadata=meta)
