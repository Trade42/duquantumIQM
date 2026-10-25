"""Verify saved Garnet records in the source and native measurement frames."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import stim
from .benchmark_headline import digest
from .decoders.hybrid import binary_rows

def verify_case(output, case):
    output = Path(output)
    source = stim.Circuit.from_file(output/f"{case['id']}.source.stim")
    native = stim.Circuit.from_file(output/f"{case['id']}.native.stim")
    for field, circuit in (("source_sha256", source), ("native_sha256", native)):
        if hashlib.sha256(str(circuit).encode()).hexdigest() != case[field]:
            raise ValueError("Circuit definition hash changed.")
    if digest(output/f"{case['id']}.decoder.stim") != case["model_sha256"]:
        raise ValueError("Physical decoder prior hash changed.")
    path = output/case["data_file"]
    if digest(path) != case["dataset_sha256"]:
        raise ValueError("Hardware dataset hash changed.")
    with np.load(path, allow_pickle=False) as arrays:
        raw = binary_rows(arrays["raw_meas"], source.num_measurements)
        nraw = binary_rows(arrays["native_raw_meas"], native.num_measurements)
        dets = binary_rows(arrays["det_events"], source.num_detectors)
        obs = binary_rows(arrays["obs_flips"], source.num_observables)
        if any(len(a) != case["shots"] for a in (raw, nraw, dets, obs)):
            raise ValueError("Returned shot count changed.")
        columns = case["native_measurement_columns"]
        if sorted(columns) != list(range(source.num_measurements)) or not np.array_equal(raw[:, columns], nraw):
            raise ValueError("Native measurement order differs.")
        for circuit, measurements in ((source, raw), (native, nraw)):
            d, o = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
            if not np.array_equal(d, dets) or not np.array_equal(o, obs):
                raise ValueError("Source/native detector or observable frame differs.")
        return dets.copy(), obs.copy()
