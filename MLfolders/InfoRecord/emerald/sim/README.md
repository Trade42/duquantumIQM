# SIMULATED: the Emerald dry run

Everything under `sim/` is **simulated**: the stand-in `IQMFakeAphrodite+profile(23e12f6b)` (IQMFakeAphrodite,
Emerald's 54 qubits and 90 couplers, with the live noise of calibration set `23e12f6b-c23c-4736-8736-b0142989c0e6`;
the 5 CZ-less couplers padded and never used), Aer matrix-product-state method, seeded. It has
**no idle noise**: Aer puts none on `delay` or on idle qubits, so E4 is checked for structure only.

It is the full record's layout at small shot counts ({"P0": 1000, "E1": 96, "E4": 48} per circuit), so every
reader and every analysis can be built before the hardware data exist. Do not mix it with hardware
data, and do not pre-train on it (see `../calibration/emerald/README.md`): every manifest says
`"label": "SIMULATED"`, every shot `simulated = 1`. The manifests' `device` is `emerald` (the
device the circuits are for); `simulator.backend` and the raw `meta.json`'s `device` name the
stand-in.

Regenerate it (about 8 min wall on 4 cores):

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
    uv run python scripts/emerald_dryrun.py --clean
```

`checks/` holds the run's own checks: `structure.json`, `noiseless.json`, `decode.json`,
`crosscheck.json`, `d5.json`, `summary.json` and the log.
