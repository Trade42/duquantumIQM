# SIMULATED: the Garnet dry run

Everything under `sim/` is **simulated**: the stand-in `IQMFakeGarnet+profile(a18a947f)` (IQMFakeGarnet with the
live noise of calibration set `a18a947f-0408-4628-81ea-1715637e434f`), Aer matrix-product-state method, seeded.
It is the full record's layout, at small shot counts ({"P0": 1000, "P1": 256, "E1": 128, "E2": 64, "E3": 256, "E4": 128, "E5": 256} per circuit), so
every reader and every analysis can be built before the hardware data exist. Do not mix it with
hardware data: every manifest says `"label": "SIMULATED"`, every shot `simulated = 1`.

Regenerate it (about 9 min wall on 4 cores):

```
flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \
    uv run python scripts/garnet_dryrun.py --clean
```

`checks/` holds the run's own checks: `structure.json`, `noiseless.json`, `mps_vs_statevector.json`,
`decode.json`, `features.json`, `sizes.json`, `summary.json` and the log.
