# resonance-check: first contact with IQM Resonance

This task was read-only and spent **zero credits**: it submitted no job of any kind (real, `:mock` or `facade_*`), and nothing in it can submit.
Snapshots taken 2026-10-24 between 19:09 and 19:13 UTC.

## Run it

```
uv run python scripts/resonance_check.py             # live: needs IQM_TOKEN (.env); writes new snapshots, then analyses them
uv run python scripts/resonance_check.py --offline   # no network, no token: analyses the newest committed snapshots
uv run python scripts/resonance_check.py --snapshot data/calibration/garnet/<file>.json   # one file
```

The live run takes about 30 s and the offline run about 3 s, so neither needs the sim lock.
The analysis printed by `--offline` is byte-identical to the one the live run printed when it wrote the committed snapshots.

**Live calls.** Only `dqiqm.hw`'s read-only helpers are called: `list_devices`, `architecture`, `calibration_set_id`, `calibration_fidelities` and `backend`.
On the read-only backend's `IQMClient`, the script also makes the GET calls those helpers rest on:

- `get_quality_metric_set(calset)`;
- `get_calibration_set(calset)`;
- `get_static_quantum_architecture()`;
- `get_health()`;
- `get_about()`.

Every call after `architecture` is pinned to the calibration set that `architecture` returned. So one snapshot describes one calibration set even if the server recalibrates during the fetch.

## Snapshots: `data/calibration/<device>/`

**File name.** The first snapshot of a calibration set is `<calibration_set_id>.json`. A later one is `<calibration_set_id>.<YYYYMMDDTHHMMSSZ>.json`. The file is opened with mode `x`, so it is never overwritten.

**Contents:**

| Key | What |
|---|---|
| `calibration_set_id`, `quality_metric_set_id`, `default_calibration_set_id_at_fetch` | which calibration set and metric set the figures describe; the server's default when fetched |
| `fetched_utc_start`, `fetched_utc_end`, `dqiqm_commit`, `versions` | when, from which commit, with which iqm-client, iqm-qubit-selector, qiskit and networkx |
| `helpers` | every call, `ok` or its exact error |
| `device_list`, `device_list_calls` | the devices the token sees, and how they were read |
| `architecture` | `hw.architecture()`: the dynamic architecture (qubits, gates, implementations, loci) |
| `static_architecture` | every coupler of the chip, calibrated or not, plus the chip's `dut_label` |
| `calibration_fidelities` | `hw.calibration_fidelities()`: iqm-qubit-selector's parsed figures |
| `quality_metric_set` | every observation of the metric set (dut_field, value, unit, uncertainty, timestamps) |
| `calibration_set` | the observations that describe the qubits: `characterization.*`, `gate_definitions.*` and every `*.duration`. The control-electronics settings (`controllers.*` and pulse shapes) are counted in `observations_total` but not kept |
| `health`, `about` | server status and version |

No token or header is in a snapshot. The script refuses to write one that contains the token or an authorization header.

**Committed.** Two snapshots per device. Each pair has identical observations, and differs only in the fetch times, `dqiqm_commit` and the health timestamp.

| File | Size | Written by |
|---|---|---|
| `garnet/a18a947f-0408-4628-81ea-1715637e434f.json` | 0.42 MB | the script before its commit (`dqiqm_commit` 7a10dd8, the base) |
| `garnet/a18a947f-0408-4628-81ea-1715637e434f.20261024T191313Z.json` | 0.42 MB | the committed script (22d63dc) |
| `emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.json` | 1.14 MB | the script before its commit (`dqiqm_commit` 7a10dd8) |
| `emerald/23e12f6b-c23c-4736-8736-b0142989c0e6.20261024T191321Z.json` | 1.14 MB | the committed script (22d63dc) |

The second file of each pair also shows the no-overwrite rule at work.

## Which field is which

Unless noted, the figures come from the metric set of the default `measure` and `prx` implementations (both `constant`/`drag_crf_sx`) and of each coupler's CZ implementation.
The CZ implementation honours `override_default_implementation`, which on Emerald lists pairs in either order (for example `QB51,QB50`).

| Figure | Observation |
|---|---|
| T1, T2 Ramsey, T2 echo | `characterization.model.<q>.t1_time`, `.t2_time`, `.t2_echo_time` (s) |
| P(1\|0), P(0\|1) | `metrics.ssro.measure.constant.<q>.error_0_to_1`, `.error_1_to_0` |
| readout assignment error | `metrics.ssro.measure.constant.<q>.assignment_error`, which is the mean of the two above |
| 1Q error | 1 − `metrics.rb.prx.drag_crf_sx.<q>.fidelity:par=d2` |
| CZ error | 1 − `metrics.irb.cz.<impl>.<a>__<b>.fidelity:par=d2` (interleaved RB); Clifford-RB `metrics.rb.clifford.uz_cz…` is given for reference |
| readout leakage, seepage, QND-ness | `metrics.rrc.measure.constant.<q>.leakage`, `.seepage`, `.qndness` |
| durations | `gates.prx.drag_crf_sx.<q>.duration`, `gates.cz.<impl>.<a>__<b>.duration`, `gates.measure.constant.<q>.duration` |

**iqm-qubit-selector's parser (`hw.calibration_fidelities`) differs from these choices:**

- its `readout` is the **heralded** readout fidelity of the `measure_fidelity` gate (`metrics.ssro.measure_fidelity.constant.<q>.fidelity:herald=measure.constant`), not the `measure` a circuit uses. On Garnet QB1 that is a 0.80% error against 1.61%;
- its `t2` is the Ramsey T2;
- its `readout_qndness` comes back empty, although the server serves `metrics.rrc.…qndness`.

A noise model or decoder prior should take the readout from the table above, not from the parser.

## Results

### Device list

The token sees `emerald`, `emerald:mock`, `garnet`, `garnet:mock`, `sirius` and `sirius:mock`.

`hw.list_devices()` **fails**:

```
iqm.station_control.client.authentication.ClientConfigurationError: Quantum computer not selected. Available quantum computers are: emerald, emerald:mock, garnet, garnet:mock, sirius, sirius:mock
```

`IQMClient(SERVER_URL)` with no `quantum_computer` resolves the device in its constructor, and that raises once the token sees more than one device. The private getter itself works: `hw.backend("garnet").client._iqm_server_client.get_quantum_computers()` returns all six, with id, alias and display name.

**The fix (in `hw.py`, not done here):** construct the client with any listed device, or call the getter without resolving one.

Every other helper worked on both devices.

### Key calibration numbers

| | Garnet median | Garnet worst | Emerald median | Emerald worst |
|---|---|---|---|---|
| qubits / calibrated couplers | 20 / 30 | | 54 / 85 (of 90) | |
| T1 (µs) | 37.1 | 15.8 (QB18) | 46.3 | 6.5 (QB35) |
| T2 echo (µs) | 19.9 | 15.6 (QB18) | 36.7 | 5.8 (QB41) |
| T2 Ramsey (µs) | 9.08 | 5.19 (QB2) | 15.5 | 2.45 (QB41); QB3 has none |
| readout assignment error | 0.0244 | 0.0522 (QB11) | 0.0118 | 0.0939 (QB3) |
| P(1\|0) | 0.0123 | 0.0270 (QB13) | 0.0075 | 0.0376 (QB47) |
| P(0\|1) | 0.0329 | 0.0912 (QB2) | 0.0145 | 0.1506 (QB3) |
| 1Q error (RB) | 6.0e-4 | 3.7e-3 (QB12) | 3.2e-4 | 5.2e-3 (QB35) |
| CZ error (IRB) | 0.0050 | 0.0120 (QB16–QB20) | 0.0035 | 0.0538 (QB35–QB43) |
| CZ error (Clifford RB) | 0.0156 | 0.0208 (QB15–QB16) | 0.0103 | 0.0637 (QB35–QB43) |
| readout leakage | 0.0011 | 0.107 (QB6); QB2 0.080 | 0.0076 | 0.028 (QB27) |
| prx / CZ / measure (ns) | 16 / 40 / 480 | measure 556 (QB2) | 24 / 40 / 360 | CZ 60 (QB44–QB50) |

**Other architecture facts:**

- Readout is asymmetric on both devices: P(0|1) is about 2.7× P(1|0) on Garnet and 1.9× on Emerald.
- Garnet serves `measure_012` (readout of |2⟩) on 4 qubits, and Emerald on 1 (QB43). Both serve `prx_12` on every qubit.
- `cc_prx` (a classically controlled prx) and `reset_wait` are listed on **every qubit of both devices** in these calibration sets.
- Emerald's CZ uses two implementations: `slepian_crf` on 83 couplers, and `crf_acstarkcrf` on QB44–QB50 and QB50–QB51.
- Five of Emerald's 90 static couplers have no CZ in this calibration set: QB6–QB7, QB15–QB23, QB24–QB32, QB49–QB50 and QB50–QB54.

### d=3 placement with zero SWAPs

The code graph is `dqiqm.surface.geometry(3)`'s ancilla-to-data graph: 17 nodes and 24 edges. Its degrees are:

- data qubits: 2, 2, 2, 2, 3, 3, 3, 3, 4;
- ancillas: 2, 2, 2, 2, 4, 4, 4, 4.

Each placement is a networkx `GraphMatcher(device, code).subgraph_monomorphisms_iter()` match, taken in the calibrated CZ graph. Placements are grouped by (data qubits, CZs used).

**Score** (lower is better), for one round plus the final data readout:

    S = Σ_{24 CZs} −ln F_CZ  +  Σ_{17 qubits} [ −ln(1 − e_readout) + t_round / T1 ]
    t_round = 4 t_CZ + 10 t_prx + t_measure   (device medians; a CX is H·CZ·H)

The score depends on the qubit set and the CZs used, not on which ancillas are X-type and which Z-type.

**Garnet: the code embeds with zero SWAPs, in exactly one placement.**

- There are 4 monomorphisms, all on one qubit set with one set of 24 CZs. They are the code graph's 4 automorphisms, and two of them swap X and Z.
- The same holds on the full static graph, since Garnet's 30 couplers are all calibrated.
- So a "top 3" on Garnet is this one placement, in two X/Z orientations.

| | |
|---|---|
| score | 0.9953 = CZ 0.1502 + readout 0.4556 + T1 0.3895 (t_round 0.800 µs) |
| qubits | QB1, QB2, QB4–QB16, QB19, QB20 (left out: QB3, QB17, QB18) |
| data | QB2, QB4, QB6, QB8, QB10, QB12, QB14, QB16, QB19 |
| ancillas | QB1, QB9, QB11, QB20 / QB5, QB7, QB13, QB15: either four can be the X-type |
| code coordinate → qubit | (1,1) QB2, (3,1) QB4, (5,1) QB8, (1,3) QB6, (3,3) QB10, (5,3) QB14, (1,5) QB12, (3,5) QB16, (5,5) QB19; ancillas (2,0) QB1, (2,2) QB5, (4,2) QB9, (6,2) QB13, (0,4) QB7, (2,4) QB11, (4,4) QB15, (4,6) QB20 |

**Notes on the Garnet placement:**

- It leaves out QB18, the worst T1 and T2 echo.
- It must use QB16–QB20, the worst CZ (0.0120). It must make QB11, the worst readout (0.052), an ancilla, read every round.
- QB6 (readout leakage 0.107) and QB2 (0.080) are data qubits.

The mentors' repetition-code path (`QPURepetitionCode.ipynb`, 19 qubits) has all 18 links on calibrated couplers.

**Emerald, beyond the brief.** It is the same function, and the challenge's target.

- There are 60 monomorphisms, giving 15 placements on 15 qubit sets in the calibrated graph.
- On all 90 static couplers there would be 92 monomorphisms (23 placements). The five uncalibrated couplers cost 8 placements.

| # | score (CZ + readout + T1) | qubits | data |
|---|---|---|---|
| 1 | 0.6511 (0.1709 + 0.1984 + 0.2818) | QB4, QB5, QB8–QB11, QB16–QB20, QB25–QB28, QB33, QB34 | QB4, QB9, QB11, QB16, QB18, QB20, QB25, QB27, QB34 |
| 2 | 0.7099 (0.1489 + 0.2142 + 0.3468) | QB13, QB14, QB19–QB22, QB27–QB31, QB36–QB39, QB44, QB45 | QB13, QB20, QB22, QB27, QB29, QB31, QB36, QB38, QB45 |
| 3 | 0.7570 (0.1418 + 0.2472 + 0.3680) | QB10, QB11, QB16–QB19, QB24–QB28, QB33–QB36, QB41, QB42 | QB10, QB17, QB19, QB24, QB26, QB28, QB33, QB35, QB42 |

All three use QB19–QB27 (CZ error 0.0235). The ancillas of #1 are QB5, QB17, QB19, QB33 / QB8, QB10, QB26, QB28.

### Cost

**Nothing is exposed.** iqm-client 35.0.3 has no method for credits, prices, quotas or balance. The health, about and device-list responses have no such field: the script scans their keys for credit, cost, price, quota, balance, billing and budget.

Sal reads the dashboard.

### IQMFakeGarnet against the live Garnet

**Topology.**

- **Same:** the 20 qubit names and the 30 couplers.
- **Different chip:** `validate_compatible_architecture(live SQA)` is False, only because the chip's `dut_label` differs (fake `M153_W0_P06_Z99`, live `M194_F0W2114_P08_B15`).
- Note: in iqm-client 35.0.3, `IQMFakeGarnet` is importable from `iqm.qiskit_iqm.fake_backends.fake_garnet` but not from `iqm.qiskit_iqm.fake_backends`.

**Noise, live / fake, per qubit:**

| Figure | Median | Range |
|---|---|---|
| T1 | 0.98 | 0.41–2.45 |
| T2 Ramsey | 1.01 | 0.52–2.14 |
| T2 echo | 2.14 | |
| P(1\|0) | 0.51 | 0.15–1.06 |
| P(0\|1) | 1.32 | 0.50–3.80 |
| 1Q error, against the fake's p/2 | 1.07 | 0.31–9.7 |
| CZ IRB error, against the fake's 3p/4 | 1.30 | 0.45–3.54 |

The fake's T2 matches the Ramsey T2, not the echo T2.

The fake's readout is symmetric (about 2.5% each way), while the live readout is not. The fake's durations are prx 20 ns and CZ 40 ns; live they are 16 and 40.

**Verdict.** The fake stands in for structure: names, couplers, placement and transpilation. Its noise is right in the median but wrong per qubit, up to 3.8× on readout and 9.7× on a 1Q gate. Its readout asymmetry is also wrong, and that matters for no-reset detectors.

For offline dataset work, use the fake's topology with today's noise. `IQMFakeGarnet().copy_with_error_profile(<profile built from the snapshot>)` builds. The script maps each benchmarked infidelity r to a depolarizing parameter r·D/(D−1), which slightly overstates the noise, and clips T2 at 2·T1.

**Emerald.** `IQMFakeAphrodite`, the team's stand-in, has exactly Emerald's 54 qubit names and its 90 static couplers. Five of those couplers are uncalibrated today. Its median T1 is 48.9 µs, against 46.3 µs live.
