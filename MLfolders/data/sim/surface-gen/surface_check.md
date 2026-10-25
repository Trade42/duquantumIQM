# surface-gen: physics checks of dqiqm.surface.memory_circuit

stim 1.16.0, pymatching 2.4.0, numpy 2.4.6; seed base 2026; 1,000,000 shots per LER row; noise p = 0.003 on the four challenge knobs unless stated.

## 1. Structure, noiseless runs, DEM and distance

Noiseless: 1,000 shots per row; 'events' counts detection events (Stim, against its reference sample); 'raw≠0' counts detector parities computed from the raw measurements that are not 0; 'obs≠0' counts shots whose observable is not 0 (raw or as a flip). Distance: len(shortest_graphlike_error()) of the DEM (decompose_errors=True) under the challenge's noise. 'flip=MR': without reset, an X_ERROR(p) after every ancilla M gives exactly the with-reset DEM (approx_equals, atol 1e-12).

| d | R  | basis | reset | contract | dets | R(d²-1) | meas | R(d²-1)+d² | obs | events | raw≠0 | obs≠0 | distance | flip=MR | pass |
|---|----|-------|-------|----------|------|---------|------|------------|-----|--------|-------|-------|----------|---------|------|
| 3 | 1  | Z     | no    | yes      | 8    | 8       | 17   | 17         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 1  | Z     | MR    | yes      | 8    | 8       | 17   | 17         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 1  | X     | no    | yes      | 8    | 8       | 17   | 17         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 1  | X     | MR    | yes      | 8    | 8       | 17   | 17         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 2  | Z     | no    | yes      | 16   | 16      | 25   | 25         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 2  | Z     | MR    | yes      | 16   | 16      | 25   | 25         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 2  | X     | no    | yes      | 16   | 16      | 25   | 25         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 2  | X     | MR    | yes      | 16   | 16      | 25   | 25         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 3  | Z     | no    | yes      | 24   | 24      | 33   | 33         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 3  | Z     | MR    | yes      | 24   | 24      | 33   | 33         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 3  | X     | no    | yes      | 24   | 24      | 33   | 33         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 3  | X     | MR    | yes      | 24   | 24      | 33   | 33         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 5  | Z     | no    | yes      | 40   | 40      | 49   | 49         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 5  | Z     | MR    | yes      | 40   | 40      | 49   | 49         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 5  | X     | no    | yes      | 40   | 40      | 49   | 49         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 5  | X     | MR    | yes      | 40   | 40      | 49   | 49         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 10 | Z     | no    | yes      | 80   | 80      | 89   | 89         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 10 | Z     | MR    | yes      | 80   | 80      | 89   | 89         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 3 | 10 | X     | no    | yes      | 80   | 80      | 89   | 89         | 1   | 0      | 0     | 0     | 3        | yes     | yes  |
| 3 | 10 | X     | MR    | yes      | 80   | 80      | 89   | 89         | 1   | 0      | 0     | 0     | 3        |         | yes  |
| 5 | 1  | Z     | no    | yes      | 24   | 24      | 49   | 49         | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 1  | Z     | MR    | yes      | 24   | 24      | 49   | 49         | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 1  | X     | no    | yes      | 24   | 24      | 49   | 49         | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 1  | X     | MR    | yes      | 24   | 24      | 49   | 49         | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 2  | Z     | no    | yes      | 48   | 48      | 73   | 73         | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 2  | Z     | MR    | yes      | 48   | 48      | 73   | 73         | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 2  | X     | no    | yes      | 48   | 48      | 73   | 73         | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 2  | X     | MR    | yes      | 48   | 48      | 73   | 73         | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 3  | Z     | no    | yes      | 72   | 72      | 97   | 97         | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 3  | Z     | MR    | yes      | 72   | 72      | 97   | 97         | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 3  | X     | no    | yes      | 72   | 72      | 97   | 97         | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 3  | X     | MR    | yes      | 72   | 72      | 97   | 97         | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 5  | Z     | no    | yes      | 120  | 120     | 145  | 145        | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 5  | Z     | MR    | yes      | 120  | 120     | 145  | 145        | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 5  | X     | no    | yes      | 120  | 120     | 145  | 145        | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 5  | X     | MR    | yes      | 120  | 120     | 145  | 145        | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 10 | Z     | no    | yes      | 240  | 240     | 265  | 265        | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 10 | Z     | MR    | yes      | 240  | 240     | 265  | 265        | 1   | 0      | 0     | 0     | 5        |         | yes  |
| 5 | 10 | X     | no    | yes      | 240  | 240     | 265  | 265        | 1   | 0      | 0     | 0     | 5        | yes     | yes  |
| 5 | 10 | X     | MR    | yes      | 240  | 240     | 265  | 265        | 1   | 0      | 0     | 0     | 5        |         | yes  |

## 2. Against stim.Circuit.generated (with reset, challenge noise, d = 3)

P_L with its 95% Wilson interval, 1,000,000 shots each, decoded by PyMatching from each circuit's own DEM. 'same samples': with one seed, ours and Stim's give bit-identical detection events and observable flips (SHA-256 of all shots). With independent seeds: z is the two-proportion z-score, and '|z|<1.96' is the brief's criterion, the one 'pass' uses; 'overlap': the two Wilson intervals intersect; 'within (old)': each P_L lies inside the other's interval, the first brief's criterion, shown for the record. For two independent estimates of one rate, it holds only when |z| < 1.96/sqrt(2) = 1.39, 83% of the time per row.

| d | R | basis | dets ours/Stim | distance ours/Stim | same circuit | by                | same samples | P_L ours                               | P_L Stim                               | z     | |z|<1.96 | overlap | within (old) | pass |
|---|---|-------|----------------|--------------------|--------------|-------------------|--------------|----------------------------------------|----------------------------------------|-------|----------|---------|--------------|------|
| 3 | 3 | Z     | 24 / 24        | 3 / 3              | yes          | flattened text    | yes          | 0.00656 [0.00641, 0.00672] (seed 3026) | 0.00655 [0.00639, 0.00671] (seed 3027) | 0.14  | yes      | yes     | yes          | yes  |
| 3 | 3 | X     | 24 / 24        | 3 / 3              | yes          | DEM approx_equals | yes          | 0.00719 [0.00703, 0.00736] (seed 3028) | 0.00727 [0.00711, 0.00744] (seed 3029) | -0.69 | yes      | yes     | yes          | yes  |
| 3 | 5 | Z     | 40 / 40        | 3 / 3              | yes          | flattened text    | yes          | 0.01031 [0.01012, 0.01051] (seed 3030) | 0.01052 [0.01032, 0.01072] (seed 3031) | -1.43 | yes      | yes     | NO           | yes  |
| 3 | 5 | X     | 40 / 40        | 3 / 3              | yes          | DEM approx_equals | yes          | 0.01187 [0.01166, 0.01208] (seed 3032) | 0.01197 [0.01175, 0.01218] (seed 3033) | -0.63 | yes      | yes     | yes          | yes  |

## 3. The no-reset trade-off

Our circuits, challenge noise p = 0.003, 1,000,000 shots each, independent seeds; P_L and eps_L per round with 95% Wilson intervals. 'idle' is idle_during_measurement: DEPOLARIZE1 on the data during each ancilla readout (both variants) and X_ERROR on the live ancillas after M (no reset only). 'record' is record_flip_probability: M(p) on every readout (an X_ERROR before MR with reset). 'same DEM': the two variants' DEMs are approx_equal. 'lower P_L' names the variant whose interval lies wholly below the other's.

| d | R | basis | idle  | record | P_L no reset               | P_L reset (MR)             | eps_L/round no reset       | eps_L/round reset          | reset/no-reset | same DEM | lower P_L               |
|---|---|-------|-------|--------|----------------------------|----------------------------|----------------------------|----------------------------|----------------|----------|-------------------------|
| 3 | 3 | Z     | 0.0   | 0.0    | 0.00629 [0.00613, 0.00644] | 0.00650 [0.00635, 0.00666] | 0.00210 [0.00205, 0.00216] | 0.00218 [0.00212, 0.00223] | 1.034          | NO       | tie (intervals overlap) |
| 3 | 3 | Z     | 0.003 | 0.0    | 0.00958 [0.00939, 0.00977] | 0.00957 [0.00938, 0.00976] | 0.00321 [0.00315, 0.00328] | 0.00321 [0.00315, 0.00328] | 0.999          | yes      | tie (intervals overlap) |
| 3 | 3 | X     | 0.0   | 0.0    | 0.00714 [0.00698, 0.00731] | 0.00741 [0.00725, 0.00758] | 0.00239 [0.00234, 0.00245] | 0.00248 [0.00243, 0.00254] | 1.038          | NO       | tie (intervals overlap) |
| 3 | 3 | X     | 0.003 | 0.0    | 0.01072 [0.01052, 0.01092] | 0.01079 [0.01059, 0.01099] | 0.00360 [0.00353, 0.00367] | 0.00362 [0.00356, 0.00369] | 1.007          | yes      | tie (intervals overlap) |
| 3 | 5 | Z     | 0.0   | 0.0    | 0.01002 [0.00983, 0.01022] | 0.01049 [0.01029, 0.01069] | 0.00202 [0.00198, 0.00206] | 0.00212 [0.00208, 0.00216] | 1.047          | NO       | no reset                |
| 3 | 5 | Z     | 0.003 | 0.0    | 0.01579 [0.01555, 0.01604] | 0.01564 [0.01540, 0.01588] | 0.00320 [0.00315, 0.00325] | 0.00317 [0.00312, 0.00322] | 0.990          | yes      | tie (intervals overlap) |
| 3 | 5 | X     | 0.0   | 0.0    | 0.01120 [0.01099, 0.01140] | 0.01176 [0.01155, 0.01198] | 0.00226 [0.00222, 0.00230] | 0.00238 [0.00233, 0.00242] | 1.051          | NO       | no reset                |
| 3 | 5 | X     | 0.003 | 0.0    | 0.01792 [0.01766, 0.01818] | 0.01809 [0.01783, 0.01835] | 0.00364 [0.00358, 0.00369] | 0.00367 [0.00362, 0.00372] | 1.009          | yes      | tie (intervals overlap) |
| 5 | 3 | Z     | 0.0   | 0.0    | 0.00176 [0.00168, 0.00184] | 0.00194 [0.00186, 0.00203] | 0.00059 [0.00056, 0.00061] | 0.00065 [0.00062, 0.00068] | 1.106          | NO       | no reset                |
| 5 | 3 | Z     | 0.003 | 0.0    | 0.00317 [0.00306, 0.00328] | 0.00308 [0.00298, 0.00319] | 0.00106 [0.00102, 0.00110] | 0.00103 [0.00099, 0.00107] | 0.972          | yes      | tie (intervals overlap) |
| 5 | 3 | X     | 0.0   | 0.0    | 0.00205 [0.00197, 0.00214] | 0.00211 [0.00202, 0.00220] | 0.00068 [0.00066, 0.00072] | 0.00071 [0.00068, 0.00074] | 1.030          | NO       | tie (intervals overlap) |
| 5 | 3 | X     | 0.003 | 0.0    | 0.00347 [0.00336, 0.00359] | 0.00341 [0.00330, 0.00353] | 0.00116 [0.00112, 0.00120] | 0.00114 [0.00110, 0.00118] | 0.984          | yes      | tie (intervals overlap) |
| 5 | 5 | Z     | 0.0   | 0.0    | 0.00301 [0.00290, 0.00312] | 0.00322 [0.00311, 0.00334] | 0.00060 [0.00058, 0.00062] | 0.00065 [0.00062, 0.00067] | 1.073          | NO       | tie (intervals overlap) |
| 5 | 5 | Z     | 0.003 | 0.0    | 0.00567 [0.00552, 0.00582] | 0.00555 [0.00541, 0.00570] | 0.00114 [0.00111, 0.00117] | 0.00112 [0.00109, 0.00115] | 0.979          | yes      | tie (intervals overlap) |
| 5 | 5 | X     | 0.0   | 0.0    | 0.00348 [0.00336, 0.00359] | 0.00371 [0.00359, 0.00383] | 0.00070 [0.00067, 0.00072] | 0.00074 [0.00072, 0.00077] | 1.067          | NO       | tie (intervals overlap) |
| 5 | 5 | X     | 0.003 | 0.0    | 0.00642 [0.00627, 0.00658] | 0.00645 [0.00629, 0.00661] | 0.00129 [0.00126, 0.00132] | 0.00130 [0.00127, 0.00133] | 1.004          | yes      | tie (intervals overlap) |

Idle sweep (d = 3, R = 5, Z basis; the reset-flip rate stays 0.003):

| d | R | basis | idle   | record | P_L no reset               | P_L reset (MR)             | eps_L/round no reset       | eps_L/round reset          | reset/no-reset | same DEM | lower P_L               |
|---|---|-------|--------|--------|----------------------------|----------------------------|----------------------------|----------------------------|----------------|----------|-------------------------|
| 3 | 5 | Z     | 0.0    | 0.0    | 0.01002 [0.00983, 0.01022] | 0.01049 [0.01029, 0.01069] | 0.00202 [0.00198, 0.00206] | 0.00212 [0.00208, 0.00216] | 1.047          | NO       | no reset                |
| 3 | 5 | Z     | 0.0015 | 0.0    | 0.01240 [0.01218, 0.01262] | 0.01305 [0.01283, 0.01327] | 0.00250 [0.00246, 0.00255] | 0.00264 [0.00259, 0.00268] | 1.052          | NO       | no reset                |
| 3 | 5 | Z     | 0.003  | 0.0    | 0.01579 [0.01555, 0.01604] | 0.01564 [0.01540, 0.01588] | 0.00320 [0.00315, 0.00325] | 0.00317 [0.00312, 0.00322] | 0.990          | yes      | tie (intervals overlap) |
| 3 | 5 | Z     | 0.006  | 0.0    | 0.02268 [0.02239, 0.02298] | 0.02171 [0.02143, 0.02200] | 0.00462 [0.00456, 0.00468] | 0.00442 [0.00436, 0.00448] | 0.957          | NO       | reset                   |
| 3 | 5 | Z     | 0.012  | 0.0    | 0.04013 [0.03975, 0.04051] | 0.03541 [0.03505, 0.03577] | 0.00830 [0.00821, 0.00838] | 0.00729 [0.00721, 0.00737] | 0.882          | NO       | reset                   |

Record-only readout flips, M(0.01), on top of the challenge noise (R = 5, Z basis, no idle). Without reset a flipped bit fires D_k and D_(k+2); with MR it is a state flip before the reset and fires D_k and D_(k+1). Compare with the idle = 0 rows above:

| d | R | basis | idle | record | P_L no reset               | P_L reset (MR)             | eps_L/round no reset       | eps_L/round reset          | reset/no-reset | same DEM | lower P_L |
|---|---|-------|------|--------|----------------------------|----------------------------|----------------------------|----------------------------|----------------|----------|-----------|
| 3 | 5 | Z     | 0.0  | 0.01   | 0.02132 [0.02103, 0.02160] | 0.01796 [0.01771, 0.01823] | 0.00434 [0.00428, 0.00440] | 0.00365 [0.00359, 0.00370] | 0.843          | NO       | reset     |
| 5 | 5 | Z     | 0.0  | 0.01   | 0.00841 [0.00823, 0.00859] | 0.00673 [0.00658, 0.00690] | 0.00169 [0.00166, 0.00173] | 0.00135 [0.00132, 0.00139] | 0.801          | NO       | reset     |

Hard checks: PASS (40/40 structure rows, 4/4 Stim rows). Wall time 37 s, 16 CPUs visible, 4 workers.
