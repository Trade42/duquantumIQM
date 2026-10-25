# Vanilla surface-code baseline on IQM Emerald

Distance 3, nine data qubits, eight reused ancillas, rounds 1–3 in X/Z memory.
Literal H-CZ-H compilation, optimization level 0, fixed native graph embedding, DD off.
Uniform p=0.003 circuit prior and complete logical-class lookup; all shots retained.
Calibration: `23e12f6b-c23c-4736-8736-b0142989c0e6`. Charged shots: 18,000 / 18,000.

| Basis | Rounds | Failures / shots | Failure probability, 95% Wilson interval |
|---|---:|---:|---:|
| X | 1 | 341 / 3000 | 11.367% (10.280–12.552%) |
| X | 2 | 681 / 3000 | 22.700% (21.237–24.233%) |
| X | 3 | 859 / 3000 | 28.633% (27.044–30.277%) |
| Z | 1 | 46 / 3000 | 1.533% (1.152–2.039%) |
| Z | 2 | 301 / 3000 | 10.033% (9.009–11.160%) |
| Z | 3 | 456 / 3000 | 15.200% (13.960–16.529%) |

Challenge fit: `P_L(r) = [1 - (1 - 2 epsilon_L)^r]/2`, amplitude fixed to one.
Profile intervals describe the fitted binomial equation. Deviance records round-dependent model mismatch.
- X: **12.352% LER** (11.773–12.953%), deviance 4.67 / 2 df.
- Z: **4.749% LER** (4.424–5.089%), deviance 108.62 / 2 df.

The prior models logical H/CZ circuit faults. Native R synthesis, timing, spatial noise variation, leakage and readout disturbance contribute model mismatch.
Saved records preserve original shot order and source/native frames. Decoder states and requests were fixed before acquisition.
