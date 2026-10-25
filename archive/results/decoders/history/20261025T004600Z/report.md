# Garnet bounded improvement confirmation

New shots: 96,000; returned: 96,000.
Primary challenge A=1 fit: R=1,2,3. R=1..7 is diagnostic only.

| Basis | Physical profile | Decoder | Window | LER | 95% profile CI | Deviance |
|---|---|---|---|---:|---|---:|
| X | adaptive | gru_0.5 | headline | 0.063203 | [0.059890049902646215, 0.0666420409794941] | 90.03 |
| X | adaptive | gru_0.5 | diagnostic | 0.068612 | [0.06579463117464002, 0.07152452313268172] | 119.96 |
| X | adaptive | mwpm | headline | 0.062387 | [0.0590971074548516, 0.06580200104541266] | 75.13 |
| X | adaptive | mwpm | diagnostic | 0.066348 | [0.06358902020272658, 0.06920076815597694] | 90.64 |
| X | adaptive | mwpm_fit | headline | 0.060656 | [0.05741972930114661, 0.06401767704305807] | 76.12 |
| X | adaptive | mwpm_fit | diagnostic | 0.065529 | [0.06279625878948923, 0.06835511314667796] | 99.65 |
| X | baseline | gru_0.5 | headline | 0.057919 | [0.054766421427593755, 0.06119498089397581] | 71.87 |
| X | baseline | gru_0.5 | diagnostic | 0.062143 | [0.05950677994280004, 0.06487020441664301] | 92.92 |
| X | baseline | mwpm | headline | 0.058078 | [0.054920348154939895, 0.06136044103692753] | 52.10 |
| X | baseline | mwpm | diagnostic | 0.062357 | [0.059714970832208634, 0.0650907521682934] | 75.10 |
| X | baseline | mwpm_fit | headline | 0.057192 | [0.05406195894676143, 0.060445483569522715] | 68.35 |
| X | baseline | mwpm_fit | diagnostic | 0.061751 | [0.059126778547616235, 0.06446547522380427] | 93.85 |
| X | baseline | released | headline | 0.058078 | [0.054920348154939895, 0.06136044103692753] | 52.10 |
| Z | baseline | gru_0.5 | headline | 0.047049 | [0.045053917990172616, 0.04910442820520393] | 240.02 |
| Z | baseline | gru_0.5 | diagnostic | 0.056370 | [0.05462052789065297, 0.05816308255371946] | 429.39 |
| Z | baseline | mwpm | headline | 0.061369 | [0.05905195755889455, 0.06374799830771373] | 366.59 |
| Z | baseline | mwpm | diagnostic | 0.067674 | [0.06569217541302548, 0.06970424512406484] | 445.65 |
| Z | baseline | mwpm_fit | headline | 0.059052 | [0.05678604081646901, 0.061380097645515504] | 375.51 |
| Z | baseline | mwpm_fit | diagnostic | 0.066136 | [0.06418676476419496, 0.06813285624408227] | 475.58 |
| Z | baseline | released | headline | 0.042547 | [0.04065867001551191, 0.044493777910905204] | 198.86 |

Promotion uses 6 predeclared Bonferroni-adjusted contrasts; a combination must beat both components.

{
  "comparisons": {
    "X": {
      "reference__candidate": {
        "first": "reference",
        "second": "candidate",
        "absolute_ler_reduction": -0.0025778278512120223,
        "ci": [
          -0.008363772935039027,
          0.003533699671497626
        ],
        "confidence": 0.9916666666666667,
        "paired": false,
        "draws": 12000,
        "seed": 7321,
        "method": "round/block-stratified failure bootstrap; refit challenge A=1 equation",
        "excludes": "calibration drift outside the two acquisition blocks"
      },
      "reference__physical": {
        "first": "reference",
        "second": "physical",
        "absolute_ler_reduction": -0.004308281931774349,
        "ci": [
          -0.010706753373550565,
          0.0015618368845074246
        ],
        "confidence": 0.9916666666666667,
        "paired": false,
        "draws": 12000,
        "seed": 7321,
        "method": "round/block-stratified failure bootstrap; refit challenge A=1 equation",
        "excludes": "calibration drift outside the two acquisition blocks"
      },
      "reference__history": {
        "first": "reference",
        "second": "history",
        "absolute_ler_reduction": 0.0008864528740822053,
        "ci": [
          -0.00014846688676109187,
          0.001976100014282915
        ],
        "confidence": 0.9916666666666667,
        "paired": true,
        "draws": 12000,
        "seed": 7321,
        "method": "round/block-stratified failure bootstrap; refit challenge A=1 equation",
        "excludes": "calibration drift outside the two acquisition blocks"
      },
      "candidate__physical": {
        "first": "candidate",
        "second": "physical",
        "absolute_ler_reduction": -0.001730454080562327,
        "ci": [
          -0.0038829018046213923,
          0.0004488958594684746
        ],
        "confidence": 0.9916666666666667,
        "paired": true,
        "draws": 12000,
        "seed": 7321,
        "method": "round/block-stratified failure bootstrap; refit challenge A=1 equation",
        "excludes": "calibration drift outside the two acquisition blocks"
      },
      "candidate__history": {
        "first": "candidate",
        "second": "history",
        "absolute_ler_reduction": 0.0034642807252942276,
        "ci": [
          -0.0024969617299852825,
          0.009556030287770943
        ],
        "confidence": 0.9916666666666667,
        "paired": false,
        "draws": 12000,
        "seed": 7321,
        "method": "round/block-stratified failure bootstrap; refit challenge A=1 equation",
        "excludes": "calibration drift outside the two acquisition blocks"
      },
      "physical__history": {
        "first": "physical",
        "second": "history",
        "absolute_ler_reduction": 0.0051947348058565546,
        "ci": [
          -0.0010405302664590315,
          0.01110274931520496
        ],
        "confidence": 0.9916666666666667,
        "paired": false,
        "draws": 12000,
        "seed": 7321,
        "method": "round/block-stratified failure bootstrap; refit challenge A=1 equation",
        "excludes": "calibration drift outside the two acquisition blocks"
      }
    },
    "Z": {}
  },
  "adopted": {
    "X": {
      "profile": "baseline",
      "family": "released",
      "promoted": false,
      "arm": "reference",
      "rule": "Combined must beat both components and released; simpler components need adjusted gain over released"
    },
    "Z": {
      "profile": "baseline",
      "family": "released",
      "promoted": false,
      "arm": "reference",
      "rule": "Combined must beat both components and released; simpler components need adjusted gain over released"
    }
  }
}

One calibration and two successive acquisition windows, not separate days/calibrations
Challenge stationary equation is descriptive whenever lack-of-fit rejects it
Native Pauli priors omit explicit inserted-DD pulse errors and idle decoherence; fitting/learning is empirical model-mismatch compensation
QND implementation unavailable; no leakage herald or qutrit oracle input
Old confirmation data was development only, fresh confirmation remained frozen
