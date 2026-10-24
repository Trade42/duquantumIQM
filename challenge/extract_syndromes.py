import numpy as np
import stim


def extract_syndromes(
    raw_meas: np.ndarray,
    stim_circuit_noisy: stim.Circuit,
    print_summary: bool = True,
) -> dict:
    """
    ONLY NEEDED FOR QISKIT. In qrisp, the DetectorExperiment returns
    detection events and observable flips directly.

    Converts raw hardware measurements into detection events and observable
    flips, after a hardware or simulation run, so after run_qiskit(). This will be executed before any decoder (PyMatching or NVIDIA
    Ising Predecoder), see decode_hardware_results().

    Parameters
    ----------
    raw_meas            : np.ndarray  shape (shots, num_measurements), bool
                          Output of counts_to_measurement_array() in run_qiskit().
    stim_circuit_noisy  : stim.Circuit  the circuit that executed on hardware
                          (no-reset on Emerald), with correct according DETECTOR definitions.
                          This is used to define the detector structure for m2d.
                          Noise annotations are ignored.
    print_summary       : bool  print a human-readable syndrome summary.

    Returns
    -------
    dict with keys:
        "det_events"   : np.ndarray (shots, num_detectors), bool
                         Detection events — the syndrome the decoder consumes.
                         True = stabilizer fired (changed relative to reference).
        "obs_flips"    : np.ndarray (shots, num_observables), bool
                         Logical observable outcomes from the final data readout.
                         True = logical flip observed.
        "raw_meas"     : np.ndarray (shots, num_measurements), bool
                         The original raw measurements, passed through unchanged.
        "num_detectors": int
        "num_shots"    : int
        "syndrome_weight_per_shot" : np.ndarray (shots,), int
                         Number of detectors that fired per shot.
                         Low mean = low error rate; mean near num_detectors/2
                         indicates the decoder is seeing noise at or above threshold.
        "detector_firing_rate" : np.ndarray (num_detectors,), float
                         Per-detector fraction of shots in which it fired.
                         Useful for spotting hot qubits or miscalibrated ancillas.
    """
    converter  = stim_circuit_noisy.compile_m2d_converter()
    det_events, obs_flips = converter.convert(
        measurements=raw_meas.astype(bool),
        separate_observables=True,
    )

    shots         = len(raw_meas)
    n_det         = det_events.shape[1]
    weights       = det_events.sum(axis=1).astype(int)   # fired detectors per shot
    firing_rates  = det_events.mean(axis=0)               # per-detector firing rate

    if print_summary:
        print("─" * 56)
        print("  Syndrome extraction summary")
        print("─" * 56)
        print(f"  Shots              : {shots}")
        print(f"  Num detectors      : {n_det}")
        print(f"  Num observables    : {obs_flips.shape[1]}")
        print(f"  Mean syndrome wt   : {weights.mean():.3f}  "
              f"(max possible {n_det})")
        print(f"  Shots w/ 0 errors  : {int((weights == 0).sum())}  "
              f"({100*(weights==0).mean():.1f}%)")
        print(f"  Shots w/ ≥1 error  : {int((weights > 0).sum())}  "
              f"({100*(weights>0).mean():.1f}%)")
        print(f"  Logical flip rate  : "
              f"{obs_flips.mean(axis=0)} (per observable)")
        print()
        print("  Per-detector firing rates (flag anything > 0.2 as hot):")
        for i, r in enumerate(firing_rates):
            flag = "  ← HOT" if r > 0.2 else ""
            print(f"    detector {i:>3}: {r:.4f}{flag}")
        print("─" * 56)

    return {
        "det_events"              : det_events,
        "obs_flips"               : obs_flips,
        "raw_meas"                : raw_meas,
        "num_detectors"           : n_det,
        "num_shots"               : shots,
        "syndrome_weight_per_shot": weights,
        "detector_firing_rate"    : firing_rates,
    }