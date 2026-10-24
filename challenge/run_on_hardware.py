from surface_code import *
from stim_to_qiskit import stim_to_qiskit
from build_emerald_qubit_rotated import *
from extract_syndromes import *
from qiskit import transpile


TOKEN = ""
SERVER_URL = "https://resonance.iqm.tech"


# ─────────────────────────────────────────────────────────────────────────────
#  5.  HARDWARE EXECUTION ON IQM RESONANCE
#
#  Two equal tracks. Pick ONE. Both return the same syndromes dict
#  (at least "det_events" and "obs_flips"), so decode_hardware_results()
#  works for either.
#
#  NOTE: Emerald does not support mid-circuit reset -> use the no-reset circuit.
# ─────────────────────────────────────────────────────────────────────────────

def run_qiskit(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
) -> dict:
    """
    Build in qiskit. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qiskit circuit → transpile → run → counts
        → raw measurements → extract_syndromes()

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token (falls back to TOKEN)
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  output of extract_syndromes()
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables), ...
    """
    from iqm.qiskit_iqm import IQMProvider

    # Build circuits, exchange for make_qiskit_circuit() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    #CREATE A CIRCUIT TO BE EXECUTED ON HARDWARE
    qc = []
    #qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)
    #qc = make_qiskit_circuit(args)


    # OPTIONAL: IF YOU WANT TO ASSIGN QUBIT MAP BY HAND
    # Map abstract Qiskit qubits to emerald physical qubits
    qubit_map  = build_emerald_qubit_map(stim_circ)

    # HIGHLY RECOMMEND PRINTING THIS, if you are using it.
    # print(qubit_map)
    # initial_layout: Qiskit needs a list where position i = physical qubit for logical i
    n_qubits   = qc.num_qubits
    layout     = [qubit_map.get(i, i) for i in range(n_qubits)]

    # Connect to IQM Resonance
    provider   = IQMProvider(
        SERVER_URL,
        quantum_computer=quantum_computer,
        token=token or TOKEN
    )
    backend    = provider.get_backend()

    # Transpile: Qiskit transpilerwill insert SWAPs for diagonal CX pairs automatically
    # think about whether this is necessary
    qc_t       = transpile(
    # arguments
    )
    print(f"Transpiled circuit depth: {qc_t.depth()}  "
          f"(ideal: {qc.depth()}, extra depth from SWAPs)")

    # Run
    job        = backend.run(qc_t, shots=shots)
    counts     = job.result().get_counts()

    # Convert counts dict → (shots, num_measurements) numpy array
    raw_meas   = counts_to_measurement_array(counts, stim_circ.num_measurements,
                                               shots)
    # m2d converter must come from the circuit that actually ran
    return extract_syndromes(raw_meas, stim_circ)



def run_qrisp(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
) -> dict:
    """
    Build in qrisp. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qrisp DetectorExperiment → IQMBackend → run
        → detection events + observable flips

    The DetectorExperiment stores the info the detectors and observables itself, so
    counts_to_measurement_array() and extract_syndromes() are not needed here.

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token 
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  same keys as extract_syndromes() from qiskit function
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables)
    """
    import os
    from iqm.qrisp_iqm import IQMBackend
    from stim_to_qrisp import stim_to_detector_experiment

    os.environ["IQM_TOKEN"] = token or TOKEN

    # Build circuits, exchange for make_qrisp_experiment() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    # CREATE AN EXPERIMENT TO BE EXECUTED ON HARDWARE
    experiment = []
    #experiment = stim_to_detector_experiment(stim_circ, no_reset=no_reset)
    #experiment = make_qrisp_experiment(distance, rounds, no_reset=no_reset)

    # Connect to IQM Resonance
    # Layout: think about the pass_manager. iqm.qrisp_iqm.vf2pp_layout finds an
    # exact embedding on the coupling map (zero SWAPs) or fails.
    backend    = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
    # further arguments
    )

    # Run the experiment on the backend and collect detection events + observable flips
    # TO BE DONE
    det_events, obs_flips = [], []

    return {"det_events": det_events, "obs_flips": obs_flips}



def decode_hardware_results(
    syndromes: dict
) -> tuple[float, float]:
    """
    Decodes the output of run_qiskit() or run_qrisp() via your desired decoder.

    NOTE: Other parameters can be important for this function, such as a noise model.
    If you decode with a detector error model (DEM): the DEM’s detectors must match the circuit that ran.


    Parameters
    ----------
    syndromes           : dict  output of run_qiskit() / run_qrisp()

    Returns
    -------
    ler : float   estimated logical error probability for this round count (fit over several round counts to get the logical error rate, see README)
    err : float   OPTIONAL 1-sigma statistical error
    """

    # USE e.g. PyMatching to decode

    ler,err = [], []
    return ler, err