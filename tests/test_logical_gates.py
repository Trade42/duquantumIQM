from itertools import product
import numpy as np
import pytest
import stim

from challenge.logical_gates import (make_logical_algorithm, plan_pauli_h,
    normalize_gate_set, transversal_cnot, logical_supports)
from challenge.surface_code import make_stim_circuit
from challenge.noise_models import NoiseModel


def test_all_small_logical_tableaus_match_compiler():
    for length in range(5):
        for ops in product(("X","Y","Z","H"),repeat=length):
            plan = plan_pauli_h(ops)
            original, compiled = stim.Circuit(), stim.Circuit()
            for gate in ops:
                original.append(gate,[0])
            if plan.hadamard:
                compiled.append("H",[0])
            if plan.pauli_z:
                compiled.append("Z",[0])
            if plan.pauli_x:
                compiled.append("X",[0])
            # Pad identity circuits to a one-qubit tableau.
            original.append("I",[0]); compiled.append("I",[0])
            assert original.to_tableau() == compiled.to_tableau()


@pytest.mark.parametrize("ops", [(),("X",),("Z",),("Y",),("H",),
    ("H","X"),("X","H"),("H","Y"),("H","X","H","Z","H"),
    ("Y","H","Z","X","H","Y","H")])
@pytest.mark.parametrize("basis", ["X","Z"])
@pytest.mark.parametrize("optimize", [False,True])
@pytest.mark.parametrize("gate_basis", ["cx","cz"])
def test_logical_algorithm_noiseless_semantics(ops,basis,optimize,gate_basis):
    algorithm = make_logical_algorithm(3,3,ops,basis=basis,optimize=optimize,gate_basis=gate_basis)
    circuit = algorithm.circuit
    circuit.detector_error_model()
    raw = circuit.compile_sampler(seed=11).sample(64)
    dets,obs = circuit.compile_m2d_converter().convert(measurements=raw,separate_observables=True)
    assert not dets.any() and not obs.any()
    assert np.all(algorithm.corrected_logical_bits(raw) == algorithm.expected_logical_bit)
    assert circuit.num_measurements == 33 and circuit.num_detectors == 24


@pytest.mark.parametrize("distance", [3,5,7])
@pytest.mark.parametrize("basis", ["X","Z"])
@pytest.mark.parametrize("no_reset", [False,True])
@pytest.mark.parametrize("optimize", [False,True])
def test_algorithm_circuit_distance(distance,basis,no_reset,optimize):
    algorithm = make_logical_algorithm(distance,distance,["H","X","H","Y","H"],
        basis=basis,no_reset=no_reset,optimize=optimize,noise=NoiseModel.uniform(.001))
    assert len(algorithm.circuit.shortest_graphlike_error()) == distance


def test_gate_sets_and_nontransversal_requests_rejected():
    with pytest.raises(ValueError,match="not in"):
        make_stim_circuit(3,3,transversal_gate_set="pauli",logical_operations=["H"])
    for gate in ("S","T"):
        with pytest.raises(ValueError):
            normalize_gate_set({"H",gate})
    with pytest.raises(ValueError,match="two identical"):
        make_stim_circuit(3,3,transversal_gate_set="css",logical_operations=["CX"])
    with pytest.raises(ValueError):
        make_stim_circuit(3,1,logical_operations=["H"])
    make_stim_circuit(3,3,transversal_gate_set={"X","H"},logical_operations=["X","H"])


def test_cnot_primitive_and_connectivity():
    c,t = list(range(9)),list(range(9,18))
    pairs = list(zip(c,t))
    circ = transversal_cnot(c,t,available_pairs=pairs)
    assert len(circ[0].targets_copy()) == 18
    tableau = circ.to_tableau()
    x = stim.PauliString("X"*9+"I"*9)
    z = stim.PauliString("I"*9+"Z"*9)
    assert tableau(x) == stim.PauliString("X"*18)
    assert tableau(z) == stim.PauliString("Z"*18)
    with pytest.raises(ValueError,match="coupling"):
        transversal_cnot(c,t,available_pairs=pairs[:-1])
    with pytest.raises(ValueError,match="disjoint"):
        transversal_cnot(c,c,available_pairs=pairs)


def test_clifford_block_has_no_hidden_repeated_hadamards():
    ops = ["H","X","H","Z"]*8+["H"]
    before = make_logical_algorithm(3,5,ops,optimize=False).circuit
    after = make_logical_algorithm(3,5,ops,optimize=True).circuit
    count = lambda c: sum(len(i.targets_copy()) for i in c if i.name in {"H","X","Y","Z"})
    assert count(before)-count(after) == 16*9+16*3
    assert sum(len(i.targets_copy())//2 for i in before if i.name == "CZ") == 24*5
    assert sum(len(i.targets_copy())//2 for i in after if i.name == "CZ") == 24*5


@pytest.mark.parametrize("basis",["X","Z"])
@pytest.mark.parametrize("optimize",[False,True])
def test_scheduled_gates_preserve_rounds_and_logical_sign(basis,optimize):
    schedule = {1:["H","X"],2:["Y"],3:["H","Z"],4:["H","Y"]}
    algorithm = make_logical_algorithm(3,5,schedule,basis=basis,optimize=optimize)
    raw = algorithm.circuit.compile_sampler(seed=3).sample(100)
    assert np.all(algorithm.corrected_logical_bits(raw) == algorithm.expected_logical_bit)
    assert algorithm.circuit.num_measurements == 49
    algorithm.circuit.detector_error_model()


def test_benchmark_noop_control_and_resource_counts():
    from challenge.benchmark_logical import workloads,operation_counts,logical_cost,compare_results
    spec = workloads()["clifford_block"]
    base = make_logical_algorithm(3,spec["rounds"],spec["operations"],optimize=False)
    opt = make_logical_algorithm(3,spec["rounds"],spec["operations"],optimize=True)
    cost = logical_cost(opt,3)
    assert cost["physical_algorithm_one_qubit_before"] == 201
    assert cost["physical_algorithm_one_qubit_after"] == 9
    assert operation_counts(base.circuit)["total_clifford"]-operation_counts(opt.circuit)["total_clifford"] == 192
    before = make_logical_algorithm(3,5,["H"],optimize=False)
    after = make_logical_algorithm(3,5,["H"],optimize=True)
    assert before.circuit == after.circuit
    result = {"probability":.01,"errors":100,"shots":10000}
    assert compare_results(result,result,identical=True)["relative_failure_reduction"] == 0


@pytest.mark.parametrize("basis",["X","Z"])
@pytest.mark.parametrize("optimize",[False,True])
def test_qiskit_logical_gates_end_to_end(basis,optimize):
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator
    from challenge.stim_to_qiskit import stim_to_qiskit
    from challenge.internal_helpers import counts_to_measurement_array
    algorithm = make_logical_algorithm(3,4,{1:["H","X"],2:["Y"]},basis=basis,optimize=optimize)
    qc,_,_ = stim_to_qiskit(algorithm.circuit)
    counts = AerSimulator(method="stabilizer",max_parallel_threads=1).run(qc,shots=128,seed_simulator=3).result().get_counts()
    raw = counts_to_measurement_array(counts,algorithm.circuit.num_measurements,128)
    assert np.all(algorithm.corrected_logical_bits(raw) == algorithm.expected_logical_bit)
    dets,obs = algorithm.circuit.compile_m2d_converter().convert(measurements=raw,separate_observables=True)
    assert not dets.any() and not obs.any()


def test_spam_fit_does_not_stick_at_amplitude_boundary():
    from challenge.decoding import fit_ler
    fit = fit_ler([1,4,7,10,13],[316,779,1346,1929,2499],[100000]*5,fit_spam=True)
    assert .99 < fit["amplitude"] < 1
    assert fit["ci95_profile"][0] < fit["epsilon_per_round"] < fit["ci95_profile"][1]
    # A small perturbation of either parameter must not improve the likelihood.
    r=np.array([1,4,7,10,13]); k=np.array([316,779,1346,1929,2499]); n=100000
    def nll(e,a):
        p=.5*(1-a*(1-2*e)**r)
        return -np.sum(k*np.log(p)+(n-k)*np.log1p(-p))
    e,a=fit["epsilon_per_round"],fit["amplitude"]
    assert nll(e,a) <= min(nll(e+1e-5,a),nll(e-1e-5,a),nll(e,a+1e-5),nll(e,a-1e-5))
