from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest
import stim

from challenge.audit_garnet_routing import (
    ARCHITECTURE, BUNDLE, CERTIFICATE, audit, audit_circuit_pair, cz_edges,
)


@pytest.fixture(scope="module")
def physical_case():
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    architecture = json.loads(ARCHITECTURE.read_text(encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["arm"] == "optimized" and c["rounds"] == 3)
    gate = architecture["gates"]["cz"]
    return dict(
        source=stim.Circuit.from_file(BUNDLE / case["files"]["source"]),
        native=stim.Circuit.from_file(BUNDLE / case["files"]["native"]),
        mapping={int(q): name for q, name in manifest["optimized_placement"].items()},
        qubit_names=architecture["qubits"],
        couplers=gate["implementations"][gate["default_implementation"]]["loci"],
        rounds=3, columns=case["native_measurement_columns"],
    )


def test_published_certificate_recomputes_offline(monkeypatch):
    original_open = Path.open

    def no_credentials(path, *args, **kwargs):
        if path.name in {"apikey.jom", ".env"}:
            raise AssertionError("Routing audit read credentials")
        return original_open(path, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Routing audit used the network")

    monkeypatch.setattr(Path, "open", no_credentials)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    result = audit()
    assert result == json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    assert result["cases_verified"] == 12
    assert result["routing_swaps_achieved"] == result["optimal_routing_swap_count"] == 0
    assert {c["native_cz"] for c in result["cases"]} == {24, 48, 72}


def test_literal_and_decomposed_swaps_are_rejected(physical_case):
    native = physical_case["native"]
    a, b = cz_edges(native)[0]
    explicit = native + stim.Circuit(f"SWAP {a} {b}")
    with pytest.raises(ValueError, match="Unexpected two-qubit gate"):
        audit_circuit_pair(**{**physical_case, "native": explicit})
    # A basis-gate SWAP has no SWAP label; its three extra CZs must still fail.
    extra = stim.Circuit(f"H {b}\nCZ {a} {b}\nH {b}\nH {a}\nCZ {a} {b}\nH {a}\nH {b}\nCZ {a} {b}\nH {b}")
    with pytest.raises(ValueError, match="edge multiset differs"):
        audit_circuit_pair(**{**physical_case, "native": native + extra})


def test_nonnative_coupler_and_noninjective_mapping_are_rejected(physical_case):
    case = deepcopy(physical_case)
    first = cz_edges(case["native"])[0]
    forbidden = {case["qubit_names"][q] for q in first}
    case["couplers"] = [pair for pair in case["couplers"] if set(pair) != forbidden]
    with pytest.raises(ValueError, match="non-native coupling"):
        audit_circuit_pair(**case)
    case = deepcopy(physical_case)
    a, b = list(case["mapping"])[:2]
    case["mapping"][a] = case["mapping"][b]
    with pytest.raises(ValueError, match="injective"):
        audit_circuit_pair(**case)


def test_reordered_entanglers_fail_even_when_gate_counts_match(physical_case):
    instructions = []
    for instruction in physical_case["native"].flattened():
        if instruction.name == "CZ":
            targets = instruction.targets_copy()
            instructions.extend(stim.CircuitInstruction("CZ", [a, b])
                                for a, b in zip(targets[::2], targets[1::2]))
        else:
            instructions.append(instruction)
    positions = [i for i, instruction in enumerate(instructions) if instruction.name == "CZ"]
    a = positions[0]
    first = {t.value for t in instructions[a].targets_copy()}
    b = next(i for i in positions[1:]
             if first & {t.value for t in instructions[i].targets_copy()}
             and first != {t.value for t in instructions[i].targets_copy()})
    instructions[a], instructions[b] = instructions[b], instructions[a]
    changed = stim.Circuit()
    for instruction in instructions:
        changed.append(instruction)
    with pytest.raises(ValueError, match="ordered CZ interactions"):
        audit_circuit_pair(**{**physical_case, "native": changed})


def test_wrong_measurement_mapping_and_calibration_are_rejected(physical_case, tmp_path):
    with pytest.raises(ValueError, match="Measurement wires"):
        audit_circuit_pair(**{**physical_case, "columns": physical_case["columns"][::-1]})
    architecture = json.loads(ARCHITECTURE.read_text(encoding="utf-8"))
    architecture["calibration_set_id"] = "different-calibration"
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(architecture), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration differ"):
        audit(architecture=path)
