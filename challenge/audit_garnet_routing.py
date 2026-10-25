"""Offline routing certificate for the published Garnet memory circuits."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import stim

from .submission import BUNDLE, REPOSITORY, verify_bundle

ARCHITECTURE = REPOSITORY / "docs/assets/garnet_architecture.json"
CERTIFICATE = REPOSITORY / "docs/assets/garnet_routing_certificate.json"


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cz_edges(circuit):
    """Inspect every two-qubit operation, including decomposable SWAP variants."""
    edges = []
    for instruction in circuit.flattened():
        if not stim.gate_data(instruction.name).is_two_qubit_gate:
            continue
        if instruction.name != "CZ":
            raise ValueError(f"Unexpected two-qubit gate: {instruction.name}")
        targets = instruction.targets_copy()
        for a, b in zip(targets[::2], targets[1::2]):
            if not a.is_qubit_target or not b.is_qubit_target:
                raise ValueError("Classically controlled two-qubit operation.")
            edges.append((a.value, b.value))
    return edges


def wire_traces(edges):
    traces = defaultdict(list)
    for a, b in edges:
        traces[a].append(b)
        traces[b].append(a)
    return dict(traces)


def measurement_qubits(circuit):
    qubits = []
    for instruction in circuit.flattened():
        if not stim.gate_data(instruction.name).produces_measurements:
            continue
        if instruction.name not in {"M", "MX"}:
            raise ValueError("Unexpected measurement primitive.")
        qubits.extend(target.value for target in instruction.targets_copy())
    return qubits


def check_kinds(source):
    """Recover X/Z check types by back-propagating their first measurements."""
    coordinates = source.get_final_qubit_coordinates()
    data = {q for q, (x, y, *_) in coordinates.items() if x % 2 == y % 2 == 1}
    prefix = stim.Circuit()
    for instruction in source.flattened():
        if stim.gate_data(instruction.name).produces_measurements:
            break
        if stim.gate_data(instruction.name).is_unitary:
            prefix.append(instruction)
    inverse = prefix.to_tableau().inverse()
    measured_basis = {}
    for instruction in source.flattened():
        if instruction.name in {"M", "MX"}:
            for target in instruction.targets_copy():
                measured_basis.setdefault(target.value, instruction.name)
    kinds = {}
    for q in set(coordinates) - data:
        observable = stim.PauliString(source.num_qubits)
        observable[q] = "X" if measured_basis[q] == "MX" else "Z"
        backwards = inverse(observable)
        support = {backwards[datum] for datum in data if backwards[datum]}
        if support not in ({1}, {3}):
            raise ValueError("Source check is not a pure CSS X or Z check.")
        kinds[q] = "X" if support == {1} else "Z"
    if Counter(kinds.values()) != Counter({"X": 4, "Z": 4}):
        raise ValueError("Expected four X checks and four Z checks.")
    return kinds


def audit_circuit_pair(source, native, mapping, qubit_names, couplers, rounds, columns):
    """Prove static native locality and zero added entanglers for one case."""
    if rounds not in (1, 2, 3):
        raise ValueError("Published Garnet certificate covers rounds 1, 2 and 3.")
    source_coords = source.get_final_qubit_coordinates()
    native_coords = native.get_final_qubit_coordinates()
    if len(source_coords) != 17 or set(source_coords) != set(mapping):
        raise ValueError("Expected exactly the seventeen source qubits.")
    if len(set(mapping.values())) != 17 or not set(mapping.values()) <= set(qubit_names):
        raise ValueError("Placement must be an injective hardware mapping.")
    if len(native_coords) != 17:
        raise ValueError("Native circuit uses an unexpected physical patch.")
    indices = {name: index for index, name in enumerate(qubit_names)}
    physical = {q: indices[name] for q, name in mapping.items()}
    if any(native_coords.get(physical[q]) != xy for q, xy in source_coords.items()):
        raise ValueError("Native coordinates disagree with the static placement.")
    for instruction in native.flattened():
        if any(target.value not in physical.values() for target in instruction.targets_copy()
               if target.is_qubit_target):
            raise ValueError("Native operation addresses a qubit outside the patch.")

    source_edges = cz_edges(source)
    native_edges = cz_edges(native)
    expected = [(physical[a], physical[b]) for a, b in source_edges]
    expected_counts = Counter(tuple(sorted(edge)) for edge in expected)
    actual_counts = Counter(tuple(sorted(edge)) for edge in native_edges)
    if len(source_edges) != 24 * rounds or len(expected_counts) != 24:
        raise ValueError("Source is not the complete 24-interaction memory schedule.")
    if any(count != rounds for count in expected_counts.values()):
        raise ValueError("A required source coupling is missing from a round.")
    if actual_counts != expected_counts:
        raise ValueError("Native CZ edge multiset differs: added, missing or rerouted gates.")
    if wire_traces(native_edges) != wire_traces(expected):
        raise ValueError("The ordered CZ interactions on a physical wire changed.")
    legal = {frozenset(edge) for edge in couplers}
    if any(frozenset((qubit_names[a], qubit_names[b])) not in legal for a, b in native_edges):
        raise ValueError("Native circuit contains a non-native coupling.")

    data = {q for q, (x, y, *_) in source_coords.items() if x % 2 == y % 2 == 1}
    checks = set(source_coords) - data
    if len(data) != 9 or len(checks) != 8:
        raise ValueError("Unexpected data/check partition.")
    support = defaultdict(set)
    for a, b in source_edges:
        if (a in data) == (b in data):
            raise ValueError("Expected data-to-check interactions only.")
        check, datum = (b, a) if a in data else (a, b)
        support[check].add(datum)
    if Counter(map(len, support.values())) != Counter({2: 4, 4: 4}):
        raise ValueError("Expected four weight-two and four weight-four checks.")
    kinds = check_kinds(source)

    source_measurements = measurement_qubits(source)
    native_measurements = measurement_qubits(native)
    expected_measurements = 8 * rounds + 9
    if len(source_measurements) != expected_measurements or len(native_measurements) != expected_measurements:
        raise ValueError("Syndrome or terminal measurements changed.")
    if sorted(columns) != list(range(expected_measurements)):
        raise ValueError("Measurement columns are not a bijection.")
    if any(q != physical[source_measurements[columns[i]]] for i, q in enumerate(native_measurements)):
        raise ValueError("Measurement wires do not preserve the fixed placement.")

    return dict(source_cz=len(source_edges), native_cz=len(native_edges),
                explicit_swaps=0, added_two_qubit_gates=0, non_native_couplings=0,
                required_couplers=24, used_qubits=17, data_qubits=9, check_qubits=8,
                check_weights={"2": 4, "4": 4},
                check_types=dict(Counter(kinds.values())),
                same_edge_multiset=True, same_order_on_every_wire=True,
                static_coordinate_mapping=True, static_measurement_mapping=True)


def audit(bundle=BUNDLE, architecture=ARCHITECTURE):
    bundle = Path(bundle)
    manifest = verify_bundle(bundle)
    architecture = json.loads(Path(architecture).read_text(encoding="utf-8"))
    if str(architecture["calibration_set_id"]) != manifest["calibration_set_id"]:
        raise ValueError("Architecture and measured circuit calibration differ.")
    gate = architecture["gates"]["cz"]
    couplers = gate["implementations"][gate["default_implementation"]]["loci"]
    mapping = {int(q): name for q, name in manifest["optimized_placement"].items()}
    cases = [case for case in manifest["cases"] if case["arm"] == "optimized"]
    inventory = Counter((case["basis"], case["rounds"]) for case in cases)
    if inventory != Counter({(basis, rounds): 2 for basis in ("X", "Z") for rounds in (1, 2, 3)}):
        raise ValueError("Expected both acquisition blocks for every X/Z round count.")
    results = []
    for case in sorted(cases, key=lambda case: case["id"]):
        source = stim.Circuit.from_file(bundle / case["files"]["source"])
        native = stim.Circuit.from_file(bundle / case["files"]["native"])
        result = audit_circuit_pair(source, native, mapping, architecture["qubits"],
                                    couplers, case["rounds"], case["native_measurement_columns"])
        for stage in ("before", "after"):
            operations = case["optimization"][stage]
            if operations.get("cz", 0) != result["native_cz"] or set(operations) - {"r", "cz", "measure", "barrier"}:
                raise ValueError("Recorded compilation inventory disagrees with the native circuit.")
        results.append(dict(id=case["id"], basis=case["basis"], rounds=case["rounds"],
                            source_file=case["files"]["source"], native_file=case["files"]["native"],
                            source_sha256=manifest["files"][case["files"]["source"]],
                            native_sha256=manifest["files"][case["files"]["native"]], **result))

    compiler = REPOSITORY / "ethan/src/iqm_surface_code/run_on_hardware.py"
    tree = ast.parse(compiler.read_text(encoding="utf-8"))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_qiskit")
    routing = [keyword.value.value for node in ast.walk(prepare)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "transpile"
               for keyword in node.keywords if keyword.arg == "routing_method" and isinstance(keyword.value, ast.Constant)]
    if routing != ["none"]:
        raise ValueError("Current compiler no longer explicitly disables routing.")
    return dict(schema_version=1, device="garnet", distance=3,
                calibration_set_id=manifest["calibration_set_id"],
                measurement_manifest_sha256=hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
                architecture_sha256_canonical=canonical_hash(architecture),
                hardware_qubits=len(architecture["qubits"]), hardware_couplers=len(couplers),
                placement=manifest["optimized_placement"], routing_method="none",
                routing_swap_lower_bound=0, routing_swaps_achieved=0, optimal_routing_swap_count=0,
                cases_verified=len(results), cases=results,
                scope="Recorded optimized d3 X/Z memory circuits at rounds 1--3; routing SWAP optimality for this fixed schedule.")


def render_layout(result):
    """Draw the certified patch on the recorded twenty-qubit device graph."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    architecture = json.loads(ARCHITECTURE.read_text(encoding="utf-8"))
    gate = architecture["gates"]["cz"]
    couplers = gate["implementations"][gate["default_implementation"]]["loci"]
    grid = [[None, "QB17", "QB12", "QB7", None],
            ["QB20", "QB16", "QB11", "QB6", None],
            ["QB19", "QB15", "QB10", "QB5", "QB2"],
            ["QB18", "QB14", "QB9", "QB4", "QB1"],
            [None, "QB13", "QB8", "QB3", None]]
    positions = {q: (column, -row) for row, cells in enumerate(grid)
                 for column, q in enumerate(cells) if q is not None}
    if set(positions) != set(architecture["qubits"]):
        raise ValueError("Display grid does not cover the device.")
    if any(sum(abs(x-y) for x, y in zip(positions[a], positions[b])) != 1 for a, b in couplers):
        raise ValueError("Display grid distorts a calibrated coupler.")
    case = next(c for c in result["cases"] if c["basis"] == "Z" and c["rounds"] == 1)
    source = stim.Circuit.from_file(BUNDLE / case["source_file"])
    mapping = {int(q): name for q, name in result["placement"].items()}
    data = {q for q, (x, y, *_) in source.get_final_qubit_coordinates().items() if x % 2 == y % 2 == 1}
    kinds = {mapping[q]: "Data" for q in data}
    kinds.update({mapping[q]: kind + " check" for q, kind in check_kinds(source).items()})
    required = {frozenset((mapping[a], mapping[b])) for a, b in cz_edges(source)}
    colors = {"Data": "#2563a6", "X check": "#9a6500", "Z check": "#167567", "Unused": "#dbe2ea"}
    markers = {"Data": "o", "X check": "s", "Z check": "D", "Unused": "o"}
    figure, axes = plt.subplots(figsize=(7.2, 6.4))
    figure.patch.set_facecolor("white")
    for a, b in couplers:
        selected = frozenset((a, b)) in required
        x, y = zip(positions[a], positions[b])
        axes.plot(x, y, color="#324b66" if selected else "#d5dce5",
                  linewidth=2.7 if selected else 1.3, zorder=1)
    for q, (x, y) in positions.items():
        kind = kinds.get(q, "Unused")
        axes.scatter([x], [y], s=930, marker=markers[kind], c=colors[kind],
                     edgecolors="white", linewidths=1.6, zorder=2)
        axes.text(x, y, q, ha="center", va="center", fontsize=9.5, weight="bold",
                  color="#3d4858" if kind == "Unused" else "white", zorder=3)
    legend = [Line2D([], [], marker=markers[k], color="none", markerfacecolor=colors[k],
                     markersize=10, label=k) for k in colors]
    axes.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.12),
                ncol=4, frameon=False, fontsize=10)
    axes.set_title("Garnet surface-code patch\n17 active qubits | 24 native couplers | 0 SWAPs",
                   loc="left", fontsize=15, pad=20, color="#172b42")
    axes.set_aspect("equal")
    axes.set_xlim(-.55, 4.55)
    axes.set_ylim(-4.55, .55)
    axes.axis("off")
    figure.subplots_adjust(left=.06, right=.97, top=.85, bottom=.16)
    for suffix in ("svg", "png"):
        figure.savefig(REPOSITORY / f"docs/assets/garnet_routing_layout.{suffix}", dpi=180,
                       facecolor="white", metadata={"Description": "Certified zero-routing Garnet patch"})
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    from examples.reproduce import network_disabled
    with network_disabled():
        result = audit()
    if args.write_certificate:
        CERTIFICATE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif json.loads(CERTIFICATE.read_text(encoding="utf-8")) != result:
        raise ValueError("Stored certificate differs from the current audited evidence.")
    if args.plot:
        render_layout(result)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Garnet: verified {result['cases_verified']} native memory circuits; 0 routing SWAPs (minimum 0).")
        for rounds in (1, 2, 3):
            print(f"X/Z R{rounds}: {24 * rounds} source CZ = {24 * rounds} native CZ; 0 added entanglers.")
    return result


if __name__ == "__main__":
    main()
