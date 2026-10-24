from internal_helpers import get_qubit_lists
import stim



"""

OPTIONAL FILE: Allows you to build a custom STIM -> EMERALD mapping.

QISKIT TRACK ONLY. The output is a Qiskit initial_layout. On the Qrisp track,
use iqm.qrisp_iqm.vf2pp_layout in the pass manager instead (exact embedding,
zero SWAPs, or an error if none exists).

"""


def build_emerald_qubit_map(
    stim_circuit: stim.Circuit,

) -> dict:
    """
    Maps Stim dense qubit indices (0..16) to Qiskit qubit indices for IQM Emerald.

    NOTE: OPTIONAL, since the qiskit transpiler automatically optimizes for least number of swap gates. Still, hand-testing is recommended.
    You will also have to adjust which parameters this taken. Default parameters form ``print_qubit_map`` target this 5x5 sub-region of the Emerald chip (OR ANY OTHER!)

        QB41 QB42 QB43 QB44 QB45   <- grid row 0 (top)
        QB33 QB34 QB35 QB36 QB37
        QB25 QB26 QB27 QB28 QB29
        QB17 QB18 QB19 QB20 QB21
        QB9  QB10 QB11 QB12 QB13   <- grid row 4 (bottom)

    This maps the code's natural diagonal connectivity directly onto the hardware grid, giving
    ALL 24 stabilizer CX pairs a Manhattan distance of 1 — no SWAPs needed.

    EXAMPLARY Resulting layout on the 5x5 hardware sub-region:

        col:   0     1     2     3     4
        row 0:  .     .   D(5)  A(13)  .       QB41-QB45
        row 1: A(2)  D(3)  A(11) D(12)  .       QB33-QB37
        row 2: D(1)  A(9)  D(10) A(18) D(19)    QB25-QB29
        row 3:  .    D(8)  A(16) D(17) A(25)    QB17-QB21
        row 4:  .    A(14) D(15)  .     .       QB9 -QB13

    (numbers are Stim qubit indices; 8 spare positions at edges)

    IQM Resonance uses 1-based QB numbers (QB1, QB2, ...).
    Qiskit uses 0-based indices.
    The output could be ready ready to pass directly to transpile(initial_layout=...)

    Parameters
    ----------

    Returns
    -------
    dict  Ready for: transpile(qc, backend, initial_layout=qubit_map)

    Notes
    -----
    Verify before hardware runs by cross-checking with backend.coupling_map.
    THIS NEEDS A FULL REWORK, if you want to use it for the Ising model, i.e. it needs to based on the Ising Models MemoryCiruict functionality, see github.com/NVIDIA/Ising-Decoding/code/qec/surface_code/memory_circuit.py
    """
    coords        = stim_circuit.get_final_qubit_coordinates()
    qubit_map = {}

    # TO BE DONE
    return qubit_map





def print_qubit_map(
    stim_circuit: stim.Circuit,
    grid_top_left: int = 41,
    grid_n_rows: int = 5,
    grid_n_cols: int = 5,
    row_stride: int = -8,
    col_stride: int = 1,
    qiskit_offset: int = -1,
):
    """
    NOTE: If you are using qiskit, the mapping is easily compared by hand, since the expected Qiskit and Resonance qubit indices are the same (except for a -1 difference).


    Prints a human-readable table of the Stim → Emerald qubit mapping
    and flags any CX pairs that require SWAPs (non-adjacent physical qubits).
    Use this to verify the mapping before submitting to hardware.

    ONLY WORKS ON SQUARE PATCHES


    Parameters
    ----------
    stim_circuit   : stim.Circuit
    grid_top_left  : Resonance QB number at (row=0, col=0). Default 41.
    grid_n_rows    : number of rows in the sub-region. Default 5.
    grid_n_cols    : number of columns in the sub-region. Default 5.
    row_stride     : QB number difference per row step downward. Default -8. I.e. row 0 starts at QB41, row 1 starts QB33
    col_stride     : QB number difference per column step rightward. Default 1.
    qiskit_offset  : added to QB numbers to get Qiskit 0-based indices.
                     Default -1 (QB1 -> 0, QB17 -> 16, QB45 -> 44).
                     Set to 0 to return raw Resonance QB numbers instead.
    """
    coords        = stim_circuit.get_final_qubit_coordinates()
    data_q, anc_q = get_qubit_lists(stim_circuit)
    qubit_map     = build_emerald_qubit_map(
        stim_circuit, grid_top_left, grid_n_rows, grid_n_cols,
        row_stride, col_stride, qiskit_offset,
    )
    all_stim_q    = sorted(set(data_q + anc_q))
    stim_to_dense = {sq: i for i, sq in enumerate(all_stim_q)}

    print("─" * 68)
    print(f"{'Stim idx':>10} {'Stim coord':>12} {'Type':>8} "
          f"{'Dense idx':>10} {'Qiskit idx':>12} {'QB (Resonance)':>16}")
    print("─" * 68)
    for stim_q in all_stim_q:
        d    = stim_to_dense[stim_q]
        qk   = qubit_map[d]
        xy   = coords[stim_q]
        qt   = "DATA " if stim_q in set(data_q) else "ANCL "
        print(f"{stim_q:>10} {str((int(xy[0]),int(xy[1]))):>12} "
              f"{qt:>8} {d:>10} {qk:>12} {'QB'+str(qk-qiskit_offset):>16}")
    print("─" * 68)

    def qiskit_to_grid(q):
        """Convert Qiskit index to (row, col) in the Emerald sub-grid.
        Derived from: q = (grid_top_left + qiskit_offset) + r*row_stride + c*col_stride
        For default params: q = 40 + r*(-8) + c*1  →  r = (40-q+q%8)//8, c = q%8
        General: solves for r and c given the stride parameters."""
        # Shift to remove qiskit_offset: get QB number back, then find grid pos
        qb = q - qiskit_offset          # Resonance QB number
        # qb = grid_top_left + r*row_stride + c*col_stride
        # Since col_stride=1: c = (qb - grid_top_left - r*row_stride)
        # Iterate rows (small loop, only grid_n_rows iterations)
        for r in range(grid_n_rows):
            c = qb - grid_top_left - r * row_stride
            if 0 <= c < grid_n_cols:
                return r, c
        return None, None  # not in this sub-grid

    # CX connectivity check
    n_native, n_swap = 0, 0
    swap_details = []
    for instr in stim_circuit.flattened():
        if instr.name == "CX":
            targets = [t.value for t in instr.targets_copy() if t.is_qubit_target]
            for i in range(0, len(targets), 2):
                sa, sb = targets[i], targets[i + 1]
                qa = qubit_map[stim_to_dense[sa]]
                qb = qubit_map[stim_to_dense[sb]]
                ra, ca = qiskit_to_grid(qa)
                rb, cb = qiskit_to_grid(qb)
                dist = abs(ra - rb) + abs(ca - cb)
                if dist == 1:
                    n_native += 1
                else:
                    n_swap += 1
                    swap_details.append((sa, sb, qa, qb, dist))

    print(f"CX gates: {n_native} native (dist=1),  "
          f"{n_swap} non-native (dist>1, need SWAP)")
    if swap_details:
        print("  Non-native pairs (sorted by distance):")
        for sa, sb, qa, qb, d in sorted(swap_details, key=lambda x: -x[4]):
            print(f"    stim({sa:>2},{sb:>2})  QB{qa-qiskit_offset:>2}↔QB{qb-qiskit_offset:>2}"
                  f"  Qiskit({qa},{qb})  dist={d}")
    print()