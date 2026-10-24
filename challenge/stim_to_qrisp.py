"""Prototype: replay a Stim circuit (e.g. stim.Circuit.generated) as a Qrisp DetectorExperiment.

Geometry, CX order, noise and detectors all come from Stim. With no_reset=True, every
mid-circuit MR becomes a plain M and the detectors are rewritten automatically:
the raw no-reset outcome is m'_j = s_j XOR m'_prev, so each referenced s_j is
replaced by {j, prev(j)} (symmetric difference). Expectations come from Stim's
reference sample of the rewritten noiseless circuit.
"""
import stim
from iqm.qrisp_iqm import DetectorExperiment
from qrisp import QuantumArray, QuantumBool, cx, h, measure, parity, reset
from qrisp.misc.stim_tools import stim_noise

GATES_1Q = {"H": h}
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR"}
SKIP = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}


def _analyse(flat: stim.Circuit, no_reset: bool):
    """Measurement bookkeeping + detector/observable index sets """
    meas_q, last_on_q, prev_removed, removed = [], {}, {}, set()
    dets, obs = [], {}
    for inst in flat:
        name = inst.name
        if name in ("M", "MR"):
            for t in inst.targets_copy():
                j = len(meas_q)
                meas_q.append(t.value)
                i = last_on_q.get(t.value)
                # previous measurement on this qubit had its reset removed -> outcomes accumulate
                prev_removed[j] = i if (no_reset and i is not None and i in removed) else None
                last_on_q[t.value] = j
                if name == "MR":
                    removed.add(j)
        elif name == "DETECTOR":
            dets.append([len(meas_q) + t.value for t in inst.targets_copy() if t.is_measurement_record_target])
        elif name == "OBSERVABLE_INCLUDE":
            k = int(inst.gate_args_copy()[0])
            obs.setdefault(k, []).extend(len(meas_q) + t.value for t in inst.targets_copy())

    def rewrite(idx_list):
        out = set()
        for j in idx_list:
            # s_j = m'_j XOR m'_prev
            out ^= {j}
            if prev_removed.get(j) is not None:
                out ^= {prev_removed[j]}
        return sorted(out)

    return meas_q, [rewrite(d) for d in dets], {k: rewrite(v) for k, v in obs.items()}



def stim_to_detector_experiment(circ: stim.Circuit, no_reset: bool = True):
    flat = circ.flattened()
    meas_q, dets, obs = _analyse(flat, no_reset)
    qubits = sorted({t.value for inst in flat for t in inst.targets_copy() if t.is_qubit_target})
    pos = {q: i for i, q in enumerate(qubits)}

    # noiseless reference of the circuit that will actually run -> parity expectations
    ref_circ = stim.Circuit(str(flat.without_noise()).replace("MR ", "M ")) if no_reset else flat.without_noise()
    ref = ref_circ.reference_sample()
    exp_of = lambda s: int(sum(int(ref[i]) for i in s) % 2)

    @DetectorExperiment
    def experiment():
        q = QuantumArray(qtype=QuantumBool(), shape=(len(qubits),))
        touched, m = set(), []
        for inst in flat:
            name, tg = inst.name, [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if name in SKIP:
                continue
            if name == "R":
                for t in tg:                       # first R is free: qubits start in |0>
                    if t in touched:
                        reset(q[pos[t]])
                    touched.add(t)
            elif name in GATES_1Q:
                for t in tg:
                    GATES_1Q[name](q[pos[t]]); touched.add(t)
            elif name == "CX":
                for a, b in zip(tg[::2], tg[1::2]):
                    cx(q[pos[a]], q[pos[b]]); touched |= {a, b}
            elif name in ("M", "MR"):
                for t in tg:
                    m.append(measure(q[pos[t]]))
                    if name == "MR" and not no_reset:
                        reset(q[pos[t]])
            elif name in NOISE:
                p = inst.gate_args_copy()[0]
                if name == "DEPOLARIZE2":
                    for a, b in zip(tg[::2], tg[1::2]):
                        stim_noise(name, p, q[pos[a]], q[pos[b]])
                else:
                    for t in tg:
                        stim_noise(name, p, q[pos[t]])
            else:
                raise NotImplementedError(name)
        D = [parity(*[m[i] for i in s], expectation=exp_of(s)) for s in dets]
        O = [parity(*[m[i] for i in obs[k]], expectation=exp_of(obs[k]), observable=True) for k in sorted(obs)]
        return D, O

    return experiment
