"""Persistent, stateful leakage reduction model on a Clifford Pauli frame.

This is a phenomenological leakage-to-Pauli model, not a qutrit simulator.
Leaked participants are randomized; interactions depolarize partners and may
spread leakage. Flags survive readout, recover stochastically per moment, and
clear at an actual reset (when enabled). An ideal Clifford backbone remains.
"""
from __future__ import annotations
import numpy as np
import stim


def sample_leakage(circuit: stim.Circuit, model, shots: int, *, seed=None,
                    qubit_map=None):
    """Sample static Pauli channels plus temporally persistent hidden flags.

    ``circuit`` includes the model's static noise. No oracle flags are sent to
    the decoder. Return measurements and aggregate leakage diagnostics only.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    rng = np.random.default_rng(seed)
    mapping = qubit_map or {}
    n = circuit.num_qubits
    x, z, leaked = (np.zeros((shots, n), dtype=bool) for _ in range(3))
    raw = circuit.without_noise().compile_sampler(seed=seed).sample(shots)
    mi, entries, exposures, readouts, moments = 0, 0, 0, 0, 0
    used = sorted(circuit.get_final_qubit_coordinates())

    def rand_pauli(q, mask):
        x[:, q] ^= mask & (rng.random(shots) < .5)
        z[:, q] ^= mask & (rng.random(shots) < .5)

    def depolarize(q, p, mask=None):
        hit = rng.random(shots) < p
        if mask is not None:
            hit &= mask
        code = rng.integers(1, 4, shots)
        x[:, q] ^= hit & (code != 3)
        z[:, q] ^= hit & (code != 1)

    for inst in circuit.flattened():
        name, args = inst.name, inst.gate_args_copy()
        targets = inst.targets_copy()
        qs = [t.value for t in targets if t.is_qubit_target]
        if name in {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if name == "TICK":
            exposures += int(leaked[:, used].sum())
            moments += 1
            recover = leaked & (rng.random(leaked.shape) < model.seepage)
            leaked[recover] = False
            # Return to computational space maximally mixed, not an ideal reset.
            x ^= recover & (rng.random(x.shape) < .5)
            z ^= recover & (rng.random(z.shape) < .5)
        elif name in {"R", "RX"}:
            x[:, qs], z[:, qs] = False, False
            if model.reset_clears_leakage:
                leaked[:, qs] = False
        elif name == "H":
            x[:, qs], z[:, qs] = z[:, qs].copy(), x[:, qs].copy()
        elif name in {"S", "S_DAG"}:
            z[:, qs] ^= x[:, qs]
        elif name in {"X", "Y", "Z", "I"}:
            pass  # Same ideal Pauli on reference and noisy trajectory.
        elif name in {"CX", "CZ"}:
            for a, b in zip(qs[::2], qs[1::2]):
                if name == "CX":
                    x[:, b] ^= x[:, a]
                    z[:, a] ^= z[:, b]
                else:
                    z[:, a] ^= x[:, b]
                    z[:, b] ^= x[:, a]
                old_a, old_b = leaked[:, a].copy(), leaked[:, b].copy()
                for q, partner_old in ((a, old_b), (b, old_a)):
                    p = model.leakage
                    if model.calibration:
                        p = 1 - (1-p)*(1-model.calibration.qubits[mapping[q]].leakage_per_gate)
                    enter = ~leaked[:, q] & ((rng.random(shots) < p) |
                                (partner_old & (rng.random(shots) < model.leakage_spread)))
                    entries += int(enter.sum())
                    leaked[:, q] |= enter
                depolarize(a, model.leakage_partner_error, leaked[:, b])
                depolarize(b, model.leakage_partner_error, leaked[:, a])
                rand_pauli(a, leaked[:, a])
                rand_pauli(b, leaked[:, b])
        elif name in {"M", "MX", "MR", "MRX"}:
            for q in qs:
                readouts += int(leaked[:, q].sum())
                rand_pauli(q, leaked[:, q])
                flip = z[:, q] if name in {"MX", "MRX"} else x[:, q]
                raw[:, mi] ^= flip ^ (rng.random(shots) < (args[0] if args else 0))
                mi += 1
                if name.startswith("MR"):
                    x[:, q], z[:, q] = False, False
                    if model.reset_clears_leakage:
                        leaked[:, q] = False
        elif name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
            for q in qs:
                mask = rng.random(shots) < args[0]
                if name != "Z_ERROR":
                    x[:, q] ^= mask
                if name != "X_ERROR":
                    z[:, q] ^= mask
        elif name == "DEPOLARIZE1":
            for q in qs:
                depolarize(q, args[0])
        elif name == "DEPOLARIZE2":
            for a, b in zip(qs[::2], qs[1::2]):
                hit = rng.random(shots) < args[0]
                code = rng.integers(1, 16, shots)
                for q, pauli in ((a, code % 4), (b, code // 4)):
                    x[:, q] ^= hit & ((pauli == 1) | (pauli == 2))
                    z[:, q] ^= hit & ((pauli == 2) | (pauli == 3))
        elif name == "PAULI_CHANNEL_1":
            for q in qs:
                r = rng.random(shots)
                x[:, q] ^= r < args[0]+args[1]
                z[:, q] ^= (r >= args[0]) & (r < sum(args))
        elif name in {"E", "CORRELATED_ERROR"}:
            mask = rng.random(shots) < args[0]
            for t in targets:
                if t.is_x_target or t.is_y_target:
                    x[:, t.value] ^= mask
                if t.is_z_target or t.is_y_target:
                    z[:, t.value] ^= mask
        else:
            raise NotImplementedError(f"Leakage sampler cannot process {name}")
    return raw, dict(model="persistent phenomenological leakage-to-Pauli",
        leakage_entries=entries, leaked_measurements=readouts,
        mean_leaked_fraction=exposures/max(1, shots*len(used)*moments),
        final_leaked_fraction=float(leaked[:, used].mean()),
        hidden_flags_given_to_decoder=False, seepage_unit="per TICK moment")
