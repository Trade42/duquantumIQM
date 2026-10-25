"""Shared machinery for the recurrent decoders (rnn_lstm, rnn_tf): layouts, the decoder class, mixtures.

The detection events of a no-reset memory circuit are arranged per round and per stabilizer
with ``data.detector_layout``: rounds 0..R-1 carry one event per stabilizer (missing where
the circuit has no detector, e.g. the non-basis checks of round 0), and the final step (t = R)
carries the basis-type checks read from the data qubits. A network sees, per round, the events,
the mask of which detectors exist, and the round's kind (first, second, bulk), so one set of
weights serves every R. torch is imported lazily.

``SequenceDecoder`` implements the backend API (CONTRACT.md). ``decode_batch`` infers R from the
input width ((d^2 - 1) R detectors) and rejects any other width.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import time
import numpy as np
import stim

from . import data
from .data import DatasetSpec, circuit_sha256

KINDS = ("first", "second", "bulk", "final")


@lru_cache(maxsize=None)
def layout_for(distance: int, rounds: int, basis: str) -> dict:
    """detector_layout of the team's ideal circuit, plus the per-round tensors the networks read."""
    spec = DatasetSpec(distance=distance, rounds=rounds, basis=basis)
    ideal = data._ideal(spec)
    lay = data.detector_layout(ideal)
    grid = lay["grid"]                                    # (R+1, S), -1 where no detector
    S = lay["num_stabilizers"]
    pad = ideal.num_detectors                             # Index of the zero column appended at gather time.
    index = np.where(grid >= 0, grid, pad).astype(np.int64)
    kinds = np.array([0 if t == 0 else 1 if t == 1 else 2 for t in range(rounds)], dtype=np.int64)
    return dict(distance=distance, rounds=rounds, basis=basis, num_detectors=ideal.num_detectors,
                num_measurements=ideal.num_measurements, S=S, index=index, exists=(grid >= 0).astype(np.float32),
                kinds=kinds, types=np.array([s["type"] == "X" for s in lay["stabilizers"]], dtype=np.int64),
                xy=np.array([[s["x"], s["y"]] for s in lay["stabilizers"]], dtype=np.float32),
                stabilizers=lay["stabilizers"], raw_index=_raw_index(ideal, lay))


def _raw_index(ideal: stim.Circuit, lay: dict) -> np.ndarray:
    """(R, S): the measurement index of each ancilla's readout in each round (the raw record variant)."""
    from challenge.internal_helpers import get_meas_order
    order = get_meas_order(ideal)
    qubit_of = [s["qubit"] for s in lay["stabilizers"]]
    seen, out = {}, np.full((lay["rounds"], lay["num_stabilizers"]), -1, dtype=np.int64)
    for j, q in enumerate(order):
        if q in qubit_of:
            k = seen.get(q, 0)
            seen[q] = k + 1
            if k < lay["rounds"]:
                out[k, qubit_of.index(q)] = j
    if (out < 0).any():
        raise ValueError("Some ancilla readouts are missing from the record")
    return out


def rounds_from_width(width: int, distance: int) -> int:
    per = distance**2 - 1
    if width <= 0 or width % per:
        raise ValueError(f"{width} detectors is not (d^2 - 1) * R for d = {distance}")
    return width // per


class LayoutCache:
    """Per (R) layout tensors on one torch device."""

    def __init__(self, distance: int, basis: str, device: str):
        self.distance, self.basis, self.device, self.cache = distance, basis, device, {}

    def get(self, rounds: int) -> dict:
        import torch
        key = (rounds, self.device)
        if key not in self.cache:
            lay = layout_for(self.distance, rounds, self.basis)
            self.cache[key] = dict(rounds=rounds, num_detectors=lay["num_detectors"],
                                   index=torch.from_numpy(lay["index"]).to(self.device),
                                   exists=torch.from_numpy(lay["exists"]).to(self.device),
                                   kinds=torch.from_numpy(lay["kinds"]).to(self.device),
                                   raw_index=torch.from_numpy(lay["raw_index"]).to(self.device))
        return self.cache[key]


def gather(dets_u8, lay: dict):
    """(N, D) uint8 events on the device -> rounds (N, R, S) and final (N, S), float32."""
    import torch
    n = dets_u8.shape[0]
    padded = torch.cat([dets_u8, torch.zeros((n, 1), dtype=dets_u8.dtype, device=dets_u8.device)], dim=1)
    grid = padded[:, lay["index"].reshape(-1)].reshape(n, *lay["index"].shape).float()
    return grid[:, :-1], grid[:, -1]


def gather_raw(raw_u8, lay: dict):
    """(N, M) uint8 raw record -> (N, R, S) ancilla readouts per round, float32."""
    n = raw_u8.shape[0]
    return raw_u8[:, lay["raw_index"].reshape(-1)].reshape(n, *lay["raw_index"].shape).float()


# ---------------------------------------------------------------- mixtures (what fit() trains on)

@dataclass(frozen=True)
class Component:
    """One noise model of a training mixture; its rounds come from the mixture."""
    noise: str
    p: float
    calibration: str | None
    weight: float

    def spec(self, distance: int, rounds: int, basis: str) -> DatasetSpec:
        return DatasetSpec(distance=distance, rounds=rounds, basis=basis, noise=self.noise, p=self.p,
                           calibration=self.calibration)


@dataclass(frozen=True)
class Mixture:
    """Synthetic training data: noise components x rounds, with a curriculum on R."""
    distance: int
    basis: str
    components: tuple
    rounds: tuple
    curriculum_fraction: float = 0.15     # Share of the steps that only see R <= curriculum_max_rounds.
    curriculum_max_rounds: int = 5

    def specs(self) -> list[DatasetSpec]:
        return [c.spec(self.distance, r, self.basis) for c in self.components for r in self.rounds]

    def describe(self) -> dict:
        return dict(distance=self.distance, basis=self.basis, rounds=list(self.rounds),
                    curriculum=dict(fraction=self.curriculum_fraction, max_rounds=self.curriculum_max_rounds),
                    components=[dict(asdict(c), slug=c.spec(self.distance, 1, self.basis).slug) for c in self.components])


# ---------------------------------------------------------------- the decoder

class SequenceDecoder:
    """Backend-API decoder around a recurrent network (subclasses supply ``make_net``)."""

    name = "sequence"
    FORMAT = "mlDecoder.sequence/1"
    DEFAULT_MODEL: dict = {}

    def __init__(self, distance: int, basis: str, config: dict, *, family: str | None = None):
        self.distance, self.basis = distance, basis
        self.config = dict(config)
        self.model_cfg = {**self.DEFAULT_MODEL, **self.config.get("model", {})}
        self.uses_raw = bool(self.model_cfg.get("raw_record", False))
        self.family = family
        self.device, self.net, self.training_info, self.checkpoint_path = "cpu", None, {}, None
        self.ignored_options = {}
        self.layouts = LayoutCache(distance, basis, "cpu")

    # subclasses
    def make_net(self):
        raise NotImplementedError

    def forward(self, net, rounds, final, raw, lay):
        """-> (main logits (N, 1), aux dict) for one batch of one R."""
        raise NotImplementedError

    # ------------------------------------------------------------ plumbing
    def _ensure_net(self):
        if self.net is None:
            self.net = self.make_net()
        return self.net

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self._ensure_net().parameters())

    def weights_sha256(self) -> str:
        digest = hashlib.sha256()
        for key, value in self._ensure_net().state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def to(self, device: str) -> "SequenceDecoder":
        self._ensure_net().to(device)
        self.device = device
        self.layouts = LayoutCache(self.distance, self.basis, device)
        return self

    # ------------------------------------------------------------ inference
    def _check(self, det_events, enable_correlations):
        if enable_correlations is not None:
            self.ignored_options["enable_correlations"] = ("accepted and ignored: a recurrent network has no matching "
                                                           f"graph; last value {enable_correlations!r}")
        x = np.asarray(det_events)
        if x.ndim != 2:
            raise ValueError(f"Expected (shots, num_detectors) detection events, got {x.shape}")
        rounds = rounds_from_width(x.shape[1], self.distance)
        if x.dtype == bool:
            x = x.view(np.uint8)
        elif x.size and (x.min() < 0 or x.max() > 1):
            raise ValueError("Detection events must be binary")
        return np.ascontiguousarray(x, dtype=np.uint8), rounds

    def _logits_tensor(self, x_u8, rounds, raw_u8=None):
        import torch
        lay = self.layouts.get(rounds)
        r, f = gather(x_u8, lay)
        raw = gather_raw(raw_u8, lay) if (self.uses_raw and raw_u8 is not None) else None
        if self.uses_raw and raw is None:
            raise ValueError("This decoder reads the raw measurement record: pass raw_meas")
        with torch.autocast(device_type=x_u8.device.type, dtype=torch.float16, enabled=x_u8.device.type == "cuda"):
            main, _ = self.forward(self.net, r, f, raw, lay)
        return main.float()

    def decode_batch(self, det_events: np.ndarray, *, enable_correlations=None, raw_meas=None,
                     chunk: int = 32768) -> np.ndarray:
        """(N, 1) bool in shot order; R from the width. Shots with no detection event predict no flip."""
        import torch
        x, rounds = self._check(det_events, enable_correlations)
        raw = None if raw_meas is None else np.ascontiguousarray(raw_meas, dtype=np.uint8)
        self._ensure_net().eval()
        out = np.zeros((len(x), 1), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                xt = torch.from_numpy(x[start:start + chunk]).to(self.device)
                rt = None if raw is None else torch.from_numpy(raw[start:start + chunk]).to(self.device)
                out[start:start + chunk] = self._predict_tensor(xt, rounds, rt).cpu().numpy()
        return out

    def _predict_tensor(self, x_u8, rounds, raw_u8=None):
        import torch
        active = (x_u8 != 0).any(dim=1)
        pred = torch.zeros((x_u8.shape[0], 1), dtype=torch.bool, device=x_u8.device)
        if bool(active.any()):
            pred[active] = self._logits_tensor(x_u8[active], rounds, None if raw_u8 is None else raw_u8[active]) > 0
        return pred

    def probabilities(self, det_events: np.ndarray, *, raw_meas=None, chunk: int = 32768) -> np.ndarray:
        """(N,) float64 P(observable flip) from the main head; 0 for an empty syndrome (the fast path's decision)."""
        import torch
        x, rounds = self._check(det_events, None)
        raw = None if raw_meas is None else np.ascontiguousarray(raw_meas, dtype=np.uint8)
        self._ensure_net().eval()
        out = np.zeros(len(x), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                xt = torch.from_numpy(x[start:start + chunk]).to(self.device)
                active = (xt != 0).any(dim=1)
                if bool(active.any()):
                    rt = None if raw is None else torch.from_numpy(raw[start:start + chunk]).to(self.device)[active]
                    logits = self._logits_tensor(xt[active], rounds, rt)[:, 0]
                    block = np.zeros(len(xt))
                    block[active.cpu().numpy()] = torch.sigmoid(logits).double().cpu().numpy()
                    out[start:start + chunk] = block
        return out

    def decode_dataset(self, dataset, *, chunk: int = 65536) -> np.ndarray:
        """Predictions for a Dataset (uses its raw record when the variant needs it)."""
        import torch
        from .train import unpack_bits
        rounds = rounds_from_width(dataset.num_detectors, self.distance)
        width = dataset.num_detectors
        if self.uses_raw and dataset.raw_packed is None:
            raise ValueError("The raw-record variant needs a Dataset with its raw record")
        self._ensure_net().eval()
        out = np.zeros((dataset.shots, 1), dtype=bool)
        with torch.no_grad():
            for start in range(0, dataset.shots, chunk):
                p = torch.from_numpy(np.ascontiguousarray(dataset.dets_packed[start:start + chunk])).to(self.device)
                raw = None
                if self.uses_raw:
                    rp = torch.from_numpy(np.ascontiguousarray(dataset.raw_packed[start:start + chunk])).to(self.device)
                    raw = unpack_bits(rp, dataset.circuit.num_measurements)
                out[start:start + chunk] = self._predict_tensor(unpack_bits(p, width), rounds, raw).cpu().numpy()
        return out

    def profile(self, det_events: np.ndarray, reps: int = 20) -> dict:
        import torch
        x, rounds = self._check(det_events, None)
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        parts = dict(transfer_in=[], inference=[], transfer_out=[])
        with torch.no_grad():
            for _ in range(reps + 2):
                sync(); t0 = time.perf_counter()
                xt = torch.from_numpy(x).to(self.device)
                sync(); t1 = time.perf_counter()
                pred = self._predict_tensor(xt, rounds)
                sync(); t2 = time.perf_counter()
                pred.cpu().numpy()
                t3 = time.perf_counter()
                for key, value in zip(parts, (t1 - t0, t2 - t1, t3 - t2)):
                    parts[key].append(value)
        return {f"{k}_us": float(1e6 * np.median(v[2:])) for k, v in parts.items()} | dict(batch=len(x), reps=reps)

    # ------------------------------------------------------------ training
    def fit(self, train_spec_or_data, val, *, device: str, seed: int, log=print) -> dict:
        """Pre-train on a Mixture, or fine-tune on a list of Datasets (real shots); then freeze.

        ``val`` is a list of Datasets (any mix of R): early stopping and checkpoint choice only.
        Fine-tuning starts from the current weights; config["train"]["l2_to_init"] pulls the
        weights towards them (AlphaQubit's weight decay relative to the pre-trained model).
        """
        import torch
        from .train import train_sequence
        if self.net is None:
            torch.manual_seed(seed)      # Initial weights come from the run seed.
        for parameter in self._ensure_net().parameters():
            parameter.requires_grad_(True)   # A loaded (frozen) model can be fine-tuned.
        self.to(device)
        started = time.perf_counter()
        info = train_sequence(self, train=train_spec_or_data, val=list(val), cfg=self.config.get("train", {}),
                              device=device, seed=seed, log=log, checkpoint_path=self.checkpoint_path)
        info["fit_seconds"] = time.perf_counter() - started
        info["seed"] = seed
        self.training_info = info
        return {k: v for k, v in info.items() if k not in ("history", "config")}

    # ------------------------------------------------------------ persistence
    def save(self, path, *, half: bool = False) -> None:
        import torch
        state = {k: (v.detach().cpu().half() if half and v.is_floating_point() else v.detach().cpu())
                 for k, v in self._ensure_net().state_dict().items()}
        torch.save(dict(format=self.FORMAT, backend=self.name, distance=self.distance, basis=self.basis,
                        family=self.family, config=self.config, training_info=self.training_info,
                        weights_sha256=self.weights_sha256(), state_dict=state, half=half), path)

    @classmethod
    def load(cls, path, circuit: stim.Circuit | None, device: str) -> "SequenceDecoder":
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != cls.FORMAT or payload.get("backend") != cls.name:
            raise ValueError(f"{path}: not a {cls.name} {cls.FORMAT} file")
        decoder = cls(payload["distance"], payload["basis"], payload["config"], family=payload.get("family"))
        if circuit is not None:
            lay = data.detector_layout(circuit)
            if lay["distance"] != decoder.distance or set(lay["type"][lay["round"] == 0]) != {decoder.basis}:
                raise ValueError(f"{path}: trained for d={decoder.distance} {decoder.basis} memory, not this circuit")
        net = decoder._ensure_net()
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in payload["state_dict"].items()})
        decoder.training_info = payload["training_info"]
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        return decoder.to(device)

    def metadata(self) -> dict:
        import torch
        info = {k: v for k, v in self.training_info.items() if k != "config"}
        return dict(name=self.name, architecture=self.describe_architecture(), parameters=self.parameter_count(),
                    distance=self.distance, basis=self.basis, family=self.family, config=self.config,
                    torch=torch.__version__, device=self.device, weights_sha256=self.weights_sha256(),
                    training=info, ignored_options=self.ignored_options,
                    inputs=("detection events and the raw measurement record (offline, full measured record)"
                            if self.uses_raw else "detection events only (no observables, no leakage flags)"),
                    rounds="any R: R inferred from the input width", empty_syndrome="fast path: predicts no flip")

    def describe_architecture(self) -> str:
        return self.name
