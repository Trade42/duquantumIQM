"""MLP baseline: detection events as floats -> 2-4 ReLU hidden layers -> one logit per observable.

Follows the feed-forward decoders of Chamberland & Ronagh (2018); see docs/mlp.md.
One model per (distance, rounds, basis, noise). Shots with no detection event
take a fast path and predict no flip, as matching does; they are dropped from
training because they carry no information. torch is imported lazily.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
import stim

from ..data import circuit_sha256

FORMAT = "mlDecoder.mlp/1"
DEFAULT_MODEL = dict(hidden_layers=3, width_factor=6, min_width=128)


def build(circuit: stim.Circuit, config: dict) -> "MLPDecoder":
    return MLPDecoder(circuit.num_detectors, circuit.num_observables, circuit_sha256(circuit), config)


class MLPDecoder:
    name = "mlp"

    def __init__(self, num_detectors: int, num_observables: int, circuit_sha: str, config: dict):
        self.num_detectors, self.num_observables, self.circuit_sha256 = num_detectors, num_observables, circuit_sha
        self.config = dict(config)
        model = {**DEFAULT_MODEL, **self.config.get("model", {})}
        if not 2 <= model["hidden_layers"] <= 4:
            raise ValueError("hidden_layers must be 2-4")
        self.hidden_layers = int(model["hidden_layers"])
        self.width = int(max(model["min_width"], round(model["width_factor"] * num_detectors)))
        self.device, self.net, self.training_info = "cpu", None, {}
        self.ignored_options, self.checkpoint_path = {}, None

    # ------------------------------------------------------------ network
    def _ensure_net(self):
        if self.net is None:
            import torch
            layers, width_in = [], self.num_detectors
            for _ in range(self.hidden_layers):
                layers += [torch.nn.Linear(width_in, self.width), torch.nn.ReLU()]
                width_in = self.width
            layers.append(torch.nn.Linear(width_in, self.num_observables))
            self.net = torch.nn.Sequential(*layers)
        return self.net

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self._ensure_net().parameters())

    def weights_sha256(self) -> str:
        digest = hashlib.sha256()
        for key, value in self._ensure_net().state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def to(self, device: str) -> "MLPDecoder":
        self._ensure_net().to(device)
        self.device = device
        return self

    # ------------------------------------------------------------ backend API
    def fit(self, train_spec_or_data, val, *, device: str, seed: int, log=print) -> dict:
        import torch
        from ..train import train_classifier
        torch.manual_seed(seed)          # Initial weights come from the run seed.
        self.net = None
        self._ensure_net()
        self.to(device)
        started = time.perf_counter()
        info = train_classifier(self.net, train=train_spec_or_data, val=val, cfg=self.config.get("train", {}),
                                device=device, seed=seed, log=log, checkpoint_path=self.checkpoint_path)
        info["fit_seconds"] = time.perf_counter() - started
        info["seed"] = seed
        self.training_info = info
        return {k: v for k, v in info.items() if k not in ("history", "config")}

    def _note_ignored(self, enable_correlations):
        if enable_correlations is not None:
            self.ignored_options["enable_correlations"] = ("accepted and ignored: an MLP has no matching graph; "
                                                           f"last value {enable_correlations!r}")

    def _check(self, det_events, enable_correlations):
        self._note_ignored(enable_correlations)
        x = np.asarray(det_events)
        if x.ndim != 2 or x.shape[1] != self.num_detectors:
            raise ValueError(f"Expected (shots, {self.num_detectors}) detection events, got {x.shape}")
        if x.dtype == bool:
            return np.ascontiguousarray(x).view(np.uint8)
        if x.size and (x.min() < 0 or x.max() > 1):
            raise ValueError("Detection events must be binary")
        return np.ascontiguousarray(x, dtype=np.uint8)

    def _logits(self, x):
        import torch
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            return self.net(x.float()).float()

    def _predict_tensor(self, x_u8):
        """x_u8: (N, D) uint8 on the device -> (N, O) bool on the device; empty syndromes predict 0."""
        import torch
        active = (x_u8 != 0).any(dim=1)
        pred = torch.zeros((x_u8.shape[0], self.num_observables), dtype=torch.bool, device=x_u8.device)
        if bool(active.any()):
            pred[active] = self._logits(x_u8[active]) > 0
        return pred

    def decode_batch(self, det_events: np.ndarray, *, enable_correlations=None, chunk: int = 65536) -> np.ndarray:
        """(N, num_obs) bool in shot order. Raises on a wrong width."""
        import torch
        x = self._check(det_events, enable_correlations)
        self._ensure_net().eval()
        out = np.zeros((len(x), self.num_observables), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                xt = torch.from_numpy(x[start:start + chunk]).to(self.device)
                out[start:start + chunk] = self._predict_tensor(xt).cpu().numpy()
        return out

    def logits(self, det_events: np.ndarray, *, chunk: int = 65536) -> np.ndarray:
        """Raw network logits (N, num_obs) float32, for analysis; NaN where the empty-syndrome fast path applies."""
        import torch
        x = self._check(det_events, None)
        self._ensure_net().eval()
        out = np.full((len(x), self.num_observables), np.nan, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                xt = torch.from_numpy(x[start:start + chunk]).to(self.device)
                active = (xt != 0).any(dim=1)
                if bool(active.any()):
                    block = np.full((len(xt), self.num_observables), np.nan, dtype=np.float32)
                    block[active.cpu().numpy()] = self._logits(xt[active]).cpu().numpy()
                    out[start:start + chunk] = block
        return out

    def decode_packed(self, packed: np.ndarray, *, enable_correlations=None, chunk: int = 262144) -> np.ndarray:
        """Bit-packed detection events (little-endian), unpacked on the device."""
        import torch
        from ..train import unpack_bits
        self._note_ignored(enable_correlations)
        if packed.ndim != 2 or packed.shape[1] != (self.num_detectors + 7) // 8:
            raise ValueError(f"Expected packed width {(self.num_detectors + 7) // 8}, got {packed.shape}")
        self._ensure_net().eval()
        out = np.zeros((len(packed), self.num_observables), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(packed), chunk):
                p = torch.from_numpy(np.ascontiguousarray(packed[start:start + chunk])).to(self.device)
                out[start:start + chunk] = self._predict_tensor(unpack_bits(p, self.num_detectors)).cpu().numpy()
        return out

    def profile(self, det_events: np.ndarray, reps: int = 20) -> dict:
        """Median host->device transfer, inference and readback times for one batch."""
        import torch
        x = self._check(det_events, None)
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        parts = dict(transfer_in=[], inference=[], transfer_out=[])
        with torch.no_grad():
            for _ in range(reps + 2):
                sync(); t0 = time.perf_counter()
                xt = torch.from_numpy(x).to(self.device)
                sync(); t1 = time.perf_counter()
                pred = self._predict_tensor(xt)
                sync(); t2 = time.perf_counter()
                pred.cpu().numpy()
                t3 = time.perf_counter()
                for key, value in zip(parts, (t1 - t0, t2 - t1, t3 - t2)):
                    parts[key].append(value)
        return {f"{k}_us": float(1e6 * np.median(v[2:])) for k, v in parts.items()} | dict(batch=len(x), reps=reps)

    def save(self, path) -> None:
        import torch
        torch.save(dict(format=FORMAT, num_detectors=self.num_detectors, num_observables=self.num_observables,
                        circuit_sha256=self.circuit_sha256, config=self.config, width=self.width,
                        hidden_layers=self.hidden_layers, training_info=self.training_info,
                        state_dict={k: v.detach().cpu() for k, v in self._ensure_net().state_dict().items()}), path)

    @classmethod
    def load(cls, path, circuit: stim.Circuit, device: str) -> "MLPDecoder":
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT:
            raise ValueError(f"{path}: not an {FORMAT} file")
        if (payload["num_detectors"], payload["num_observables"]) != (circuit.num_detectors, circuit.num_observables):
            raise ValueError(f"{path}: trained for a different detector/observable count")
        if payload["circuit_sha256"] != circuit_sha256(circuit):
            raise ValueError(f"{path}: trained on a different circuit")
        decoder = cls(payload["num_detectors"], payload["num_observables"], payload["circuit_sha256"], payload["config"])
        decoder._ensure_net().load_state_dict(payload["state_dict"])
        decoder.training_info = payload["training_info"]
        decoder.net.eval()
        for parameter in decoder.net.parameters():
            parameter.requires_grad_(False)
        return decoder.to(device)

    def metadata(self) -> dict:
        import torch
        widths = [self.num_detectors] + [self.width]*self.hidden_layers + [self.num_observables]
        info = {k: v for k, v in self.training_info.items() if k != "config"}
        return dict(name=self.name, architecture="MLP " + "-".join(map(str, widths)) + " (ReLU)",
                    parameters=self.parameter_count(), hidden_layers=self.hidden_layers, width=self.width,
                    config=self.config, torch=torch.__version__, device=self.device,
                    weights_sha256=self.weights_sha256(), training=info, ignored_options=self.ignored_options,
                    inputs="detection events only (no observables, no leakage flags)",
                    empty_syndrome="fast path: predicts no flip without running the network")
