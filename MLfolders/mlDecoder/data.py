"""Dataset specs, the team's circuits, seeded splits, frozen test sets, detector layout.

Circuits are built exactly as ``challenge.experiment.simulate_memory`` builds them,
and Pauli/leakage shots are sampled through the same calls. Detection events are
stored bit-packed (little-endian bits, as Stim and PyMatching use); ``Dataset.dets``
unpacks them to ``(N, num_detectors)`` uint8 on access.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import stim

from challenge.logical_gates import make_logical_algorithm
from challenge.noise_models import noise_suite, add_noise
from challenge.leakage import sample_leakage
from challenge.internal_helpers import get_qubit_lists, get_meas_order
from . import hardware

DATA_SEED = 20261024           # Project-wide entropy; test/pilot shots depend only on spec.
DATA_VERSION = 1               # Bump to publish new dataset IDs after a sampling change.
SPLITS = {"train": 1, "val": 2, "test": 3, "pilot": 4, "test2": 5}  # test2: independent replication sets
STREAM_CHUNK = 2**31 - 1       # Chunk index reserved for streamed training shots.
LEAKAGE_STREAM_BASE = 10**9   # Chunk indices of fresh leakage training shots (never a pool chunk).
PLACEMENT_TEMPLATE_ROUNDS = 10  # Calibrated models: one fixed patch, as main() uses max(rounds).
DEFAULT_CHUNK = 100_000        # Shots per seeded chunk (Stim samplers).
LEAKAGE_CHUNK = 10_000         # Smaller chunks so leakage sets spread over workers.


@dataclass(frozen=True)
class DatasetSpec:
    distance: int = 3; rounds: int = 3; basis: str = "Z"
    noise: str = "uniform"            # a noise_suite key
    p: float = 0.001
    gate_basis: str = "cz"; no_reset: bool = True
    calibration: str | None = None    # "synthetic:<seed>", or a device patch "inforecord:..." (hardware.DeviceNoise)

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @property
    def slug(self) -> str:
        cal = ("" if self.calibration is None else "_" + hardware.DeviceNoise.parse(self.calibration).slug()
               if hardware.is_device(self.calibration) else "_" + self.calibration.replace(":", ""))
        return (f"d{self.distance}_r{self.rounds}_{self.basis}_{self.noise}_p{self.p:g}"
                f"_{self.gate_basis}{'_nr' if self.no_reset else '_r'}{cal}")

    @classmethod
    def from_dict(cls, values) -> "DatasetSpec":
        return cls(**values)


def results_root() -> Path:
    """``mlDecoder/results`` (git-ignored); MLDECODER_RESULTS overrides it, e.g. in tests."""
    return Path(os.environ.get("MLDECODER_RESULTS", Path(__file__).resolve().parent / "results"))


def sha256_bytes(data) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def circuit_sha256(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


# ---------------------------------------------------------------- circuits

@lru_cache(maxsize=None)
def _calibration(calibration: str):
    if hardware.is_device(calibration):
        return hardware.snapshot(calibration)
    from challenge.calibration import synthetic_snapshot
    kind, _, seed = calibration.partition(":")
    if kind != "synthetic" or not seed.isdigit():
        raise ValueError(f"Unsupported calibration {calibration!r}; use 'synthetic:<seed>' or 'inforecord:...'.")
    return synthetic_snapshot(8, 8, int(seed))


def _ideal(spec: DatasetSpec) -> stim.Circuit:
    return make_logical_algorithm(spec.distance, spec.rounds, (), basis=spec.basis,
        no_reset=spec.no_reset, gate_basis=spec.gate_basis, transversal_gate_set="pauli_h",
        logical_after_round=1, optimize=False).circuit


def _check_device_spec(spec: DatasetSpec) -> None:
    noise = hardware.DeviceNoise.parse(spec.calibration)
    if not spec.noise.startswith("calibrated_"):
        raise ValueError(f"A device calibration goes with a calibrated_* model, not {spec.noise!r} "
                         "(the plain models take calibration=None)")
    if (noise.leak == "readout") != (spec.noise == "calibrated_leakage"):
        raise ValueError("leak=readout goes with calibrated_leakage, and only with it")


@lru_cache(maxsize=None)
def _qubit_map(spec: DatasetSpec):
    """The team's placement, chosen the way experiment.main() chooses it; a device patch's by coordinates."""
    if spec.calibration is None:
        return None
    if hardware.is_device(spec.calibration):
        _check_device_spec(spec)
        return hardware.qubit_map(spec.calibration, _ideal(spec))
    from challenge.build_emerald_qubit_rotated import rank_placements
    from challenge.surface_code import make_stim_circuit
    template = make_stim_circuit(spec.distance, max(PLACEMENT_TEMPLATE_ROUNDS, spec.rounds),
                                 gate_basis=spec.gate_basis)
    return rank_placements(template, calibration=_calibration(spec.calibration), rows=8, cols=8)[0].stim_to_physical


def noise_model(spec: DatasetSpec):
    models = noise_suite(spec.p, _calibration(spec.calibration) if spec.calibration else None)
    if spec.noise not in models:
        raise ValueError(f"Unknown noise {spec.noise!r}; available: {', '.join(models)}")
    return models[spec.noise]


@lru_cache(maxsize=None)
def uses_leakage_sampler(spec: DatasetSpec) -> bool:
    """The test simulate_memory applies to choose its leakage sampling path."""
    model, qubit_map = noise_model(spec), _qubit_map(spec)
    return bool(model.leakage or (model.calibration and
        any(model.calibration.qubits[p].leakage_per_gate for p in qubit_map.values())))


def default_chunk(spec: DatasetSpec) -> int:
    return LEAKAGE_CHUNK if uses_leakage_sampler(spec) else DEFAULT_CHUNK


@lru_cache(maxsize=256)
def _circuits(spec: DatasetSpec):
    ideal = _ideal(spec)
    model, mapping = noise_model(spec), _qubit_map(spec)
    noisy = add_noise(ideal, model, qubit_map=mapping)
    if hardware.is_device(spec.calibration):
        noisy = hardware.readout_window(noisy, model, mapping, spec.calibration)  # mlDecoder's extension.
    return ideal, noisy


@lru_cache(maxsize=256)
def _asymmetric(spec: DatasetSpec):
    """readout=asym: (the noisy circuit without readout flips, P(1|0) and P(0|1) per measurement), else None."""
    if not hardware.is_device(spec.calibration) or hardware.DeviceNoise.parse(spec.calibration).readout != "asym":
        return None
    ideal, noisy = _circuits(spec)
    p10, p01 = hardware.readout_flip_table(spec.calibration, _qubit_map(spec), ideal)
    return hardware.strip_readout_flips(noisy), p10, p01


def _flip_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(0xA5,)))


def make_circuits(spec: DatasetSpec) -> tuple[stim.Circuit, stim.Circuit]:
    """(ideal, noisy), as simulate_memory builds them (no logical operations).

    A device calibration adds mlDecoder's readout-window extension (hardware.readout_window)
    after the team's add_noise; the ideal circuit is the team's either way.
    """
    ideal, noisy = _circuits(spec)
    return ideal.copy(), noisy.copy()


# ---------------------------------------------------------------- seeds and sampling

def _spec_words(spec: DatasetSpec) -> tuple[int, ...]:
    digest = bytes.fromhex(spec.sha())
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4))


def seed_sequence(spec: DatasetSpec, split: str, seed: int = 0, chunk: int = 0) -> np.random.SeedSequence:
    """Independent stream per (spec, split, run seed, chunk)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {tuple(SPLITS)}")
    if split in {"test", "test2", "pilot"} and seed != 0:
        raise ValueError("Test and pilot shots are canonical: seed must be 0.")
    return np.random.SeedSequence(DATA_SEED, spawn_key=(*_spec_words(spec), DATA_VERSION,
                                                        SPLITS[split], int(seed), int(chunk)))


def chunk_seed(spec: DatasetSpec, split: str, seed: int = 0, chunk: int = 0) -> int:
    return int(seed_sequence(spec, split, seed, chunk).generate_state(1, np.uint64)[0] >> np.uint64(1))


def pack(bits: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(bits, dtype=np.uint8), axis=1, bitorder="little")


def unpack(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1, count=width, bitorder="little")


def sample_raw(spec: DatasetSpec, n: int, seed: int):
    """One seeded draw of the raw measurement record (N, M) bool, plus leakage diagnostics or None.

    readout=asym specs sample without readout flips, then flip each bit by the value read.
    """
    ideal, noisy = _circuits(spec)
    asym = _asymmetric(spec)
    circuit = asym[0] if asym else noisy
    if uses_leakage_sampler(spec):
        raw, diagnostic = sample_leakage(circuit, noise_model(spec), n, seed=seed, qubit_map=_qubit_map(spec))
    else:
        raw, diagnostic = circuit.compile_sampler(seed=seed).sample(n), None
    if asym:
        hardware.apply_readout_flips(raw, asym[1], asym[2], _flip_rng(seed))
    return raw, diagnostic


def sample_shots(spec: DatasetSpec, n: int, seed: int, *, packed: bool = True):
    """One seeded draw: (dets, obs, leakage diagnostics or None). Top level for workers."""
    ideal, noisy = _circuits(spec)
    if uses_leakage_sampler(spec) or _asymmetric(spec):
        raw, diagnostic = sample_raw(spec, n, seed)
        dets, obs = ideal.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    else:
        diagnostic = None
        dets, obs = noisy.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    dets, obs = dets.view(np.uint8), obs.view(np.uint8)
    return (pack(dets) if packed else dets), obs, diagnostic


def sample_with_raw(spec: DatasetSpec, n: int, seed: int):
    """One seeded draw through the raw record: (packed dets, obs, diagnostics, packed raw). Top level for workers."""
    ideal, _ = _circuits(spec)
    raw, diagnostic = sample_raw(spec, n, seed)
    dets, obs = ideal.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    return pack(dets.view(np.uint8)), obs.view(np.uint8), diagnostic, pack(raw.view(np.uint8))


def sample_batch(spec: DatasetSpec, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Fresh shots (unpacked uint8), seeded from ``rng`` (streamed training)."""
    dets, obs, _ = sample_shots(spec, n, int(rng.integers(0, 2**63)), packed=False)
    return dets, obs


class Stream:
    """One seeded Stim sampler drawing fresh Pauli-model shots, bit-packed.

    readout=asym specs draw measurement records, flip them by the value read, and convert.
    """

    def __init__(self, spec: DatasetSpec, seed: int = 0, split: str = "train"):
        if uses_leakage_sampler(spec):
            raise ValueError("The leakage sampler is slow: train from a pre-generated pool.")
        self.spec, self.seed = spec, seed
        ideal, noisy = _circuits(spec)
        self.num_observables = noisy.num_observables
        stream_seed = chunk_seed(spec, split, seed, STREAM_CHUNK)
        self.asym = _asymmetric(spec)
        if self.asym:
            self.sampler = self.asym[0].compile_sampler(seed=stream_seed)
            self.converter = ideal.compile_m2d_converter()
            self.rng = _flip_rng(stream_seed)
        else:
            self.sampler = noisy.compile_detector_sampler(seed=stream_seed)

    def next(self, n: int):
        """(packed dets, unpacked uint8 obs)."""
        if self.asym:
            raw = hardware.apply_readout_flips(self.sampler.sample(n), self.asym[1], self.asym[2], self.rng)
            dets, obs = self.converter.convert(measurements=raw, separate_observables=True)
            return pack(dets.view(np.uint8)), obs.view(np.uint8)
        dets, obs = self.sampler.sample(n, separate_observables=True, bit_packed=True)
        return dets, unpack(obs, self.num_observables)


# ---------------------------------------------------------------- datasets

@dataclass
class Dataset:
    spec: DatasetSpec; split: str     # "train" | "val" | "test"
    dets_packed: np.ndarray           # (N, ceil(num_detectors/8)) uint8, little-endian bits
    obs: np.ndarray                   # (N, num_observables) uint8
    detector_coords: np.ndarray       # (num_detectors, 3): x, y, t
    circuit: stim.Circuit             # the noisy circuit (for real shots: the decoder prior's)
    manifest: dict
    raw_packed: np.ndarray | None = None  # (N, ceil(num_measurements/8)) raw record, when kept (real shots)

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    @property
    def raw(self) -> np.ndarray | None:
        """(N, num_measurements) uint8 raw measurements, or None."""
        return None if self.raw_packed is None else unpack(self.raw_packed, self.circuit.num_measurements)

    @property
    def shots(self) -> int:
        return len(self.obs)

    @property
    def dets(self) -> np.ndarray:
        """(N, num_detectors) uint8, unpacked on every access."""
        return unpack(self.dets_packed, self.num_detectors)

    def batches(self, size: int):
        for start in range(0, self.shots, size):
            stop = min(start + size, self.shots)
            yield start, stop, self.dets_packed[start:stop]


def _coords(circuit: stim.Circuit) -> np.ndarray:
    coords = circuit.get_detector_coordinates()
    return np.array([coords[k][:3] for k in range(circuit.num_detectors)], dtype=np.float32)


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("stim", "numpy", "pymatching", "scipy")}


def _map_chunks(spec, sizes, seeds, workers, pool=None, keep_raw=False):
    """Sample chunks in order; leakage chunks go to worker processes when allowed."""
    fn = sample_with_raw if keep_raw else sample_shots
    if pool is None and uses_leakage_sampler(spec) and workers > 1 and len(sizes) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(sizes))) as own:
            return list(own.map(fn, [spec]*len(sizes), sizes, seeds))
    if pool is not None:
        return list(pool.map(fn, [spec]*len(sizes), sizes, seeds))
    return [fn(spec, n, s) for n, s in zip(sizes, seeds)]


def make_dataset(spec: DatasetSpec, split: str, shots: int, *, seed: int | None = None,
                 chunk_shots: int | None = None, workers: int = 1, log=None, keep_raw: bool = False) -> Dataset:
    """Generate ``shots`` shots in seeded chunks; leakage chunks may use worker processes.

    ``keep_raw`` samples every shot through its raw measurement record (sample_with_raw) and keeps
    it, for decoders that read the full measured record; it is part of the dataset's identity.
    """
    seed = 0 if seed is None else int(seed)
    chunk_shots = chunk_shots or default_chunk(spec)
    if shots < 1 or chunk_shots < 1:
        raise ValueError("shots and chunk_shots must be positive")
    ideal, noisy = make_circuits(spec)
    sizes = [min(chunk_shots, shots - start) for start in range(0, shots, chunk_shots)]
    seeds = [chunk_seed(spec, split, seed, k) for k in range(len(sizes))]
    started = time.perf_counter()
    leakage = uses_leakage_sampler(spec)
    parts = _map_chunks(spec, sizes, seeds, workers, keep_raw=keep_raw)
    dets = np.concatenate([p[0] for p in parts])
    obs = np.concatenate([p[1] for p in parts])
    raw = np.concatenate([p[3] for p in parts]) if keep_raw else None
    elapsed = time.perf_counter() - started
    manifest = dict(dataset_id=dataset_id(spec, split, shots, seed, chunk_shots, keep_raw=keep_raw),
        spec=asdict(spec), spec_sha256=spec.sha(), split=split, shots=shots, seed=seed,
        data_seed=DATA_SEED, data_version=DATA_VERSION, chunk_shots=chunk_shots, chunk_seeds=seeds,
        sampler=("raw record (sample_raw), then ideal.compile_m2d_converter().convert; " if keep_raw else "")
                + ("challenge.leakage.sample_leakage, then ideal.compile_m2d_converter().convert"
                   if leakage else "noisy.compile_detector_sampler(seed).sample" if not keep_raw
                   else "noisy.compile_sampler(seed).sample")
                + (" (readout=asym: no readout flips in the sampled circuit; each bit flipped by P(1|0) or P(0|1) "
                   "of the value read, then the ideal m2d converter)" if _asymmetric(spec) else ""),
        num_detectors=noisy.num_detectors, num_observables=noisy.num_observables,
        dets_encoding="np.packbits(axis=1, bitorder='little')",
        dets_sha256=sha256_bytes(dets), obs_sha256=sha256_bytes(obs),
        circuit_sha256=circuit_sha256(noisy), ideal_sha256=circuit_sha256(ideal),
        noise_model={k: v for k, v in asdict(noise_model(spec)).items() if k != "calibration"},
        qubit_map=_qubit_map(spec), versions=_versions(), generation_seconds=elapsed,
        workers=workers if leakage else 1, hidden_flags_given_to_decoder=False, keep_raw=keep_raw,
        raw_sha256=None if raw is None else sha256_bytes(raw), num_measurements=noisy.num_measurements)
    if leakage and parts[0][2] is not None:
        diags = [p[2] for p in parts]
        manifest["leakage"] = dict(model=diags[0]["model"],
            leakage_entries=sum(d["leakage_entries"] for d in diags),
            leaked_measurements=sum(d["leaked_measurements"] for d in diags),
            mean_leaked_fraction=sum(n*d["mean_leaked_fraction"] for n, d in zip(sizes, diags))/shots,
            final_leaked_fraction=sum(n*d["final_leaked_fraction"] for n, d in zip(sizes, diags))/shots)
    if log:
        log(f"  generated {split} {spec.slug}: {shots:,} shots in {elapsed:.1f} s")
    return Dataset(spec, split, dets, obs, _coords(noisy), noisy, manifest, raw)


def dataset_id(spec: DatasetSpec, split: str, shots: int, seed: int = 0, chunk_shots: int | None = None, *,
               keep_raw: bool = False) -> str:
    fields = dict(spec=spec.canonical(), split=split, shots=shots, seed=seed,
                  chunk_shots=chunk_shots or default_chunk(spec), data_seed=DATA_SEED, data_version=DATA_VERSION)
    if keep_raw:
        fields["keep_raw"] = True       # Absent (not False) for plain sets, so stage-0 IDs are unchanged.
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]


def dataset_dir(spec, split, shots, seed=0, chunk_shots=None, *, keep_raw=False) -> Path:
    return results_root() / "datasets" / dataset_id(spec, split, shots, seed, chunk_shots, keep_raw=keep_raw)


def save_dataset(dataset: Dataset, directory: Path) -> None:
    tmp = directory.with_name(directory.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    arrays = dict(dets_packed=dataset.dets_packed, obs=dataset.obs)
    if dataset.raw_packed is not None:
        arrays["raw_packed"] = dataset.raw_packed
    np.savez_compressed(tmp / "data.npz", **arrays)
    (tmp / "manifest.json").write_text(json.dumps(dataset.manifest, indent=2) + "\n", encoding="utf-8")
    if directory.exists():
        shutil.rmtree(tmp)
        raise FileExistsError(f"{directory} already exists; shared datasets are read-only.")
    tmp.rename(directory)


def load_dataset(directory: Path, spec: DatasetSpec) -> Dataset:
    """Reload a saved dataset; refuse any spec, array or circuit hash mismatch."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["spec"] != asdict(spec):
        raise ValueError(f"{directory}: manifest spec {manifest['spec']} != {asdict(spec)}")
    with np.load(directory / "data.npz") as data:
        dets, obs = data["dets_packed"], data["obs"]
        raw = data["raw_packed"] if "raw_packed" in data.files else None
    _, noisy = make_circuits(spec)
    checks = dict(dets_sha256=sha256_bytes(dets), obs_sha256=sha256_bytes(obs),
                  circuit_sha256=circuit_sha256(noisy))
    if manifest.get("keep_raw"):
        checks["raw_sha256"] = None if raw is None else sha256_bytes(raw)
    for key, value in checks.items():
        if manifest[key] != value:
            raise ValueError(f"{directory}: {key} mismatch (manifest {manifest[key][:12]}, now {value[:12]})")
    if len(obs) != manifest["shots"]:
        raise ValueError(f"{directory}: shot count mismatch")
    return Dataset(spec, manifest["split"], dets, obs, _coords(noisy), noisy, manifest, raw)


def cached_dataset(spec: DatasetSpec, split: str, shots: int, *, seed: int = 0,
                   chunk_shots: int | None = None, workers: int = 1, log=None, keep_raw: bool = False) -> Dataset:
    directory = dataset_dir(spec, split, shots, seed, chunk_shots, keep_raw=keep_raw)
    if (directory / "manifest.json").exists():
        return load_dataset(directory, spec)
    dataset = make_dataset(spec, split, shots, seed=seed, chunk_shots=chunk_shots, workers=workers, log=log,
                           keep_raw=keep_raw)
    save_dataset(dataset, directory)
    return dataset


def frozen_test_set(spec: DatasetSpec, shots: int, *, workers: int = 1, log=None, keep_raw: bool = False) -> Dataset:
    """Generated once, then frozen under results/datasets/<sha12>/ (data.npz + manifest.json)."""
    return cached_dataset(spec, "test", shots, seed=0, workers=workers, log=log, keep_raw=keep_raw)


EXTERNAL_SPLITS = ("finetune", "validation", "test")


def external_dataset(spec: DatasetSpec, dets: np.ndarray, obs: np.ndarray, manifest: dict, *,
                     split: str = "test", detector_table: dict | None = None, raw_meas: np.ndarray | None = None,
                     source: str = "external") -> Dataset:
    """Wrap externally measured shots (e.g. hardware) as a Dataset, checked against the team's circuit.

    Splits: ``finetune`` (training), ``validation`` (the fine-tuning side's held-out shots: early
    stopping only, never a test) and ``test`` (a whole time-separated batch, never trained or
    early-stopped on). ``detector_table`` (if given) must match detector_layout's rounds and (x, y).
    ``spec`` names the decoder prior: its noisy circuit is the Dataset's circuit.
    """
    if split not in EXTERNAL_SPLITS:
        raise ValueError(f"External shots are {', '.join(EXTERNAL_SPLITS)} data only, not {split!r}")
    _, noisy = make_circuits(spec)
    dets, obs = np.asarray(dets, dtype=np.uint8), np.asarray(obs, dtype=np.uint8)
    if dets.ndim != 2 or dets.shape[1] != noisy.num_detectors or obs.shape != (len(dets), noisy.num_observables):
        raise ValueError(f"Expected dets (N, {noisy.num_detectors}) and obs (N, {noisy.num_observables})")
    if detector_table is not None:
        layout = detector_layout(noisy)
        for key in ("round", "x", "y"):
            if not np.array_equal(np.asarray(detector_table[key], dtype=float), layout[key].astype(float)):
                raise ValueError(f"External detector table disagrees with the circuit on {key!r}")
    packed = pack(dets)
    raw_packed = None
    if raw_meas is not None:
        raw_meas = np.asarray(raw_meas, dtype=np.uint8)
        if raw_meas.shape != (len(dets), noisy.num_measurements):
            raise ValueError(f"Expected raw_meas (N, {noisy.num_measurements})")
        raw_packed = pack(raw_meas)
    meta = dict(manifest, source=source, split=split, spec=asdict(spec), shots=len(dets),
                dets_sha256=sha256_bytes(packed), obs_sha256=sha256_bytes(obs), circuit_sha256=circuit_sha256(noisy),
                raw_meas_sha256=None if raw_packed is None else sha256_bytes(raw_packed))
    meta.setdefault("dataset_id", "ext-" + hashlib.sha256((meta["dets_sha256"] + meta["obs_sha256"] + split
                                                            + spec.canonical()).encode()).hexdigest()[:12])
    return Dataset(spec, split, packed, obs, _coords(noisy), noisy, meta, raw_packed)


def inforecord_dataset(spec: DatasetSpec, run: "hardware.Run", split: str) -> Dataset:
    """One split of an InfoRecord run (hardware.load_run) as a Dataset, in shot order.

    The run's circuit must equal the team's for the spec (detectors, observables, measurement
    order), and its device, patch and placement the spec's calibration (hardware.check_run_against).
    The spec names the decoder prior (e.g. calibrated_coherence on the device patch).
    """
    run_spec = run.spec
    if run_spec.get("family") != "surface_memory" or not run_spec.get("no_reset", False):
        raise ValueError(f"{run.path}: not a no-reset surface-code memory run")
    if (int(run_spec["d"]), int(run_spec["rounds"]), run_spec["basis"]) != (spec.distance, spec.rounds, spec.basis):
        raise ValueError(f"{run.path}: d/rounds/basis {run_spec['d']}/{run_spec['rounds']}/{run_spec['basis']} "
                         f"!= spec {spec.distance}/{spec.rounds}/{spec.basis}")
    ideal, noisy = make_circuits(spec)
    hardware.check_run_against(run, ideal, spec.calibration)
    mask = run.split == split
    if not mask.any():
        raise ValueError(f"{run.path}: no {split} shots")
    table = dict(round=[int(d["round"]) for d in run.detectors], x=[float(d["x"]) for d in run.detectors],
                 y=[float(d["y"]) for d in run.detectors])
    meta = dict(run.describe(), label=run.label, run_manifest_sha256=hashlib.sha256(
                    (run.path / "manifest.json").read_bytes()).hexdigest(),
                shot_indices=np.flatnonzero(mask).tolist(), decoder_prior=spec.slug,
                hidden_flags_given_to_decoder=False)
    meta["dataset_id"] = f"ir-{run.manifest['run_id']}-{split}"
    return external_dataset(spec, run.dets[mask], run.obs[mask], meta, split=split, detector_table=table,
                            raw_meas=None if run.raw_meas is None else run.raw_meas[mask], source=run.label)


# ---------------------------------------------------------------- test-set size

def pilot(spec: DatasetSpec, *, max_shots: int, target_failures: int = 30, chunk_shots: int | None = None,
          workers: int = 1, log=None) -> dict:
    """Correlated-MWPM pilot P_L on the canonical pilot stream, cached per spec and budget."""
    chunk_shots = chunk_shots or default_chunk(spec)
    path = results_root() / "datasets" / "pilots" / f"{spec.sha()[:12]}_{max_shots}_{target_failures}_{chunk_shots}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from challenge.decoding import make_matcher
    _, noisy = make_circuits(spec)
    matcher = make_matcher(noisy, correlated=True)
    shots = failures = chunk = 0
    started = time.perf_counter()
    parallel = uses_leakage_sampler(spec) and workers > 1
    pool = ProcessPoolExecutor(max_workers=workers) if parallel else None
    try:
        while shots < max_shots and failures < target_failures:
            sizes = [min(chunk_shots, max_shots - shots - k*chunk_shots) for k in range(workers if parallel else 1)]
            sizes = [n for n in sizes if n > 0]
            seeds = [chunk_seed(spec, "pilot", 0, chunk + k) for k in range(len(sizes))]
            for dets, obs, _ in _map_chunks(spec, sizes, seeds, 1, pool):
                pred = matcher.decode_batch(dets, bit_packed_shots=True, enable_correlations=True)
                failures += int(np.any(pred != obs, axis=1).sum())
                shots += len(obs)
            chunk += len(sizes)
    finally:
        if pool:
            pool.shutdown()
    result = dict(spec=asdict(spec), shots=shots, failures=failures, p_hat=failures/shots,
                  max_shots=max_shots, target_failures=target_failures, chunk_shots=chunk_shots,
                  seconds=time.perf_counter() - started, decoder="correlated MWPM (challenge.decoding.make_matcher)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if log:
        log(f"  pilot {spec.slug}: {failures}/{shots:,} = {failures/shots:.3g}")
    return result


def choose_test_shots(pilot_result: dict, *, target_failures: int = 100, safety: float = 1.25,
                      min_shots: int = 100_000, max_shots: int = 10_000_000) -> dict:
    """N so that MWPM expects >= target failures at the pilot rate, capped at max_shots."""
    p = pilot_result["p_hat"]
    wanted = max_shots if p == 0 else math.ceil(safety * target_failures / p)
    shots = int(min(max_shots, max(min_shots, wanted)))
    rounded = int(min(max_shots, math.ceil(shots / DEFAULT_CHUNK) * DEFAULT_CHUNK))
    return dict(shots=rounded, wanted=wanted, cap_binds=p == 0 or wanted > max_shots, expected_failures=rounded*p,
                target_failures=target_failures, safety=safety, min_shots=min_shots, max_shots=max_shots)


# ---------------------------------------------------------------- detector layout

@lru_cache(maxsize=64)
def _ancilla_types(distance: int, basis: str) -> dict:
    """Ancilla qubit -> 'X'/'Z' from the team's CX-basis generator (control = X check)."""
    from challenge.surface_code import make_stim_circuit
    template = make_stim_circuit(distance, 1, basis, True, gate_basis="cx")
    _, anc = get_qubit_lists(template)
    coords = template.get_final_qubit_coordinates()
    types = {}
    for inst in template.flattened():
        if inst.name == "CX":
            qs = [t.value for t in inst.targets_copy()]
            for a, b in zip(qs[::2], qs[1::2]):
                for q, kind in ((a, "X"), (b, "Z")):
                    if q in anc and types.setdefault(q, kind) != kind:
                        raise ValueError(f"Ancilla {q} acts as both X and Z check")
    if set(types) != set(anc):
        raise ValueError("Some ancillas have no CX interaction in the template")
    return {q: (kind, tuple(coords[q])) for q, kind in types.items()}


def detector_layout(circuit: stim.Circuit) -> dict:
    """Per detector: round (final data detectors at t = rounds), stabilizer id, X/Z type, (x, y).

    Read from the circuit's own measurement records and coordinates; X/Z check types
    come from the team's CX-basis generator for the same distance (same qubit indices).
    """
    data, anc = get_qubit_lists(circuit)
    data_set, anc_set = set(data), set(anc)
    distance = math.isqrt(len(anc) + 1)
    if distance**2 - 1 != len(anc) or len(data) != distance**2:
        raise ValueError("Not a rotated surface-code memory circuit")
    meas_qubit = get_meas_order(circuit)
    meas_round, counts = [], {}
    for q in meas_qubit:
        meas_round.append(counts.get(q, 0))
        counts[q] = counts.get(q, 0) + 1
    rounds = max(counts[q] for q in anc)
    qcoords = circuit.get_final_qubit_coordinates()
    types_z, types_x = _ancilla_types(distance, "Z"), _ancilla_types(distance, "X")
    if types_z != types_x:
        raise ValueError("X- and Z-memory templates disagree on check types")
    for q in anc:
        if q not in types_z or types_z[q][1] != tuple(qcoords[q]):
            raise ValueError(f"Ancilla {q} is not where the generator's template puts it")
    types_z = {q: kind for q, (kind, _) in types_z.items()}
    stabilizers = sorted(anc)
    stab_index = {q: i for i, q in enumerate(stabilizers)}
    coords = circuit.get_detector_coordinates()
    n, det = 0, 0
    rnd, stab, final = (np.empty(circuit.num_detectors, dtype=np.int32) for _ in range(3))
    for inst in circuit.flattened():
        if inst.name in {"M", "MX", "MY", "MR", "MRX", "MRY"}:
            n += sum(t.is_qubit_target for t in inst.targets_copy())
        elif inst.name == "DETECTOR":
            recs = [n + t.value for t in inst.targets_copy()]
            qs = {meas_qubit[r] for r in recs}
            checks = qs & anc_set
            if len(checks) != 1 or qs - anc_set - data_set:
                raise ValueError(f"Detector {det} spans checks {sorted(checks)}")
            (q,) = checks
            is_final = bool(qs & data_set)
            rnd[det] = rounds if is_final else max(meas_round[r] for r in recs if meas_qubit[r] == q)
            stab[det], final[det] = stab_index[q], is_final
            if tuple(coords[det][:2]) != tuple(qcoords[q]):
                raise ValueError(f"Detector {det} coordinates {coords[det]} != check {q} at {qcoords[q]}")
            det += 1
    if det != circuit.num_detectors:
        raise ValueError("Detector count mismatch")
    grid = np.full((rounds + 1, len(stabilizers)), -1, dtype=np.int32)
    for k in range(det):
        if grid[rnd[k], stab[k]] != -1:
            raise ValueError(f"Two detectors at round {rnd[k]}, check {stab[k]}")
        grid[rnd[k], stab[k]] = k
    kinds = np.array([types_z[stabilizers[s]] for s in stab])
    xy = np.array([qcoords[stabilizers[s]] for s in stab], dtype=np.float32)
    return dict(distance=distance, rounds=rounds, num_detectors=det, num_stabilizers=len(stabilizers),
        round=rnd, stabilizer=stab, is_final=final.astype(bool), type=kinds, is_z=kinds == "Z",
        x=xy[:, 0], y=xy[:, 1], grid=grid,
        stabilizers=[dict(id=i, qubit=q, x=float(qcoords[q][0]), y=float(qcoords[q][1]), type=types_z[q])
                     for i, q in enumerate(stabilizers)])
