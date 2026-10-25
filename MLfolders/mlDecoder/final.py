"""The final ML decoders: load the committed weights, decode detection events.

    from mlDecoder import final
    decoder = final.load("garnet", "pA", "Z")       # weights from mlDecoder/weights/, on the CPU by default
    flips = decoder.decode_batch(det_events)        # (shots, 1) bool, in shot order; R from the width
    cascade = final.load("garnet", "pA", "Z", cascade=True)   # LSTM -> Relay-BP below tau -> correlated MWPM
    detail = cascade.decode_detailed(det_events)    # predictions, the layer that decided each shot, p, confidence

``det_events`` are the circuit's detection events, (shots, (d^2 - 1) R) binary: InfoRecord's
``det_events.npy``, or the m2d conversion of the team's ``make_stim_circuit`` record. One model per
(device, patch, basis) serves every R. A model trained on the raw record also needs
``raw_meas=`` (the measurement record, Stim order).

python -m mlDecoder.final list                         # what is committed
python -m mlDecoder.final decode --device garnet --patch pA --basis Z --dets run/det_events.npy [--obs ...]
python -m mlDecoder.final promote --backend rnn_lstm --run <run id> [--groups garnet_pA_Z ...]

``promote`` copies trained models (pre-training or fine-tuning runs) into ``mlDecoder/weights/`` as
float16 and records each file's sha256, size, parameter count and provenance in ``weights/manifest.json``.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np

WEIGHTS = Path(__file__).resolve().parent / "weights"
MANIFEST = WEIGHTS / "manifest.json"


def manifest() -> dict:
    if not MANIFEST.exists():
        return dict(format="mlDecoder.weights/1", models=[])
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def available() -> list[dict]:
    return manifest()["models"]


def entry(device: str, patch: str, basis: str, backend: str | None = None) -> dict:
    found = [m for m in available() if (m["device"], m["patch"], m["basis"]) == (device, patch, basis)
             and (backend is None or m["backend"] == backend)]
    if not found:
        raise KeyError(f"No committed model for {device} {patch} {basis}" + (f" ({backend})" if backend else "")
                       + f"; available: {[(m['device'], m['patch'], m['basis'], m['backend']) for m in available()]}")
    preferred = [m for m in found if m.get("default")]
    return (preferred or found)[0]


def load(device: str, patch: str, basis: str, *, backend: str | None = None, torch_device: str = "cpu",
         cascade: bool = False, tau: float | None = None):
    """The committed decoder for a device patch and basis (the manifest's default, unless a backend is named).

    ``cascade=True`` wraps it as the two-layer decoder (cascade.Cascade): shots whose confidence
    |2p - 1| is below the stored tau (validation-tuned) go to Relay-BP on the stored decoder prior's
    DEM, and correlated MWPM takes the shots Relay-BP does not converge on. ``tau`` overrides it.
    """
    m = entry(device, patch, basis, backend)
    path = WEIGHTS / m["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != m["sha256"]:
        raise ValueError(f"{path}: sha256 {digest[:12]} does not match the manifest's {m['sha256'][:12]}")
    module = importlib.import_module(f".models.{m['backend']}", __package__)
    from . import data
    cls = type(module.build(data.make_circuits(data.DatasetSpec(basis=basis))[0], {}))
    decoder = cls.load(path, None, torch_device)
    decoder.final_entry = m
    if not cascade:
        return decoder
    from .cascade import Cascade
    c = m.get("cascade")
    if not c and tau is None:
        raise KeyError(f"No cascade threshold stored for {device} {patch} {basis}; pass tau=")
    out = Cascade(decoder, (c or {}).get("prior") or m["prior"], tau if tau is not None else c["tau"],
                  (c or {}).get("relay_config", {}))
    out.final_entry = m
    return out


def promote(backend: str, run: str, groups=None, *, default: bool = True, note: str = "") -> list[dict]:
    """Copy a run's trained models into mlDecoder/weights/ (float16) and record them in the manifest."""
    from . import data
    run_dir = data.results_root() / backend / run
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    module = importlib.import_module(f".models.{backend}", __package__)
    WEIGHTS.mkdir(exist_ok=True)
    man = manifest()
    added = []
    for name, g in result["groups"].items():
        if groups and name not in groups or not g.get("model_file"):
            continue
        grp = g["group"]
        cls = type(module.build(data.make_circuits(data.DatasetSpec(basis=grp["basis"]))[0], {}))
        decoder = cls.load(run_dir / g["model_file"], None, "cpu")
        target = f"{grp['device']}/{backend}_{grp['patch']}_{grp['basis']}.pt"
        (WEIGHTS / grp["device"]).mkdir(exist_ok=True)
        decoder.save(WEIGHTS / target, half=True)
        stage = "finetuned" if result.get("mode") == "finetune" else "pretrained"
        prior = result.get("config", {}).get("prior") or dict(noise="calibrated_coherence", p=0.001,
                                                               options=dict(t2="ramsey", readout="asym", qnd="excess"))
        from . import hardware
        prior = dict(noise=prior["noise"], p=float(prior["p"]),
                     calibration=hardware.device_calibration(grp["device"], grp["calset"], grp["patch"],
                                                             window_ns=grp["window_ns"], **prior.get("options", {})))
        tuned = g.get("cascade")
        record = dict(device=grp["device"], patch=grp["patch"], basis=grp["basis"], backend=backend, file=target,
                      sha256=hashlib.sha256((WEIGHTS / target).read_bytes()).hexdigest(),
                      bytes=(WEIGHTS / target).stat().st_size, parameters=decoder.parameter_count(), dtype="float16",
                      stage=stage, source_run=f"{backend}/{run}", source_commit=result["git"]["commit"],
                      calibration_set=grp["calset"], default=default, note=note,
                      inputs=decoder.metadata()["inputs"], prior=prior,
                      cascade=None if not tuned else dict(tau=tuned["tau"], rule=tuned["rule"],
                                                          prior=tuned.get("prior") or prior,
                                                          prior_choice=tuned.get("prior_choice"),
                                                          relay_config=result.get("config", {}).get("cascade", {}).get("relay", {}),
                                                          validation_shots=tuned["validation_shots"]))
        man["models"] = [m for m in man["models"] if not (m["file"] == target)]
        if default:
            for m in man["models"]:
                if (m["device"], m["patch"], m["basis"]) == (grp["device"], grp["patch"], grp["basis"]):
                    m["default"] = False
        man["models"].append(record)
        added.append(record)
    man["models"].sort(key=lambda m: (m["device"], m["patch"], m["basis"], m["backend"]))
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    return added


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    d = sub.add_parser("decode")
    for key in ("device", "patch", "basis"):
        d.add_argument(f"--{key}", required=True)
    d.add_argument("--backend")
    d.add_argument("--dets", required=True, type=Path)
    d.add_argument("--obs", type=Path, help="Observable flips, to report P_L")
    d.add_argument("--out", type=Path)
    p = sub.add_parser("promote")
    p.add_argument("--backend", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--groups", nargs="*")
    p.add_argument("--note", default="")
    p.add_argument("--not-default", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for m in available():
            print(f"{m['device']:8s} {m['patch']:3s} {m['basis']}  {m['backend']:9s} {m['stage']:10s} "
                  f"{m['parameters']:>8,} params  {m['bytes']/1024:7.1f} KiB  {'default' if m.get('default') else ''}  {m['file']}")
        return 0
    if args.command == "promote":
        for m in promote(args.backend, args.run, args.groups, default=not args.not_default, note=args.note):
            print(f"promoted {m['file']} ({m['bytes']/1024:.1f} KiB, {m['stage']})")
        return 0
    decoder = load(args.device, args.patch, args.basis, backend=args.backend)
    dets = np.load(args.dets)
    flips = decoder.decode_batch(dets)
    if args.out:
        np.save(args.out, flips.astype(np.uint8))
    print(f"{len(flips)} shots decoded with {decoder.final_entry['backend']} ({decoder.final_entry['stage']}); "
          f"predicted flip fraction {flips.mean():.4f}")
    if args.obs:
        obs = np.load(args.obs).astype(bool)
        print(f"P_L = {np.any(flips != obs, axis=1).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
