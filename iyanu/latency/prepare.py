"""Convert a compiled decoder (.npz) and a held-out test job into the binaries decode_bench.c reads.

    python -m iyanu.latency.prepare <tag>_compiled.npz <test job dir> <basis> <rounds> <out dir>
"""
from __future__ import annotations
import json
import struct
import sys
from pathlib import Path
import numpy as np


def write_table(npz, path):
    z = np.load(npz)
    meta = json.loads(str(z["meta"]))
    n_layers = len(meta["mlp_layers"])
    with open(path, "wb") as f:
        f.write(b"DQLT")
        f.write(struct.pack("<IIQI", 0 if meta["mode"] == "dense" else 1, meta["n_det"], meta["entries"], n_layers))
        if meta["mode"] == "dense":
            f.write(z["dense"].astype(np.uint8).tobytes())
        else:
            if meta["n_det"] > 32:
                raise ValueError("C decoder handles at most 32 detectors")
            f.write(z["keys"].astype("<u4").tobytes())
            f.write(z["bits"].astype(np.uint8).tobytes())
        for i in range(n_layers):
            w, b = z[f"w{i}"].astype("<f4"), z[f"b{i}"].astype("<f4")
            f.write(struct.pack("<II", *w.shape))
            f.write(w.tobytes())
            f.write(b.tobytes())
    return meta


def write_shots(job_dir, basis, rounds, path):
    from iyanu.finetune_hw import load
    from iyanu.compiled import keys
    te = load([job_dir])[(basis, rounds)]
    k = keys(te["dets"]).astype("<u4")
    with open(path, "wb") as f:
        f.write(b"DQSH")
        f.write(struct.pack("<Q", len(k)))
        f.write(k.tobytes())
        f.write(te["obs"][:, 0].astype(np.uint8).tobytes())
    return len(k)


if __name__ == "__main__":
    npz, job, basis, rounds, out = sys.argv[1:6]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = write_table(npz, out/"table.bin")
    n = write_shots(job, basis, int(rounds), out/"shots.bin")
    print(json.dumps(dict(meta, test_shots=n, table=str(out/"table.bin"), shots=str(out/"shots.bin"))))
