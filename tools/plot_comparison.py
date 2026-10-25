"""Render the submission's measured failure curves and Wilson intervals."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main():
    repo = Path(__file__).resolve().parents[1]
    data = json.loads((repo / "benchmarks/garnet/evaluation.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "duquantumIQM"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for basis, ax in zip(("X", "Z"), axes):
        for arm, label, color in (("vanilla", "Vanilla surface code", "#b65c19"),
                                  ("optimized", "Our pipeline", "#14698b")):
            fit = data["measurements"][arm][basis]
            probability = np.asarray(fit["probabilities"])
            interval = np.asarray(fit["wilson_ci95"])
            ax.errorbar(data["rounds"], probability,
                        yerr=np.stack((probability - interval[:, 0], interval[:, 1] - probability)),
                        color=color, marker="o", markersize=6, linewidth=2,
                        elinewidth=1.4, capsize=4, label=label)
        ax.set(title=f"{basis} memory", xlabel="Syndrome rounds", xticks=[1, 2, 3],
               xlim=(.8, 3.2), ylim=(0, .46))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, loc="upper left", fontsize=10)
    axes[0].set_ylabel("Decoded logical failure probability")
    fig.suptitle("Surface-code memory on IQM Garnet", fontsize=16, weight="bold")
    fig.text(.5, .025, "Distance 3 · no mid-circuit reset · 95% Wilson intervals · 54,000 measured shots",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, .065, 1, .94))
    target = repo / "docs/assets/garnet_comparison"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=180)
    svg = target.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
