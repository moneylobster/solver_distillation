"""Plot Figure 3 panels (FID vs NFE, DLMS vs handcrafted baselines) from
results/baselines.csv and results/dlms.csv.

Example:
  uv run python scripts/make_figures.py --datasets cifar10,ffhq,imagenet64
"""

import csv
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dlms.bootstrap import RESULTS_DIR  # noqa: E402

STYLE = {  # paper Fig. 3 color coding
    "deis": dict(label="DEIS", color="red", marker="s"),
    "dpmpp": dict(label="DPM-Solver++", color="orange", marker="^"),
    "unipc": dict(label="UniPC", color="blue", marker="v"),
    "dlms": dict(label="DLMS", color="green", marker="o"),
}
TITLES = {
    "cifar10": "CIFAR10 32×32 (pixel space, unconditional)",
    "ffhq": "FFHQ 64×64 (pixel space, unconditional)",
    "imagenet64": "ImageNet-64 64×64 (pixel space, conditional)",
}


def read_rows(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            with open(p) as f:
                rows.extend(csv.DictReader(f))
    return rows


@click.command()
@click.option("--datasets", type=str, default="cifar10,ffhq,imagenet64", show_default=True)
@click.option("--outdir", type=click.Path(), default=None, help="Default: results/figures.")
def main(datasets, outdir):
    outdir = Path(outdir) if outdir else RESULTS_DIR / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows([RESULTS_DIR / "baselines.csv", RESULTS_DIR / "dlms.csv"])

    datasets = [d.strip() for d in datasets.split(",")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for solver, style in STYLE.items():
            pts = sorted((int(r["nfe"]), float(r["fid"])) for r in rows
                         if r["dataset"] == ds and r["solver"] == solver)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        markersize=5, linewidth=1.5, **style)
        ax.set_xlabel("NFE")
        ax.set_ylabel("FID")
        ax.set_title(TITLES.get(ds, ds), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig3.{ext}", dpi=200)
    print(f"Wrote {outdir / 'fig3.png'}")


if __name__ == "__main__":
    main()
