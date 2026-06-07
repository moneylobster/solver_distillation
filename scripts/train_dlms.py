"""Train a DLMS designer network (paper Algorithm 1).

Examples:
  # Paper-faithful CIFAR-10 run at NFE 5 (20k trajectories):
  uv run python scripts/train_dlms.py --dataset cifar10 --nfe 5

  # Quick local smoke run:
  uv run python scripts/train_dlms.py --dataset cifar10 --nfe 4 \\
      --total-traj 200 --batch 16 --batch-gpu 8 --tag smoke

  # Table 3 ablation example:
  uv run python scripts/train_dlms.py --dataset cifar10 --nfe 4 --no-afs --tag wo_afs
"""

import click
import torch

from dlms.bootstrap import RESULTS_DIR
from dlms.config import make_config
from dlms.train import train_dlms


@click.command()
@click.option("--dataset", "dataset_name", type=click.Choice(["cifar10", "ffhq", "imagenet64"]), required=True)
@click.option("--nfe", type=int, required=True, help="Target sampling NFE (paper sweeps 4-10).")
@click.option("--total-traj", type=int, default=None, help="Training trajectories (default: preset, 20k).")
@click.option("--batch", type=int, default=None, help="Trajectories per optimizer step (default: preset).")
@click.option("--batch-gpu", type=int, default=0, help="Microbatch size for gradient accumulation (0 = batch).")
@click.option("--lr", type=float, default=5e-3, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--scale-range", type=float, default=0.2, show_default=True, help="s_n in [1-r, 1+r].")
# Practical-technique / ablation switches (paper Table 3):
@click.option("--afs/--no-afs", default=True, show_default=True)
@click.option("--bottleneck/--no-bottleneck", "use_bottleneck", default=True, show_default=True)
@click.option("--high-order-init/--no-high-order-init", default=True, show_default=True)
@click.option("--inception-loss/--no-inception-loss", default=True, show_default=True)
@click.option("--time-scaling/--no-time-scaling", "learn_scale", default=True, show_default=True)
@click.option("--adaptive-schedule/--no-adaptive-schedule", "learn_schedule", default=True, show_default=True)
@click.option("--outdir", type=click.Path(), default=None, help="Run directory (default: results/runs/<name>).")
@click.option("--tag", type=str, default="", help="Suffix for the default run directory name.")
def main(dataset_name, nfe, total_traj, batch, batch_gpu, lr, seed, scale_range,
         afs, use_bottleneck, high_order_init, inception_loss, learn_scale,
         learn_schedule, outdir, tag):
    overrides = dict(nfe=nfe, batch_gpu=batch_gpu, lr=lr, seed=seed, scale_range=scale_range,
                     afs=afs, use_bottleneck=use_bottleneck, high_order_init=high_order_init,
                     inception_loss=inception_loss, learn_scale=learn_scale,
                     learn_schedule=learn_schedule)
    if total_traj is not None:
        overrides["total_traj"] = total_traj
    if batch is not None:
        overrides["batch"] = batch
    cfg = make_config(dataset_name, **overrides)

    if outdir is None:
        name = f"{dataset_name}-nfe{nfe}" + (f"-{tag}" if tag else "")
        outdir = RESULTS_DIR / "runs" / name
    snap = train_dlms(cfg, outdir, device=torch.device("cuda"))
    print(f"Final snapshot: {snap}")


if __name__ == "__main__":
    main()
