"""Generate images with a trained DLMS designer network (paper Algorithm 2).

Examples:
  # 50k images for FID (paper protocol, seeds 0-49999):
  uv run python scripts/generate_dlms.py --snapshot results/runs/cifar10-nfe5/snapshot-020000.pt \\
      --variant ema_2 --seeds 0-49999 --outdir results/samples/cifar10-nfe5-ema2

  # Quick 64-image grid:
  uv run python scripts/generate_dlms.py --snapshot ... --seeds 0-63 --grid grid.png
"""

import click
import torch

from dlms.generate import generate_images, parse_int_list, save_grid
from dlms.models import create_edm_model
from dlms.solver import dlms_sampler
from dlms.train import load_predictor, snapshot_variants


@click.command()
@click.option("--snapshot", type=click.Path(exists=True), required=True)
@click.option("--variant", type=str, default="raw", show_default=True,
              help="Weight set: raw or ema_<halflife> (e.g. ema_1, ema_2, ema_3).")
@click.option("--seeds", type=str, default="0-49999", show_default=True)
@click.option("--batch", "batch_size", type=int, default=64, show_default=True)
@click.option("--outdir", type=click.Path(), default=None, help="PNG output dir (FID layout).")
@click.option("--grid", "grid_path", type=click.Path(), default=None, help="Save one PNG grid instead.")
@click.option("--list-variants", is_flag=True, help="List snapshot weight sets and exit.")
def main(snapshot, variant, seeds, batch_size, outdir, grid_path, list_variants):
    if list_variants:
        print("\n".join(snapshot_variants(snapshot)))
        return
    assert (outdir is None) != (grid_path is None), "provide exactly one of --outdir / --grid"
    device = torch.device("cuda")
    g_phi = load_predictor(snapshot, variant=variant, device=device)
    net = create_edm_model(g_phi.dataset_name, device=device)
    seeds = parse_int_list(seeds)

    def sample_fn(net, latents, class_labels):
        return dlms_sampler(net, latents, g_phi=g_phi, num_steps=g_phi.num_steps,
                            sigma_min=g_phi.sigma_min, sigma_max=g_phi.sigma_max,
                            afs=g_phi.afs, class_labels=class_labels,
                            max_order=g_phi.max_order)

    nfe = g_phi.num_steps - (2 if g_phi.afs else 1)
    print(f"Sampling {len(seeds)} images: dataset={g_phi.dataset_name}, NFE={nfe}, "
          f"num_steps={g_phi.num_steps}, afs={g_phi.afs}, variant={variant}")
    if grid_path is not None:
        from dlms.generate import StackedRandomGenerator

        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn([len(seeds), net.img_channels, net.img_resolution,
                             net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[len(seeds)], device=device)]
        images = sample_fn(net, latents, class_labels)
        save_grid(images, grid_path)
        print(f"Saved grid to {grid_path}")
    else:
        generate_images(net, sample_fn, outdir, seeds, batch_size=batch_size, device=device)
        print(f"Saved {len(seeds)} images to {outdir}")


if __name__ == "__main__":
    main()
