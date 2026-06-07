"""Compute FID for a directory of generated PNGs against the official EDM
reference statistics (downloaded automatically).

Example:
  uv run python scripts/compute_fid.py --images results/samples/cifar10-nfe5-ema2 \\
      --dataset cifar10 --fid-num 50000
"""

import json

import click
import torch

from dlms.fid_utils import FID_REFS, compute_fid


@click.command()
@click.option("--images", "image_dir", type=click.Path(exists=True), required=True)
@click.option("--dataset", "dataset_or_ref", type=str, required=True,
              help=f"Dataset key ({', '.join(FID_REFS)}) or a reference-npz path/URL.")
@click.option("--fid-num", type=int, default=50000, show_default=True,
              help="Number of images to use (paper: 50k; lower for drafts).")
@click.option("--batch", "batch_size", type=int, default=250, show_default=True)
@click.option("--out", "out_path", type=click.Path(), default=None, help="Optional JSON output file.")
def main(image_dir, dataset_or_ref, fid_num, batch_size, out_path):
    fid = compute_fid(image_dir, dataset_or_ref, num_expected=fid_num,
                      batch_size=batch_size, device=torch.device("cuda"))
    print(f"FID: {fid:g}")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(dict(images=str(image_dir), dataset=dataset_or_ref,
                           num=fid_num, fid=fid), f, indent=2)


if __name__ == "__main__":
    main()
