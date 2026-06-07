"""Select the best EMA variant of a trained snapshot by a cheap FID proxy
(paper Sec. 3.3 keeps EMAs with half-lives 1/2/3 kimg plus the raw weights
and picks the best-performing set).

Generates `--fid-num` images (default 5k, seeds 0..fid_num-1) per variant and
reports FID against the reference statistics; writes selection.json next to
the snapshot. Rerun the winner at 50k for the final reported number.

Example:
  uv run python scripts/select_ema.py --snapshot results/runs/cifar10-nfe5/snapshot-020000.pt
"""

import json
import shutil
import tempfile
from pathlib import Path

import click
import torch

from dlms.fid_utils import compute_fid
from dlms.generate import generate_images
from dlms.models import create_edm_model
from dlms.solver import dlms_sampler
from dlms.train import load_predictor, snapshot_variants


@click.command()
@click.option("--snapshot", type=click.Path(exists=True), required=True)
@click.option("--fid-num", type=int, default=5000, show_default=True)
@click.option("--batch", "batch_size", type=int, default=64, show_default=True)
@click.option("--keep-images", is_flag=True, help="Keep per-variant image dirs (next to snapshot).")
def main(snapshot, fid_num, batch_size, keep_images):
    device = torch.device("cuda")
    snapshot = Path(snapshot)
    variants = snapshot_variants(snapshot)
    results = {}
    net = None
    for variant in variants:
        g_phi = load_predictor(snapshot, variant=variant, device=device)
        if net is None:
            net = create_edm_model(g_phi.dataset_name, device=device)

        def sample_fn(net, latents, class_labels):
            return dlms_sampler(net, latents, g_phi=g_phi, num_steps=g_phi.num_steps,
                                sigma_min=g_phi.sigma_min, sigma_max=g_phi.sigma_max,
                                afs=g_phi.afs, class_labels=class_labels,
                                max_order=g_phi.max_order)

        if keep_images:
            outdir = snapshot.parent / f"ema-select-{variant}"
            outdir.mkdir(parents=True, exist_ok=True)
        else:
            outdir = Path(tempfile.mkdtemp(prefix=f"dlms-{variant}-"))
        print(f"[{variant}] generating {fid_num} images...")
        generate_images(net, sample_fn, str(outdir), list(range(fid_num)),
                        batch_size=batch_size, device=device)
        fid = compute_fid(str(outdir), g_phi.dataset_name, num_expected=fid_num, device=device)
        results[variant] = fid
        print(f"[{variant}] FID-{fid_num//1000}k: {fid:g}")
        if not keep_images:
            shutil.rmtree(outdir, ignore_errors=True)

    best = min(results, key=results.get)
    out = dict(snapshot=str(snapshot), fid_num=fid_num, fids=results, best=best)
    sel_path = snapshot.parent / "selection.json"
    sel_path.write_text(json.dumps(out, indent=2))
    print(f"Best variant: {best} (FID {results[best]:g}); wrote {sel_path}")


if __name__ == "__main__":
    main()
