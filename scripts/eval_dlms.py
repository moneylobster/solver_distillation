"""Evaluate trained DLMS runs: generate images, compute FID, append to
results/dlms.csv (resumable; existing rows are skipped).

Expects run directories created by train_dlms.py. Picks the weight variant
from the run's selection.json (written by select_ema.py) unless --variant is
given. Use --solver-label for ablation rows (Table 3), e.g. dlms_wo_afs.

Examples:
  # Table 5 row / Fig 3 curve for CIFAR-10 (after training NFE 4..10):
  uv run python scripts/eval_dlms.py --dataset cifar10 --nfes 4-10

  # An ablation run trained with --tag wo_afs:
  uv run python scripts/eval_dlms.py --dataset cifar10 --nfes 4,6,8,10 \\
      --run-pattern "{dataset}-nfe{nfe}-wo_afs" --solver-label dlms_wo_afs
"""

import csv
import json
import shutil
from pathlib import Path

import click
import torch

from dlms.bootstrap import RESULTS_DIR, SAMPLES_DIR
from dlms.fid_utils import compute_fid
from dlms.generate import generate_images, parse_int_list
from dlms.models import create_edm_model
from dlms.solver import dlms_sampler
from dlms.train import load_predictor


def latest_snapshot(run_dir):
    snaps = sorted(Path(run_dir).glob("snapshot-*.pt"))
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {run_dir}")
    return snaps[-1]


def load_done(csv_path):
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["solver"], int(row["nfe"])))
    return done


@click.command()
@click.option("--dataset", "dataset_name", type=click.Choice(["cifar10", "ffhq", "imagenet64"]), required=True)
@click.option("--nfes", type=str, default="4-10", show_default=True)
@click.option("--runs-dir", type=click.Path(), default=None, help="Default: results/runs.")
@click.option("--run-pattern", type=str, default="{dataset}-nfe{nfe}", show_default=True)
@click.option("--solver-label", type=str, default="dlms", show_default=True)
@click.option("--variant", type=str, default=None,
              help="Weight variant; default: best from selection.json, else raw.")
@click.option("--seeds", type=str, default="0-49999", show_default=True)
@click.option("--fid-num", type=int, default=50000, show_default=True)
@click.option("--batch", "batch_size", type=int, default=128, show_default=True)
@click.option("--keep-images", is_flag=True)
@click.option("--csv-path", type=click.Path(), default=None, help="Default: results/dlms.csv.")
def main(dataset_name, nfes, runs_dir, run_pattern, solver_label, variant, seeds,
         fid_num, batch_size, keep_images, csv_path):
    device = torch.device("cuda")
    runs_dir = Path(runs_dir) if runs_dir else RESULTS_DIR / "runs"
    csv_path = Path(csv_path) if csv_path else RESULTS_DIR / "dlms.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(csv_path)
    seed_list = parse_int_list(seeds)
    net = None

    for nfe in parse_int_list(nfes):
        if (dataset_name, solver_label, nfe) in done:
            print(f"[skip] {dataset_name} {solver_label} NFE {nfe}")
            continue
        run_dir = runs_dir / run_pattern.format(dataset=dataset_name, nfe=nfe)
        snap = latest_snapshot(run_dir)
        var = variant
        if var is None:
            sel = run_dir / "selection.json"
            var = json.loads(sel.read_text())["best"] if sel.exists() else "raw"
        g_phi = load_predictor(snap, variant=var, device=device)
        if net is None:
            net = create_edm_model(dataset_name, device=device)

        def sample_fn(net, latents, class_labels):
            return dlms_sampler(net, latents, g_phi=g_phi, num_steps=g_phi.num_steps,
                                sigma_min=g_phi.sigma_min, sigma_max=g_phi.sigma_max,
                                afs=g_phi.afs, class_labels=class_labels,
                                max_order=g_phi.max_order)

        outdir = SAMPLES_DIR / dataset_name / f"{solver_label}_nfe{nfe}"
        print(f"[run ] {dataset_name} {solver_label} NFE {nfe} ({snap.name}, {var})")
        generate_images(net, sample_fn, str(outdir), seed_list, batch_size=batch_size, device=device)
        fid = compute_fid(str(outdir), dataset_name, num_expected=fid_num, device=device)
        print(f"[fid ] {dataset_name} {solver_label} NFE {nfe}: {fid:g}")

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["dataset", "solver", "nfe", "fid", "fid_num", "variant", "snapshot"])
            writer.writerow([dataset_name, solver_label, nfe, f"{fid:.4f}", fid_num, var, str(snap)])
        if not keep_images:
            shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
