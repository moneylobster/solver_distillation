"""Generate the full-experiment job manifest (one independent bash command
chain per line) consumed by scripts/runner.sh on the cluster.

Each line is self-contained (train -> select_ema -> eval for one run), so any
number of GPUs can work through the manifest concurrently via the runner's
work-stealing locks. Heaviest jobs are emitted first for good load balance.

Contents (paper-required scope):
- 21 main runs:      {cifar10, ffhq, imagenet64} x NFE 4..10 (Table 5 / Fig 3a-c)
- 24 ablation runs:  cifar10, 6 variants x NFE {4,6,8,10}    (Table 3)
- 9 baseline sweeps: {deis, dpmpp, unipc} x 3 datasets        (Fig 3 curves)
- 1 calibration:     iPNDM-4 cifar10 NFE 6 (README of diff-solvers: FID 7.05)

Example:
  uv run python scripts/make_manifest.py            # -> scripts/manifest.txt
  uv run python scripts/make_manifest.py --no-baselines   # cite baselines instead
"""

from pathlib import Path

import click

# Generation/eval batch sizes (NOT the training batch, which is a paper
# hyperparameter). Sampling is per-seed deterministic, so these only affect
# throughput — sized to saturate 80 GB-class GPUs (A100/H100):
# cifar10 ~17 GB, ffhq ~22 GB, imagenet64 ~24 GB.
GEN_BATCH = {"cifar10": 1024, "ffhq": 512, "imagenet64": 256}

ABLATIONS = [  # (tag, train flag, eval solver label)
    ("wo_afs", "--no-afs", "dlms_wo_afs"),
    ("wo_bottleneck", "--no-bottleneck", "dlms_wo_bottleneck"),
    ("wo_high_order", "--no-high-order-init", "dlms_wo_high_order"),
    ("wo_inception", "--no-inception-loss", "dlms_wo_inception"),
    ("wo_time_scaling", "--no-time-scaling", "dlms_wo_time_scaling"),
    ("wo_adaptive_schedule", "--no-adaptive-schedule", "dlms_wo_adaptive_schedule"),
]


def main_chain(ds, nfe):
    b = GEN_BATCH[ds]
    run = f"results/runs/{ds}-nfe{nfe}"
    return (
        f"python scripts/train_dlms.py --dataset {ds} --nfe {nfe}"
        f" && python scripts/select_ema.py --snapshot $(ls -1v {run}/snapshot-*.pt | tail -1) --batch {b}"
        f" && python scripts/eval_dlms.py --dataset {ds} --nfes {nfe} --batch {b}"
    )


def ablation_chain(tag, flag, label, nfe):
    b = GEN_BATCH["cifar10"]
    run = f"results/runs/cifar10-nfe{nfe}-{tag}"
    return (
        f"python scripts/train_dlms.py --dataset cifar10 --nfe {nfe} {flag} --tag {tag}"
        f" && python scripts/select_ema.py --snapshot $(ls -1v {run}/snapshot-*.pt | tail -1) --batch {b}"
        f" && python scripts/eval_dlms.py --dataset cifar10 --nfes {nfe}"
        f" --run-pattern '{{dataset}}-nfe{{nfe}}-{tag}' --solver-label {label} --batch {b}"
    )


def baseline_line(ds, solver, nfes="4-10"):
    return (f"python scripts/run_baselines.py --dataset {ds} --solvers {solver}"
            f" --nfes {nfes} --batch {GEN_BATCH[ds]}")


@click.command()
@click.option("--out", type=click.Path(), default="scripts/manifest.txt", show_default=True)
@click.option("--baselines/--no-baselines", default=True, show_default=True,
              help="Include baseline FID sweeps (alternatively cite diff-solvers README).")
def main(out, baselines):
    lines = []
    # Heaviest first: imagenet64, then ffhq (high NFE first within each).
    for ds in ("imagenet64", "ffhq"):
        for nfe in range(10, 3, -1):
            lines.append(main_chain(ds, nfe))
        if baselines:
            for solver in ("deis", "dpmpp", "unipc"):
                lines.append(baseline_line(ds, solver))
    # CIFAR-10 ablations (Table 3), then mains, then baselines.
    for tag, flag, label in ABLATIONS:
        for nfe in (10, 8, 6, 4):
            lines.append(ablation_chain(tag, flag, label, nfe))
    for nfe in range(10, 3, -1):
        lines.append(main_chain("cifar10", nfe))
    if baselines:
        for solver in ("deis", "dpmpp", "unipc"):
            lines.append(baseline_line("cifar10", solver))
    lines.append(baseline_line("cifar10", "ipndm", nfes="6"))  # pipeline calibration

    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
