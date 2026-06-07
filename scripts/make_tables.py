"""Assemble Table 5 (DLMS vs handcrafted solvers) and Table 3 (ablations on
CIFAR-10) as Markdown from results/baselines.csv and results/dlms.csv.

Table 3 rows come from ablation evaluations recorded with --solver-label,
e.g. dlms_wo_afs, dlms_wo_bottleneck, dlms_wo_high_order, dlms_wo_inception,
dlms_wo_time_scaling, dlms_wo_adaptive_schedule.

Example:
  uv run python scripts/make_tables.py
"""

import csv
from pathlib import Path

import click

from dlms.bootstrap import RESULTS_DIR

TABLE5_SOLVERS = ["deis", "dpmpp", "unipc", "dlms"]
SOLVER_NAMES = {"deis": "DEIS", "dpmpp": "DPM-Solver++", "unipc": "UniPC", "dlms": "DLMS"}
TABLE3_ROWS = [
    ("dlms", "DLMS"),
    ("dlms_wo_afs", "w/o AFS"),
    ("dlms_wo_bottleneck", "w/o bottleneck feature"),
    ("dlms_wo_high_order", "w/o high-order initialization"),
    ("dlms_wo_inception", "w/o Inception distance"),
    ("dlms_wo_time_scaling", "w/o time scaling"),
    ("dlms_wo_adaptive_schedule", "w/o adaptive time schedule"),
]


def read_rows(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            with open(p) as f:
                rows.extend(csv.DictReader(f))
    return rows


def fid_lookup(rows):
    return {(r["dataset"], r["solver"], int(r["nfe"])): float(r["fid"]) for r in rows}


def md_table(header, body_rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body_rows]
    return "\n".join(lines)


@click.command()
@click.option("--datasets", type=str, default="cifar10,ffhq,imagenet64", show_default=True)
@click.option("--nfes", type=str, default="4,5,6,7,8,9,10", show_default=True)
@click.option("--outdir", type=click.Path(), default=None, help="Default: results/tables.")
def main(datasets, nfes, outdir):
    outdir = Path(outdir) if outdir else RESULTS_DIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    fids = fid_lookup(read_rows([RESULTS_DIR / "baselines.csv", RESULTS_DIR / "dlms.csv"]))
    datasets = [d.strip() for d in datasets.split(",")]
    nfe_list = [int(n) for n in nfes.split(",")]

    def cell(ds, solver, nfe):
        v = fids.get((ds, solver, nfe))
        return f"{v:.2f}" if v is not None else "-"

    # Table 5.
    body = []
    for ds in datasets:
        for solver in TABLE5_SOLVERS:
            body.append([ds, SOLVER_NAMES[solver]] + [cell(ds, solver, n) for n in nfe_list])
    t5 = "### Table 5: FID, DLMS vs handcrafted solvers\n\n" + md_table(
        ["Dataset", "Solver"] + [f"NFE {n}" for n in nfe_list], body)
    (outdir / "table5.md").write_text(t5, encoding="utf-8")

    # Table 3 (CIFAR-10 ablations, NFE 4/6/8/10).
    abl_nfes = [n for n in (4, 6, 8, 10) if n in nfe_list] or [4, 6, 8, 10]
    body = [[name] + [cell("cifar10", label, n) for n in abl_nfes]
            for label, name in TABLE3_ROWS]
    t3 = "### Table 3: ablation study on CIFAR-10\n\n" + md_table(
        ["Variant"] + [f"NFE {n}" for n in abl_nfes], body)
    (outdir / "table3.md").write_text(t3, encoding="utf-8")

    print(t5 + "\n\n" + t3)
    print(f"\nWrote {outdir / 'table5.md'} and {outdir / 'table3.md'}")


if __name__ == "__main__":
    main()
