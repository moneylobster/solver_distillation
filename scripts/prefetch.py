"""Download every artifact the experiments need at runtime. CPU-only — run
this on a cluster LOGIN node (compute nodes may lack internet access).

Fetches into shared-filesystem caches:
- EDM checkpoints  -> external/diff-sampler/amed-solver-main/src/<dataset>/
- Inception detector -> results/detector_cache/
- FID reference stats -> results/fid_refs/

Example:
  uv run python scripts/prefetch.py                       # all three datasets
  uv run python scripts/prefetch.py --datasets cifar10
"""

import click

from dlms.bootstrap import RESULTS_DIR, amed_cwd
from dlms.fid_utils import load_ref_stats
from dlms.inception import DETECTOR_URL


@click.command()
@click.option("--datasets", type=str, default="cifar10,ffhq,imagenet64", show_default=True)
def main(datasets):
    datasets = [d.strip() for d in datasets.split(",")]

    with amed_cwd():
        from torch_utils.download_util import check_file_by_key

        for ds in datasets:
            path, _ = check_file_by_key(ds)
            print(f"[ok] {ds} checkpoint: {path}")

    import dnnlib  # via bootstrap sys.path

    cache_dir = RESULTS_DIR / "detector_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with dnnlib.util.open_url(DETECTOR_URL, cache_dir=str(cache_dir)):
        pass
    print(f"[ok] Inception detector: {cache_dir}")

    for ds in datasets:
        load_ref_stats(ds)
        print(f"[ok] FID reference stats: {ds}")

    print("All artifacts cached; compute jobs can now run offline.")


if __name__ == "__main__":
    main()
