"""Path bootstrap for reusing the diff-sampler toolbox.

The AMED-Solver subproject (external/diff-sampler/amed-solver-main) is not a
package, so we add it to sys.path. Its `torch_utils.download_util` downloads
checkpoints relative to the *current working directory*, so we provide a
context manager that temporarily chdirs into the AMED directory; all
diff-sampler subprojects then share the same `src/` model cache.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFF_SAMPLER = PROJECT_ROOT / "external" / "diff-sampler"
AMED_DIR = DIFF_SAMPLER / "amed-solver-main"
DIFF_SOLVERS_DIR = DIFF_SAMPLER / "diff-solvers-main"
RESULTS_DIR = PROJECT_ROOT / "results"

# Transient generated-image trees (50k PNGs per FID evaluation) go here.
# Override with DLMS_SAMPLES_DIR to keep them off shared/quota'd filesystems —
# on HPC clusters point this at node-local NVMe (each eval generates and
# consumes its images within one job on one node, then deletes them).
SAMPLES_DIR = Path(os.environ.get("DLMS_SAMPLES_DIR", RESULTS_DIR / "samples"))

if not AMED_DIR.is_dir():
    raise ImportError(
        f"diff-sampler submodule not found at {AMED_DIR}. "
        "Run: git submodule update --init --recursive"
    )

if str(AMED_DIR) not in sys.path:
    sys.path.insert(0, str(AMED_DIR))


@contextmanager
def amed_cwd():
    """Temporarily chdir into amed-solver-main (for check_file_by_key downloads)."""
    prev = os.getcwd()
    os.chdir(AMED_DIR)
    try:
        yield
    finally:
        os.chdir(prev)
