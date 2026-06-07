"""Pretrained diffusion model loading (EDM models only, single-GPU, no DDP).

Mirrors the EDM branch of amed-solver-main/training/training_loop.py::create_model
without any torch.distributed dependency. Checkpoints are downloaded
automatically into external/diff-sampler/amed-solver-main/src/<dataset>/ via
diff-sampler's own download helper, so all subprojects share one cache.
"""

import pickle

import torch

from .bootstrap import amed_cwd

EDM_DATASETS = ("cifar10", "ffhq", "afhqv2", "imagenet64")

# Bottleneck hook target inside EDM's SongUNet/DhariwalUNet encoder, as used by
# AMED-Plugin (solvers_amed.py::init_hook). Class-conditional ImageNet-64 uses
# the ADM architecture whose deepest encoder block is '8x8_block2'.
def bottleneck_module(net, class_conditional):
    return net.model.enc["8x8_block2" if class_conditional else "8x8_block3"]


def create_edm_model(dataset_name, device=torch.device("cuda")):
    """Load a pretrained EDM model; returns the wrapped denoiser `net`.

    The returned net satisfies: denoised = net(x, sigma, class_labels)
    with x in [-1,1]-ish data space and sigma the EDM noise level.
    """
    if dataset_name not in EDM_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset_name!r}; expected one of {EDM_DATASETS}")

    with amed_cwd():
        # Imported lazily so that bootstrap has set sys.path first.
        from torch_utils.download_util import check_file_by_key

        model_path, _ = check_file_by_key(dataset_name)
        import dnnlib

        with dnnlib.util.open_url(str(model_path)) as f:
            net = pickle.load(f)["ema"].to(device)

    net.sigma_min = 0.002
    net.sigma_max = 80.0
    net.eval().requires_grad_(False)
    return net


class BottleneckHook:
    """Forward hook capturing the channel-pooled UNet bottleneck feature.

    After each net(...) call, `pooled()` returns the latest feature as a
    (B, 8, 8) tensor (mean over channels), matching AMED-Plugin.
    The hook output is detached (gradients never flow into g_phi through h).
    """

    def __init__(self, net, class_conditional):
        self.outputs = []
        self.handle = bottleneck_module(net, class_conditional).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.outputs.append(output.detach())

    def pooled(self):
        return torch.mean(self.outputs[-1], dim=1)

    def clear(self):
        self.outputs.clear()

    def remove(self):
        self.handle.remove()
