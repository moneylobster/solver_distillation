"""Deterministic image generation for FID evaluation.

Faithful single-GPU port of amed-solver-main/sample.py: per-seed stacked
random generators (so results are independent of batch size), identical
latent/label sampling, identical uint8 quantization, identical PNG layout
(1000-seed subdirectories). The paper evaluates FID on a fixed seed set
(0..49999) for fairness across solvers.
"""

import os

import PIL.Image
import torch
from tqdm import tqdm


class StackedRandomGenerator:
    """Wrapper for torch.Generator giving each sample its own seed-derived
    generator (verbatim from diff-sampler's sample.py)."""

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s):
    """'0-4,9' -> [0, 1, 2, 3, 4, 9]."""
    if isinstance(s, list):
        return s
    import re

    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


@torch.no_grad()
def generate_images(net, sample_fn, outdir, seeds, batch_size=64,
                    device=torch.device("cuda"), subdirs=True, verbose=True):
    """Generate images for `seeds` and write PNGs under `outdir`.

    sample_fn(net, latents, class_labels) -> images in model space (~[-1,1]).
    """
    os.makedirs(outdir, exist_ok=True)
    batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
    for batch_seeds in tqdm(batches, unit="batch", disable=not verbose):
        B = len(batch_seeds)
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([B, net.img_channels, net.img_resolution, net.img_resolution],
                            device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[B], device=device)]

        images = sample_fn(net, latents, class_labels)

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f"{seed - seed % 1000:06d}") if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            PIL.Image.fromarray(image_np, "RGB").save(os.path.join(image_dir, f"{seed:06d}.png"))


def save_grid(images, path, gridw=None):
    """Save a batch of model-space images as one PNG grid."""
    import numpy as np

    n = images.shape[0]
    gridw = gridw or int(n ** 0.5)
    gridh = (n + gridw - 1) // gridw
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    H, W, C = images_np.shape[1:]
    canvas = np.zeros((gridh * H, gridw * W, C), dtype=np.uint8)
    for i, img in enumerate(images_np):
        r, c = divmod(i, gridw)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PIL.Image.fromarray(canvas, "RGB").save(path)
