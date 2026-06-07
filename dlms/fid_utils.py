"""FID computation (single-GPU port of amed-solver-main/fid.py).

Uses the same NVIDIA Inception-v3 detector and the official EDM reference
statistics, so numbers are directly comparable to the paper and to the
diff-sampler README tables.
"""

from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm

from .bootstrap import RESULTS_DIR
from .inception import load_inception

FID_REFS = {
    "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz",
    "afhqv2": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz",
    "imagenet64": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz",
}


def load_ref_stats(dataset_or_path):
    """Load (mu_ref, sigma_ref) from a dataset key, local npz, or URL."""
    import dnnlib

    src = FID_REFS.get(dataset_or_path, dataset_or_path)
    cache_dir = RESULTS_DIR / "fid_refs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with dnnlib.util.open_url(str(src), cache_dir=str(cache_dir)) as f:
        ref = dict(np.load(f))
    return ref["mu"], ref["sigma"]


def iter_image_batches(image_dir, batch_size=250, num_expected=None):
    """Yield uint8 NCHW tensors from PNGs under image_dir (recursive, sorted)."""
    import PIL.Image

    files = sorted(Path(image_dir).rglob("*.png"))
    if num_expected is not None:
        if len(files) < num_expected:
            raise RuntimeError(f"found {len(files)} images in {image_dir}, expected {num_expected}")
        files = files[:num_expected]
    if len(files) < 2:
        raise RuntimeError(f"found only {len(files)} images in {image_dir}")
    for i in range(0, len(files), batch_size):
        imgs = [np.array(PIL.Image.open(f).convert("RGB")) for f in files[i:i + batch_size]]
        yield torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)


def calculate_inception_stats(image_dir, num_expected=None, batch_size=250,
                              device=torch.device("cuda"), verbose=True):
    detector = load_inception(device)
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    count = 0
    batches = iter_image_batches(image_dir, batch_size, num_expected)
    for images in tqdm(batches, unit="batch", disable=not verbose):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(device), return_features=True).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
        count += images.shape[0]
    mu /= count
    sigma -= mu.ger(mu) * count
    sigma /= count - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def compute_fid(image_dir, dataset_or_ref, num_expected=None, batch_size=250,
                device=torch.device("cuda"), verbose=True):
    mu_ref, sigma_ref = load_ref_stats(dataset_or_ref)
    mu, sigma = calculate_inception_stats(image_dir, num_expected=num_expected,
                                          batch_size=batch_size, device=device, verbose=verbose)
    return calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref)
