"""Inception feature network for the final-step Inception distance loss and FID.

Uses NVIDIA's inception-2015-12-05.pkl (same detector as the diff-sampler /
EDM FID pipeline), cached under results/. The detector takes uint8-range
NCHW images (any spatial size; it resizes internally) and returns 2048-dim
features.

For the *loss* we feed float tensors in [0, 255] range computed as
x*127.5+128 WITHOUT clamping or uint8 casting, so gradients can flow back to
the designer network. Differentiability of the pickled TorchScript model is
verified by tests/test_gradients.py (and was spike-tested during development);
if it ever breaks, swap in torchvision's inception_v3 features for the loss
only (FID must keep the NVIDIA detector for comparability).
"""

import pickle

import torch

from .bootstrap import RESULTS_DIR

DETECTOR_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1"
    "/files/metrics/inception-2015-12-05.pkl"
)

_detector_cache = {}


def load_inception(device=torch.device("cuda")):
    """Load (and cache) the NVIDIA Inception-v3 FID detector."""
    key = str(device)
    if key not in _detector_cache:
        import dnnlib  # via bootstrap sys.path

        cache_dir = RESULTS_DIR / "detector_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with dnnlib.util.open_url(DETECTOR_URL, cache_dir=str(cache_dir), verbose=True) as f:
            _detector_cache[key] = pickle.load(f).to(device)
    return _detector_cache[key]


def inception_features(detector, images_255):
    """2048-dim features of float images in [0,255] (B,C,H,W). Differentiable."""
    if images_255.shape[1] == 1:  # grayscale -> RGB
        images_255 = images_255.repeat(1, 3, 1, 1)
    return detector(images_255, return_features=True)


def to_uint8_range(x):
    """Map model-space samples (~[-1,1]) to [0,255] floats, unclamped (differentiable)."""
    return x * 127.5 + 128
