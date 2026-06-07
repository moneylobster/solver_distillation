import pytest
import torch

import dlms  # noqa: F401  (bootstrap sys.path)
from dlms.models import create_edm_model


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def cifar_net(device):
    torch.backends.cudnn.benchmark = False  # bit-exactness across calls
    return create_edm_model("cifar10", device=device)


@pytest.fixture()
def latents(device):
    g = torch.Generator(device).manual_seed(0)
    return torch.randn(4, 3, 32, 32, generator=g, device=device)
