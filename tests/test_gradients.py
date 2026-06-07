"""Test D: gradient flow and memory behavior of the Algorithm 1 training step.

- gradients reach the designer-network heads (and, with random init, every
  parameter), including through the final-step Inception distance;
- the frozen diffusion UNet receives no gradients;
- computation graphs do not span iterations: peak memory is essentially
  independent of the number of steps N.
"""

import torch

from dlms.config import DLMSConfig
from dlms.inception import load_inception
from dlms.network import DLMSPredictor
from dlms.train import run_trajectory_batch


def make(cfg, device, **net_kwargs):
    torch.manual_seed(0)
    g_phi = DLMSPredictor(
        num_steps=cfg.num_steps, max_order=cfg.max_order,
        afs=cfg.afs, use_bottleneck=cfg.use_bottleneck,
        high_order_init=cfg.high_order_init, learn_schedule=cfg.learn_schedule,
        learn_scale=cfg.learn_scale, **net_kwargs,
    ).to(device).train().requires_grad_(True)
    return g_phi


def run_batch(cifar_net, g_phi, cfg, device, batch=2, detector=None):
    torch.manual_seed(1)
    latents = torch.randn(batch, 3, 32, 32, device=device)
    return run_trajectory_batch(cifar_net, g_phi, cfg, latents, detector=detector)


def test_grads_reach_heads_and_not_unet(cifar_net, device):
    cfg = DLMSConfig(nfe=4, afs=True, inception_loss=True)
    g_phi = make(cfg, device)
    detector = load_inception(device)
    info = run_batch(cifar_net, g_phi, cfg, device, detector=detector)
    assert len(info["losses"]) == cfg.num_steps - 2  # iterations n = 2..N

    for name in ("fc_coef", "fc_time", "fc_scale"):
        head = getattr(g_phi, name)
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0, name
        assert head.bias.grad is not None and head.bias.grad.abs().sum() > 0, f"{name}.bias"

    for p in cifar_net.parameters():
        assert not p.requires_grad and p.grad is None


def test_grads_reach_all_params_random_init(cifar_net, device):
    cfg = DLMSConfig(nfe=4, afs=True, high_order_init=False, inception_loss=False)
    g_phi = make(cfg, device)
    run_batch(cifar_net, g_phi, cfg, device)
    for name, p in g_phi.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"


def test_memory_flat_in_num_steps(cifar_net, device):
    peaks = {}
    for nfe in (4, 8):
        cfg = DLMSConfig(nfe=nfe, afs=True, inception_loss=False)
        g_phi = make(cfg, device)
        run_batch(cifar_net, g_phi, cfg, device, batch=8)  # warmup/alloc
        g_phi.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        run_batch(cifar_net, g_phi, cfg, device, batch=8)
        torch.cuda.synchronize()
        peaks[nfe] = torch.cuda.max_memory_allocated(device)
    assert peaks[8] < peaks[4] * 1.25, f"memory grows with N: {peaks}"


def test_loss_decreases_quick(cifar_net, device):
    """Tiny optimization sanity: a few Adam steps reduce the summed loss."""
    cfg = DLMSConfig(nfe=4, afs=True, inception_loss=False)
    g_phi = make(cfg, device)
    opt = torch.optim.Adam(g_phi.parameters(), lr=5e-3)
    first = last = None
    for it in range(8):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(123)  # same batch every iteration
        latents = torch.randn(8, 3, 32, 32, device=device)
        info = run_trajectory_batch(cifar_net, g_phi, cfg, latents)
        total = sum(info["losses"])
        first = total if first is None else first
        last = total
        opt.step()
    assert last < first, f"loss did not decrease: {first} -> {last}"
