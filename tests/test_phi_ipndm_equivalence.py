"""Test A: the eps-mode reference path of dlms_sampler is bit-exact with
diff-sampler's ipndm_sampler. Validates schedule handling, buffer warmup
order, and the sampling loop plumbing."""

import torch

from dlms.solver import dlms_sampler


def _run_ipndm(net, latents, num_steps, afs):
    import solvers_amed

    return solvers_amed.ipndm_sampler(
        net, latents, class_labels=None, num_steps=num_steps,
        sigma_min=0.002, sigma_max=80.0, schedule_type="polynomial",
        schedule_rho=7, afs=afs, max_order=4)


def test_eps_mode_bit_exact(cifar_net, latents):
    for num_steps in (5, 8):
        ref = _run_ipndm(cifar_net, latents, num_steps, afs=False)
        ours = dlms_sampler(cifar_net, latents, fixed_coef_mode="eps",
                            num_steps=num_steps, afs=False)
        assert torch.equal(ours, ref), f"mismatch at num_steps={num_steps}"


def test_eps_mode_bit_exact_afs(cifar_net, latents):
    ref = _run_ipndm(cifar_net, latents, num_steps=6, afs=True)
    ours = dlms_sampler(cifar_net, latents, fixed_coef_mode="eps",
                        num_steps=6, afs=True)
    assert torch.equal(ours, ref)
