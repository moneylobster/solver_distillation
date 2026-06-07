"""Test B: high-order initialization exactness (paper Sec. 3.3).

An untrained DLMSPredictor must emit exactly the PLMS/iPNDM coefficients, the
uniform time schedule and s = 1, and sampling with it must be bit-identical
to the fixed-coefficient 'data' reference mode of dlms_sampler.
"""

import torch

from dlms.network import DLMSPredictor
from dlms.solver import (coefs_from_tail, dlms_sampler, next_time_from_fraction,
                         plms_tail, uniform_schedule_logits)

NUM_STEPS = 8


def make_net(device, **kwargs):
    torch.manual_seed(0)
    return DLMSPredictor(num_steps=NUM_STEPS, **kwargs).to(device)


def test_init_outputs_exact(device):
    g_phi = make_net(device)
    B = 3
    sched_logits = uniform_schedule_logits(NUM_STEPS, 0.002, 80.0).to(device)
    t_cur = torch.full((B,), 80.0, device=device)
    realized = [t_cur]
    for step_idx in range(NUM_STEPS - 1):
        h = torch.randn(B, 8, 8, device=device)
        coefs, t_next, s_next = g_phi(h, t_cur, step_idx)
        order = min(step_idx + 1, 4)
        expected = coefs_from_tail(plms_tail(order, device=device)).unsqueeze(0).expand(B, -1)
        assert torch.equal(coefs, expected), f"coefs mismatch at step {step_idx}"
        assert torch.equal(s_next, torch.ones(B, device=device)), f"s != 1 at step {step_idx}"
        if step_idx == NUM_STEPS - 2:
            expected_t = torch.full((B,), 0.002, device=device)
        else:
            frac = torch.sigmoid(sched_logits[step_idx]).expand(B)
            expected_t = next_time_from_fraction(t_cur, frac, 0.002)
        assert torch.equal(t_next, expected_t), f"t_next mismatch at step {step_idx}"
        t_cur = t_next
        realized.append(t_cur)
    # The realized schedule matches the time_uniform reference closely on the
    # intermediate points; the final point is deliberately forced to
    # sigma_min (the reference schedule ends slightly above it, ~0.0020132).
    from solver_utils import get_schedule
    t_ref = get_schedule(NUM_STEPS, 0.002, 80.0, device=device,
                         schedule_type="time_uniform", schedule_rho=1)
    for n in range(1, NUM_STEPS - 1):
        assert torch.allclose(realized[n], t_ref[n].expand(B), rtol=1e-4), f"schedule point {n}"
    assert torch.equal(realized[-1], torch.full((B,), 0.002, device=device))


def test_param_count_about_9k():
    g_phi = DLMSPredictor(num_steps=NUM_STEPS)
    n = sum(p.numel() for p in g_phi.parameters())
    assert 8_000 <= n <= 10_000, f"param count {n} not ~9k"


def test_untrained_sampler_equals_data_mode(cifar_net, latents, device):
    for afs in (False, True):
        g_phi = make_net(device, afs=afs)
        ref = dlms_sampler(cifar_net, latents, fixed_coef_mode="data",
                           num_steps=NUM_STEPS, afs=afs)
        ours = dlms_sampler(cifar_net, latents, g_phi=g_phi,
                            num_steps=NUM_STEPS, afs=afs)
        assert torch.equal(ours, ref), f"sampler mismatch (afs={afs})"


def test_ablation_flags(device):
    B = 2
    t_cur = torch.full((B,), 40.0, device=device)
    h = torch.randn(B, 8, 8, device=device)

    # w/o adaptive schedule: t comes from the uniform schedule even with a
    # randomly initialized time head.
    g = make_net(device, learn_schedule=False, high_order_init=False)
    with torch.no_grad():
        g.fc_time.weight.add_(torch.randn_like(g.fc_time.weight))
    _, t1, _ = g(h, t_cur, 2)
    sched_logits = uniform_schedule_logits(NUM_STEPS, 0.002, 80.0).to(device)
    expected_t = next_time_from_fraction(t_cur, torch.sigmoid(sched_logits[2]).expand(B), 0.002)
    assert torch.equal(t1, expected_t)

    # w/o time scaling: s == 1 regardless of head weights.
    g = make_net(device, learn_scale=False, high_order_init=False)
    with torch.no_grad():
        g.fc_scale.weight.add_(torch.randn_like(g.fc_scale.weight))
    _, _, s = g(h, t_cur, 2)
    assert torch.equal(s, torch.ones(B, device=device))

    # w/o bottleneck: outputs are independent of h.
    g = make_net(device, use_bottleneck=False)
    with torch.no_grad():  # give heads nonzero weights so h could matter
        for fc in (g.fc_coef, g.fc_time, g.fc_scale):
            fc.weight.add_(torch.randn_like(fc.weight))
    out1 = g(torch.randn(B, 8, 8, device=device), t_cur, 2)
    out2 = g(torch.randn(B, 8, 8, device=device), t_cur, 2)
    for a, b in zip(out1, out2):
        assert torch.equal(a, b)

    # w/o high-order init: untrained coefficients are NOT the PLMS ones
    # (base tail is zeros => DDIM-like), and heads are randomly initialized.
    g = make_net(device, high_order_init=False)
    coefs, _, _ = g(torch.zeros(B, 8, 8, device=device), t_cur, 3)
    assert coefs.shape == (B, 4)
    assert not torch.allclose(coefs, coefs_from_tail(plms_tail(4, device=device)).expand(B, -1))
    assert torch.allclose(coefs.sum(dim=-1), torch.ones(B, device=device), atol=1e-6)
