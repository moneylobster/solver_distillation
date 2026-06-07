"""DLMS prediction formula (paper Eqs. 8-9) and sampler (Algorithm 2).

Conventions (matching diff-sampler):
- `num_steps` = number of time-grid points t_0..t_{N}, i.e. N = num_steps - 1
  prediction steps. NFE = N without AFS, N - 1 with AFS.
- EDM noise schedule: alpha_t = 1, sigma_t = t, lambda_t = -log t. With these,
  Eq. 9 reduces to  x_n = (t_n/t_{n-1}) * x_{n-1} + (1 - t_n/t_{n-1}) * D_n.
- The buffer Q holds raw denoiser outputs (data predictions), newest last.

Besides the learned sampler, two fixed-coefficient reference modes exist for
sanity testing:
- 'eps':  bit-exact replica of diff-sampler's ipndm_sampler (PLMS in
          epsilon space). Validates plumbing/schedules/seeds only - the DLMS
          formula (Eq. 8) lives in data space and is NOT algebraically equal
          to this for order >= 2.
- 'data': Eq. 8-9 with fixed PLMS coefficients and the uniform init schedule;
          equals what an untrained (high-order-initialized) designer network
          produces, bit-exactly.
"""

import math

import torch

from . import bootstrap  # noqa: F401  (sys.path for solver_utils)
from .models import BottleneckHook

# Adams-Bashforth / PLMS (iPNDM) coefficient tails (a_2..a_p) per order.
# a_1 is implied by the constraint sum_k a_k = 1.
PLMS_TAIL = {
    1: [],
    2: [-1.0 / 2],
    3: [-16.0 / 12, 5.0 / 12],
    4: [-59.0 / 24, 37.0 / 24, -9.0 / 24],
}


def plms_tail(order, device=None):
    return torch.tensor(PLMS_TAIL[order], device=device, dtype=torch.float32)


def coefs_from_tail(tail):
    """Prepend a_1 = 1 - sum(tail). tail: (..., p'-1) -> coefs: (..., p')."""
    a1 = 1.0 - tail.sum(dim=-1, keepdim=True)
    return torch.cat([a1, tail], dim=-1)


def phi_step(x_cur, t_cur, t_next, coefs, Q):
    """One DLMS prediction step (Eq. 8-9, EDM schedule).

    x_cur:  (B, C, H, W)
    t_cur, t_next: broadcastable to (B, 1, 1, 1)
    coefs:  (B, p') or (p',); coefs[..., 0] weights the NEWEST buffer entry
    Q:      list of past denoiser outputs, newest last; uses Q[-1]..Q[-p']
    """
    if coefs.dim() == 1:
        coefs = coefs.unsqueeze(0)
    order = coefs.shape[-1]
    assert len(Q) >= order, f"buffer has {len(Q)} entries, need {order}"
    D = sum(coefs[:, k].reshape(-1, 1, 1, 1) * Q[-1 - k] for k in range(order))
    ratio = t_next / t_cur
    return ratio * x_cur + (1 - ratio) * D


def lambda_of(t):
    return -torch.log(t) if torch.is_tensor(t) else -math.log(t)


def next_time_from_fraction(t_cur, frac, sigma_min):
    """t_next from the remaining-interval fraction in lambda = -log t space.

    frac in (0,1): 0 -> stay at t_cur, 1 -> jump to sigma_min.
    Strictly monotone, never leaves (sigma_min, t_cur).
    """
    lam_cur = lambda_of(t_cur)
    lam_min = lambda_of(torch.as_tensor(sigma_min, dtype=t_cur.dtype, device=t_cur.device))
    return torch.exp(-(lam_cur + frac * (lam_min - lam_cur)))


def uniform_schedule_logits(num_steps, sigma_min, sigma_max, device=None):
    """Per-step logits c_n such that sequentially applying
    sigmoid(c_n) through next_time_from_fraction reproduces the uniform
    (Ho et al. 2020 / 'time_uniform') schedule on `num_steps` grid points.

    Returns (num_steps - 1,) logits for steps n = 1..N. The entry for the
    final step is unused at sampling time (t_N is forced to sigma_min).
    """
    from solver_utils import get_schedule

    t_ref = get_schedule(num_steps, sigma_min, sigma_max, device=device,
                         schedule_type="time_uniform", schedule_rho=1)
    lam = -t_ref.log()
    lam_min = -math.log(sigma_min)
    frac = (lam[1:] - lam[:-1]) / (lam_min - lam[:-1])
    frac = frac.clamp(1e-6, 1 - 1e-6)  # final entry would be exactly 1 (logit inf)
    return torch.log(frac / (1 - frac))


def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    """Denoiser call, dispatching on the wrapper type (mirrors solvers_amed)."""
    if hasattr(net, "guidance_type"):  # LDM / Stable-Diffusion wrappers
        return net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    return net(x, t, class_labels=class_labels)


# ----------------------------------------------------------------------------
# Algorithm 2: Distilled Solver Sampling.

@torch.no_grad()
def dlms_sampler(
    net,
    latents,
    g_phi=None,
    *,
    num_steps,
    sigma_min=0.002,
    sigma_max=80.0,
    afs=False,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    max_order=4,
    fixed_coef_mode=None,       # None (use g_phi) | 'data' | 'eps' (tests)
    return_inters=False,
    return_schedule=False,
):
    """Sample with a trained designer network g_phi (or fixed-coef reference).

    latents: N(0, I) noise of the data shape; scaled by sigma_max internally.
    Returns the final sample; optionally the trajectory / realized schedule.
    """
    assert (g_phi is not None) != (fixed_coef_mode is not None), \
        "provide exactly one of g_phi / fixed_coef_mode"
    # Guided (CFG) models double the batch inside the wrapper; BottleneckHook
    # does not slice that, so only unguided/EDM-style models are supported.
    assert condition is None and unconditional_condition is None, \
        "guided (LDM/SD) models are not supported by this sampler yet"
    if fixed_coef_mode == "eps":
        assert not return_schedule
        return _ipndm_reference_sampler(
            net, latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            afs=afs, class_labels=class_labels, condition=condition,
            unconditional_condition=unconditional_condition, max_order=max_order,
            return_inters=return_inters)

    device = latents.device
    B = latents.shape[0]
    N = num_steps - 1
    if fixed_coef_mode == "data":
        # Computed on CPU then moved, exactly like DLMSPredictor's buffer, so
        # the 'data' mode is bit-identical to an untrained designer network.
        sched_logits = uniform_schedule_logits(num_steps, sigma_min, sigma_max).to(device)

    t_cur = torch.full((B,), sigma_max, dtype=torch.float32, device=device)
    x = latents.to(torch.float32) * sigma_max
    s_cur = torch.ones((B,), dtype=torch.float32, device=device)
    Q = []
    inters = [x.unsqueeze(0)]
    schedule = [t_cur.clone()]

    class_conditional = class_labels is not None
    for n in range(1, N + 1):
        step_idx = n - 1
        if afs and n == 1:
            Q.append(torch.zeros_like(x))
            h = torch.zeros((B, 8, 8), device=device)
        else:
            hook = BottleneckHook(net, class_conditional)
            denoised = get_denoised(net, x, s_cur * t_cur, class_labels=class_labels,
                                    condition=condition, unconditional_condition=unconditional_condition)
            Q.append(denoised.to(torch.float32))
            h = hook.pooled().to(torch.float32)
            hook.remove()
        if len(Q) > max_order:
            Q.pop(0)

        order = min(n, max_order)
        if g_phi is not None:
            coefs, t_next, s_next = g_phi(h, t_cur, step_idx)
        else:
            coefs = coefs_from_tail(plms_tail(order, device=device))
            frac = torch.sigmoid(sched_logits[step_idx])
            t_next = (torch.full_like(t_cur, sigma_min) if n == N
                      else next_time_from_fraction(t_cur, frac, sigma_min))
            s_next = torch.ones_like(s_cur)

        x = phi_step(x, t_cur.reshape(-1, 1, 1, 1), t_next.reshape(-1, 1, 1, 1), coefs, Q)
        t_cur, s_cur = t_next, s_next
        if return_inters:
            inters.append(x.unsqueeze(0))
        schedule.append(t_cur.clone())

    out = torch.cat(inters, dim=0) if return_inters else x
    if return_schedule:
        return out, torch.stack(schedule, dim=0)
    return out


# ----------------------------------------------------------------------------
# Bit-exact replica of diff-sampler's ipndm_sampler (epsilon-space PLMS),
# used only by tests/test_phi_ipndm_equivalence.py to validate plumbing.

@torch.no_grad()
def _ipndm_reference_sampler(net, latents, *, num_steps, sigma_min, sigma_max, afs,
                             class_labels, condition, unconditional_condition,
                             max_order, return_inters):
    from solver_utils import get_schedule

    assert 1 <= max_order <= 4
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type="polynomial", schedule_rho=7, net=net)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        use_afs = afs and len(buffer_model) == 0
        if use_afs:
            d_cur = x_cur / ((1 + t_cur ** 2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels,
                                    condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, len(buffer_model) + 1)
        if order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next
