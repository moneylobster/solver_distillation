"""DLMS solver distillation - paper Algorithm 1.

Single-GPU, dataloader-free training: latents are drawn on the fly, the
teacher trajectory is computed online (the time schedule is adaptive, so it
cannot be precomputed), and the designer network is updated once per batch of
trajectories with gradients accumulated over the N prediction steps.

Stop-gradient placement (flagged as critical in the paper; one reading choice —
the bottleneck feature h is always detached at the hook, following AMED-Plugin;
Algorithm 1 does not list h among its explicit sg() targets, so a literal
reading would keep the within-iteration path loss -> coefs_b -> h. See README):
- the state entering each iteration (x, Q, t, h) is detached, so computation
  graphs never span iterations (flat memory) and the schedule cannot collapse;
- the previous step's designer outputs are recomputed with gradients enabled
  (lines 8-9 of Algorithm 1), which is how {a_k}, t_{n-1}, s_{n-1} receive
  gradient through the current step's denoiser call;
- t_n is detached immediately (line 13); its gradient arrives only via the
  next iteration's recomputation;
- the teacher runs entirely under no_grad.
"""

import json
import time
from pathlib import Path

import torch

from .config import DLMSConfig
from .ema import EMA
from .inception import inception_features, load_inception, to_uint8_range
from .models import BottleneckHook, create_edm_model
from .network import DLMSPredictor
from .solver import get_denoised, phi_step
from .teacher import DPMppTeacher


def run_trajectory_batch(net, g_phi, cfg, latents, class_labels=None,
                         detector=None, loss_scale=1.0):
    """One batch of trajectories through Algorithm 1.

    Calls loss.backward() once per step n (gradients accumulate on g_phi).
    Returns a logging dict with per-step losses and the realized schedule.
    """
    device = latents.device
    B = latents.shape[0]
    N = cfg.num_steps - 1
    class_conditional = class_labels is not None
    pixel_space = not hasattr(net, "guidance_type")
    col = lambda t: t.reshape(-1, 1, 1, 1)

    # Initialization (lines 2-6): t_0 = T, s_0 = 1, first buffer entry.
    x_stu = (latents.to(torch.float32) * cfg.sigma_max).detach()  # x^S at t_{n-2}
    t_prev2 = torch.full((B,), cfg.sigma_max, device=device)      # t_{n-2}
    Q = []
    if cfg.afs:  # virtual first step: x_theta = 0, h = 0
        Q.append(torch.zeros_like(x_stu))
        h_prev2 = torch.zeros((B, 8, 8), device=device)
    else:
        with torch.no_grad():
            hook = BottleneckHook(net, class_conditional)
            den = get_denoised(net, x_stu, t_prev2, class_labels=class_labels)
            Q.append(den.to(torch.float32))
            h_prev2 = hook.pooled().to(torch.float32)
            hook.remove()
    x_tea = x_stu.clone()
    teacher = DPMppTeacher(net, num_student_steps=N, M=cfg.M,
                           max_order=cfg.teacher_max_order, class_labels=class_labels)

    losses = []
    sched = [cfg.sigma_max]
    for n in range(2, N + 1):
        # (line 8) Recompute the previous step's designer outputs WITH grad.
        coefs_a, t_prev, s_prev = g_phi(h_prev2, t_prev2, n - 2)
        # (line 9) Rebuild x^S_{t_{n-1}} from the detached state.
        x_mid = phi_step(x_stu, col(t_prev2), col(t_prev), coefs_a, Q)
        # (lines 10-11) Grad-enabled denoiser call -> buffer entry + bottleneck.
        hook = BottleneckHook(net, class_conditional)
        den = get_denoised(net, x_mid, s_prev * t_prev, class_labels=class_labels)
        Q.append(den.to(torch.float32))
        if len(Q) > cfg.max_order:
            Q.pop(0)
        h_prev = hook.pooled().to(torch.float32)
        hook.remove()
        # (lines 12-13) Designer outputs for the current step; stop-grad t_n.
        coefs_b, t_n, s_n = g_phi(h_prev, t_prev, n - 1)
        t_n = t_n.detach()
        # (line 14) Predict x^S_{t_n}.
        x_n = phi_step(x_mid, col(t_prev), col(t_n), coefs_b, Q)
        # (lines 15-18) Teacher trajectory on the fly (no grad).
        t_prev_sg = t_prev.detach()
        if n == 2:
            x_tea = teacher.advance(x_tea, t_prev2, t_prev_sg)
        x_tea = teacher.advance(x_tea, t_prev_sg, t_n)
        # (lines 19-20) Square distance; Inception distance at the final step
        # for pixel-space models (Sec. 3.3).
        if n == N and cfg.inception_loss and pixel_space and detector is not None:
            f_stu = inception_features(detector, to_uint8_range(x_n))
            with torch.no_grad():
                f_tea = inception_features(detector, to_uint8_range(x_tea))
            loss = (f_stu - f_tea).pow(2).sum(dim=1)
        else:
            loss = (x_n - x_tea).pow(2).flatten(1).sum(dim=1)
        loss.mean().mul(loss_scale).backward()
        losses.append(loss.mean().item())
        # (line 21) Detach buffers and state for the next iteration.
        x_stu = x_mid.detach()
        t_prev2 = t_prev_sg
        h_prev2 = h_prev  # hook output is already detached
        Q = [q.detach() for q in Q]
        sched.append(t_prev_sg.mean().item())
    sched.append(float(t_n.mean().item()))
    return dict(losses=losses, schedule=sched)


# ----------------------------------------------------------------------------


def save_snapshot(path, g_phi, emas, optimizer, cfg, cur_traj):
    data = dict(
        config=g_phi.config_dict(),
        train_config=cfg.asdict(),
        cur_traj=cur_traj,
        raw=g_phi.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    for ema in emas:
        data[f"ema_{ema.halflife_kimg:g}"] = ema.state_dict()
    torch.save(data, path)


def load_predictor(path, variant="raw", device=torch.device("cuda")):
    """Rebuild a DLMSPredictor from a snapshot. variant: 'raw' or 'ema_<hl>'."""
    data = torch.load(path, map_location=device, weights_only=False)
    g_phi = DLMSPredictor(**data["config"]).to(device)
    if variant not in data:
        raise KeyError(f"variant {variant!r} not in snapshot; have {sorted(data)}")
    g_phi.load_state_dict(data[variant])
    g_phi.eval().requires_grad_(False)
    return g_phi


def snapshot_variants(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return ["raw"] + [k for k in data if k.startswith("ema_")]


# ----------------------------------------------------------------------------


def train_dlms(cfg: DLMSConfig, run_dir, device=None, verbose=True):
    device = device or torch.device("cuda")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.asdict(), indent=2))
    log = print if verbose else (lambda *a, **k: None)

    torch.manual_seed(cfg.seed)
    net = create_edm_model(cfg.dataset_name, device=device)
    pixel_space = not hasattr(net, "guidance_type")

    g_phi = DLMSPredictor(
        num_steps=cfg.num_steps, max_order=cfg.max_order,
        sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        dataset_name=cfg.dataset_name, afs=cfg.afs, scale_range=cfg.scale_range,
        use_bottleneck=cfg.use_bottleneck, high_order_init=cfg.high_order_init,
        learn_schedule=cfg.learn_schedule, learn_scale=cfg.learn_scale,
    ).to(device).train().requires_grad_(True)
    n_params = sum(p.numel() for p in g_phi.parameters())
    log(f"Designer network: {n_params} parameters; num_steps={cfg.num_steps} (NFE {cfg.nfe}, afs={cfg.afs})")

    optimizer = torch.optim.Adam(g_phi.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    emas = [EMA(g_phi, hl) for hl in cfg.ema_halflives_kimg]
    detector = load_inception(device) if (cfg.inception_loss and pixel_space) else None

    batch_gpu = cfg.batch_gpu or cfg.batch
    assert cfg.batch % batch_gpu == 0, "batch must be divisible by batch_gpu"
    rounds = cfg.batch // batch_gpu

    stats_path = run_dir / "stats.jsonl"
    start_time = time.time()
    cur_traj = 0
    last_tick_traj = 0
    last_snapshot_traj = 0
    while cur_traj < cfg.total_traj:
        optimizer.zero_grad(set_to_none=True)
        batch_logs = []
        for _ in range(rounds):
            latents = torch.randn(batch_gpu, net.img_channels, net.img_resolution,
                                  net.img_resolution, device=device)
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[
                    torch.randint(net.label_dim, size=(batch_gpu,), device=device)]
            info = run_trajectory_batch(net, g_phi, cfg, latents, class_labels,
                                        detector=detector, loss_scale=1.0 / rounds)
            batch_logs.append(info)
        for param in g_phi.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        for ema in emas:
            ema.update(g_phi, cfg.batch)
        cur_traj += cfg.batch

        if cur_traj - last_tick_traj >= cfg.tick_traj or cur_traj >= cfg.total_traj:
            last_tick_traj = cur_traj
            n_steps = len(batch_logs[0]["losses"])
            mean_losses = [sum(b["losses"][i] for b in batch_logs) / rounds for i in range(n_steps)]
            entry = dict(
                traj=cur_traj,
                time_sec=round(time.time() - start_time, 1),
                loss_per_step=[round(v, 5) for v in mean_losses],
                schedule=[round(v, 5) for v in batch_logs[-1]["schedule"]],
                peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                if device.type == "cuda" else 0.0,
            )
            with open(stats_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            log(f"traj {cur_traj:>6d}/{cfg.total_traj}  "
                f"loss/step {' '.join(f'{v:.4f}' for v in mean_losses)}  "
                f"t {entry['time_sec']:.0f}s  vram {entry['peak_vram_gb']:.2f}GB")

        if (cur_traj - last_snapshot_traj >= cfg.snapshot_traj) or cur_traj >= cfg.total_traj:
            last_snapshot_traj = cur_traj
            snap_path = run_dir / f"snapshot-{cur_traj:06d}.pt"
            save_snapshot(snap_path, g_phi, emas, optimizer, cfg, cur_traj)
            log(f"saved {snap_path}")

    final_path = run_dir / f"snapshot-{cur_traj:06d}.pt"
    return final_path
