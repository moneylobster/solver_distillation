"""Test C: the incremental DPMppTeacher equals diff-sampler's dpm_pp_sampler
run on the equivalent fine grid.

When the student times follow a logsnr (lambda-uniform) schedule, the
teacher's lambda-uniform sub-steps make its fine grid identical to a logsnr
schedule with (M+1)*N + 1 points, so the trajectories must coincide (up to
float32 round-off in the time computation: not bit-exact, but very tight).
"""

import torch

from dlms.teacher import DPMppTeacher

N = 5      # student prediction steps
M = 4      # interpolation sub-steps


def test_teacher_matches_dpm_pp_sampler(cifar_net, latents, device):
    import solvers_amed
    from solver_utils import get_schedule

    fine_steps = (M + 1) * N + 1
    ref_traj = solvers_amed.dpm_pp_sampler(
        cifar_net, latents, num_steps=fine_steps, sigma_min=0.002, sigma_max=80.0,
        schedule_type="logsnr", afs=False, return_inters=True,
        max_order=3, predict_x0=True, lower_order_final=True)
    # Teacher checkpoints at the student times = every (M+1)-th fine point.
    ref_slice = [ref_traj[i * (M + 1)] for i in range(1, N + 1)]

    t_student = get_schedule(N + 1, 0.002, 80.0, device=device, schedule_type="logsnr")
    teacher = DPMppTeacher(cifar_net, num_student_steps=N, M=M, max_order=3)
    x = latents * t_student[0]
    outs = []
    for n in range(1, N + 1):
        x = teacher.advance(x, t_student[n - 1].expand(latents.shape[0]),
                            t_student[n].expand(latents.shape[0]))
        outs.append(x)

    for n, (ours, ref) in enumerate(zip(outs, ref_slice), start=1):
        diff = (ours - ref).abs().max().item()
        assert diff < 1e-3, f"teacher mismatch at student step {n}: max abs diff {diff}"
