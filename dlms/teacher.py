"""On-the-fly DPM-Solver++ teacher (paper Sec. 3.1, Algorithm 1).

The teacher solves the same PF-ODE as the student but with M extra
interpolation time steps inserted between consecutive student times
t_{n-1} -> t_n. Because the student's time schedule is adaptive (predicted by
g_phi during training), the teacher trajectory cannot be precomputed; this
class advances it segment by segment while keeping the multistep history
buffer across segments, so the whole trajectory is one continuous
DPM-Solver++(3M) run on the fine grid - identical to diff-sampler's
dpm_pp_sampler run on that grid (verified in tests/test_teacher_equivalence.py).

Sub-step placement within a segment is uniform in lambda = -log t (an
implementation choice; the paper only says "M interpolation time steps").
"""

import torch

from . import bootstrap  # noqa: F401
from .solver import get_denoised


class DPMppTeacher:
    """One teacher trajectory (reusable for a single batch).

    Holds the persistent model-output/time buffers and the global sub-step
    counter needed for DPM-Solver++'s order warmup and `lower_order_final`.
    """

    def __init__(self, net, num_student_steps, M=4, max_order=3,
                 class_labels=None, condition=None, unconditional_condition=None,
                 predict_x0=True, lower_order_final=True):
        assert 1 <= max_order <= 3
        self.net = net
        self.M = M
        self.max_order = max_order
        self.class_labels = class_labels
        self.condition = condition
        self.unconditional_condition = unconditional_condition
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        # Fine grid: (M+1) sub-intervals per student step => total grid points.
        self.total_points = (M + 1) * num_student_steps + 1
        self.buffer_model = []
        self.buffer_t = []
        self.step_cur = 0  # number of completed fine-grid steps

    @torch.no_grad()
    def advance(self, x, t_from, t_to):
        """Advance the teacher state from t_from to t_to (per-sample times).

        x: (B, C, H, W); t_from, t_to: (B,) or scalar tensors.
        Returns the teacher sample at t_to.
        """
        from solver_utils import dpm_pp_update, dynamic_thresholding_fn

        t_from = torch.as_tensor(t_from, device=x.device, dtype=torch.float32).reshape(-1)
        t_to = torch.as_tensor(t_to, device=x.device, dtype=torch.float32).reshape(-1)
        lam_from, lam_to = -t_from.log(), -t_to.log()

        for j in range(self.M + 1):
            w0 = j / (self.M + 1)
            w1 = (j + 1) / (self.M + 1)
            t_cur = torch.exp(-(lam_from + w0 * (lam_to - lam_from))) if j > 0 else t_from
            t_next = torch.exp(-(lam_from + w1 * (lam_to - lam_from))) if j < self.M else t_to

            denoised = get_denoised(self.net, x, t_cur, class_labels=self.class_labels,
                                    condition=self.condition,
                                    unconditional_condition=self.unconditional_condition)
            d_cur = (x - denoised) / t_cur.reshape(-1, 1, 1, 1)
            self.buffer_model.append(dynamic_thresholding_fn(denoised) if self.predict_x0 else d_cur)
            self.buffer_t.append(t_cur)
            if len(self.buffer_model) > self.max_order:
                self.buffer_model.pop(0)
                self.buffer_t.pop(0)

            self.step_cur += 1
            if self.lower_order_final:
                order = (self.step_cur if self.step_cur < self.max_order
                         else min(self.max_order, self.total_points - self.step_cur))
            else:
                order = min(self.max_order, self.step_cur)
            x = dpm_pp_update(x, self.buffer_model, self.buffer_t, t_next, order,
                              predict_x0=self.predict_x0)
        return x
