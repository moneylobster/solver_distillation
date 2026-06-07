"""DLMS designer network g_phi (paper Sec. 3.1-3.3, Fig. 2).

g_phi(h_{t_{n-1}}, t_{n-1}, n-1) -> ({a_k}_{k=1}^{p'}, t_n, s_n), p' = min(n, p).

Architecture mirrors AMED-Plugin's predictor (~9k parameters): a two-layer
bottleneck-feature encoder, a positional time embedding, a learned step-index
embedding, concatenated and fed to per-quantity linear heads.

Output parameterizations (implementation choices; the paper underspecifies
these — see README):
- coefficients: the head emits deltas u_2..u_p added to the Adams-Bashforth
  (PLMS/iPNDM) tail of the active order; a_1 = 1 - sum(others) enforces
  sum_k a_k = 1 by construction and admits negative coefficients.
- next time step: the head emits a logit offset v; the fraction
  f = sigmoid(c_n + v) of the remaining lambda-interval is consumed, where
  c_n are precomputed logits reproducing the uniform (Ho et al. 2020) time
  schedule at v = 0. Guarantees sigma_min < t_n < t_{n-1}. The final step is
  forced to t_N = sigma_min.
- time scale: s = 1 + scale_range * (2*sigmoid(w) - 1), exactly 1 at w = 0.

With "high-order initialization" (Sec. 3.3) all heads are zero-initialized,
so the untrained network reproduces PLMS coefficients, the uniform schedule
and s = 1 exactly (verified bit-exactly in tests/test_network_init.py).
"""

import numpy as np
import torch
from torch.nn.functional import silu

from .solver import coefs_from_tail, next_time_from_fraction, plms_tail, uniform_schedule_logits

# ----------------------------------------------------------------------------
# Layers vendored from amed-solver-main/training/networks.py (EDM-style init),
# minus the persistence decorators (we snapshot plain state_dicts instead).


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal",
                 init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------


class DLMSPredictor(torch.nn.Module):
    def __init__(
        self,
        num_steps,                      # number of time-grid points (N + 1)
        max_order=4,                    # p
        sigma_min=0.002,
        sigma_max=80.0,
        dataset_name=None,              # metadata only
        afs=True,                       # metadata: sampler must match training
        bottleneck_dim=64,
        hidden_dim=128,
        bottleneck_out=4,
        noise_channels=8,
        step_emb_dim=8,
        scale_range=0.2,                # s_n in [1 - r, 1 + r]
        # Ablation switches (paper Table 3):
        use_bottleneck=True,
        high_order_init=True,
        learn_schedule=True,
        learn_scale=True,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.max_order = max_order
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.dataset_name = dataset_name
        self.afs = afs
        self.scale_range = scale_range
        self.use_bottleneck = use_bottleneck
        self.high_order_init = high_order_init
        self.learn_schedule = learn_schedule
        self.learn_scale = learn_scale

        # Bottleneck-feature encoder (kaiming_normal defaults, as in AMED).
        self.enc_layer0 = Linear(bottleneck_dim, hidden_dim)
        self.enc_layer1 = Linear(hidden_dim, bottleneck_out)

        # Time-step embedding (as in AMED: positional embedding of sigma).
        init = dict(init_mode="xavier_uniform")
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_layer0 = Linear(noise_channels, noise_channels, **init)

        # Step-index embedding. Heads are zero-initialized under high-order
        # init, so a random embedding does not perturb the initial outputs.
        self.step_emb = torch.nn.Embedding(num_steps - 1, step_emb_dim)

        trunk_dim = bottleneck_out + noise_channels + step_emb_dim
        head_init = dict(init_weight=0, init_bias=0) if high_order_init else init
        self.fc_coef = Linear(trunk_dim, max_order - 1, **head_init)
        self.fc_time = Linear(trunk_dim, 1, **head_init)
        self.fc_scale = Linear(trunk_dim, 1, **head_init)

        # Adams-Bashforth coefficient tails per order (zeros => order-1 / DDIM
        # baseline when high-order init is disabled).
        for order in range(1, max_order + 1):
            tail = plms_tail(order) if high_order_init else torch.zeros(order - 1)
            self.register_buffer(f"base_tail_{order}", tail)

        # Logits of the uniform (Ho et al. 2020) reference time schedule.
        self.register_buffer("sched_logits",
                             uniform_schedule_logits(num_steps, sigma_min, sigma_max))

    def forward(self, h, t_cur, step_idx):
        """h: (B, 8, 8) or (B, 64) channel-pooled bottleneck feature (contents
        ignored if use_bottleneck=False or the AFS virtual step passes zeros);
        t_cur: (B,) current sigma; step_idx: int in [0, num_steps-2].

        Returns (coefs (B, p'), t_next (B,), s_next (B,)).
        """
        B = t_cur.shape[0]
        order = min(step_idx + 1, self.max_order)

        h = h.reshape(B, -1).to(torch.float32)
        if not self.use_bottleneck:
            h = torch.zeros_like(h)
        feat = self.enc_layer1(silu(self.enc_layer0(h)))

        emb = self.map_noise(t_cur.reshape(-1).to(torch.float32))
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = silu(self.map_layer0(emb))

        idx = torch.full((B,), step_idx, dtype=torch.long, device=t_cur.device)
        trunk = torch.cat([feat, emb, self.step_emb(idx)], dim=1)

        # Prediction coefficients {a_k}.
        base = getattr(self, f"base_tail_{order}")
        deltas = self.fc_coef(trunk)[:, : order - 1]
        coefs = coefs_from_tail(base.unsqueeze(0) + deltas)

        # Next time step t_n.
        if step_idx == self.num_steps - 2:  # final step: force t_N = sigma_min
            t_next = torch.full_like(t_cur, self.sigma_min)
        else:
            v = self.fc_time(trunk)[:, 0] if self.learn_schedule else 0.0
            frac = torch.sigmoid(self.sched_logits[step_idx] + v)
            t_next = next_time_from_fraction(t_cur, frac, self.sigma_min)

        # Time scaling factor s_n.
        if self.learn_scale:
            w = self.fc_scale(trunk)[:, 0]
            s_next = 1.0 + self.scale_range * (2.0 * torch.sigmoid(w) - 1.0)
        else:
            s_next = torch.ones_like(t_cur)

        return coefs, t_next.reshape(B), s_next.reshape(B)

    def config_dict(self):
        """Constructor kwargs, for snapshotting."""
        return dict(
            num_steps=self.num_steps, max_order=self.max_order,
            sigma_min=self.sigma_min, sigma_max=self.sigma_max,
            dataset_name=self.dataset_name, afs=self.afs,
            scale_range=self.scale_range, use_bottleneck=self.use_bottleneck,
            high_order_init=self.high_order_init, learn_schedule=self.learn_schedule,
            learn_scale=self.learn_scale,
        )
