"""Training configuration and per-dataset presets (paper Appendix B)."""

from dataclasses import asdict, dataclass, field


def num_steps_for_nfe(nfe, afs):
    """Time-grid points for a target NFE. With AFS the t_0 denoiser call is
    free, so the grid gains one extra point."""
    return nfe + 2 if afs else nfe + 1


@dataclass
class DLMSConfig:
    dataset_name: str = "cifar10"
    nfe: int = 5                          # target NFE; num_steps derived
    max_order: int = 4                    # p
    M: int = 4                            # teacher interpolation sub-steps
    teacher_max_order: int = 3            # DPM-Solver++(3M)
    total_traj: int = 20_000              # total training trajectories
    batch: int = 64                       # trajectories per optimizer step
    batch_gpu: int = 0                    # microbatch (0 = batch)
    lr: float = 5e-3
    seed: int = 0
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    scale_range: float = 0.2
    ema_halflives_kimg: tuple = (1.0, 2.0, 3.0)
    tick_traj: int = 500                  # progress print/log interval
    snapshot_traj: int = 2_500            # snapshot interval
    # Practical techniques / ablation switches (paper Sec. 3.3 / Table 3):
    afs: bool = True
    use_bottleneck: bool = True
    high_order_init: bool = True
    inception_loss: bool = True           # final-step Inception distance (pixel space)
    learn_scale: bool = True
    learn_schedule: bool = True

    @property
    def num_steps(self):
        return num_steps_for_nfe(self.nfe, self.afs)

    def asdict(self):
        return asdict(self)


# Per-dataset presets from Appendix B. `batch` is the paper-faithful 64 (their
# total across 8 V100s is unstated; 64 fits common single-GPU setups for
# CIFAR; lower batch_gpu via CLI on small-VRAM GPUs).
PRESETS = {
    "cifar10":    dict(dataset_name="cifar10", max_order=4, M=4, total_traj=20_000, batch=64),
    "ffhq":       dict(dataset_name="ffhq", max_order=4, M=4, total_traj=20_000, batch=32),
    "imagenet64": dict(dataset_name="imagenet64", max_order=4, M=4, total_traj=20_000, batch=16),
}


def make_config(dataset_name, **overrides):
    kwargs = dict(PRESETS[dataset_name])
    kwargs.update(overrides)
    return DLMSConfig(**kwargs)
