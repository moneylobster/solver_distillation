"""Local GPU benchmark: sampling throughput, training step time, inception throughput.

Usage: uv run python scripts/bench_local.py <dataset> [--sample-batch B] [--train-batch B]
       uv run python scripts/bench_local.py inception
"""

import argparse
import sys
import time

import torch

DATASET_RES = {"cifar10": 32, "ffhq": 64, "imagenet64": 64}


def sync_reset():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3


def labels_for(net, B, device):
    if getattr(net, "label_dim", 0):
        return torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, (B,), device=device)]
    return None


def bench_sampling(dataset, batch, device):
    from dlms.models import create_edm_model
    from dlms.solver import dlms_sampler

    net = create_edm_model(dataset, device=device)
    res = DATASET_RES[dataset]
    while batch >= 1:
        try:
            torch.manual_seed(0)
            x = torch.randn(batch, 3, res, res, device=device)
            cl = labels_for(net, batch, device)
            with torch.no_grad():
                dlms_sampler(net, x, fixed_coef_mode="data", num_steps=7,
                             afs=True, class_labels=cl)  # warmup
            sync_reset()
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    dlms_sampler(net, x, fixed_coef_mode="data", num_steps=7,
                                 afs=True, class_labels=cl)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            sec = sum(times) / len(times)
            print(f"SAMPLE {dataset} batch={batch} sec/batch={sec:.3f} "
                  f"imgs/sec={batch/sec:.2f} times={[f'{t:.3f}' for t in times]} "
                  f"peakVRAM={peak_gb():.2f}GB", flush=True)
            return net
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"SAMPLE {dataset} OOM at batch={batch}, halving", flush=True)
            batch //= 2
    return net


def bench_training(dataset, batch, device, net):
    from dlms.config import DLMSConfig
    from dlms.network import DLMSPredictor
    from dlms.train import run_trajectory_batch

    cfg = DLMSConfig(dataset_name=dataset, nfe=5, afs=True, inception_loss=False)
    res = DATASET_RES[dataset]
    while batch >= 1:
        try:
            torch.manual_seed(0)
            g_phi = DLMSPredictor(
                num_steps=cfg.num_steps, max_order=cfg.max_order,
                afs=cfg.afs, use_bottleneck=cfg.use_bottleneck,
                high_order_init=cfg.high_order_init, learn_schedule=cfg.learn_schedule,
                learn_scale=cfg.learn_scale, dataset_name=dataset,
            ).to(device).train().requires_grad_(True)

            def step():
                torch.manual_seed(1)
                latents = torch.randn(batch, 3, res, res, device=device)
                cl = labels_for(net, batch, device)
                run_trajectory_batch(net, g_phi, cfg, latents, class_labels=cl)
                g_phi.zero_grad(set_to_none=True)

            step()  # warmup
            sync_reset()
            times = []
            for _ in range(2):
                torch.cuda.synchronize()
                t0 = time.time()
                step()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            sec = sum(times) / len(times)
            print(f"TRAIN {dataset} batch={batch} sec/batch={sec:.3f} "
                  f"sec/traj={sec/batch:.4f} times={[f'{t:.3f}' for t in times]} "
                  f"peakVRAM={peak_gb():.2f}GB "
                  f"sec_per_20k={20000/batch*sec:.0f}", flush=True)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"TRAIN {dataset} OOM at batch={batch}, halving", flush=True)
            batch //= 2


def bench_inception(device):
    from dlms.inception import load_inception

    detector = load_inception(device)
    for res, batch in ((32, 100), (64, 100)):
        x = torch.randint(0, 255, (batch, 3, res, res), device=device).to(torch.float32)
        with torch.no_grad():
            detector(x, return_features=True)  # warmup
        sync_reset()
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                detector(x, return_features=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        sec = sum(times) / len(times)
        print(f"INCEPTION res={res} batch={batch} sec/batch={sec:.4f} "
              f"imgs/sec={batch/sec:.1f} peakVRAM={peak_gb():.2f}GB", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--sample-batch", type=int, default=None)
    p.add_argument("--train-batch", type=int, default=None)
    p.add_argument("--skip-sample", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda")

    if args.dataset == "inception":
        bench_inception(device)
        return

    defaults = {"cifar10": (128, 16), "ffhq": (64, 8), "imagenet64": (32, 4)}
    sb, tb = defaults[args.dataset]
    sb = args.sample_batch or sb
    tb = args.train_batch or tb

    from dlms.models import create_edm_model
    if args.skip_sample:
        net = create_edm_model(args.dataset, device=device)
    else:
        net = bench_sampling(args.dataset, sb, device)
    if not args.skip_train:
        torch.cuda.empty_cache()
        bench_training(args.dataset, tb, device, net)


if __name__ == "__main__":
    main()
