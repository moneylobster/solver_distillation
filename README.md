# DLMS: Linear Multistep Solver Distillation — Reimplementation

Reimplementation of: Yuchen Liang, Xiangzhong Fang, Hanting Chen, Yunhe Wang. "Linear Multistep Solver Distillation for Fast Sampling of Diffusion Models." ICLR 2025. [OpenReview](https://openreview.net/forum?id=vkOFOUDLTn)

Course project for CENG502.

## Paper Description

This paper introduces an improvement on the multistep solvers used to speed up sampling from denoising diffusion models. This technique involves writing out the diffusion process as a Probability Flow ODE, and then using a multistep solver to solve this ODE (as opposed to single-step Euler which is how diffusion models are usually evaluated), specifically the exponential integration method. In this method, previous evaluations of the ODE are used to fit a polynomial which is used to solve for the next step (to put it concisely). Previous works in this line of research (DEIS, DPM-Solver++) use handcrafted formulas to weight the past evaluations of the denoiser to solve the ODE, while this paper trains a small network which outputs these weights and the timing information for the next output, at each solution step. This network is trained by distilling using a teacher DPM-Solver++'s outputs, and the network is trained to match the trajectory while being given less denoiser evaluations than its teacher. The resulting model can match/beat the FID of the other methods while doing less denoiser evaluations. It's also noteworthy that the sampler optimization process takes much less time (10x faster) compared to RL-based alternatives in the literature.

## Reproduced results (scope)

- Table 5 rows for CIFAR-10, FFHQ-64 and ImageNet-64
- Figure 3 (a–c)
- Table 3

Baseline curves (DEIS, DPM-Solver++, UniPC) are re-run with the [diff-sampler](https://github.com/zju-pi/diff-sampler) toolbox at its recommended settings (which match the paper's baseline numbers).

### Results

**NOTE: The evaluation/FID calculations are still underway at time of upload, with less than an hour left until completion. I will update the README with the results once it has finished.**

Paper targets for reference (Table 5, DLMS FID at NFE 4–10):

| Dataset     | 4     | 5    | 6    | 7    | 8    | 9    | 10   |
|-------------|-------|------|------|------|------|------|------|
| CIFAR-10    | 4.52  | 3.23 | 2.81 | 2.53 | 2.43 | 2.37 | 2.24 |
| FFHQ-64     | 9.63  | 6.85 | 5.82 | 5.16 | 4.81 | 4.23 | 4.12 |
| ImageNet-64 | 10.07 | 7.16 | 7.08 | 6.31 | 5.93 | 4.57 | 4.30 |

## Code structure

`dlms` contains the multi-step solver implementation.
- `bootstrap.py`: Path utils
- `config.py`: Training configurations
- `ema.py`: EMA implementation (training process optimizes over EMA half-life as a hyperparameter)
- `fid_utils.py`: FID calculation
- `generate.py`: Seeded image generation
- `inception.py`: Inception network for FID calculation
- `models.py`: Loads diffusion models
- `network.py`: DLMS designer network implementation
- `solver.py`: DLMS solver implementation
- `teacher.py`: DPM-Solver++ used as teacher
- `train.py`: Distillation/training process
`scripts` contains scripts to generate/test/train various parts of the codebase.
`slurm` contains slurm batch scripts.
`apptainer` has a file to generate an apptainer container for the dependencies.
`tests` contains generated tests to verify code works as expected
`external/diff-sampler` is the git submodule containing implementations of the other multistep methods
`results` contains the outputs once they have been generated.

## Installation

```bash
git clone --recursive https://github.com/moneylobster/solver_distillation
uv sync
```

### Running on a SLURM cluster

I used the TRUBA HPC cluster to generate my results.

```bash
# Build the container
apptainer build --fakeroot dlms.sif apptainer/dlms.def

# Copy to cluster

# Download checkpoints/FID assets, create work schedule (job manifest)
apptainer exec dlms.sif bash -c 'export PYTHONPATH=$PWD; python scripts/prefetch.py'
apptainer exec dlms.sif bash -c 'export PYTHONPATH=$PWD; python scripts/make_manifest.py'

# Edit slurm file and fill in queue and username, then run. (Takes up multiple GPUs)
sbatch slurm/dlms_array.sbatch

# Copy outputs back
rsync -a --exclude joblocks results/ /arf/home/$USER/dlms-results/

# make_figures.py / make_tables.py to generate figures/tables.
```

## Running

One DLMS solver is trained per (dataset, NFE). Full workflow for one number:

```bash
# Train (Algorithm 1; ~minutes on a fast GPU for CIFAR-10)
uv run python scripts/train_dlms.py --dataset cifar10 --nfe 5
# Pick the best EMA variant by FID-5k proxy (writes selection.json)
uv run python scripts/select_ema.py --snapshot results/runs/cifar10-nfe5/snapshot-020032.pt
# Generate 50k images + FID (uses selection.json automatically)
uv run python scripts/eval_dlms.py --dataset cifar10 --nfes 5
```

Sweeps (Table 5 / Fig 3a–c):

```bash
for n in 4 5 6 7 8 9 10:  train_dlms --dataset cifar10 --nfe $n ; select_ema ; done
uv run python scripts/eval_dlms.py --dataset cifar10 --nfes 4-10
uv run python scripts/run_baselines.py --dataset cifar10 --solvers deis,dpmpp,unipc --nfes 4-10
# same with --dataset ffhq / imagenet64 (lower --batch-gpu on small GPUs)
uv run python scripts/make_figures.py            # results/figures/fig3.png
uv run python scripts/make_tables.py             # results/tables/table5.md, table3.md
```

Table 3 ablations (CIFAR-10, NFE 4/6/8/10): train with one switch off, then evaluate under a matching label:

```bash
uv run python scripts/train_dlms.py --dataset cifar10 --nfe 4 --no-afs --tag wo_afs
uv run python scripts/eval_dlms.py --dataset cifar10 --nfes 4 \
    --run-pattern "{dataset}-nfe{nfe}-wo_afs" --solver-label dlms_wo_afs
```

Switches: `--no-afs`, `--no-bottleneck`, `--no-high-order-init`, `--no-inception-loss`, `--no-time-scaling`, `--no-adaptive-schedule` (labels `dlms_wo_afs`, `dlms_wo_bottleneck`, `dlms_wo_high_order`, `dlms_wo_inception`, `dlms_wo_time_scaling`, `dlms_wo_adaptive_schedule`).

A sanity check that validates the whole eval pipeline against published
numbers (diff-solvers README: iPNDM-4 on CIFAR-10 at NFE 6 → FID 7.05):

```bash
uv run python scripts/run_baselines.py --dataset cifar10 --solvers ipndm --nfes 6
```

## Attribution

LLMs were used to assist in code generation during this project.

`dlms/network.py` copies two layers (EDM-style `Linear`, `PositionalEmbedding`) from  [zju-pi/diff-sampler](https://github.com/zju-pi/diff-sampler)'s AMED-Solver code, `dlms/generate.py` and `dlms/fid_utils.py` are modifications of diff-sampler's `sample.py`/`fid.py` to keep the evaluation protocol identical.
