#!/bin/bash

#SBATCH -J dlms
#SBATCH -A <username>
#SBATCH -p <queue>
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH -o results/slurm-%A_%a.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

if [ -d /localscratch ]; then
    export TMPDIR="/localscratch/$USER"
else
    export TMPDIR="/tmp/$USER"
fi
mkdir -p "$TMPDIR"
export DLMS_SAMPLES_DIR="$TMPDIR/dlms-samples-$SLURM_JOB_ID"
trap 'rm -rf "$DLMS_SAMPLES_DIR"' EXIT

command -v apptainer >/dev/null 2>&1 || module load apps/apptainer 2>/dev/null || true

apptainer exec --nv --bind "$PWD" --bind "$TMPDIR" dlms.sif \
    bash -c 'export PYTHONPATH="$PWD"; bash scripts/runner.sh scripts/manifest.txt'
