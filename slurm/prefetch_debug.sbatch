#!/bin/bash
#SBATCH -J dlms-prefetch
#SBATCH -A <username>
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o results/slurm-prefetch-%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

command -v apptainer >/dev/null 2>&1 || module load apps/apptainer 2>/dev/null || true

# Connectivity check: all artifact hosts must be reachable from this node.
for host in https://nvlabs-fi-cdn.nvidia.com https://api.ngc.nvidia.com; do
    if ! curl -sI --max-time 15 "$host" >/dev/null; then
        echo "ERROR: no route to $host from $(hostname)." >&2
        echo "Compute nodes appear to lack internet access; run the prefetch" >&2
        echo "on the login node instead (see header of this script)." >&2
        exit 1
    fi
done

apptainer exec --bind "$PWD" dlms.sif \
    bash -c 'export PYTHONPATH="$PWD"; python scripts/prefetch.py'

echo "Prefetch complete. Cached artifacts:"
du -sh external/diff-sampler/amed-solver-main/src/* results/detector_cache results/fid_refs 2>/dev/null
