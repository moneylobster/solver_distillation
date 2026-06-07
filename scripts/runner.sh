#!/usr/bin/env bash
# Work-stealing job runner: each instance (one per GPU) walks the manifest and
# claims unclaimed lines via atomic `mkdir` locks on the shared filesystem,
# so any number of concurrent runners self-balance without coordination.
#
# Usage (one instance per GPU, from the repo root):
#   bash scripts/runner.sh [scripts/manifest.txt]
#
# State lives in results/joblocks/job-NNN/: log.txt + DONE or FAILED marker.
# Re-running skips claimed jobs; to retry failures, delete their lock dirs:
#   find results/joblocks -name FAILED -exec dirname {} \; | xargs rm -rf
set -u

MANIFEST="${1:-scripts/manifest.txt}"
LOCKROOT="results/joblocks"
mkdir -p "$LOCKROOT"

# Unique rendezvous port per runner: diff-solvers' sample.py initializes
# torch.distributed (world size 1) and defaults to port 29500, which would
# collide between concurrent runners on the same node.
export MASTER_PORT=$((20000 + RANDOM % 20000))

# Unbuffered Python: stdout is block-buffered when redirected to log files,
# which makes sparse training prints invisible until process exit.
export PYTHONUNBUFFERED=1

lineno=0
claimed=0
while IFS= read -r cmd; do
    lineno=$((lineno + 1))
    case "$cmd" in ""|\#*) continue;; esac
    lock="$LOCKROOT/job-$(printf '%03d' "$lineno")"
    if mkdir "$lock" 2>/dev/null; then
        claimed=$((claimed + 1))
        printf '%s\n' "$cmd" > "$lock/cmd.txt"
        echo "[runner $$] line $lineno: $cmd"
        if bash -c "$cmd" >"$lock/log.txt" 2>&1; then
            touch "$lock/DONE"
            echo "[runner $$] line $lineno: DONE"
        else
            touch "$lock/FAILED"
            echo "[runner $$] line $lineno: FAILED (see $lock/log.txt)"
        fi
    fi
done < "$MANIFEST"

echo "[runner $$] manifest exhausted ($claimed jobs executed here)."
