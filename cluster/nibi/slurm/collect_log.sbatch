#!/bin/bash
#SBATCH --job-name=uniner-collect
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

set -euo pipefail

JOBID=${JOBID:-}
if [ -z "$JOBID" ]; then
  echo "ERROR: Set JOBID via --export=JOBID=<id> when submitting." >&2
  exit 2
fi

REPO_DIR="$SLURM_SUBMIT_DIR"
LOG_SRC="uniner-hf-${JOBID}.out"
LOG_DST_DIR="${REPO_DIR}/cluster/nibi/logs"
LOG_DST="${LOG_DST_DIR}/uniner-hf-${JOBID}.out"

mkdir -p "$LOG_DST_DIR"

if [ ! -f "$LOG_SRC" ]; then
  echo "WARN: Log file $LOG_SRC not found in $REPO_DIR" >&2
  ls -ltr | tail -n 20 || true
  exit 3
fi

cp "$LOG_SRC" "$LOG_DST"

cd "$REPO_DIR"
git add "$LOG_DST"
git commit -m "Add HF inference log for job ${JOBID}" || true
git push || true

echo "[COLLECTED] $LOG_DST pushed to repo."
