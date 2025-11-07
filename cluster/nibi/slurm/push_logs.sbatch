#!/bin/bash
#SBATCH --job-name=uniner-pushlogs
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
echo "[INFO] Listing .out files in $(pwd):"
ls -ltr *.out || true

git add *.out || true
git commit -m "Push Slurm .out logs from submit dir" || true
git push || true

echo "[DONE] Pushed logs (if any)."
