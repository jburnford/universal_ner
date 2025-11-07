#!/bin/bash
#SBATCH --job-name=uniner-train
#SBATCH --account=def-jic823
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
echo "[INFO] Job $SLURM_JOBID on $(hostname) started at $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Optional: load site-specific modules (edit as needed)
module purge || true
module load python/3.12
module load cuda/12.2
module load gcc arrow/18.1.0

# Activate conda/mamba environment (choose one)
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate uniner-train
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate uniner-train
elif [ -f "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate" ]; then
  source "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate"
else
  echo "[ERROR] No conda/micromamba found in PATH." >&2
  exit 1
fi

# Put caches and temp on fast local storage
export HF_HOME=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hf
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export MPLCONFIGDIR=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mpl
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

# Move to repo root
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Sanity: show GPU/driver
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "[INFO] Python: $(python --version)"
echo "[INFO] Torch:  $(python -c 'import torch,sys; print(torch.__version__, torch.cuda.is_available())' || echo 'not installed')"

# Training example (FastChat-based, from upstream README)
# Requires the uniner-train environment and a compatible Torch+CUDA install.
# Adjust MODEL/DATA paths, world size, and hyperparameters as needed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
MODEL=yahma/llama-7b-hf
DATA=./src/train/playground/data/ner.json

mkdir -p saved_models
torchrun --nproc_per_node=1 --master_port=20001 src/train/fastchat/train/train_mem.py \
    --model_name_or_path ${MODEL}  \
    --data_path ${DATA} \
    --run_name universalner \
    --bf16 True \
    --output_dir saved_models/universalner \
    --dataloader_num_workers 8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config src/train/fsdp_config.json \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True

echo "[INFO] Training job completed at $(date)"
