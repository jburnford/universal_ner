Universal NER on Nibi (DRAC) — Setup and Run

This directory provides cluster-friendly templates to run the Universal NER repo on the Nibi cluster (Slurm).

What you need to fill in
- Slurm account/partition/QOS names and resource sizes in the `slurm/*.sbatch` files.
- The actual training/inference entrypoint command(s) for this repo.
- Optional: any required `module load` lines specific to Nibi (CUDA, cuDNN, NCCL, etc.).

Recommended layout on Nibi
- Code: place the cloned repo under `$HOME/projects/universal-ner` (or a similar path you prefer).
- Data: use `$PROJECT` or `$SCRATCH` for large datasets and outputs.

Clone and prepare the repo on Nibi
1) SSH to Nibi: `ssh nibi`
2) Choose a directory and clone your mirror of the repository (the GitHub repo you set up):
   - `mkdir -p $HOME/projects && cd $HOME/projects`
   - `git clone <YOUR_GITHUB_REPO_URL>.git universal-ner`
   - `cd universal-ner`

Environments (pick one per task)
- Inference/Evaluation (vLLM):
  - micromamba: `micromamba create -y -n uniner-infer -f cluster/nibi/env/infer.yml && micromamba activate uniner-infer`
  - conda: `conda env create -n uniner-infer -f cluster/nibi/env/infer.yml && conda activate uniner-infer`
  - Install PyTorch matching Nibi’s CUDA:
    - CUDA 12.1: `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
    - CUDA 11.8: `pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
    - CPU only: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
  - Then install the repo (editable): `pip install -e .`

- Training (FastChat-based):
  - micromamba: `micromamba create -y -n uniner-train -f cluster/nibi/env/train.yml && micromamba activate uniner-train`
  - conda: `conda env create -n uniner-train -f cluster/nibi/env/train.yml && conda activate uniner-train`
  - Install PyTorch (same CUDA rules as above), then `pip install -e .`

Hugging Face cache and logs
- For batch jobs, set caches to a fast local dir to avoid home quota issues:
  - `export HF_HOME=${SLURM_TMPDIR:-$TMPDIR}/hf`
  - `export TRANSFORMERS_CACHE=${HF_HOME}`
  - `export HF_DATASETS_CACHE=${HF_HOME}/datasets`
  - These exports are already included in the Slurm templates.

Submitting jobs
- Edit `cluster/nibi/slurm/train_gpu.sbatch`, `infer_cpu.sbatch`, or `infer_gpu.sbatch` to fill in:
  - `#SBATCH --account=...`, `--partition=...` (or `--qos`), time, memory, GPUs/CPUs.
  - Any Nibi-specific `module load` lines.
  - The final `python ...` command(s) to run your training/inference.
- Submit with `sbatch cluster/nibi/slurm/train_gpu.sbatch` (or the other script).
- Check status with `squeue -u $USER`.
- Inspect logs with `tail -f slurm-%j.out` (job ID replaces `%j`).

Interactive debugging (optional)
- Start an interactive GPU session if your allocation allows it:
  - `salloc --account=<acct> --partition=<gpu_part> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00`
  - Then activate the env and run your command directly to iterate faster.

Notes on data and outputs
- Point training/inference to datasets under `$PROJECT` or `$SCRATCH`.
- Write checkpoints and intermediate files to a project/scratch path, not `$HOME`.

Troubleshooting
- CUDA/driver mismatch: ensure your PyTorch build (cu118/cu121) matches the CUDA runtime modules on Nibi (`nvidia-smi` inside the job should show driver ≥ required).
- Out-of-memory: reduce `batch_size` and/or gradient accum steps; request more `--mem`/`--gres=gpu:*` if available.
- Permission/quota: move caches (`HF_HOME`, etc.) to `$SLURM_TMPDIR` or `$SCRATCH`.

Dependency summary (from upstream repo)
- Inference/Eval: `vllm`, `transformers>=4.30.1`, `fire`, `tokenizers>=0.13.1`, `gradio`, `sentencepiece`.
- Training: `accelerate`, `fastapi`, `gradio==3.23`, `httpx`, `markdown2[all]`, `nh3`, `numpy`,
  `prompt_toolkit>=3.0.0`, `pydantic`, `requests`, `rich>=10.0.0`, `sentencepiece`, `shortuuid`,
  `tokenizers>=0.12.1`, `torch` (per CUDA), `transformers>=4.28.0,<4.29.0`, `uvicorn`, `wandb`.

Nibi tips
- Determine CUDA: inside an interactive GPU node run `nvidia-smi` and `module avail cuda` to choose `cu118` or `cu121` wheels.
- GCC toolchain: vLLM requires `gcc >= 5` at runtime; if needed, `module load gcc/9` or similar.
