#!/bin/bash
# ============================================================
# W&B Sweep: flip2_top5 (LLM, pLLM, one-hot)
# Sweeps n_iters, n_train_iter, seed across 3 configs.
#
# 1. Register the sweep first:
#      wandb sweep configs/sweep_top5.yaml
# 2. Paste the sweep ID into SWEEP_ID below, then:
#      sbatch jobs/sweep_top5.sh
#
# Each array task runs one wandb agent that picks up trials
# from the sweep queue until it is exhausted.
# Total runs: 3 configs x 1 n_iters x 3 n_train_iter x 3 seeds = 27
# ============================================================
#SBATCH --job-name=sweep-top5
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0-8   # 3 parallel agents; raise if you want more parallelism

# ---- Fill in after: wandb sweep configs/sweep_top5.yaml ----
WANDB_ENTITY="liac"
SWEEP_ID="4e1i1bbe"
# ------------------------------------------------------------

CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: SWEEP_ID is not set. Run 'wandb sweep configs/sweep_top5.yaml' first."
    exit 1
fi

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : ${SLURM_ARRAY_JOB_ID:-N/A} / ${SLURM_ARRAY_TASK_ID:-0}"
echo "Sweep ID   : $SWEEP_ID"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date       : $(date)"
echo "========================================"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface"

wandb agent "${WANDB_ENTITY}/gollum_flip2/${SWEEP_ID}"

echo "--- Agent done: $(date) ---"
