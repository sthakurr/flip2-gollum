#!/bin/bash
# ============================================================
# FLIP2 One-Hot Baseline: all 5 datasets x 5 seeds
#
# Representation : protein one-hot (20 AA x sequence length)
# Surrogate      : plain GP (no LLM, no finetuning)
# Datasets       : alpha-amylase (x2 splits), ired, nucB, trpB
# Seeds          : 5 per dataset
# Total          : 5 datasets x 5 seeds = 25 array tasks
#
# Submit: sbatch jobs/flip2_onehot.sh
# ============================================================
#SBATCH --job-name=flip2-onehot
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
WANDB_PROJECT="gollum_flip2"
BASE_CONFIG="configs/flip2_onehot_top5.yaml"
# ------------------------------------

DATASETS=("alpha-amylase/one_to_many_train.csv"    "ired/two_to_many_train.csv"       "nucB/two_to_many_train.csv"    "alpha-amylase/close_to_far_train.csv"      "alpha-amylase/far_to_close_train.csv"      "alpha-amylase/by_mutation_train.csv"  "trpB/two_to_many_train.csv" "trpB/by_position_train.csv")
SEEDS=(10 150 230 430 500 1 15 23 43 50)

N_DATASETS=${#DATASETS[@]}
N_SEEDS=${#SEEDS[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

DATASET="${DATASETS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TRAIN_PATH="data/flip2/${DATASET}"
TEST_PATH="${DATA_PATH/_train.csv/_test.csv}"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Dataset    : $DATASET  ($DATA_PATH)"
echo "Seed       : $SEED"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date       : $(date)"
echo "========================================"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

TMP_CONFIG=$(mktemp /tmp/gollum_flip2_onehot_XXXXXX.yaml)

python3 - <<PYEOF
import yaml

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["init_args"]["train_path"]      = "$TRAIN_PATH"
cfg["data"]["init_args"]["test_path"] = "$TEST_PATH"
cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG  (dataset=$TRAIN_PATH, seed=$SEED)")
PYEOF

python train.py \
    --config "$TMP_CONFIG" \
    --seed   "$SEED" \
    --group  "flip2_onehot"

rm -f "$TMP_CONFIG"
echo "--- Done (dataset=$DATASET, seed=$SEED): $(date) ---"
