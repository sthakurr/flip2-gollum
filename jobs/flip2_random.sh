#!/bin/bash
# ============================================================
# FLIP2 Random Baseline: all 5 datasets x 5 seeds
#
# Representation : none (random selection, no model)
# Datasets       : alpha-amylase (x2 splits), ired, nucB, trpB
# Seeds          : 5 per dataset (run sequentially via --n_seeds)
# Total          : 5 array tasks (one per dataset)
#
# Submit: sbatch jobs/flip2_random.sh
# ============================================================
#SBATCH --job-name=flip2-random
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --account=a131
#SBATCH --array=0-2

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
WANDB_PROJECT="gollum-flip2-final"
BASE_CONFIG="configs/flip2_random.yaml"
N_SEEDS=5
BASE_SEED=1
# ------------------------------------
    # "alpha-amylase/one_to_many_train.csv"
    # "alpha-amylase/close_to_far_train.csv"
    # "ired/two_to_many_train.csv"
    # "nucB/two_to_many_train.csv"
DATASETS=("alpha-amylase/far_to_close_train.csv"  "alpha-amylase/by_mutation_train.csv"  "trpB/two_to_many_train.csv")

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATA_PATH="data/flip2/${DATASET}"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Dataset    : $DATASET  ($DATA_PATH)"
echo "N seeds    : $N_SEEDS (starting from seed $BASE_SEED)"
echo "Node       : $(hostname)"
echo "Date       : $(date)"
echo "========================================"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

TMP_CONFIG=$(mktemp /tmp/gollum_flip2_random_XXXXXX.yaml)

python3 - <<PYEOF
import yaml

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["init_args"]["data_path"] = "$DATA_PATH"
cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG  (dataset=$DATA_PATH)")
PYEOF

python random_baseline.py \
    --config   "$TMP_CONFIG" \
    --seed     "$BASE_SEED" \
    --n_seeds  "$N_SEEDS" \
    --group    "flip2_random"

rm -f "$TMP_CONFIG"
echo "--- Done (dataset=$DATASET): $(date) ---"
