#!/bin/bash
#SBATCH --job-name=gollum-rep-comparison
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0-24         # 5 arms x 5 seeds = 25 parallel runs

# --- Adjust these to your setup ---
CONDA_BASE="/iopsstor/scratch/cscs/ssaumya/piflow/miniforge3"
CONDA_ENV="gollum"
ARMS=(
  "configs/flip2_arms/onehot.yaml"
  "configs/flip2_arms/esm2_dense.yaml"
  "configs/flip2_arms/esm2_pca.yaml"
  "configs/flip2_arms/esmc_dense.yaml"
  "configs/flip2_arms/esmc_sae.yaml"
)
SEEDS=(23 43 44 45 46)
MODE="both"                  # gate | bo | both (both = Phase-1 BO collect, then test eval)
SWEEP_GROUP="flip2-rep-comparison"
# -----------------------------------

N_SEEDS=${#SEEDS[@]}
ARM_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

ARM=${ARMS[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

set -euo pipefail
mkdir -p logs

echo "Array job:  $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Arm:        $ARM"
echo "Seed:       $SEED"
echo "Mode:       $MODE"
echo "Date:       $(date)"
echo "---"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ESM-C arms need the `esm` package and (for the SAE arm) trained weights; if
# they're unavailable the run errors clearly and the other array tasks proceed.
python train.py \
    --config "$ARM" \
    --mode "$MODE" \
    --seed "$SEED" \
    --group "$SWEEP_GROUP" || echo "Arm $ARM (seed $SEED) failed — skipping."

echo "--- Done (arm=$ARM, seed=$SEED): $(date) ---"
