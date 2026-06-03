#!/bin/bash
# ============================================================
# flip2_predictive.sh
#
# Predictive metrics (R², NLPD, Spearman ρ)
# on alpha-amylase one-to-many split — one-hot, frozen, finetuned variants.
#
# Setups (MODEL_IDX):
#   0 : one-hot + GP             (flip2_onehot.yaml)
#   1 : ESM2-650M frozen + GP    (flip2_llm.yaml)
#   3 : t5-base    frozen + GP   (flip2_llm.yaml)
#   5 : ESM2-650M finetuned + GP  (flip2_pllmphi.yaml) 
#   7 : t5-base    finetuned + GP  (flip2_pllmphi.yaml)
#
# Seeds: 1  15  23  43  50
# Total: 5 setups × 5 seeds = 25 array tasks (0–24)
#
# Submit all:        sbatch --array=0-24 jobs/flip2_predictive.sh
# Submit one-hot:    sbatch --array=0-4  jobs/flip2_predictive.sh
# Submit LLMs only:  sbatch --array=5-24 jobs/flip2_predictive.sh
#
# Final R², NLPD, MSLL, Spearman ρ (train + test) logged under
# prefix "final_train/" and "final_test/" in WandB.
# ============================================================
#SBATCH --job-name=flip2-predictive
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0-24

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
WANDB_PROJECT="gollum_flip2"
SPLIT_NAME="alpha-amylase/one_to_many"
N_ITERS=10
TEST_N_SEQS=200   # use more than 96 for final table values
# ------------------------------------

SEEDS=(1 15 23 43 50)
N_SEEDS=${#SEEDS[@]}

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
SEED="${SEEDS[$SEED_IDX]}"

TRAIN_DATA_PATH="data/flip2/${SPLIT_NAME}_train.csv"
TEST_DATA_PATH="data/flip2/${SPLIT_NAME}_test.csv"

# Per-setup config
case "$MODEL_IDX" in
  0) BASE_CONFIG="configs/flip2_onehot.yaml";   MODEL_TAG="one_hot"          ;;
  1) BASE_CONFIG="configs/flip2_llm.yaml";      MODEL_TAG="esm2_frozen"      ;;
  2) BASE_CONFIG="configs/flip2_llm.yaml";      MODEL_TAG="t5_base_frozen"   ;;
  3) BASE_CONFIG="configs/flip2_pllmphi.yaml";  MODEL_TAG="esm2_finetuned"   ;;
  4) BASE_CONFIG="configs/flip2_pllmphi.yaml";  MODEL_TAG="t5_base_finetuned";;
  *) echo "Unknown MODEL_IDX=$MODEL_IDX"; exit 1 ;;
esac

LLM_MODELS=(
    ""                               # 0: one-hot (no model)
    "facebook/esm2_t33_650M_UR50D"  # 1: ESM2 frozen
    "t5-base"                        # 2: t5-base frozen
    "facebook/esm2_t33_650M_UR50D"  # 3: ESM2 finetuned
    "t5-base"                        # 4: t5-base finetuned
)
LLM_POOLINGS=(
    ""         # 0: one-hot
    "average"  # 1: ESM2 frozen
    "average"  # 2: t5-base frozen
    "average"  # 3: ESM2 finetuned
    "average"  # 4: t5-base finetuned
)
EMBED_DIMS=(
    0     # 0: one-hot (unused)
    1280  # 1: ESM2 frozen
    768   # 2: t5-base frozen
    1280  # 3: ESM2 finetuned
    768   # 4: t5-base finetuned
)
MODEL_NAME="${LLM_MODELS[$MODEL_IDX]}"
POOLING="${LLM_POOLINGS[$MODEL_IDX]}"
EMBED_DIM="${EMBED_DIMS[$MODEL_IDX]}"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : ${SLURM_ARRAY_JOB_ID:-local} / ${SLURM_ARRAY_TASK_ID:-0}"
echo "Setup      : $MODEL_TAG (idx=$MODEL_IDX)"
echo "Model      : ${MODEL_NAME:-one_hot}"
echo "Split      : $SPLIT_NAME"
echo "Seed       : $SEED"
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

TMP_CONFIG=$(mktemp /tmp/gollum_predictive_XXXXXX.yaml)

python3 - <<PYEOF
import yaml

MODEL_IDX       = int("$MODEL_IDX")
MODEL_NAME      = "$MODEL_NAME"
POOLING         = "$POOLING"
EMBED_DIM       = int("$EMBED_DIM")
TRAIN_DATA_PATH = "$TRAIN_DATA_PATH"
TEST_DATA_PATH  = "$TEST_DATA_PATH"
N_ITERS         = int("$N_ITERS")

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

# Patch data paths
cfg["data"]["init_args"]["data_path"]      = TRAIN_DATA_PATH
cfg["data"]["init_args"]["test_data_path"] = TEST_DATA_PATH

# Patch featurizer for all LLM-based configs
if MODEL_IDX > 0:
    feat = cfg["data"]["init_args"]["featurizer"]["init_args"]
    feat["model_name"]     = MODEL_NAME
    feat["pooling_method"] = POOLING

# For DeepGP (pllmphi): also patch finetuning_model model_name, pooling, input_dim
is_deepgp = "DeepGP" in cfg["surrogate_model"]["class_path"]
if is_deepgp:
    ft = cfg["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]
    ft["model_name"]     = MODEL_NAME
    ft["pooling_method"] = POOLING
    ft["input_dim"]      = EMBED_DIM

cfg["n_iters"]       = N_ITERS
cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written → model=$MODEL_TAG  split=$SPLIT_NAME  seed=$SEED")
PYEOF

SPLIT_TAG="${SPLIT_NAME//\//_}"
GROUP="predictive_${SPLIT_TAG}"

python train.py \
    --config      "$TMP_CONFIG" \
    --seed        "$SEED" \
    --group       "$GROUP" \
    --wandb_project "$WANDB_PROJECT" \
    --test_n_seqs "$TEST_N_SEQS"

rm -f "$TMP_CONFIG"
echo "--- Done ($MODEL_TAG, seed=$SEED): $(date) ---"
