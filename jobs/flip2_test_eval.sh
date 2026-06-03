#!/bin/bash
# ============================================================
# FLIP2 Test-Eval: surrogate Spearman correlation on test split
#
# Runs 3 BO epochs on the training pool, then evaluates the
# surrogate on the first 96 sequences of the test split and
# reports / plots the Spearman correlation.
#
# Models   : ESM2-650M, ProtT5-XL, t5-base, ESMC-600M
# Seeds    : 5 per model
# Total    : 4 models x 5 seeds = 20 array tasks
#
# SPLIT_NAME controls which dataset/split to evaluate.
# Set it here or override at submit time:
#   sbatch --export=ALL,SPLIT_NAME=alpha-amylase/close_to_far jobs/flip2_test_eval.sh
#
# Valid SPLIT_NAME values (must have matching _train.csv and _test.csv):
#   trpB/one_to_many
#   alpha-amylase/one_to_many
#   alpha-amylase/close_to_far
#   ired/two_to_many
#   nucB/two_to_many
#
# Submit: sbatch jobs/flip2_test_eval.sh
# ============================================================
# Array size guide:
#   LLM models (flip2_pllmphi.yaml)  → --array=0-19   (4 models x 5 seeds)
#   One-hot    (flip2_onehot.yaml)   → --array=0-4    (1 model  x 5 seeds)
#SBATCH --job-name=flip2-test-eval
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
#SBATCH --array=0        # 4 models x 5 seeds

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
WANDB_PROJECT="gollum_flip2"
BASE_CONFIG="configs/flip2_pllmphi.yaml"
SPLIT_NAME="${SPLIT_NAME:-alpha-amylase/close_to_far}"   # override via --export
N_ITERS=10
TEST_N_SEQS=96
# ------------------------------------

MODELS=(
    "facebook/esm2_t33_650M_UR50D"
    "Rostlab/prot_t5_xl_uniref50"
    "t5-base"
    "EvolutionaryScale/esmc-600m-2024-12"
)

MODEL_POOLING=(
    "average"   # ESM2
    "average"   # ProtT5
    "average"   # t5-base
    "average"   # ESMC
)

EMBEDDING_SIZES=(1280 1024 768 1152)   # ESM2, ProtT5, t5-base, ESMC

SEEDS=(1 15 23 43 50)

N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

MODEL="${MODELS[$MODEL_IDX]}"
POOLING="${MODEL_POOLING[$MODEL_IDX]}"
EMBED_DIM="${EMBEDDING_SIZES[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

TRAIN_DATA_PATH="data/flip2/${SPLIT_NAME}_train.csv"
TEST_DATA_PATH="data/flip2/${SPLIT_NAME}_test.csv"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL  (dim=$EMBED_DIM)"
echo "Split      : $SPLIT_NAME"
echo "Train data : $TRAIN_DATA_PATH"
echo "Test data  : $TEST_DATA_PATH"
echo "n_iters    : $N_ITERS"
echo "test_n_seqs: $TEST_N_SEQS"
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

TMP_CONFIG=$(mktemp /tmp/gollum_flip2_testeval_XXXXXX.yaml)

python3 - <<PYEOF
import yaml

MODEL           = "$MODEL"
POOLING         = "$POOLING"
EMBED_DIM       = int("$EMBED_DIM")
TRAIN_DATA_PATH = "$TRAIN_DATA_PATH"
TEST_DATA_PATH  = "$TEST_DATA_PATH"
N_ITERS         = int("$N_ITERS")

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

# Featurizer — only patch model/pooling for LLM-based configs
feat_args = cfg["data"]["init_args"]["featurizer"]["init_args"]
if "model_name" in feat_args:
    feat_args["model_name"]    = MODEL
    feat_args["pooling_method"] = POOLING

# Data paths (always)
cfg["data"]["init_args"]["data_path"]      = TRAIN_DATA_PATH
cfg["data"]["init_args"]["test_data_path"] = TEST_DATA_PATH

# Finetuning model — only for DeepGP configs
sm_args = cfg["surrogate_model"]["init_args"]
if "finetuning_model" in sm_args:
    ft = sm_args["finetuning_model"]["init_args"]
    ft["model_name"]     = MODEL
    ft["pooling_method"] = POOLING
    ft["input_dim"]      = EMBED_DIM

# BO iterations
cfg["n_iters"] = N_ITERS

cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG  (model={MODEL}, split=$SPLIT_NAME, seed=$SEED, n_iters={N_ITERS})")
PYEOF

SPLIT_TAG="${SPLIT_NAME//\//_}"

python train.py \
    --config      "$TMP_CONFIG" \
    --seed        "$SEED" \
    --group       "flip2_eval_${SPLIT_TAG}" \
    --test_n_seqs "$TEST_N_SEQS"

rm -f "$TMP_CONFIG"
echo "--- Done (model=$MODEL, split=$SPLIT_NAME, seed=$SEED): $(date) ---"
