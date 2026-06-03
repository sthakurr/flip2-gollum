#!/bin/bash
# ============================================================
# FLIP2 Frozen-LLM Baseline
#
# Frozen (no finetuning): embeddings computed once at startup,
# plain GP fitted each BO iteration.
#
# Models   : ESMC-600M, ESM2-650M, ProtT5-XL, t5-base
# Datasets : alpha-amylase (x4 splits), ired, nucB, trpB (x3 splits)
# Seeds    : 5 per combination
# Total    : 4 models x 9 datasets x 5 seeds = 180 array tasks
#
# Submit: sbatch jobs/flip2_frozen_llm.sh
# ============================================================
#SBATCH --job-name=flip2-frozen
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0-19        # 2 models x 1 datasets x 10 seeds = 180 tasks

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
WANDB_PROJECT="gollum_flip2"
# ------------------------------------

# ESMC uses flip2_llm_esmc.yaml (native ESM tokenizer + get_huggingface_embeddings ESMC branch)
# All HF models use flip2_llm.yaml (get_huggingface_embeddings, plain GP)
# "EvolutionaryScale/esmc-600m-2024-12"
# "Rostlab/prot_t5_xl_uniref50"
MODELS=(
    "t5-base"
    "facebook/esm2_t33_650M_UR50D"            # 1  ESM2   – hf config
)

MODEL_CONFIGS=(
    "configs/flip2_llm_top5.yaml"        # ESM2
    "configs/flip2_llm_top5.yaml"        # ProtT5
    
)

MODEL_POOLING=(
    "average"   # ESM2
    "average"   # ProtT5
    "average"   # t5-base
)

DATASETS=("ired/two_to_many_train.csv")

SEEDS=(10 150 230 430 500 1 15 23 43 50)

N_MODELS=${#MODELS[@]}
N_DATASETS=${#DATASETS[@]}
N_SEEDS=${#SEEDS[@]}

# Decompose flat task ID → (model, dataset, seed)
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DATASETS * N_SEEDS) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (N_DATASETS * N_SEEDS) ))
DATASET_IDX=$(( REMAINDER / N_SEEDS ))
SEED_IDX=$(( REMAINDER % N_SEEDS ))

MODEL="${MODELS[$MODEL_IDX]}"
BASE_CONFIG="${MODEL_CONFIGS[$MODEL_IDX]}"
POOLING="${MODEL_POOLING[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TRAIN_PATH="data/flip2/${DATASET}"
TEST_PATH="${TRAIN_PATH/_train.csv/_test.csv}"
# DATA_PATH="/capstor/store/cscs/swissai/a131/ssaumya/gollum/data/flip2/${DATASET}"
# TEST_DATA_PATH="${DATA_PATH/_train.csv/_test.csv}"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL"
echo "Config     : $BASE_CONFIG"
echo "Dataset    : $DATASET  ($TRAIN_PATH)"
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

TMP_CONFIG=$(mktemp /tmp/gollum_flip2_frozen_XXXXXX.yaml)

python3 - <<PYEOF
import yaml

MODEL          = "$MODEL"
TRAIN_PATH      = "$TRAIN_PATH"
TEST_PATH = "$TEST_PATH"
POOLING        = "$POOLING"

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["init_args"]["featurizer"]["init_args"]["model_name"]     = MODEL
cfg["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = POOLING
cfg["data"]["init_args"]["train_path"]      = TRAIN_PATH
cfg["data"]["init_args"]["test_path"] = TEST_PATH
cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG  (model={MODEL}, dataset=$TRAIN_PATH, seed=$SEED)")
PYEOF

python train.py \
    --config "$TMP_CONFIG" \
    --seed   "$SEED" \
    --group  "flip2_frozen"

rm -f "$TMP_CONFIG"
echo "--- Done (model=$MODEL, dataset=$DATASET, seed=$SEED): $(date) ---"
