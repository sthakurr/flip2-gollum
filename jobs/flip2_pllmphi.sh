#!/bin/bash
# ============================================================
# FLIP2 Benchmark: all models x all 4 enzyme datasets x seeds
#
# Models   : ESMC-600M, ESM2-650M, ProtT5-XL, t5-base
# Datasets : alpha-amylase, ired, nucB, trpB
# Seeds    : 10 per combination
# Total    : 4 models x 2 datasets x 10 seeds = 80 array tasks
#
# Submit: sbatch jobs/flip2_pllmphi.sh
# ============================================================
#SBATCH --job-name=flip2-pllmphi
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
#SBATCH --array=1-59        # 3 models x 2 datasets x 10 seeds

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
SWEEP_GROUP="gollum_flip2_benchmark_v2"
WANDB_PROJECT="gollum_flip2"
# ------------------------------------

# Models: ESMC uses flip2_pllmphi.yaml (esm tokenizer); HF models use flip2_hf.yaml
# "t5-base"
# "facebook/esm2_t33_650M_UR50D"
# "Rostlab/prot_t5_xl_uniref50"
# "EvolutionaryScale/esmc-600m-2024-12"
MODELS=( 
    "t5-base"
    "facebook/esm2_t33_650M_UR50D"
)

# Which base config each model uses
MODEL_CONFIGS=(
    "configs/flip2_pllmphi_top5.yaml"
    "configs/flip2_pllmphi_top5.yaml"         
)

# Pooling method for the static featurizer (used only for HF models in get_tokens mode)
MODEL_POOLING=(
    "average"   
    "average"   
    "average"
    "average"
)

# Dataset names and their training-split CSV names
# "alpha-amylase/one_to_many_train.csv"    "ired/two_to_many_train.csv"            "nucB/two_to_many_train.csv"    "alpha-amylase/close_to_far_train.csv"    "alpha-amylase/close_to_far_train.csv"      "alpha-amylase/far_to_close_train.csv"      "alpha-amylase/by_mutation_train.csv"   "ired/two_to_many_train.csv"   "nucB/two_to_many_train.csv"    
# DATASETS=("alpha-amylase/one_to_many_train.csv"    "ired/two_to_many_train.csv"       "nucB/two_to_many_train.csv"    "alpha-amylase/close_to_far_train.csv"      "alpha-amylase/far_to_close_train.csv"      "alpha-amylase/by_mutation_train.csv" )
DATASETS=("alpha-amylase/close_to_far_train.csv" "ired/two_to_many_train.csv")
#SPLITS=(  "one_to_many"      "two_to_many"     "two_to_many"     "one_to_many")

SEEDS=(1 15 23 43 50 10 150 230 430 500)

BATCH_SIZE=96
N_ITERS=3

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
# SPLIT="${SPLITS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
# DATA_PATH="data/flip2/${DATASET}"
# TEST_DATA_PATH="${DATA_PATH/_train.csv/_test.csv}"
TRAIN_PATH="data/flip2/${DATASET}"
TEST_PATH="${TRAIN_PATH/_train.csv/_test.csv}"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL"
echo "Dataset    : $DATASET  ($TRAIN_PATH)"
echo "Seed       : $SEED"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date       : $(date)"
echo "========================================"

# Activate conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface"

# Build a temporary config with model_name, input_dim, pooling, and data_path patched
TMP_CONFIG=$(mktemp /tmp/gollum_flip2_XXXXXX.yaml)

python3 - <<PYEOF
import yaml, sys

MODEL           = "$MODEL"
TRAIN_PATH      = "$TRAIN_PATH"
TEST_PATH       = "$TEST_PATH"
POOLING         = "$POOLING"

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

# ---- Patch model names ----
cfg["data"]["init_args"]["featurizer"]["init_args"]["model_name"] = MODEL
cfg["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = POOLING
cfg["data"]["init_args"]["train_path"] = TRAIN_PATH
cfg["data"]["init_args"]["test_path"] = TEST_PATH
cfg["bo"]["init_args"]["batch_size"] = $BATCH_SIZE
cfg["n_iters"] = $N_ITERS
cfg["save_checkpoints"] = False

ft = cfg["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]
ft["model_name"] = MODEL
ft["pooling_method"] = POOLING

# ---- Auto-set input_dim from known embedding sizes ----
EMBEDDING_SIZES = {
    "EvolutionaryScale/esmc-600m-2024-12": 1152,
    "facebook/esm2_t33_650M_UR50D":        1280,
    "Rostlab/prot_t5_xl_uniref50":         1024,
    "t5-base":                             768,
    "EvolutionaryScale/esm3-sm-open-v1":   1024,
}
if MODEL in EMBEDDING_SIZES:
    ft["input_dim"] = EMBEDDING_SIZES[MODEL]

cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG  (model={MODEL}, dataset=$DATASET, seed=$SEED)")
PYEOF

# Run training
python train.py \
    --config  "$TMP_CONFIG" \
    --seed    "$SEED" \
    --group   "flip2_pllmphi"

rm -f "$TMP_CONFIG"
echo "--- Done (model=$MODEL, dataset=$DATASET, seed=$SEED): $(date) ---"
