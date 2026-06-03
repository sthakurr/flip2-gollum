#!/bin/bash
# ============================================================
# FLIP2 Benchmark Test: all 4 models x 1 seed
# Logs to wandb project gollum_flip2_test to verify the full
# pipeline is working before submitting the full sweep.
#
# Submit: sbatch jobs/flip2_benchmark_test.sh
# ============================================================
#SBATCH --job-name=flip2-test
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0        # one task per model

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
SEED=1
DATASET="alpha-amylase"
SPLIT="close_to_far"
SWEEP_GROUP="flip2_test"
WANDB_PROJECT="gollum_flip2_test"
# ------------------------------------

MODELS=(
    "EvolutionaryScale/esmc-600m-2024-12"   # 0  ESMC
    "facebook/esm2_t33_650M_UR50D"           # 1  ESM2
    "Rostlab/prot_t5_xl_uniref50"            # 2  ProtT5
    "t5-base"                                # 3  t5-base
)

MODEL_CONFIGS=(
    "configs/flip2_hf.yaml"
    "configs/flip2_hf.yaml"        # ESM2
    "configs/flip2_hf.yaml"        # ProtT5
    "configs/flip2_hf.yaml"        # t5-base
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BASE_CONFIG="${MODEL_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
DATA_PATH="data/flip2/${DATASET}/${SPLIT}_train.csv"

set -euo pipefail
mkdir -p logs

echo "========================================"
# echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL"
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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface"

EMBEDDING_SIZES=(
    "t5-base:768"
    "EvolutionaryScale/esmc-600m-2024-12:1152"
    "facebook/esm2_t33_650M_UR50D:1280"
    "Rostlab/prot_t5_xl_uniref50:1024"
)

TMP_CONFIG=$(mktemp /tmp/gollum_flip2_test_XXXXXX.yaml)

python3 - <<PYEOF
import yaml

MODEL     = "$MODEL"
DATA_PATH = "$DATA_PATH"

EMBEDDING_SIZES = {
    "EvolutionaryScale/esmc-600m-2024-12": 1152,
    "facebook/esm2_t33_650M_UR50D":        1280,
    "Rostlab/prot_t5_xl_uniref50":         1024,
    "t5-base":                             768,
}

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["init_args"]["featurizer"]["init_args"]["model_name"] = MODEL
cfg["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = "average"
cfg["data"]["init_args"]["data_path"] = DATA_PATH

ft = cfg["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]
ft["model_name"] = MODEL
ft["pooling_method"] = "average"
if MODEL in EMBEDDING_SIZES:
    ft["input_dim"] = EMBEDDING_SIZES[MODEL]

cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG")
PYEOF

python train.py \
    --config "$TMP_CONFIG" \
    --seed   "$SEED" \
    --group  "$SWEEP_GROUP"

rm -f "$TMP_CONFIG"
echo "--- Done (model=$MODEL, seed=$SEED): $(date) ---"
