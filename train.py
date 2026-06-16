"""
Training script for Gollum BO. Can run in "gate" mode to just evaluate surrogate quality on a held-aside test split, "bo" mode to run the Phase-2 BO loop, or "both" to do BO then gate. Example
Usage: python train.py --config configs/flip2_arms/esm2_dense.yaml --seed 1 --n_iters 3 --mode both
"""
import warnings
from botorch.exceptions import InputDataWarning

import os
import re
import torch
import logging

warnings.filterwarnings("ignore", category=InputDataWarning)
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
warnings.filterwarnings(
    "ignore",
    message="ExpectedImprovement has known numerical issues that lead to suboptimal optimization performance"
)

class IgnoreDeviceFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


logger.addFilter(IgnoreDeviceFilter())


warnings.filterwarnings(
    "ignore",
    message=re.escape(
        "You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. "
        "To properly utilize them, you should set "
        "`torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision "
        "for performance. For more details, read "
        "https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html"
    ),
    category=UserWarning,
    module="torch",
)

warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
    module="pytorch_lightning.trainer.connectors.data_connector",
)

from gollum.data.module import BaseDataModule
from gollum.bo.optimizer import BotorchOptimizer


from gollum.metrics import (
    calculate_data_stats,
    log_bo_metrics,
    log_data_stats,
    log_surrogate_eval,
    log_prior_correlation,
)



torch.set_float32_matmul_precision("high")


from pytorch_lightning import seed_everything
import wandb
from tqdm import tqdm
from gollum.utils.config import flatten

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
)
from gollum.utils.config import instantiate_class
from botorch.acquisition import AcquisitionFunction
from gollum.surrogate_models.gp import SurrogateModel



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_EMBEDDING_SIZES = {
    "WhereIsAI/UAE-Large-V1": 1024,
    "nomic-ai/modernbert-embed-base": 768,
    "Qwen/Qwen2-7B-Instruct": 3584,
    "t5-base": 768,
    "mistralai/Mistral-7B-Instruct-v0.2": 4096,
    "text-embedding-3-large": 3072,
    "nomic-ai/modernbert-embed-base;get_huggingface_embeddings;normalize:False;pooling:cls": 768,
    "Qwen/Qwen2-7B-Instruct;get_huggingface_embeddings;normalize:False;pooling:last_token": 3584,
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": 768,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "Rostlab/prot_t5_xl_uniref50": 1024,
}


def configure_embedding_size(config, model_name):
    if model_name in MODEL_EMBEDDING_SIZES:
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"][
            "input_dim"
        ] = MODEL_EMBEDDING_SIZES[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in supported models.")
    return config


def configure_pooling_method(config, model_name):
    hugging_face_models = {
        "nomic-ai/modernbert-embed-base": "cls",
        "mixedbread-ai/mxbai-embed-large-v1": "cls",
        "WhereIsAI/UAE-Large-V1": "cls",
        "Alibaba-NLP/gte-Qwen1.5-7B-instruct": "last_token_pool",
        "GT4SD/multitask-text-and-chemistry-t5-base-augm": "average",
        "GT4SD/multitask-text-and-chemistry-t5-base-augm-from-rxn": "average",
        "t5-base": "average",
        "Qwen/Qwen2-7B-Instruct": "last_token_pool",
        "facebook/esm2_t33_650M_UR50D": "average",
        "Rostlab/prot_t5_xl_uniref50": "average",
    }

    if model_name in hugging_face_models:
        config["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = (
            hugging_face_models[model_name]
        )
    else:
        raise ValueError(
            f"Model {model_name} not found in supported models. Please specify pooling method manually."
        )

    if (
        config["surrogate_model"]["class_path"]
        in ["gollum.surrogate_models.gp.DeepGP"]
    ):
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"][
            "pooling_method"
        ] = hugging_face_models[model_name]

    return config


def configure_benchmark_datasets(config):
    """Configure dataset settings based on benchmark name."""
    benchmark = config["benchmark"]
    if benchmark.startswith("bh"):
        reaction_num = benchmark[-1]

        config["data"]["init_args"][
            "data_path"
        ] = f"data/reactions/buchwald-hartwig/bh_reaction_{reaction_num}_procedure_template_basic.csv"
        
        config["data"]["init_args"]["target_column"] = "objective"
        config["data"]["init_args"]["maximize"] = True
    

    return config


def validate_configuration(config):
    """Validate that the configuration is consistent."""
    surrogate_class = config["surrogate_model"]["class_path"]
    
    featurizer_config = config["data"]["init_args"]["featurizer"]["init_args"]
    representation = featurizer_config.get("representation")
    
    # check for invalid configurations
    if surrogate_class == "gollum.surrogate_models.gp.GP" and representation == "get_tokens":
        raise ValueError("Standard GP or PLLM shouldn't use 'get_tokens'. This is for trainable LLM models only.")
    if surrogate_class == "gollum.surrogate_models.gp.DeepGP" and representation != "get_tokens":
        raise ValueError("DeepGP surrogate requires 'get_tokens' representation.")

    # Validate the one-word kernel selector, if used.
    surrogate_args = config["surrogate_model"].get("init_args", {}) or {}
    kernel = surrogate_args.get("kernel")
    if kernel is not None:
        from gollum.surrogate_models.kernels import VALID_KERNELS

        if kernel not in VALID_KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'; choose one of {VALID_KERNELS}."
            )
        if surrogate_args.get("covar_module") is not None:
            print(
                "WARNING: both 'covar_module' and 'kernel' are set; the explicit "
                "covar_module takes precedence and 'kernel' is ignored."
            )
    
    # Ensure model embedding sizes are correct
    model_name = featurizer_config.get("model_name")
    if model_name in MODEL_EMBEDDING_SIZES:
        embedding_size = MODEL_EMBEDDING_SIZES[model_name]
        
        if "surrogate_model" in config and "init_args" in config["surrogate_model"]:
            if "finetuning_model" in config["surrogate_model"]["init_args"]:
                current_dim = config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"].get("input_dim")
                if current_dim != embedding_size:
                    print(f"Updating input_dim from {current_dim} to {embedding_size} for {model_name}")
                    config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]["input_dim"] = embedding_size
    
    return config


def setup_data(config):
    initializer = instantiate_class(
        config["data"]["init_args"]["initializer"], seed=config["seed"]
    )
    featurizer = instantiate_class(config["data"]["init_args"]["featurizer"])
    dm = instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=config["data"]["init_args"]["normalize_input"],
        maximize=config["data"]["init_args"]["maximize"],
    )

    return dm




def setup_bo_optimizer(config, design_space):
    bo_config = config["bo"]["init_args"]
    surrogate_model_config = config["surrogate_model"]
    acquisition_config = config["acquisition"]
    bo = BotorchOptimizer(
        design_space=design_space,
        surrogate_model_config=surrogate_model_config,
        acq_function_config=acquisition_config,
        batch_strategy=bo_config["batch_strategy"],
        batch_size=bo_config["batch_size"],
    )
    return bo


def run_gate(config, dm, bo):
    """Surrogate-quality gate: fit the surrogate on the current train set (the
    Phase-1-collected points) and evaluate how well it ranks the held-aside test
    split. Answers "does this representation transfer train->test?". When run
    after run_bo (mode=both) the train set is the full 96 + n_iters*batch points.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dm.test_x is None:
        raise ValueError(
            "Gate mode needs a held-aside test set; set respect_split: true."
        )
    train_x = dm.train_x.clone().to(device)
    train_y = dm.train_y.clone().to(device)
    test_x = dm.test_x.clone().to(device)
    test_y = dm.test_y.clone().to(device)

    print(f"Gate: fitting on {train_x.shape[0]} train points, "
          f"evaluating on {test_x.shape[0]} test points")
    bo.train_surrogate_model(train_x, train_y)
    posterior = bo.surrogate_model.predict(test_x, return_posterior=True)
    metrics = log_surrogate_eval(
        posterior, test_y, stage="test", epoch=config.get("n_iters", 0)
    )
    # Ober-style prior-correlation diagnostic: how does a random train point
    # correlate with other train points vs test points under the fitted kernel?
    # Comment out when not needed.
    log_prior_correlation(
        bo.surrogate_model, train_x, test_x, stage="test",
        epoch=config.get("n_iters", 0),
    )
    print("Gate metrics (train->test):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def run_bo(config, dm, bo, data_stats, n_iters=None):
    """Phase-1 BO loop: iteratively acquire candidates from the held-out design
    space (the remaining train pool when respect_split is set)."""
    n_iters = n_iters if n_iters is not None else config["n_iters"]
    for i in tqdm(range(n_iters), colour="blue"):
        train_x = dm.train_x.clone().to("cuda")
        train_y = dm.train_y.clone().to("cuda")
        design_space = dm.heldout_x.clone().to("cuda")

        ## this trains the model, updates acqf and returns the next point to evaluate
        x_next = bo.suggest_next_experiments(train_x, train_y, design_space)
        x_next = torch.stack(x_next)

        log_bo_metrics(data_stats, dm.train_y, epoch=i)

        matches = (design_space.unsqueeze(0).to("cuda") == x_next).all(dim=-1)
        indices = matches.nonzero(as_tuple=True)[1].to("cpu")

        if not torch.all(matches.sum(dim=-1) == 1):
            print("Unable to find a unique match for some x_next in the dataset.")

        wandb.log(
            {
                "evaluated_suggestions": wandb.Histogram(dm.heldout_y[indices]),
                "epoch": i,
            }
        )

        x_next = x_next.squeeze(1)

        # update indices tracking
        evaluated_original_indices = dm.heldout_indices[indices]
        dm.train_indexes = np.append(dm.train_indexes, evaluated_original_indices)
        dm.heldout_indices = np.delete(dm.heldout_indices, indices)

        dm.train_x = dm.x[dm.train_indexes]
        dm.train_y = dm.y[dm.train_indexes]
        dm.heldout_x = dm.x[dm.heldout_indices]
        dm.heldout_y = dm.y[dm.heldout_indices]

        train_df = dm.data.loc[dm.train_indexes].copy()
        heldout_df = dm.data.loc[dm.heldout_indices.tolist()].copy()
        dm.data = pd.concat([train_df, heldout_df])

        df_values = dm.data.loc[dm.train_indexes][dm.target_column].values
        tensor_values = dm.train_y.squeeze().cpu().numpy()

        is_consistent = np.allclose(df_values, tensor_values)
        assert is_consistent, "DataFrame values don't match tensor values"

        assert len(np.unique(dm.train_indexes)) == len(
            dm.train_indexes
        ), "Duplicates found in dm.train_indexes"
        assert len(np.unique(dm.heldout_indices)) == len(
            dm.heldout_indices
        ), "Duplicates found in dm.heldout_indices"

        # Check for any common indices between dm.train_indexes and dm.heldout_indices
        common_indices = np.intersect1d(dm.train_indexes, dm.heldout_indices)
        assert (
            len(common_indices) == 0
        ), f"Common indices found between train and heldout: {common_indices}"
        # In respect_split mode the test rows are held aside (not in train/heldout).
        n_test = 0 if getattr(dm, "test_indices", None) is None else len(dm.test_indices)
        total_indices = len(dm.train_indexes) + len(dm.heldout_indices) + n_test
        assert total_indices == len(dm.x), "Mismatch in the total number of indices"

    log_bo_metrics(data_stats, dm.train_y, epoch=n_iters)


def _build_phase2_seed(config, dm, seed_source, seed_size, device):
    """Build the Phase-2 GP seed + the test design space for a given baseline.

    Returns (train_x, train_y, design_x, design_y, acquired_test_y), where
    `acquired_test_y` holds TEST labels already observed at the start (only
    non-empty for `none`, whose cold start spends an initial random TEST batch).

    seed_source:
      - "phase1_bo"   : the Phase-1 BO-collected train points (dm.train_*).
      - "random_train": `seed_size` random points from the full train pool.
      - "all_train"   : the entire train pool.
      - "none"        : no train warm-start; bootstrap from `seed_size` random
                        TEST points (design space = the remaining test).
    """
    test_x = dm.test_x.clone().to(device)
    test_y = dm.test_y.clone().to(device)
    empty_y = test_y.new_empty((0, test_y.shape[-1]))

    if seed_source == "phase1_bo":
        return (dm.train_x.clone().to(device), dm.train_y.clone().to(device),
                test_x, test_y, empty_y)

    if seed_source in ("random_train", "all_train"):
        train_pool = np.concatenate(
            [np.asarray(dm.train_indexes), np.asarray(dm.heldout_indices.cpu())]
        )
        if seed_source == "random_train":
            rng = np.random.default_rng(config["seed"])
            k = min(seed_size, len(train_pool))
            idx = np.sort(rng.choice(train_pool, size=k, replace=False))
        else:
            idx = np.sort(train_pool)
        seed_x = dm.x[idx].clone().to(device)
        seed_y = dm.y[idx].clone().to(device)
        return seed_x, seed_y, test_x, test_y, empty_y

    if seed_source == "none":
        # cold start from test set
        rng = np.random.default_rng(int(config["seed"]))
        test_y_flat = test_y.squeeze()
        median = test_y_flat.median()
        eligible = torch.where(test_y_flat <= median)[0].cpu().numpy()
        k = min(seed_size, len(eligible))
        init_idx_np = np.sort(rng.choice(eligible, size=k, replace=False))
        init_idx = torch.as_tensor(init_idx_np, device=device, dtype=torch.long)
        seed_x = test_x[init_idx].clone()
        seed_y = test_y[init_idx].clone()
        mask = torch.ones(test_x.shape[0], dtype=torch.bool, device=device)
        mask[init_idx] = False
        return seed_x, seed_y, test_x[mask], test_y[mask], seed_y.clone()

    raise ValueError(
        f"Unknown seed_source '{seed_source}'; choose "
        "phase1_bo | random_train | all_train | none."
    )


def run_phase2(config, dm, bo, n_iters, epoch_offset=0,
               seed_source="phase1_bo", seed_size=None):
    """
    Phase-2 BO over the test split. The GP is re-fit each iteration on the
    growing set (seed + Phase-2 test acquisitions). 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dm.test_x is None:
        raise ValueError("Phase 2 needs a held-aside test set; set respect_split: true.")

    batch = config["bo"]["init_args"].get("batch_size", 96)
    if seed_size is None:
        seed_size = config.get("phase2_seed_size")
    if seed_size is None:
        if seed_source == "none":
            seed_size = batch
        else:
            p1 = config.get("phase1_iters") or config.get("n_iters", 3)
            seed_size = len(dm.train_indexes) + p1 * batch

    test_stats = calculate_data_stats(dm.test_x, dm.test_y)
    f_max = dm.test_y.max().item()

    # Top-q% coverage denominators: how many test points fall in each top band.
    test_y_flat = dm.test_y.squeeze()
    coverage_bands = {}  # label (e.g. 5) -> (threshold, band_size)
    for q, label in [(0.99, 1), (0.95, 5), (0.90, 10)]:
        thr = test_stats[f"target_q{int(q * 100)}"]
        band_size = int((test_y_flat >= thr).sum().item())
        coverage_bands[label] = (thr, max(band_size, 1))

    train_x, train_y, design_x, design_y, acquired_test_y = _build_phase2_seed(
        config, dm, seed_source, seed_size, device
    )

    print(
        f"Phase 2 [seed_source={seed_source}]: seeded with {train_x.shape[0]} points, "
        f"discovering over {design_x.shape[0]} test candidates for {n_iters} iters "
        f"(test max {f_max:.4f})"
    )

    def _log_metrics(epoch, new_y):
        best = acquired_test_y.max().item() if acquired_test_y.numel() else float("nan")
        wandb.log(
            {
                "phase2/best_test": best,
                "phase2/simple_regret": f_max - best,
                "phase2/n_acquired": int(acquired_test_y.shape[0]),
                "phase2/evaluated_suggestions": wandb.Histogram(new_y.cpu()),
                "epoch": epoch,
            }
        )
        log_bo_metrics(test_stats, acquired_test_y, epoch=epoch, prefix="phase2/")
        flat = acquired_test_y.squeeze()
        for label, (thr, band_size) in coverage_bands.items():
            n_hit = int((flat >= thr.to(flat.device)).sum().item())
            wandb.log({f"phase2/coverage_top{label}": n_hit / band_size, "epoch": epoch})

    # Log the starting point (iter -1 relative): for `none` the random test init
    # already contributes coverage; for train seeds acquired_test_y is empty.
    if acquired_test_y.numel():
        _log_metrics(epoch_offset, acquired_test_y)

    for i in tqdm(range(n_iters), colour="green"):
        if design_x.shape[0] == 0:
            print("Phase 2: test design space exhausted; stopping early.")
            break

        x_next = bo.suggest_next_experiments(train_x, train_y, design_x)
        x_next = torch.stack(x_next)  # (batch, 1, D)

        matches = (design_x.unsqueeze(0) == x_next).all(dim=-1)  # (batch, N)
        indices = torch.unique(matches.nonzero(as_tuple=True)[1])

        new_x = design_x[indices]
        new_y = design_y[indices]

        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)
        acquired_test_y = torch.cat([acquired_test_y, new_y], dim=0)
        # Device-safe row removal (indices live on the same device as design_x).
        keep = torch.ones(design_x.shape[0], dtype=torch.bool, device=design_x.device)
        keep[indices] = False
        design_x = design_x[keep]
        design_y = design_y[keep]

        _log_metrics(epoch_offset + 1 + i, new_y)

    final_best = acquired_test_y.max().item() if acquired_test_y.numel() else float("nan")
    print(
        f"Phase 2 done [seed_source={seed_source}]: best test value {final_best:.4f} "
        f"(test max {f_max:.4f}, final simple regret {f_max - final_best:.4f})"
    )


def make_run_name(config, mode):
    """Build a meaningful W&B run name. Uses an explicit ``name`` from the config
    if present (the representation arms set this), otherwise derives one from the
    featurizer representation / model / PCA so different arms don't collide."""
    base = config.get("name")
    if not base:
        feat = config["data"]["init_args"]["featurizer"]["init_args"]
        rep = feat.get("representation", "feat")
        model_name = feat.get("model_name")
        if rep in ("get_huggingface_embeddings", "get_esmc_embeddings"):
            base = (model_name or rep).split("/")[-1]
        elif rep == "get_esmc_sae_features":
            base = f"{(model_name or 'esmc').split('/')[-1]}_sae"
        elif rep == "onehot":
            base = "onehot"
        else:
            base = rep
        reduce_dim = config["data"]["init_args"].get("reduce_dim")
        if reduce_dim:
            base += f"_pca{reduce_dim}"
    kernel = (config["surrogate_model"].get("init_args", {}) or {}).get("kernel")
    kernel_tag = f"_{kernel.replace('matern_', '')}" if kernel else ""
    return f"{base}{kernel_tag}_{mode}_seed{config['seed']}"


def train(config):
    if config.get("benchmark", None) is not None:
        config = configure_benchmark_datasets(config)

    if config.get("data_path", None) is not None:
        config["data"]["init_args"]["data_path"] = config["data_path"]

    if config.get("init_method", None) is not None:
        config["data"]["init_args"]["initializer"]["init_args"]["method"] = config["init_method"]

    config = validate_configuration(config)
    wandb_config = flatten(config)

    mode = config.get("mode", "bo") or "bo"
    run_name = make_run_name(config, mode)

    with wandb.init(
        project="flip2_gollum_seed_source_comparison", config=wandb_config, group=config["group"], name=run_name
    ) as run:

        dm = setup_data(config)
        bo = setup_bo_optimizer(config, design_space=dm.heldout_x)

        data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)

        # Phase iteration counts: `full` uses separate phase1/phase2 budgets;
        # other modes fall back to n_iters.
        phase1_iters = config.get("phase1_iters") or config.get("n_iters", 3)
        phase2_iters = config.get("phase2_iters") or config.get("n_iters", 3)

        if mode == "full":
            # `or` (not get-with-default): the CLI registers seed_source as
            # present-but-None when unset, so a plain .get would return None.
            seed_source = config.get("seed_source") or "phase1_bo"
            if seed_source == "phase1_bo":
                run_bo(config, dm, bo, data_stats, n_iters=phase1_iters)
                run_gate(config, dm, bo)
            run_phase2(
                config, dm, bo, phase2_iters,
                epoch_offset=phase1_iters + 1, seed_source=seed_source,
            )
        else:
            # For `both`: Phase-1 BO collection over train, then gate-evaluate
            # the resulting GP on the held-aside test split.
            if mode in ("bo", "both"):
                run_bo(config, dm, bo, data_stats, n_iters=phase1_iters)
            if mode in ("gate", "both"):
                run_gate(config, dm, bo)

        # Save finetuned model if using DeepGP
        if config["surrogate_model"]["class_path"] == "gollum.surrogate_models.gp.DeepGP" and config["save_model"] == True:
            model_save_path = os.path.join(
                "checkpoints", run.name if run else "default", "finetuned_model.pt"
            )
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(
                bo.surrogate_model.finetuning_model.state_dict(), model_save_path
            )
            print(f"Saved finetuned model to {model_save_path}")

            # Visualize embeddings if requested
            if config.get("visualize", False) and config.get("full_data_path"):
                from gollum.visualization import visualize_embeddings

                ft_config = config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]
                model_name = ft_config["model_name"]
                pooling_method = ft_config["pooling_method"]

                output_dir = os.path.join(
                    "plots", "embeddings", run.name if run else "default"
                )
                visualize_embeddings(
                    full_data_path=config["full_data_path"],
                    model_name=model_name,
                    pooling_method=pooling_method,
                    model_state_path=model_save_path,
                    finetuning_model_config=ft_config,
                    output_dir=output_dir,
                    random_state=config.get("seed", 42),
                )

        logger.setLevel(logging.INFO)
        wandb.finish()


def get_train_dimension(train_x):
    return train_x.shape[-1]


def main():
    # Initialize the parser with a description
    parser = ArgumentParser(
        description="Training script",
        default_config_files=[],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--benchmark", type=str, help="Run a specific benchmark")

    # parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_argument(
        "--data_path", type=str, help="Data file csv path"
    )
    parser.add_argument(
        "--phase1_iters", type=int, help="Phase-1 (train) BO iterations (full mode)"
    )
    parser.add_argument(
        "--phase2_iters", type=int, help="Phase-2 (test) BO iterations (full mode)"
    )
    parser.add_argument(
        "--seed_source",
        type=str,
        default="phase1_bo",
        help="Phase-2 seed baseline: phase1_bo | random_train | all_train | none",
    )
    parser.add_argument(
        "--phase2_seed_size",
        type=int,
        help="Seed size for random_train / none (default: matched to Phase-1 budget)",
    )
    parser.add_argument("--kernel", type=str, help="One-word kernel selector (e.g. matern_stuyver)")
    parser.add_argument("--init_method", type=str, help="BO initialization method (e.g. true_random, sobol)")
    parser.add_argument(
        "--mode",
        type=str,
        default="bo",
        help="gate (fit train, rank test), bo (Phase-1 collect), both (bo+gate), "
        "or full (Phase-1 train BO -> Phase-2 test BO)",
    )
    parser.add_argument("--group", type=str, help="Wandb group runs")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name base")
    parser.add_argument("--save_model", type=bool, default=False, help="Save the finetuned model after training")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize embeddings after training")
    parser.add_argument("--full_data_path", type=str, default=None, help="Path to full dataset with split labels (for visualization)")

    parser.add_subclass_arguments(BaseDataModule, "data", instantiate=False)
    parser.add_subclass_arguments(SurrogateModel, "surrogate_model", instantiate=False)

    parser.add_subclass_arguments(
        AcquisitionFunction,
        "acquisition",
        instantiate=False,
        skip=["model", "best_f"],
    )
    parser.add_subclass_arguments(BotorchOptimizer, "bo", instantiate=False)

    # parse arguments
    args = parser.parse_args()
    seed_everything(args["seed"], workers=True)
    train(args.as_dict())


if __name__ == "__main__":
    main()
