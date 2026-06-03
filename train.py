import warnings
from botorch.exceptions import InputDataWarning

import math
import os
import re
import time
import torch
import logging

warnings.filterwarnings("ignore", category=InputDataWarning)
logger = logging.getLogger(__name__)
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


from gollum.data.module import BaseDataModule
from gollum.bo.optimizer import BotorchOptimizer


from gollum.metrics import (
    calculate_data_stats,
    log_bo_metrics,
    log_data_stats,
)

try:
    from gollum.metrics.analysis import calculate_distances, compute_thresholds
    _DISTANCE_METRICS_AVAILABLE = True
except Exception:
    _DISTANCE_METRICS_AVAILABLE = False



torch.set_float32_matmul_precision("high")


def seed_everything(seed: int, workers: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

# Okabe-Ito color-blind safe palette (Wong 2011, Nature Methods)
from scipy.stats import spearmanr

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
    "EvolutionaryScale/esmc-600m-2024-12": 1152,
    "EvolutionaryScale/esm3-sm-open-v1": 1024,
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


def run_train_phase(config, featurizer, n_init, n_train_iter):
    """Phase 1: sequential BO on the training split.

    Starts with n_init random seed points and selects batch_size points per
    iteration for n_train_iter iterations, simulating a real BO campaign on
    the training distribution before moving to the test candidate space.

    Returns (train_x, train_y, train_df) of all accumulated selected points.
    """
    init_args = config["data"]["init_args"]
    train_path = init_args["train_path"]
    target_col = init_args["target_column"]
    input_col  = init_args["input_column"]
    maximize   = init_args.get("maximize", True)

    train_df = pd.read_csv(train_path).reset_index(drop=True)
    if not maximize:
        train_df[target_col] = -train_df[target_col]
    n_all = len(train_df)

    print(f"[Phase 1] Featurizing {n_all} training sequences...")
    all_x = torch.from_numpy(featurizer.featurize(train_df[input_col])).to(torch.float64)
    all_y = torch.from_numpy(train_df[target_col].values).to(torch.float64).unsqueeze(-1)

    # Random n_init seed (respects seed_everything via numpy random state)
    init_idx   = np.sort(np.random.choice(n_all, size=n_init, replace=False))
    heldout_mask = np.ones(n_all, dtype=bool)
    heldout_mask[init_idx] = False
    heldout_idx = np.where(heldout_mask)[0]

    train_x = all_x[init_idx].clone()
    train_y = all_y[init_idx].clone()
    selected_rows = list(init_idx)

    bo_p1 = setup_bo_optimizer(config, design_space=all_x[heldout_idx])

    for i in tqdm(range(n_train_iter), colour="green", desc="Phase 1 (train BO)"):
        _, candidate_positions = bo_p1.suggest_next_experiments(
            train_x.clone().to("cuda"),
            train_y.clone().to("cuda"),
            all_x[heldout_idx].clone().to("cuda"),
            epoch=i,
        )
        positions      = np.asarray(candidate_positions)
        selected_global = heldout_idx[positions]

        train_x = torch.cat([train_x, all_x[selected_global]], dim=0)
        train_y = torch.cat([train_y, all_y[selected_global]], dim=0)
        selected_rows.extend(selected_global.tolist())
        heldout_idx = np.delete(heldout_idx, positions)

        print(
            f"[Phase 1 iter {i}] +{len(positions)} pts → "
            f"train={len(train_x)}, remaining train pool={len(heldout_idx)}"
        )

    return train_x, train_y, train_df.iloc[selected_rows].reset_index(drop=True)


def setup_data(config):
    init_args = config["data"]["init_args"]
    featurizer = instantiate_class(init_args["featurizer"])
    kwargs = dict(
        featurizer=featurizer,
        normalize_input=init_args["normalize_input"],
        maximize=init_args["maximize"],
    )
    if "initializer" in init_args:
        kwargs["initializer"] = instantiate_class(
            init_args["initializer"], seed=config["seed"]
        )
    dm = instantiate_class(config["data"], **kwargs)
    return dm


def _distance_metrics(train_x, train_y, max_n: int = 512):
    """Return pairwise L2 distance metrics between high/low scoring points.

    Subsamples to max_n points before computing the N×N distance matrix to
    avoid OOM on large test splits.
    """
    x_cpu = train_x.cpu().float()
    y_cpu = train_y.cpu().squeeze().float()
    if len(x_cpu) > max_n:
        x_cpu = x_cpu[:max_n]
        y_cpu = y_cpu[:max_n]
    low_thr, high_thr = compute_thresholds(y_cpu, low_quantile=0.2, high_quantile=0.8)
    hh, hl, ll, avg = calculate_distances(
        x_cpu, y_cpu,
        high_score_threshold=high_thr.item(),
        low_score_threshold=low_thr.item(),
    )
    metrics = {"distances/avg": avg.item()}
    if hh.numel() > 0:
        metrics["distances/hh_mean"] = hh.mean().item()
    if hl.numel() > 0:
        metrics["distances/hl_mean"] = hl.mean().item()
    if ll.numel() > 0:
        metrics["distances/ll_mean"] = ll.mean().item()
    return metrics


def _compute_split_metrics(surrogate, x, y, prefix, chunk_size=256):
    """Return R², NLPD (diagonal Gaussian, obs-noise-inclusive), MSLL, Spearman ρ.

    Processes x in chunks so large test splits (e.g. 200k+ sequences) don't OOM.
    x and y should be CPU tensors; each chunk is moved to CUDA inside the loop.
    """
    all_means, all_vars = [], []
    with torch.no_grad():
        for x_chunk in x.split(chunk_size):
            m, v = surrogate.predict(x_chunk.to("cuda"), observation_noise=True, return_var=True)
            all_means.append(m.squeeze().cpu())
            all_vars.append(v.squeeze().cpu())

    mu  = torch.cat(all_means)
    var = torch.cat(all_vars).clamp(min=1e-9)
    y_t = y.squeeze().cpu()

    ss_res = ((y_t - mu) ** 2).sum()
    ss_tot = ((y_t - y_t.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()

    two_pi = torch.tensor(2.0 * math.pi)
    nlpd = (0.5 * (torch.log(two_pi * var) + (y_t - mu) ** 2 / var)).mean().item()

    baseline_var = y_t.var().clamp(min=1e-9)
    baseline_nlpd = (0.5 * (torch.log(two_pi * baseline_var) + (y_t - y_t.mean()) ** 2 / baseline_var)).mean().item()
    msll = nlpd - baseline_nlpd

    rho, _ = spearmanr(mu.numpy(), y_t.numpy())

    return {
        f"{prefix}/r2": r2,
        f"{prefix}/nlpd": nlpd,
        f"{prefix}/msll": msll,
        f"{prefix}/spearman_rho": float(rho),
    }


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


def train(config):
    if config.get("benchmark", None) is not None:
        config = configure_benchmark_datasets(config)
    
    config = validate_configuration(config)
    wandb_config = flatten(config)
    
    model_name = config["data"]["init_args"]["featurizer"]["init_args"].get("model_name", "one_hot")
    model_short = model_name.split("/")[-1]
    run_name = f"{model_short}_seed{config['seed']}"

    with wandb.init(
        project=config.get("wandb_project", "gollum"),
        config=wandb_config,
        group=config["group"],
        name=run_name,
        settings=wandb.Settings(init_timeout=300),
    ) as run:
        # Decouple BO-iteration metrics from the high-frequency LR step counter.
        # DeepGP.forward() calls wandb.log() on every optimizer step, so the
        # global step counter races far ahead of the BO iteration index.
        # define_metric tells W&B to use the "epoch" field as the x-axis for all
        # BO metrics instead of the global step — eliminating the monotonic-step warning.
        wandb.define_metric("epoch")
        for _prefix in ("test/*", "train/*", "top*", "quantile*", "distances/*", "evaluated_suggestions"):
            wandb.define_metric(_prefix, step_metric="epoch")
        wandb.define_metric("finetuning/contrastiveness", step_metric="epoch")

        dm = setup_data(config)

        # --- Phase 1: sequential BO on training split (optional) ---
        n_train_iter_p1 = config.get("n_train_iter", 0)
        _phase2_seed_size = 0  # number of Phase 1 points prepended to dm.train_y
        if n_train_iter_p1 > 0:
            n_init = config.get("n_init", 10)
            phase1_x, phase1_y, phase1_df = run_train_phase(
                config, dm.featurizer, n_init, n_train_iter_p1
            )
            n_phase1 = len(phase1_x)
            n_test   = len(dm.heldout_x)

            # Snapshot test split before rebuilding dm
            _test_x  = dm.heldout_x.clone()
            _test_y  = dm.heldout_y.clone()
            _test_df = dm.data.loc[dm.heldout_indices.tolist()].copy().reset_index(drop=True)
            _test_df.index = range(n_phase1, n_phase1 + n_test)

            phase1_df = phase1_df.reset_index(drop=True)

            # Rebuild dm: Phase 1 seed + test candidate space
            dm.x             = torch.cat([phase1_x, _test_x], dim=0)
            dm.y             = torch.cat([phase1_y, _test_y], dim=0)
            dm.data          = pd.concat([phase1_df, _test_df])
            dm.train_indexes = np.arange(n_phase1)
            dm.heldout_indices = torch.arange(n_phase1, n_phase1 + n_test)
            dm.train_x       = phase1_x
            dm.train_y       = phase1_y
            dm.heldout_x     = _test_x
            dm.heldout_y     = _test_y

            _phase2_seed_size = n_phase1
            print(
                f"[Phase 2 setup] seed={n_phase1} pts (from train BO), "
                f"candidate space={n_test} test seqs"
            )

        bo = setup_bo_optimizer(config, design_space=dm.heldout_x)

        # In 2-phase BO, compute stats from the test split only so that coverage
        # and thresholds reflect the test distribution, not the combined pool.
        if _phase2_seed_size > 0:
            data_stats = calculate_data_stats(dm.heldout_x, dm.heldout_y)
        else:
            data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)

        # Pre-cache test tensors for per-iteration Spearman tracking
        eval_spearman_per_iter = config.get("eval_spearman_per_iter", True)
        test_n = config.get("test_n_seqs", 96)
        _test_x_gpu = dm.test_x[:test_n].to("cuda") if dm.test_x is not None else None
        _test_y_np  = dm.test_y[:test_n].squeeze().cpu().numpy() if dm.test_y is not None else None
        spearman_history = []  # list of (n_train, rho)

        # Set up per-iteration checkpoint directory
        is_deep_gp = config["surrogate_model"]["class_path"] == "gollum.surrogate_models.gp.DeepGP"
        save_ckpt = is_deep_gp and config.get("save_checkpoints", config["surrogate_model"].get("save_checkpoints", False))
        model_save_dir = None
        if save_ckpt:
            run_tag = f"{run.name}_{config['n_iters']}iters" if run else "default"
            # Derive split subdir from train_path (SplitDataModule) or data_path (BaseDataModule)
            # e.g. "data/flip2/trpB/one_to_many_train.csv" → "trpB/one_to_many"
            _dp = config["data"]["init_args"].get(
                "train_path", config["data"]["init_args"].get("data_path", "")
            )
            _dp_parts = os.path.normpath(_dp).split(os.sep)
            # parts[-1] is filename, parts[-2] is protein, parts[-3] is split; skip leading "data/flip2"
            split_subdir = os.path.join(*_dp_parts[-3:-1]) if len(_dp_parts) >= 3 else "unknown_split"
            model_save_dir = os.path.join("/iopsstor/scratch/cscs/ssaumya/gollum_models/", split_subdir, run_tag)
            os.makedirs(model_save_dir, exist_ok=True)

        # Start the training loop
        for i in tqdm(range(config["n_iters"]), colour="blue"):
            t_iter = time.perf_counter()
            print(f"\n[Iter {i}] Starting iteration {i} at {t_iter:.1f}s")

            train_x = dm.train_x.clone().to("cuda")
            train_y = dm.train_y.clone().to("cuda")
            design_space = dm.heldout_x.clone().to("cuda")

            _, candidate_positions = bo.suggest_next_experiments(train_x, train_y, design_space, epoch=i)
            indices = torch.tensor(candidate_positions)

            # Save per-iteration checkpoint
            if model_save_dir is not None:
                iter_ckpt = os.path.join(model_save_dir, f"iter_{i}", "finetuned_model.pt")
                os.makedirs(os.path.dirname(iter_ckpt), exist_ok=True)
                torch.save(bo.surrogate_model.finetuning_model.state_dict(), iter_ckpt)
                print(f"[Ckpt] Saved iter {i} model → {iter_ckpt}")

            # Spearman eval on test (surrogate already trained above — no extra cost)
            if _test_x_gpu is not None and eval_spearman_per_iter:
                with torch.no_grad():
                    _pred_mean, _ = bo.surrogate_model.predict(_test_x_gpu)
                _rho, _ = spearmanr(_pred_mean.squeeze().cpu().numpy(), _test_y_np)
                spearman_history.append((train_x.shape[0], _rho))
                print(f"[Test Eval] iter={i}  n_train={train_x.shape[0]}  Spearman ρ={_rho:.4f}")
                wandb.log({"test/spearman_rho": _rho, "test/n_train": train_x.shape[0], "epoch": i})

            # t_bookkeeping = time.perf_counter()
            # print(f"Bookeeping timer started at {t_bookkeeping:.1f}s")

            extra = {"evaluated_suggestions": wandb.Histogram(dm.heldout_y[indices])}
            if _DISTANCE_METRICS_AVAILABLE and len(dm.x) >= 5:
                sm = bo.surrogate_model
                if hasattr(sm, "finetuning_model"):
                    with torch.no_grad():
                        sm.finetuning_model.eval()
                        _emb = sm.finetuning_model(
                            dm.x.to("cuda")
                        ).cpu().float()
                        sm.finetuning_model.train()
                    extra.update(_distance_metrics(_emb, dm.y))
                else:
                    extra.update(_distance_metrics(dm.x, dm.y))
            try:
                sm = bo.surrogate_model
                ls = sm.covar_module.base_kernel.lengthscale.detach().squeeze()
                ls_mean = ls.mean().item()
                extra["gp/lengthscale_mean"] = ls_mean
                if ls.ndim > 0 and ls.numel() > 1:
                    extra["gp/lengthscale_min"] = ls.min().item()
                    extra["gp/lengthscale_max"] = ls.max().item()
                extra["gp/outputscale"] = sm.covar_module.outputscale.detach().item()
                extra["gp/noise"]       = sm.likelihood.noise_covar.noise.detach().item()
                if _test_x_gpu is not None and eval_spearman_per_iter and ls_mean > 0:
                    _n_annd = min(len(train_x), 512)
                    with torch.no_grad():
                        if hasattr(sm, "finetuning_model"):
                            sm.finetuning_model.eval()
                            _emb_tr = sm.finetuning_model(train_x[:_n_annd]).float()
                            _emb_te = sm.finetuning_model(_test_x_gpu).float()
                        else:
                            # for non-deep GPs (one-hot, static) use raw feature vectors
                            _emb_tr = train_x[:_n_annd].float()
                            _emb_te = _test_x_gpu.float()
                    _nn_dists = [
                        torch.cdist(_emb_te[j : j + 64], _emb_tr, p=2).min(dim=1).values
                        for j in range(0, len(_emb_te), 64)
                    ]
                    _annd = torch.cat(_nn_dists).mean().item()
                    extra["gp/annd"]         = _annd
                    extra["gp/annd_over_ls"] = _annd / ls_mean
                    del _emb_tr, _emb_te
                    torch.cuda.empty_cache()
            except Exception:
                pass
            log_bo_metrics(data_stats, dm.train_y[_phase2_seed_size:], epoch=i, extra=extra)

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
            total_indices = len(dm.train_indexes) + len(dm.heldout_indices)
            assert total_indices == len(dm.x), "Mismatch in the total number of indices"

        log_bo_metrics(data_stats, dm.train_y[_phase2_seed_size:], epoch=config["n_iters"])

        # Final retrain on fully accumulated data → last Spearman point + plots
        if _test_x_gpu is not None:
            n_final = len(dm.train_x)
            print(f"\n[Test Eval] Final retrain on {n_final} points, predicting on {test_n} test seqs...")
            bo.train_surrogate_model(dm.train_x.clone().to("cuda"), dm.train_y.clone().to("cuda"))

            with torch.no_grad():
                pred_mean, _ = bo.surrogate_model.predict(_test_x_gpu)
            pred_np = pred_mean.squeeze().cpu().numpy()

            corr_final, pval = spearmanr(pred_np, _test_y_np)
            spearman_history.append((n_final, corr_final))
            print(f"[Test Eval] Final  n_train={n_final}  Spearman ρ={corr_final:.4f}  (p={pval:.4e})")
            wandb.log({"test/spearman_rho": corr_final, "test/n_train": n_final,
                       "epoch": config["n_iters"]})


            print(f"[Diag] pred_mean std={pred_mean.std():.6f}  range=[{pred_mean.min():.4f}, {pred_mean.max():.4f}]")
            print(f"[Diag] lengthscale={bo.surrogate_model.covar_module.base_kernel.lengthscale.item():.4f}")

            # ── Final manuscript metrics: R², NLPD, MSLL, Spearman ρ ──
            # "heldout" = sequences in the exploration pool never used to fit the GP
            # "test"    = completely separate held-out test split
            print("\n" + "=" * 65)
            print("FINAL METRICS  (obs-noise-inclusive NLPD)")
            print("=" * 65)
            final_metrics = {}
            for split_prefix, sx, sy in [
                ("final_heldout", dm.heldout_x, dm.heldout_y),
                ("final_test",    dm.test_x,    dm.test_y),
            ]:
                try:
                    m = _compute_split_metrics(bo.surrogate_model, sx, sy, prefix=split_prefix)
                    final_metrics.update(m)
                    for k, v in sorted(m.items()):
                        print(f"  {k:<42} {v:+.4f}")
                except Exception as exc:
                    print(f"[{split_prefix}] Metrics computation failed: {exc}")
            wandb.log(final_metrics)
            print("=" * 65 + "\n")

            # ── Kernel covariance heatmap ──────────────────────────────
            try:
                from gollum.metrics.analysis import plot_kernel_heatmap
                _splits = [("heldout", dm.heldout_x)]
                if dm.test_x is not None:
                    _splits.append(("test", dm.test_x))
                _pdf_path = None
                if is_deep_gp and save_ckpt:
                    _pdf_path = os.path.join(model_save_dir, "kernel_heatmap.pdf")
                plot_kernel_heatmap(
                    model=bo.surrogate_model,
                    train_x=dm.train_x,
                    splits=_splits,
                    n_subsample=config.get("kernel_heatmap_n", 256),
                    save_path=_pdf_path,
                )
            except Exception as exc:
                print(f"[kernel_heatmap] Skipped: {exc}")
            # ──────────────────────────────────────────────────────────

        # Save finetuned model if using DeepGP
        if is_deep_gp:
            if save_ckpt:
                model_save_path = os.path.join(model_save_dir, "final", "finetuned_model.pt")
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
                    "/iopsstor/scratch/cscs/ssaumya/gollum_models/", run.name if run else "default"
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
        default_config_files=["configs/bochemian.yaml"],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--benchmark", type=str, help="Run a specific benchmark")

    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_argument("--n_train_iter", type=int, default=0, help="Phase 1: BO iterations on training split (0 = disabled)")
    parser.add_argument("--n_init", type=int, default=10, help="Phase 1: initial random seed size for training BO")
   
    parser.add_argument("--group", type=str, help="Wandb group runs")
    parser.add_argument("--wandb_project", type=str, default="gollum", help="Wandb project name")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize embeddings after training")
    parser.add_argument("--full_data_path", type=str, default=None, help="Path to full dataset with split labels (for visualization)")
    parser.add_argument("--save_checkpoints", type=bool, default=False, help="Save finetuned model checkpoints for DeepGP runs")
    parser.add_argument("--test_n_seqs", type=int, default=96, help="Number of test sequences to evaluate surrogate on after BO loop")
    parser.add_argument("--eval_spearman_per_iter", type=bool, default=True, help="Compute Spearman ρ on test set after every BO iteration; set false to only compute at the end")

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
