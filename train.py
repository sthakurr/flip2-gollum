import warnings
from botorch.exceptions import InputDataWarning

import os
import re
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
from gollum.surrogate_models.gp import SurrogateModel, pdist_hist, tsne_plot, kernel_blocks, kernel_hist, spearman
from sklearn.manifold import TSNE



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




def _log_distance_metrics(train_x, train_y, epoch=0):
    """Log pairwise L2 distance metrics between high/low scoring training points."""
    x_cpu = train_x.cpu().float()
    y_cpu = train_y.cpu().squeeze().float()
    low_thr, high_thr = compute_thresholds(y_cpu, low_quantile=0.2, high_quantile=0.8)
    hh, hl, ll, avg = calculate_distances(
        x_cpu, y_cpu,
        high_score_threshold=high_thr.item(),
        low_score_threshold=low_thr.item(),
    )
    dist_log = {"distances/avg": avg.item(), "epoch": epoch}
    if hh.numel() > 0:
        dist_log["distances/hh_mean"] = hh.mean().item()
    if hl.numel() > 0:
        dist_log["distances/hl_mean"] = hl.mean().item()
    if ll.numel() > 0:
        dist_log["distances/ll_mean"] = ll.mean().item()
    wandb.log(dist_log)


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
    
    model_name = config["data"]["init_args"]["featurizer"]["init_args"]["model_name"]
    model_short = model_name.split("/")[-1]
    run_name = f"{model_short}_seed{config['seed']}"

    with wandb.init(
        project=config.get("wandb_project", "gollum_mech"),
        config=wandb_config,
        group=config["group"],
        name=run_name,
    ) as run:
        # uniform x-axes across the metrics
        run.define_metric("epoch")
        run.define_metric("train_step")
        # pdists/ and ls/ hold both axes, so their keys are explicit (no glob overlap)
        for key in ["grads/*", "lr/*", "loss/*", "embed/*",
                    "pdists/mean", "pdists/median", "pdists/hist", "ls/lengthscale", "ls/ratio_train"]:
            run.define_metric(key, step_metric="train_step")
        for key in ["top5pct_coverage", "train/*", "top_*", "quantile_*", "evaluated_suggestions",
                    "distances/*", "embed_all/*", "pdists_all/*", "ls_all/*", "tsne/*", "kernel/*", "post/*",
                    "pdists/mean_by_epoch", "pdists/median_by_epoch", "ls/lengthscale_by_epoch", "ls/ratio_train_by_epoch", "ls/ratio_test_by_epoch", "ls/ratio_by_max_pdist"]:
            run.define_metric(key, step_metric="epoch")

        dm = setup_data(config)
        bo = setup_bo_optimizer(config, design_space=dm.heldout_x)
        
        data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)

        prev_tsne = None
        # Start the training loop
        for i in tqdm(range(config["n_iters"]), colour="blue"):
            train_x = dm.train_x.clone().to("cuda")
            train_y = dm.train_y.clone().to("cuda")
            design_space = dm.heldout_x.clone().to("cuda")

            ## this trains the model, updates acqf and returns the next point to evaluate
            _, candidate_positions = bo.suggest_next_experiments(train_x, train_y, design_space)
            indices = torch.tensor(candidate_positions)

            # whole-set embedding stats with the just-fitted model, once per BO iteration
            if hasattr(bo.surrogate_model, "embed"):
                emb = bo.surrogate_model.embed(dm.x)
                d = torch.pdist(emb)  # full N^2/2 pairs: fine for small sets (BH ~4k rows), subsample for large ones
                ls = getattr(bo.surrogate_model.covar_module, "base_kernel",
                             bo.surrogate_model.covar_module).lengthscale.mean().item()
                d_train = torch.pdist(emb[dm.train_indexes])
                d_test = torch.pdist(emb[torch.as_tensor(dm.heldout_indices)])  # candidates only, no train pairs mixed in
                e = emb.to(bo.surrogate_model.train_x)  # kernel params' device/dtype
                kb = kernel_blocks(getattr(bo.surrogate_model.covar_module, "base_kernel", bo.surrogate_model.covar_module),
                                   e[dm.train_indexes], e[torch.as_tensor(dm.heldout_indices)])
                # |dy| in the same pair order as kb (pdist == row-major upper triangle); test uses true yields (diagnostic only)
                y_tr, y_te = dm.y[dm.train_indexes], dm.y[torch.as_tensor(dm.heldout_indices)]
                dy = {"train-train": torch.pdist(y_tr), "train-test": (y_tr - y_te.T).abs().flatten(),
                      "test-test": torch.pdist(y_te)}
                wandb.log({
                    # < 0: kernel-similar pairs have similar yields; ~0: kernel is yield-blind
                    **{f"kernel/{name}_dy_spearman": spearman(v, dy[name]) for name, v in kb.items()},
                    **{f"kernel/{name}_mean": v.mean().item() for name, v in kb.items()},
                    **{f"kernel/{name}_var": v.var().item() for name, v in kb.items()},
                    "kernel/hist": kernel_hist(kb, f"kernel values, iter {i}"),
                    "epoch": i,
                })
                first = bo.surrogate_model._first_log
                if i == 0 and first:
                    # campaign start at epoch -1: first step of first fit (pretrained embeddings, initial hypers)
                    wandb.log({f"{k}_by_epoch": first[k] for k in
                               ["pdists/mean", "pdists/median", "ls/lengthscale", "ls/ratio_train"]} | {"epoch": -1})
                wandb.log({
                    "embed_all/min": emb.min().item(),
                    "embed_all/max": emb.max().item(),
                    "pdists_all/mean": d.mean().item(),
                    "pdists_all/median": d.median().item(),
                    "pdists_all/hist": pdist_hist(d, f"whole set, iter {i}"),
                    # final lengthscale vs whole-set spread: is the kernel meaningful where acquisition operates
                    "ls_all/ratio": ls / d.median().item(),
                    # >1: even the farthest pair in the whole space stays strongly correlated
                    "ls/ratio_by_max_pdist": ls / d.max().item(),
                    # end-of-fit state of the per-step train metrics, one point per BO iteration
                    "pdists/mean_by_epoch": d_train.mean().item(),
                    "pdists/median_by_epoch": d_train.median().item(),
                    "ls/lengthscale_by_epoch": ls,
                    "ls/ratio_train_by_epoch": ls / d_train.median().item(),
                    "ls/ratio_test_by_epoch": ls / d_test.median().item(),
                    "epoch": i,
                })

                # candidate posterior as analytic EI sees it (latent f, no observation noise), in y units
                with torch.no_grad():
                    posts = [bo.surrogate_model.posterior(c.to("cuda")) for c in dm.heldout_x.split(512)]
                mu = torch.cat([p.mean.squeeze(-1) for p in posts]).cpu()
                sd = torch.cat([p.variance.squeeze(-1) for p in posts]).clamp_min(0).sqrt().cpu()
                prior_sd = (getattr(bo.surrogate_model.covar_module, "outputscale", torch.tensor(1.0)).sqrt().item()
                            * dm.train_y.std().item())  # Standardize: prior var in y units = outputscale * var(train y)
                z = (mu[indices[0]] - dm.train_y.max()) / sd[indices[0]]  # EI's z at the chosen candidate
                wandb.log({
                    "post/std_median": sd.median().item(),
                    "post/std_min": sd.min().item(),
                    "post/std_max": sd.max().item(),
                    "post/std_rel_median": sd.median().item() / prior_sd,
                    "post/mean_std": mu.std().item(),
                    "post/mean_y_spearman": spearman(mu, dm.heldout_y.squeeze(-1)),
                    "post/ei_z_chosen": z.item(),
                    "epoch": i,
                })

                # warm-start from previous map (rescaled to tiny std, as tsne expects) so maps stay comparable across iters
                init = "pca" if prev_tsne is None else prev_tsne / prev_tsne[:, 0].std() * 1e-4
                prev_tsne = TSNE(n_components=2, init=init, random_state=config["seed"]).fit_transform(emb.numpy())
                wandb.log({
                    "tsne/map": tsne_plot(prev_tsne, dm.y.cpu().numpy(), dm.train_indexes, dm.heldout_indices[indices],
                                          f"iter {i}"),
                    "epoch": i,
                })

            log_bo_metrics(data_stats, dm.train_y, epoch=i)

            wandb.log(
                {
                    "evaluated_suggestions": wandb.Histogram(dm.heldout_y[indices]),
                    "epoch": i,
                }
            )

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

            if _DISTANCE_METRICS_AVAILABLE and len(dm.train_indexes) >= 5:
                _log_distance_metrics(dm.train_x, dm.train_y, epoch=i)

        log_bo_metrics(data_stats, dm.train_y, epoch=config["n_iters"])

        # Save finetuned model if using DeepGP
        if config["surrogate_model"]["class_path"] == "gollum.surrogate_models.gp.DeepGP":
            save_ckpt = config.get("save_checkpoints", config["surrogate_model"].get("save_checkpoints", False))
            if save_ckpt:
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
        default_config_files=["configs/bochemian.yaml"],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--benchmark", type=str, help="Run a specific benchmark")

    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
   
    parser.add_argument("--group", type=str, help="Wandb group runs")
    parser.add_argument("--wandb_project", type=str, default="gollum_mech", help="Wandb project name")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize embeddings after training")
    parser.add_argument("--full_data_path", type=str, default=None, help="Path to full dataset with split labels (for visualization)")
    parser.add_argument("--save_checkpoints", type=bool, default=False, help="Save finetuned model checkpoints for DeepGP runs")

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
