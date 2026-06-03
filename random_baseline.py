import argparse
import yaml
import numpy as np
import torch
import wandb
from tqdm import tqdm

from gollum.metrics import calculate_data_stats, log_bo_metrics, log_data_stats
from gollum.utils.config import instantiate_class, flatten


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_data(config, seed):
    initializer = instantiate_class(
        config["data"]["init_args"]["initializer"], seed=seed
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


def _hypergeometric_expected_coverage(pool_size, n_selected):
    """Expected top-5% coverage under uniform random selection without replacement.

    Under hypergeometric sampling, E[found_top5] = n_top * n_selected / pool_size,
    so E[coverage] = n_selected / pool_size regardless of n_top.
    """
    return min(n_selected / max(pool_size, 1), 1.0)


def run_one_seed(config, seed):
    seed_everything(seed)

    batch_size = config.get("batch_size", 96)
    n_iters = config["n_iters"]

    dm = setup_data(config, seed)

    data_stats = calculate_data_stats(dm.x, dm.y)
    log_data_stats(data_stats)

    pool_size = len(dm.heldout_indices)
    n_top = int(data_stats["total_q95_count"].item())
    print(f"Pool: {pool_size} sequences  |  Top-5% count: {n_top}  |  Batch: {batch_size}  |  Iters: {n_iters}")

    # Convert heldout_indices to numpy to match train.py bookkeeping behaviour
    if isinstance(dm.heldout_indices, torch.Tensor):
        dm.heldout_indices = dm.heldout_indices.numpy()
    if isinstance(dm.train_indexes, torch.Tensor):
        dm.train_indexes = dm.train_indexes.numpy()

    for i in tqdm(range(n_iters), colour="blue"):
        n_heldout = len(dm.heldout_indices)
        indices = np.random.choice(n_heldout, size=min(batch_size, n_heldout), replace=False)

        # Log metrics for the current training set (before adding new batch).
        # n_selected_from_heldout = how many heldout points have been moved to train so far.
        n_selected_from_heldout = pool_size - n_heldout
        expected_cov = _hypergeometric_expected_coverage(pool_size, n_selected_from_heldout)
        log_bo_metrics(data_stats, dm.train_y, epoch=i, extra={"random/expected_coverage": expected_cov})

        # Pool update — identical to train.py lines 351-384
        evaluated_original_indices = dm.heldout_indices[indices]
        dm.train_indexes = np.append(dm.train_indexes, evaluated_original_indices)
        dm.heldout_indices = np.delete(dm.heldout_indices, indices)

        dm.train_x = dm.x[dm.train_indexes]
        dm.train_y = dm.y[dm.train_indexes]
        dm.heldout_x = dm.x[dm.heldout_indices]
        dm.heldout_y = dm.y[dm.heldout_indices]

        assert len(np.unique(dm.train_indexes)) == len(dm.train_indexes), "Duplicates in train_indexes"
        assert len(np.intersect1d(dm.train_indexes, dm.heldout_indices)) == 0, "Overlap between train and heldout"

    # Final state
    log_bo_metrics(data_stats, dm.train_y, epoch=n_iters)
    final_coverage = (dm.train_y >= data_stats["target_q95"]).sum().item() / max(n_top, 1)
    print(f"[Seed {seed}] Final top-5% coverage: {final_coverage:.4f}")


def run(config):
    wandb_config = flatten(config)
    n_seeds = config.get("n_seeds", 1)
    base_seed = config["seed"]
    seeds = [1, 15, 23, 43, 50]

    for seed in seeds:
        run_name = f"random_seed{seed}"
        with wandb.init(
            project=config.get("wandb_project", "gollum"),
            config={**wandb_config, "seed": seed},
            group=config["group"],
            name=run_name,
            reinit=True,
        ):
            run_one_seed(config, seed)


def main():
    parser = argparse.ArgumentParser(description="Random baseline for BO coverage evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, help="Override base random seed")
    parser.add_argument("--n_seeds", type=int, help="Override number of seeds to run sequentially")
    parser.add_argument("--n_iters", type=int, help="Override number of BO iterations")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--group", type=str, help="Override wandb group")
    parser.add_argument("--wandb_project", type=str, help="Override wandb project name")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in ("seed", "n_seeds", "n_iters", "batch_size", "group", "wandb_project"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    run(config)


if __name__ == "__main__":
    main()
