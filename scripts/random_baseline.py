"""Model-free RANDOM baseline for the FLIP2 BO benchmarks.

Standalone — does not touch train.py. It builds the same data module and logs the
same metrics as the BO runs (train/best_so_far, train/coverage_top{1,5,10} for
Phase 1; phase2/best_test, phase2/simple_regret, phase2/train/best_so_far,
phase2/coverage_top{1,5,10} for Phase 2), but selects batch_size uniformly-random
design points each iteration instead of fitting a surrogate.

Mirrors run_bo / run_phase2 in train.py exactly, minus the surrogate/acquisition,
so the curves are directly comparable. Runs on CPU (featurization is cheap).

Usage:
  python random_baseline.py --config configs/flip2_arms/onehot/onehot.yaml \
      --data_path data/flip2/nucB/two_to_many.csv --seed 0 --name nucB_two_to_many__random
"""
import argparse

import numpy as np
import torch
import wandb
import yaml
from pytorch_lightning import seed_everything

from gollum.utils.config import instantiate_class, flatten
from gollum.metrics import calculate_data_stats, log_bo_metrics, log_data_stats

PROJECT = "gollum-flip2-final"           # same W&B project as the BO runs
COVERAGE_QS = [(0.99, 1), (0.95, 5), (0.90, 10)]


def setup_data(config):
    """Same construction as train.setup_data (replicated to stay standalone)."""
    initializer = instantiate_class(
        config["data"]["init_args"]["initializer"], seed=config["seed"]
    )
    featurizer = instantiate_class(config["data"]["init_args"]["featurizer"])
    return instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=config["data"]["init_args"]["normalize_input"],
        maximize=config["data"]["init_args"]["maximize"],
    )


def _coverage_bands(y_flat, stats):
    """thr + band_size per top-q band, exactly as run_bo / run_phase2 compute them."""
    bands = {}
    for q, label in COVERAGE_QS:
        thr = stats[f"target_q{int(q * 100)}"]
        band_size = int((y_flat >= thr).sum().item())
        bands[label] = (thr, max(band_size, 1))
    return bands


def _log_coverage(prefix, bands, acquired_y, epoch):
    flat = acquired_y.squeeze()
    for label, (thr, band_size) in bands.items():
        n_hit = int((flat >= thr).sum().item()) if acquired_y.numel() else 0
        wandb.log({f"{prefix}coverage_top{label}": n_hit / band_size, "epoch": epoch})


def run_phase1_random(dm, data_stats, n_iters, batch, rng):
    """Phase-1 random acquisition over the train design space (dm.heldout_*)."""
    full_y = torch.cat([dm.train_y, dm.heldout_y], dim=0)
    pool_stats = calculate_data_stats(
        torch.cat([dm.train_x, dm.heldout_x], dim=0), full_y
    )
    bands = _coverage_bands(full_y.squeeze(), pool_stats)

    train_y = dm.train_y.clone()
    heldout_y = dm.heldout_y.clone()

    for i in range(n_iters):
        log_bo_metrics(data_stats, train_y, epoch=i)
        _log_coverage("train/", bands, train_y, i)

        n = heldout_y.shape[0]
        if n == 0:
            continue
        k = min(batch, n)
        idx = rng.choice(n, size=k, replace=False)
        train_y = torch.cat([train_y, heldout_y[idx]], dim=0)
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        heldout_y = heldout_y[torch.from_numpy(keep)]

    log_bo_metrics(data_stats, train_y, epoch=n_iters)
    _log_coverage("train/", bands, train_y, n_iters)


def run_phase2_random(dm, n_iters, epoch_offset, batch, rng):
    """Phase-2 random acquisition over the held-aside test split (dm.test_*)."""
    test_stats = calculate_data_stats(dm.test_x, dm.test_y)
    f_max = dm.test_y.max().item()
    bands = _coverage_bands(dm.test_y.squeeze(), test_stats)

    design_y = dm.test_y.clone()
    acquired = dm.test_y.new_empty((0, dm.test_y.shape[-1]))

    for i in range(n_iters):
        n = design_y.shape[0]
        if n == 0:
            print("Phase 2: test design space exhausted; stopping early.")
            break
        k = min(batch, n)
        idx = rng.choice(n, size=k, replace=False)
        acquired = torch.cat([acquired, design_y[idx]], dim=0)
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        design_y = design_y[torch.from_numpy(keep)]

        epoch = epoch_offset + i
        best = acquired.max().item()
        wandb.log({
            "phase2/best_test": best,
            "phase2/simple_regret": f_max - best,
            "phase2/n_acquired": int(acquired.shape[0]),
            "epoch": epoch,
        })
        log_bo_metrics(test_stats, acquired, epoch=epoch, prefix="phase2/")
        _log_coverage("phase2/", bands, acquired, epoch)


def main():
    ap = argparse.ArgumentParser(description="Random (model-free) BO baseline")
    ap.add_argument("--config", required=True, help="YAML providing the data block + batch_size")
    ap.add_argument("--data_path", default=None, help="Override data CSV path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="W&B run-name base (e.g. <dataset>__random)")
    ap.add_argument("--group", default=None, help="W&B group")
    ap.add_argument("--phase1_iters", type=int, default=None)
    ap.add_argument("--phase2_iters", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    if args.data_path is not None:
        config["data"]["init_args"]["data_path"] = args.data_path
    # train.py exposes these as top-level CLI args, so they appear as flat keys in
    # the logged config. Mirror them so W&B grouping/filtering matches the BO runs.
    config["data_path"] = config["data"]["init_args"]["data_path"]
    config["name"] = args.name

    p1 = args.phase1_iters or config.get("phase1_iters") or config.get("n_iters", 3)
    p2 = args.phase2_iters or config.get("phase2_iters") or config.get("n_iters", 3)
    batch = config["bo"]["init_args"].get("batch_size", 96)

    seed_everything(args.seed, workers=True)
    dm = setup_data(config)
    if dm.test_x is None:
        raise ValueError("Random baseline needs a held-aside test set; set respect_split: true.")

    run_name = f"{args.name or 'random'}_random_full_seed{args.seed}"
    rng = np.random.default_rng(args.seed)

    # Flatten exactly like train.py so config keys (e.g. data.init_args.data_path)
    # match the BO runs and group/filter correctly in W&B.
    wandb_config = flatten(config)
    wandb_config["rep"] = "random"
    wandb_config["mode"] = "full"

    with wandb.init(project=PROJECT, name=run_name, group=args.group,
                    config=wandb_config):
        data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)
        run_phase1_random(dm, data_stats, p1, batch, rng)
        run_phase2_random(dm, p2, epoch_offset=p1 + 1, batch=batch, rng=rng)


if __name__ == "__main__":
    main()
