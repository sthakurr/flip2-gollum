"""The BO phases: Phase-1 acquisition over the train pool, the train->test
ranking eval, and Phase-2 acquisition over the test split.
"""
import os

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from gollum.bo.config import CHECKPOINT_DIR
from gollum.featurization.mutation import _consensus
from gollum.metrics import (
    calculate_data_stats,
    log_bo_metrics,
    log_surrogate_eval,
)
from gollum.reasoning.agent import select_acquisitions_with_llm


def _run_out_dir():
    """Per-run output dir under CHECKPOINT_DIR, falling back to ./logs when that
    path isn't writable (e.g. the cluster scratch path on a local machine)."""
    name = wandb.run.name if wandb.run else "default"
    for base in (CHECKPOINT_DIR, "logs"):
        try:
            out = os.path.join(base, name)
            os.makedirs(out, exist_ok=True)
            return out
        except OSError:
            continue
    raise OSError("no writable output directory for run logs")


def log_acq_topk(dm, scores, iteration, top_n=200):
    """Append the top-`top_n` design-space sequences by acquisition value to
    CHECKPOINT_DIR/<run>/acq_topk_sequences.csv (one block per BO iteration).
    No-op if `scores` is None (acquisition paths that don't set them)."""
    if scores is None:
        return
    import csv
    scores = scores.reshape(-1).cpu()
    top = torch.topk(scores, min(top_n, scores.numel()))
    orig = np.asarray(dm.heldout_indices)[top.indices.numpy()]
    path = os.path.join(_run_out_dir(), "acq_topk_sequences.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["iter", "rank", "acq_value", "index", "sequence", "fitness"])
        for rank, (idx, val) in enumerate(zip(orig, top.values.tolist())):
            row = dm.data.loc[int(idx)]
            w.writerow([iteration, rank, val, int(idx),
                        row[dm.input_column], row[dm.target_column]])


def log_acquired(dm, original_indices, iteration):
    """Append the Phase-1 batch acquired at `iteration` to
    CHECKPOINT_DIR/<run>/acquired_sequences.csv, with each sequence's true
    fitness. `original_indices` index into dm.data."""
    import csv
    path = os.path.join(_run_out_dir(), "acquired_sequences.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["iter", "index", "sequence", "fitness"])
        for idx in np.asarray(original_indices).tolist():
            row = dm.data.loc[int(idx)]
            w.writerow([iteration, int(idx),
                        row[dm.input_column], row[dm.target_column]])


def log_llm_reasoning(reasoning, iteration):
    """Print and append the LLM's reasoning trace for `iteration`'s re-ranking
    call to CHECKPOINT_DIR/<run>/llm_reasoning.csv. No-op if empty
    (provider/model exposed none)."""
    if not reasoning:
        return
    print(f"[iter {iteration}] LLM reasoning: {reasoning}", flush=True)
    import csv
    path = os.path.join(_run_out_dir(), "llm_reasoning.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["iter", "reasoning"])
        w.writerow([iteration, reasoning])


def run_test_eval(config, dm, bo):
    """How well does the Phase-1 model rank the held-aside test split?

    Re-fits the surrogate on the full Phase-1 train set (Phase-1's last fit
    predates the final acquired batch), then scores its predicted ranking of the
    test set (spearman / kendall / recovery@k) and the calibration of its
    posterior (nlpd / msll / qce / r2 / mse). Answers "does this representation
    transfer data_path -> test_path?".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dm.test_x is None:
        raise ValueError("test eval needs a held-aside test set; provide test_path.")
    train_x = dm.train_x.clone().to(device)
    train_y = dm.train_y.clone().to(device)
    test_x = dm.test_x.clone().to(device)
    test_y = dm.test_y.clone().to(device)

    print(f"Test eval: fitting on {train_x.shape[0]} train points, "
          f"ranking {test_x.shape[0]} test points")
    bo.train_surrogate_model(train_x, train_y)

    # Runs right after Phase-1, so log it at the last Phase-1 epoch.
    epoch = config.get("phase1_iters") or config.get("n_iters", 0)

    posterior = bo.surrogate_model.predict(test_x, return_posterior=True)
    metrics = log_surrogate_eval(posterior, test_y, stage="test", epoch=epoch)

    print("Test metrics (train->test):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def _predict_gp_stats(surrogate_model, x, chunk_size=32):
    """Predicted fitness (mean, std) for each row of `x`, processed in chunks.

    For static (precomputed-embedding) surrogates this is a cheap lookup, but
    for a fine-tuned (LoRA) surrogate each call is a real forward pass through
    the featurizer -- scoring the whole pool in one batch can OOM alongside the
    fine-tuning itself, so this chunks the pool instead.
    """
    means, stds = [], []
    for start in range(0, x.shape[0], chunk_size):
        mean, variance = surrogate_model.predict(x[start : start + chunk_size])
        means.append(mean.reshape(-1))
        stds.append(variance.reshape(-1).clamp_min(0).sqrt())
    return torch.cat(means).tolist(), torch.cat(stds).tolist()


def _rerank_with_llm(dm, bo, design_space, pool_positions, acq_scores, wild_type,
                      reasoning_config, batch_size, iteration):
    """Re-rank the acquisition function's candidate pool with an LLM, log its
    reasoning trace, and return the chosen positions into the design space (a
    `batch_size` subset of `pool_positions`, LLM-ranked).

    Candidate/history dicts are built fresh from `dm`'s current state each
    call, so the LLM always sees the live acquisition history. If
    `reasoning_config["include_gp_stats"]` is set, each candidate also carries
    the just-fit surrogate's predicted fitness (mean +/- std) so prompts that
    reference it (e.g. prompts/p1.txt) have real numbers to reason over.
    """
    pool_positions_list = pool_positions.tolist()
    original_indices = dm.heldout_indices[pool_positions_list]

    gp_mean = gp_std = None
    if reasoning_config.get("include_gp_stats"):
        pool_x = design_space[pool_positions.to(design_space.device)]
        chunk_size = reasoning_config.get("gp_stats_chunk_size", 32)
        gp_mean, gp_std = _predict_gp_stats(bo.surrogate_model, pool_x, chunk_size)

    candidates = []
    for j, (pos, orig_idx) in enumerate(zip(pool_positions_list, original_indices)):
        row = dm.data.loc[int(orig_idx)]
        candidate = {"sequence": row[dm.input_column], "_position": pos}
        if acq_scores is not None:
            candidate["acquisition_value"] = float(acq_scores[pos])
        if gp_mean is not None:
            candidate["gp_mean"] = gp_mean[j]
            candidate["gp_std"] = gp_std[j]
        candidates.append(candidate)

    history = [
        {"sequence": row[dm.input_column], "fitness": float(row[dm.target_column])}
        for _, row in dm.data.loc[dm.train_indexes].iterrows()
    ]

    prompt_path = reasoning_config.get("prompt_path")
    extra_args = {"prompt_path": prompt_path} if prompt_path else {}
    chosen, llm_reasoning = select_acquisitions_with_llm(
        candidates, history, wild_type, reasoning_config["llm_config"], batch_size,
        ensemble_size=reasoning_config.get("ensemble_size", 1),
        **extra_args,
    )
    log_llm_reasoning(llm_reasoning, iteration)
    return torch.tensor([c["_position"] for c in chosen], dtype=torch.long)


def _log_recovery_at_10(dm, pool_positions, before_positions, after_positions, iteration):
    """Debug-print recovery@10: of the true top-10 (by ground-truth fitness)
    within the `pool_positions` candidate pool, how many land in the pre-rerank
    (raw acquisition top-batch_size) vs post-rerank (LLM top-batch_size)
    selection. Ground-truth fitness is read here only for this printout -- it
    never reaches the surrogate, acquisition function, or LLM prompt.
    """
    orig = dm.heldout_indices[pool_positions.tolist()]
    fitness = dm.data.loc[orig][dm.target_column].to_numpy()
    k = min(10, len(orig))
    true_top10 = set(orig[np.argsort(fitness)[-k:]].tolist())

    before_orig = set(dm.heldout_indices[before_positions.tolist()].tolist())
    after_orig = set(dm.heldout_indices[after_positions.tolist()].tolist())

    print(
        f"[iter {iteration}] recovery@10 (true top-{k} within pool of {len(orig)}): "
        f"before-rerank={len(true_top10 & before_orig)}/{k}, "
        f"after-rerank={len(true_top10 & after_orig)}/{k}"
    )


def run_bo(config, dm, bo, data_stats, n_iters=None):
    """Phase-1 BO loop: iteratively acquire candidates from the held-out design
    space (the remaining train pool)."""
    n_iters = n_iters if n_iters is not None else config["n_iters"]

    # Optional LLM re-ranking: the acquisition function proposes a
    # `pool_size`-large shortlist, an LLM re-ranks it using the wild-type
    # sequence + acquisition history as context, and the top `batch_size` of
    # that ranking becomes the actual acquired batch. Off by default.
    reasoning = config.get("reasoning")
    wild_type = _consensus(dm.data[dm.input_column].tolist()) if reasoning else None
    if reasoning is not None and reasoning["pool_size"] < bo.batch_size:
        raise ValueError(
            f"reasoning.pool_size ({reasoning['pool_size']}) must be >= "
            f"batch_size ({bo.batch_size})."
        )

    full_train_x = torch.cat([dm.train_x, dm.heldout_x], dim=0)
    full_train_y = torch.cat([dm.train_y, dm.heldout_y], dim=0)
    train_pool_stats = calculate_data_stats(full_train_x, full_train_y)
    train_y_flat = full_train_y.squeeze()
    coverage_bands = {}  # label (e.g. 5) -> (threshold, band_size)
    for q, label in [(0.99, 1), (0.95, 5), (0.90, 10)]:
        thr = train_pool_stats[f"target_q{int(q * 100)}"]
        band_size = int((train_y_flat >= thr).sum().item())
        coverage_bands[label] = (thr, max(band_size, 1))

    def _log_train_coverage(epoch):
        flat = dm.train_y.squeeze()
        for label, (thr, band_size) in coverage_bands.items():
            n_hit = int((flat >= thr.to(flat.device)).sum().item())
            wandb.log({f"train/coverage_top{label}": n_hit / band_size, "epoch": epoch})

    # Opt-in latent diagnostics (DeepGP only): once per BO epoch, embed the whole
    # train+heldout pool with that epoch's just-fit model to plot latent-space
    # evolution + d_hh/d_ll/d_hl, and/or save the finetuned model.
    diag = None
    if (config["surrogate_model"]["class_path"] == "gollum.surrogate_models.gp.DeepGP"
            and any(config.get(f) for f in ("save_epoch_models", "visualize_latent", "plot_distances"))):
        from gollum.visualization.latent import LatentDiagnostics
        diag = LatentDiagnostics(
            out_dir=os.path.join(CHECKPOINT_DIR, wandb.run.name if wandb.run else "default", "latent"),
            viz_x=torch.cat([dm.train_x, dm.heldout_x], dim=0).to("cuda", torch.float64),
            viz_y=torch.cat([dm.train_y, dm.heldout_y], dim=0),
            save_models=bool(config.get("save_epoch_models")),
            viz_latent=bool(config.get("visualize_latent")),
            plot_dist=bool(config.get("plot_distances")),
        )

    for i in tqdm(range(n_iters), colour="blue"):
        train_x = dm.train_x.clone().to("cuda")
        train_y = dm.train_y.clone().to("cuda")
        design_space = dm.heldout_x.clone().to("cuda")

        ## this trains the model, updates acqf and returns the next point to evaluate
        if reasoning is not None:
            batch_size = bo.batch_size
            bo.batch_size = reasoning["pool_size"]
            x_next = bo.suggest_next_experiments(train_x, train_y, design_space)
            bo.batch_size = batch_size
        else:
            x_next = bo.suggest_next_experiments(train_x, train_y, design_space)
        if diag is not None:
            diag.record(bo.surrogate_model, i)
        log_acq_topk(dm, getattr(bo, "last_acq_scores", None), i)  # comment out to disable
        x_next = torch.stack(x_next)

        log_bo_metrics(data_stats, dm.train_y, epoch=i)
        _log_train_coverage(i)

        matches = (design_space.unsqueeze(0).to("cuda") == x_next).all(dim=-1)
        indices = matches.nonzero(as_tuple=True)[1].to("cpu")

        if not torch.all(matches.sum(dim=-1) == 1):
            print("Unable to find a unique match for some x_next in the dataset.")

        if reasoning is not None:
            pool_positions = indices
            before_positions = pool_positions[: bo.batch_size]
            indices = _rerank_with_llm(
                dm, bo, design_space, pool_positions, getattr(bo, "last_acq_scores", None),
                wild_type, reasoning, bo.batch_size, i,
            )
            _log_recovery_at_10(dm, pool_positions, before_positions, indices, i)

        x_next = x_next.squeeze(1)

        # update indices tracking
        evaluated_original_indices = dm.heldout_indices[indices]
        log_acquired(dm, evaluated_original_indices, i)
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
        # With a test_path the test rows are held aside (not in train/heldout).
        n_test = 0 if getattr(dm, "test_indices", None) is None else len(dm.test_indices)
        total_indices = len(dm.train_indexes) + len(dm.heldout_indices) + n_test
        assert total_indices == len(dm.x), "Mismatch in the total number of indices"

    log_bo_metrics(data_stats, dm.train_y, epoch=n_iters)
    _log_train_coverage(n_iters)
    if diag is not None:
        diag.finalize()


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
        raise ValueError("Phase 2 needs a held-aside test set; provide test_path.")

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

        _log_metrics(epoch_offset + i, new_y)

    final_best = acquired_test_y.max().item() if acquired_test_y.numel() else float("nan")
    print(
        f"Phase 2 done [seed_source={seed_source}]: best test value {final_best:.4f} "
        f"(test max {f_max:.4f}, final simple regret {f_max - final_best:.4f})"
    )
