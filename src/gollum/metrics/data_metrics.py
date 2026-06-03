import torch
import wandb


def calculate_data_stats(x, y):
    """
    Calculate statistics of the dataset.
    """
    stats = {
        "target_stat_max": torch.max(y.float()),
        "target_stat_mean": torch.mean(y.float()),
        "target_stat_std": torch.std(y.float()),
        "target_stat_var": torch.var(y.float()),
        "input_stat_feature_dimension": x.shape[-1],
        "input_stat_n_points": x.shape[0],
    }
    # Adding quantiles and top values
    for q in [0.75, 0.9, 0.95, 0.99]:
        stats[f"target_q{int(q * 100)}"] = torch.quantile(y.float(), q)
    for n in [1, 3, 5, 10]:
        k = min(n, y.shape[0])
        top_values, _ = torch.topk(y, k, dim=0)
        stats[f"top_{n}"] = top_values[-1]
    return stats


def calculate_data_stats_numpy(x_np, y_np):
    """
    Wrapper for calculate_data_stats that accepts numpy arrays.

    Args:
        x_np (np.ndarray): Input features as numpy array
        y_np (np.ndarray): Target values as numpy array
    """
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    return calculate_data_stats(x, y)


def log_data_stats(data_metrics):
    """
    Log data statistics to WandB summary.
    """
    for key, value in data_metrics.items():
        wandb.summary[key] = value.item() if torch.is_tensor(value) else value


def log_bo_metrics(data_stats, train_y, epoch=0, prefix=""):
    """
    Log bo-specific metrics (quantiles, top counts and best so far) to WandB.

    ``prefix`` namespaces the keys (e.g. "phase2/") so Phase-1 and Phase-2 BO
    don't overwrite each other's curves.
    """
    log_best_so_far(train_y, epoch, prefix=prefix)
    log_top_n_counts(data_stats, train_y, epoch, prefix=prefix)
    log_quantile_counts(data_stats, train_y, epoch, prefix=prefix)


def log_best_so_far(train_y, epoch=0, prefix=""):
    """
    Log the best-so-far value to WandB.
    """
    best_so_far = torch.max(train_y).item()
    wandb.log({f"{prefix}train/best_so_far": best_so_far, "epoch": epoch})


def _to_device(threshold, train_y):
    """Match the threshold's device to train_y (data_stats may be built from a
    different device than the y being scored, e.g. CPU test stats vs CUDA y)."""
    if torch.is_tensor(threshold):
        return threshold.to(train_y.device)
    return threshold


def log_top_n_counts(data_stats, train_y, epoch=0, prefix=""):
    """
    Log the count of top N values to WandB.
    """
    for n in [1, 3, 5, 10]:
        threshold = _to_device(data_stats[f"top_{n}"], train_y)
        count = (train_y >= threshold).sum().item()
        wandb.log({f"{prefix}top_{n}_count": count, "epoch": epoch})


def log_quantile_counts(data_stats, train_y, epoch=0, prefix=""):
    """
    Log the count of quantiles to WandB.
    """
    for q in [0.75, 0.9, 0.95, 0.99]:
        threshold = _to_device(data_stats[f"target_q{int(q * 100)}"], train_y)
        count = (train_y >= threshold).sum().item()
        wandb.log({f"{prefix}quantile_{int(q * 100)}_count": count, "epoch": epoch})
