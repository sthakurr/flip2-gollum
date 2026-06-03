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
    # Total count of points in the top 5% of the full dataset (denominator for coverage)
    stats["total_q95_count"] = (y.float() >= stats["target_q95"]).sum()
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


def log_bo_metrics(data_stats, train_y, epoch=0, extra=None):
    """
    Log bo-specific metrics (quantiles, top counts and best so far) to WandB.
    All metrics are batched into a single wandb.log call to avoid duplicate steps.
    """
    metrics = {"epoch": epoch}
    metrics.update(_best_so_far_metrics(train_y))
    metrics.update(_top_n_count_metrics(data_stats, train_y))
    metrics.update(_quantile_count_metrics(data_stats, train_y))
    metrics.update(_top5pct_coverage_metrics(data_stats, train_y))
    if extra:
        metrics.update(extra)
    wandb.log(metrics)


def _top5pct_coverage_metrics(data_stats, train_y):
    if train_y.numel() == 0:
        return {}
    total = data_stats["total_q95_count"].item()
    if total > 0:
        threshold = data_stats["target_q95"]
        found = (train_y >= threshold).sum().item()
        return {"top5pct_coverage": found / total}
    return {}


def _best_so_far_metrics(train_y):
    if train_y.numel() == 0:
        return {}
    return {"train/best_so_far": torch.max(train_y).item()}


def _top_n_count_metrics(data_stats, train_y):
    if train_y.numel() == 0:
        return {}
    metrics = {}
    for n in [1, 3, 5, 10]:
        threshold = data_stats[f"top_{n}"]
        metrics[f"top_{n}_count"] = (train_y >= threshold).sum().item()
    return metrics


def _quantile_count_metrics(data_stats, train_y):
    if train_y.numel() == 0:
        return {}
    metrics = {}
    for q in [0.75, 0.9, 0.95, 0.99]:
        threshold = data_stats[f"target_q{int(q * 100)}"]
        metrics[f"quantile_{int(q * 100)}_count"] = (train_y >= threshold).sum().item()
    return metrics


# Keep old names as thin wrappers for any external callers
def log_top5pct_coverage(data_stats, train_y, epoch=0):
    wandb.log({"top5pct_coverage": _top5pct_coverage_metrics(data_stats, train_y).get("top5pct_coverage"), "epoch": epoch})

def log_best_so_far(train_y, epoch=0):
    wandb.log({"train/best_so_far": torch.max(train_y).item(), "epoch": epoch})

def log_top_n_counts(data_stats, train_y, epoch=0):
    wandb.log({**_top_n_count_metrics(data_stats, train_y), "epoch": epoch})

def log_quantile_counts(data_stats, train_y, epoch=0):
    wandb.log({**_quantile_count_metrics(data_stats, train_y), "epoch": epoch})
