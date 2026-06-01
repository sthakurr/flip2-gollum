"""Ranking / recovery metrics for evaluating how well a surrogate trained on the
train split transfers to (ranks) the held-out test split.

These complement the regression/calibration metrics in ``model_metrics.py``
(NLPD/MSLL/QCE/R2) with rank-based and top-k recovery measures, which are what
actually matter for Bayesian Optimization: we care whether the surrogate ranks
the high performers above the rest, not its absolute error.
"""
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from gollum.metrics.model_metrics import calculate_model_fit_metrics


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).squeeze()


def spearman(preds, targets):
    """Spearman rank correlation between predictions and true targets."""
    preds, targets = _to_numpy(preds), _to_numpy(targets)
    if preds.size < 2:
        return float("nan")
    return float(spearmanr(preds, targets).correlation)


def kendall(preds, targets):
    """Kendall's tau rank correlation between predictions and true targets."""
    preds, targets = _to_numpy(preds), _to_numpy(targets)
    if preds.size < 2:
        return float("nan")
    return float(kendalltau(preds, targets).correlation)


def top_k_recovery(preds, targets, k):
    """Fraction of the true top-k items that appear in the predicted top-k.

    Recovery@k = |topk(preds) ∩ topk(targets)| / k. This is the BO-relevant
    question: if we picked the k points the surrogate likes most, how many of
    the truly best k would we have grabbed?
    """
    preds, targets = _to_numpy(preds), _to_numpy(targets)
    k = int(min(k, preds.size))
    if k < 1:
        return float("nan")
    pred_top = set(np.argsort(preds)[-k:].tolist())
    true_top = set(np.argsort(targets)[-k:].tolist())
    return len(pred_top & true_top) / k


def precision_at_k(preds, targets, k, quantile=0.9):
    """Fraction of the predicted top-k that are truly "good".

    "Good" = target at or above ``quantile`` of the target distribution. Unlike
    recovery@k this rewards the surrogate for spending its budget on genuinely
    high performers even if it misses the exact identities of the top-k.
    """
    preds, targets = _to_numpy(preds), _to_numpy(targets)
    k = int(min(k, preds.size))
    if k < 1:
        return float("nan")
    threshold = np.quantile(targets, quantile)
    pred_top = np.argsort(preds)[-k:]
    return float(np.mean(targets[pred_top] >= threshold))


def calculate_ranking_metrics(preds, targets, ks=(1, 3, 5, 10), stage="test"):
    """Bundle the rank-correlation and recovery metrics into a flat dict."""
    metrics = {
        f"{stage}/spearman": spearman(preds, targets),
        f"{stage}/kendall": kendall(preds, targets),
    }
    for k in ks:
        metrics[f"{stage}/recovery@{k}"] = top_k_recovery(preds, targets, k)
        metrics[f"{stage}/precision@{k}"] = precision_at_k(preds, targets, k)
    return metrics


def _underlying_mvn(posterior):
    """The gpytorch MultivariateNormal behind a BoTorch posterior. The gpytorch
    metric functions expect this (with a 1-D y), not the BoTorch posterior whose
    mean/variance carry an extra trailing output dim."""
    for attr in ("mvn", "distribution"):
        if hasattr(posterior, attr):
            return getattr(posterior, attr)
    return posterior


def log_surrogate_eval(posterior, y, stage="test", ks=(1, 3, 5, 10), epoch=0):
    """Evaluate a fitted surrogate's posterior against true targets ``y`` and log
    to W&B. Combines regression/calibration metrics (NLPD/MSLL/QCE/R2/MSE) with
    the rank/recovery metrics. Returns the merged metrics dict.
    """
    import wandb

    # Fit/calibration metrics need the underlying MVN + 1-D targets so the
    # gpytorch reductions collapse to scalars (not a per-point vector).
    mvn = _underlying_mvn(posterior)
    y_1d = y.squeeze(-1) if y.dim() > 1 else y
    fit_metrics = calculate_model_fit_metrics(mvn, y_1d, stage=stage)
    rank_metrics = calculate_ranking_metrics(posterior.mean, y, ks=ks, stage=stage)
    metrics = {**fit_metrics, **rank_metrics}

    if wandb.run is not None:
        wandb.log({**metrics, "epoch": epoch})
    return metrics
