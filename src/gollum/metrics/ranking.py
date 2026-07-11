"""Ranking / recovery metrics for evaluating how well a surrogate trained on the
train split transfers to (ranks) the held-out test split.

These complement the regression/calibration metrics in ``model_metrics.py``
(NLPD/MSLL/QCE/R2) with rank-based and top-k recovery measures, which are what
actually matter for Bayesian Optimization: we care whether the surrogate ranks
the high performers above the rest, not its absolute error.
"""
import math

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


def _embed_for_kernel(model, x):
    """Map raw inputs into the space the kernel actually acts on.

    DeepGP: push through the trained feature map (finetuning_model) and apply
    the same normalization ``forward`` applies before the kernel, so the kernel
    is evaluated in the space its lengthscales were fit in. Plain GP: apply the
    BoTorch input transform (Normalize). Returns a detached tensor on the
    kernel's device.
    """
    model.eval()
    with torch.no_grad():
        if getattr(model, "finetuning_model", None) is not None:
            emb = model.finetuning_model(x)
            if getattr(model, "minmax_embeddings", False):
                mn, rng = getattr(model, "_embed_min", None), getattr(model, "_embed_range", None)
                if mn is not None and rng is not None:
                    emb = (emb - mn) / rng
            elif getattr(model, "normalise_embeddings", False):
                mean, std = model._embed_mean, model._embed_std
                if mean is not None and std is not None:
                    emb = (emb - mean) / std
            elif getattr(model, "scale_embeddings", False):
                emb = model.scale_to_bounds(emb)
        elif hasattr(model, "transform_inputs"):
            emb = model.transform_inputs(x)
        else:
            emb = x
    return emb


def _base_kernel_lengthscale(covar_module):
    """Mean fitted lengthscale of the (possibly ScaleKernel-wrapped, possibly
    ARD) base kernel. Returns a float, or None if no lengthscale is exposed.
    """
    ls = getattr(getattr(covar_module, "base_kernel", None), "lengthscale", None)
    if ls is None:
        ls = getattr(covar_module, "lengthscale", None)
    if ls is None:
        return None
    return ls.detach().float().mean().item()


def log_prior_correlation(
    model, train_x, test_x, stage="test", n_ref=8, far_frac=0.1, seed=0, epoch=0
):
    """Ober-style prior-correlation diagnostic for a fitted surrogate.

    For random *train* reference points, compute the normalized kernel
    correlation rho(ref, x) = K(ref,x) / sqrt(K(ref,ref) K(x,x)) against the
    train set and the held-aside test set, evaluated in the space the kernel
    acts on (DeepGP feature map / GP input transform). A healthy SE-like kernel
    decays with distance; the DKL pathology keeps rho high/flat for far points.

    Prints a console diagnostic (fitted lengthscale vs pairwise/NN distance, the
    "flatness" number) and returns a small summary dict.
    """
    emb_tr = _embed_for_kernel(model, train_x)
    emb_te = _embed_for_kernel(model, test_x)
    emb = torch.cat([emb_tr, emb_te], dim=0)
    n_tr = emb_tr.shape[0]

    with torch.no_grad():
        K = model.covar_module(emb).evaluate()
        d = torch.sqrt(torch.diag(K).clamp_min(1e-12))
        rho = (K / d.unsqueeze(0) / d.unsqueeze(1)).cpu()
        dist = torch.cdist(emb, emb).cpu()

    g = torch.Generator().manual_seed(seed)
    refs = torch.randperm(n_tr, generator=g)[: min(n_ref, n_tr)]

    rho_tr, rho_te, far_tr, far_te = [], [], [], []
    for r in refs.tolist():
        rr = rho[r].clone()
        rr[r] = float("nan")  # drop self
        tr_vals = rr[:n_tr]
        te_vals = rr[n_tr:]
        tr_vals = tr_vals[~torch.isnan(tr_vals)]
        rho_tr.append(tr_vals)
        rho_te.append(te_vals)
        # "flatness": correlation to the farthest far_frac of all points
        dr = dist[r].clone()
        dr[r] = -1.0
        k_far = max(1, int(far_frac * dr.shape[0]))
        far_idx = dr.topk(k_far).indices
        far_tr.append(rho[r][far_idx[far_idx < n_tr]])
        far_te.append(rho[r][far_idx[far_idx >= n_tr]])

    rho_tr = torch.cat(rho_tr)
    rho_te = torch.cat(rho_te)
    far_all = torch.cat([torch.cat(far_tr), torch.cat(far_te)])

    iu = torch.triu_indices(dist.shape[0], dist.shape[1], offset=1)
    all_pair_dist = dist[iu[0], iu[1]]
    median_dist = all_pair_dist.median().item()
    # Gate-relevant scale: for each test point, its distance to the NEAREST
    # train point (this, not the global median, governs train->test prediction).
    tt_block = dist[:n_tr, n_tr:]  # (n_train, n_test)
    nn_test_to_train = tt_block.min(dim=0).values if tt_block.numel() else dist.new_empty(0)
    median_nn_dist = nn_test_to_train.median().item() if nn_test_to_train.numel() else float("nan")
    lengthscale = _base_kernel_lengthscale(model.covar_module)
    ratio = lengthscale / median_dist if (lengthscale and median_dist) else float("nan")
    nn_ratio = (
        lengthscale / median_nn_dist
        if (lengthscale and median_nn_dist and not math.isnan(median_nn_dist))
        else float("nan")
    )

    summary = {
        f"prior_corr/{stage}/rho_train_mean": rho_tr.mean().item(),
        f"prior_corr/{stage}/rho_test_mean": rho_te.mean().item(),
        f"prior_corr/{stage}/rho_far_absmean": far_all.abs().mean().item(),
        f"prior_corr/{stage}/lengthscale_mean": lengthscale,
        f"prior_corr/{stage}/median_dist": median_dist,
        f"prior_corr/{stage}/median_nn_dist": median_nn_dist,
        f"prior_corr/{stage}/lengthscale_over_dist": ratio,
        f"prior_corr/{stage}/lengthscale_over_nn_dist": nn_ratio,
    }
    print(
        f"[prior_corr/{stage}] fitted lengthscale={lengthscale:.3g}, "
        f"median pairwise dist={median_dist:.3g}, ratio={ratio:.3g} "
        f"({'OVER-SMOOTHING' if ratio > 2 else 'ok'}); "
        f"median test->NN-train dist={median_nn_dist:.3g}, nn_ratio={nn_ratio:.3g}; "
        f"rho_far_absmean={summary[f'prior_corr/{stage}/rho_far_absmean']:.3g}"
    )
    return summary
