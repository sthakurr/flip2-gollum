"""Kernel/embedding diagnostics for the GP fit.

Answers "is the kernel saturated?": if the lengthscale is large relative to the
spread of the embeddings, K approaches 1 everywhere and the GP cannot tell its
inputs apart. The ratio ``kernel/lengthscale_to_median_dist`` is that check;
everything else here is the supporting evidence (distance spread, K spectrum,
LoRA weight growth).

Two cadences:

* per fit step (`gp_diagnostics`) -- hyperparameters, kernel-matrix stats and
  LoRA norms, all read off tensors the forward pass already produced, so no
  extra LLM forward.
* per BO round (`dataset_diagnostics`) -- embedding geometry over the whole
  dataset rather than the acquired points, which costs one eval-mode forward
  (~1-2 fit steps' worth of compute).
"""
import torch


def _quantiles(v, qs):
    """Quantiles via sort. torch.quantile hard-errors above 2**24 elements and
    a full pairwise-distance vector passes that at n>~5800."""
    s = v.reshape(-1).sort().values
    n = s.numel()
    idx = [min(n - 1, max(0, int(round(q * (n - 1))))) for q in qs]
    return [s[i].item() for i in idx]


def _subsample(n, max_points, device, seed=0):
    """Fixed-seed row subsample, so the series stays comparable across rounds."""
    if n <= max_points:
        return None
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=g)[:max_points].to(device)


def distance_stats(emb, y=None, top_q=0.05, bot_q=0.5, min_group=5,
                   max_points=3000, prefix="embed"):
    """Pairwise Euclidean distance stats in the embedding space.

    `emb` is the kernel input (n, d). With `y` (n,) also returns stratified
    distances: within the top `top_q` fraction, within the bottom `bot_q`
    fraction, and between the two groups. Group sizes are floored at
    `min_group` so an early round's top 5% is not two points; the `*_n_pairs`
    counts say when a stratum is too small to trust.

    `max_points` caps n before the O(n^2) distance computation.
    """
    keep = _subsample(emb.shape[0], max_points, emb.device)
    if keep is not None:
        emb = emb[keep]
        if y is not None:
            y = y.reshape(-1)[keep]

    stats = {}
    d = torch.pdist(emb)
    if d.numel() == 0:
        return stats

    p10, median, p90 = _quantiles(d, [0.1, 0.5, 0.9])
    stats[f"{prefix}/dist_median"] = median
    stats[f"{prefix}/dist_mean"] = d.mean().item()
    stats[f"{prefix}/dist_p10"] = p10
    stats[f"{prefix}/dist_p90"] = p90
    stats[f"{prefix}/dist_n_points"] = emb.shape[0]
    del d

    if y is None:
        return stats

    y = y.reshape(-1)
    n = y.numel()
    order = torch.argsort(y, descending=True)
    top = emb[order[: min(max(min_group, int(top_q * n)), n)]]
    bot = emb[order[-min(max(min_group, int(bot_q * n)), n) :]]

    groups = {
        "top": torch.pdist(top),
        "bot": torch.pdist(bot),
        "cross": torch.cdist(top, bot).reshape(-1),
    }
    for name, dist in groups.items():
        stats[f"{prefix}/dist_{name}_n_pairs"] = dist.numel()
        if dist.numel():
            stats[f"{prefix}/dist_{name}_median"] = _quantiles(dist, [0.5])[0]
            stats[f"{prefix}/dist_{name}_mean"] = dist.mean().item()
    return stats


def kernel_stats(K, n_eig=20, eig=True):
    """Off-diagonal stats and the spectrum of a kernel matrix.

    `eig=False` skips the eigendecomposition (the only part that is not free at
    large n). Individual eigenvalues are returned as `eig_top/NN` keys; the
    caller drops them on steps where per-step vectors are not wanted.
    """
    n = K.shape[-1]
    stats = {}
    if n > 1:
        off = K[~torch.eye(n, dtype=torch.bool, device=K.device)]
        stats["median"] = off.median().item()
        stats["mean"] = off.mean().item()
        stats["min"] = off.min().item()
        stats["max"] = off.max().item()
    stats["trace"] = torch.diagonal(K, dim1=-2, dim2=-1).sum().item()

    if eig:
        evals = torch.linalg.eigvalsh(K.to(torch.float64)).flip(-1)
        top = evals[:n_eig]
        stats["eig_top1"] = top[0].item()
        stats["eig_topk_sum"] = top.sum().item()
        # Spectral entropy of the normalized spectrum -> effective rank. A
        # kernel collapsed to "everything correlates" puts all its mass on one
        # eigenvalue, so eff_rank -> 1.
        p = evals.clamp_min(0)
        total = p.sum()
        if total > 0:
            p = p / total
            entropy = -(p * p.clamp_min(1e-12).log()).sum()
            stats["eff_rank"] = entropy.exp().item()
        for i, v in enumerate(top.tolist()):
            stats[f"eig_top/{i:02d}"] = v
    return stats


def lora_norms(finetuning_model, per_layer=False):
    """Frobenius norms of the LoRA A and B matrices.

    Returns aggregates over the adapted layers; `per_layer=True` adds one key
    per adapted matrix. Empty dict when the featurizer has no adapters
    (``trainable: false`` arms, or a finetuner frozen by
    ``finetune_start_iter``).
    """
    a_norms, b_norms, layers = [], [], {}
    for name, param in finetuning_model.named_parameters():
        if "lora_A" in name:
            bucket, tag = a_norms, "A"
        elif "lora_B" in name:
            bucket, tag = b_norms, "B"
        else:
            continue
        norm = param.detach().float().norm().item()
        bucket.append(norm)
        if per_layer:
            key = name.split(".lora_")[0].replace(".", "/")
            layers[f"lora/layer/{key}/{tag}"] = norm

    if not a_norms and not b_norms:
        return {}

    stats = {}
    for tag, norms in (("A", a_norms), ("B", b_norms)):
        if norms:
            t = torch.tensor(norms)
            stats[f"lora/{tag}_fro_total"] = t.norm().item()
            stats[f"lora/{tag}_fro_mean"] = t.mean().item()
            stats[f"lora/{tag}_fro_max"] = t.max().item()
    stats["lora/n_adapted_layers"] = len(a_norms)
    stats.update(layers)
    return stats


def singular_values(emb, k=20, prefix="embed"):
    """Top-`k` singular values of the centered embedding matrix, plus the
    variance they explain. Distinguishes "variation concentrating into fewer
    directions" from "everything shrinking uniformly". Uses a randomized SVD:
    an exact one on a 3000x1280 float64 matrix costs ~1-2s, this costs ~50ms.
    """
    x = emb.to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    total = (x ** 2).sum()
    k = min(k, *x.shape)
    _, sv, _ = torch.pca_lowrank(x, q=min(k + 10, *x.shape), center=False)
    top = sv[:k]

    stats = {
        f"{prefix}/sv_top1": top[0].item(),
        f"{prefix}/sv_topk_sum": top.sum().item(),
        f"{prefix}/sv_total_var": total.item(),
    }
    if total > 0:
        stats[f"{prefix}/sv_topk_var_ratio"] = ((top ** 2).sum() / total).item()
        p = (top ** 2) / (top ** 2).sum()
        entropy = -(p * p.clamp_min(1e-12).log()).sum()
        stats[f"{prefix}/sv_eff_rank"] = entropy.exp().item()
    for i, v in enumerate(top.tolist()):
        stats[f"{prefix}/sv_top/{i:02d}"] = v
    return stats


def gp_diagnostics(model, emb, n_eig=20, eig=True, full=False):
    """Per-fit-step diagnostics as a flat wandb-loggable dict.

    `emb` is the kernel input the forward pass already computed. Embedding
    geometry is deliberately absent: those are logged per BO round over the
    whole dataset by `dataset_diagnostics`, not over the acquired points.

    `full=True` adds the per-eigenvalue and per-layer keys and is meant for the
    first and last step of a fit, not every step.
    """
    log = {}
    base = getattr(model.covar_module, "base_kernel", model.covar_module)
    lengthscale = base.lengthscale.detach()
    log["kernel/lengthscale_mean"] = lengthscale.mean().item()
    if lengthscale.numel() > 1:
        log["kernel/lengthscale_min"] = lengthscale.min().item()
        log["kernel/lengthscale_max"] = lengthscale.max().item()
    if hasattr(model.covar_module, "outputscale"):
        log["kernel/outputscale"] = model.covar_module.outputscale.detach().mean().item()
    log["kernel/noise"] = model.likelihood.noise.detach().mean().item()

    # scale_to_bounds re-fits its affine map on every train-mode call, so
    # absolute distances drift for reasons that are not geometry; logging the
    # coefficient keeps that drift separable from a real change.
    coef = getattr(getattr(model, "scale_to_bounds", None), "coefficient", None)
    if isinstance(coef, torch.Tensor):
        log["embed/scale_coefficient"] = coef.detach().mean().item()

    with torch.no_grad():
        # K over the train points -- the GP's own covariance matrix only exists
        # there. An n^2*d matmul on detached embeddings, negligible next to the
        # LLM step, and it keeps the fit's autograd graph out of it.
        corr = base(emb.detach()).to_dense()
        outputscale = log.get("kernel/outputscale", 1.0)
        for tag, mat in (("K", corr * outputscale), ("corr", corr)):
            for k, v in kernel_stats(mat, n_eig=n_eig, eig=eig).items():
                if k.startswith("eig_top/") and not full:
                    continue
                log[f"kernel/{tag}_{k}"] = v

    log.update(lora_norms(model.finetuning_model, per_layer=full))
    return log


def dataset_diagnostics(model, x, y, batch_size=64, max_points=3000, n_sv=20,
                        prefix="embed"):
    """Per-BO-round embedding geometry over the whole dataset.

    Costs one eval-mode forward over `x` (all rows, not the acquired set), then
    distance stats, top-`n_sv` singular values, and the lengthscale-to-median-
    distance saturation ratio. Restores the model's previous train/eval mode so
    it is safe to call between fits.

    NOTE: does not use `model.predict`, which permanently freezes the
    finetuner's parameters.
    """
    was_training = model.finetuning_model.training
    model.finetuning_model.eval()
    try:
        with torch.no_grad():
            chunks = [
                model.embed_eval(x[i : i + batch_size]).detach()
                for i in range(0, x.shape[0], batch_size)
            ]
        emb = torch.cat(chunks, dim=0)
    finally:
        model.finetuning_model.train(was_training)

    log = distance_stats(emb, y, max_points=max_points, prefix=prefix)
    log.update(singular_values(emb, k=n_sv, prefix=prefix))

    base = getattr(model.covar_module, "base_kernel", model.covar_module)
    lengthscale = base.lengthscale.detach().mean().item()
    median = log.get(f"{prefix}/dist_median")
    if median:
        # The saturation number: lengthscale in units of the typical
        # inter-point distance. Above ~5-10 the kernel is flat everywhere and
        # the GP cannot discriminate.
        log["kernel/lengthscale_to_median_dist"] = lengthscale / median
    return log
