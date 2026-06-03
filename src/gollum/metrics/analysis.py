import gc
import io
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from pytorch_metric_learning import distances as dist
from gollum.surrogate_models.gp import SurrogateModel


def compute_cumulative_metrics(
    group: pd.DataFrame, theoretical_max: float = 100.0
) -> Dict[str, float]:
    """
    Compute cumulative metrics for a group of runs.
    """
    metrics = {}

    # Compute area under the best-so-far curve (higher is better)
    metrics["cumulative_best"] = np.trapz(group["train/best_so_far"])

    max_value = group["train/best_so_far"].max()
    theoretical_max_area = max_value * len(group["train/best_so_far"])
    metrics["normalized_cumulative_best"] = (
        metrics["cumulative_best"] / theoretical_max_area
    )

    # Compute cumulative counts (higher is better)
    bo_metrics_final_epoch = group[group["epoch"] == group["epoch"].max()]
    for count_col in [
        "quantile_99_count",
        "quantile_95_count",
        "quantile_90_count",
        "quantile_75_count",
    ]:
        if count_col in group.columns:
            metrics[f"cumulative_{count_col}"] = group[count_col].sum()
            metrics[f"final_{count_col}"] = bo_metrics_final_epoch[count_col].mean()

    # Compute mean final metrics
    model_metrics_final_epoch = group[group["epoch"] == group["epoch"].max() - 1]
    for metric in [
        "train/nlpd",
        "train/mse",
        "train/r2",
        "train/msll",
        "train/qce",
        "valid/nlpd",
        "valid/mse",
        "valid/r2",
        "valid/msll",
        "valid/qce",
        "covar_module.base_kernel.lengthscale",
        "covar_module.outputscale",
        "likelihood.noise_covar.noise",
    ]:
        metrics[f"final_{metric}"] = model_metrics_final_epoch[metric].mean()

    return metrics


def analyze_results(
    df: pd.DataFrame, param_cols=None, sort_by_column="cumulative_best"
) -> pd.DataFrame:
    """
    Analyze sweep results and return a summary dataframe sorted by performance.
    """

    # Compute metrics for each parameter configuration
    results = []
    for params, group in df.groupby(param_cols, dropna=False):
        if len(param_cols) == 1:
            param_dict = {param_cols[0]: params}
        else:
            param_dict = dict(zip(param_cols, params))

        # Compute metrics across all seeds
        seed_metrics = []
        for seed, seed_group in group.groupby("seed"):
            metrics = compute_cumulative_metrics(
                seed_group, theoretical_max=seed_group["summary_target_stat_max"].max()
            )
            seed_metrics.append(metrics)

        # Average metrics across seeds
        avg_metrics = {
            k: np.mean([m[k] for m in seed_metrics]) for k in seed_metrics[0].keys()
        }
        std_metrics = {
            f"{k}_std": np.std([m[k] for m in seed_metrics])
            for k in seed_metrics[0].keys()
        }
        # TODO save only one
        count_metrics = {
            f"{k}_cnt": len([m[k] for m in seed_metrics])
            for k in seed_metrics[0].keys()
        }

        results.append({**param_dict, **avg_metrics, **std_metrics, **count_metrics})

    results_df = pd.DataFrame(results)

    # Sort by cumulative best-so-far (can be changed to other metrics)
    results_df = results_df.sort_values(sort_by_column, ascending=False)

    return results_df


def compute_thresholds(y, low_quantile=0.2, high_quantile=0.8):
    """
    Computes thresholds using 5th and 95th percentiles.

    Args:
        y: Objective values
        low_quantile: Quantile for low threshold (default: 0.05 for 5th percentile)
        high_quantile: Quantile for high threshold (default: 0.95 for 95th percentile)
    """
    if torch.is_tensor(y):
        low_threshold = torch.quantile(y, low_quantile)
        high_threshold = torch.quantile(y, high_quantile)
    else:
        low_threshold = np.quantile(y, low_quantile)
        high_threshold = np.quantile(y, high_quantile)

    return low_threshold, high_threshold


def calculate_distances(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    high_score_threshold: float = 70,
    low_score_threshold: float = 10,
    model: Optional[SurrogateModel] = None,
):
    """
    Calculates distances between points in latent space based on their score categories.

    Args:
        embeddings: Embeddings in latent space (torch.Tensor)
        scores: Corresponding scores (torch.Tensor)
        high_score_threshold: Threshold for high scores (default: 70)
        low_score_threshold: Threshold for low scores (default: 10)
        model: Optional surrogate model for kernel similarity

    Returns:
        Tuple of (high-high distances, high-low distances, low-low distances, average distance)
    """
    # convert numpy
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)

    # kernel similrity
    if model:
        with torch.no_grad():
            distances = model.covar_module(
                embeddings.to(model.covar_module.device)
            ).evaluate()
    else:
        # Calculate pairwise L2 distances
        distances = torch.cdist(embeddings, embeddings, p=2)

    # Get masks for high and low scores
    high_mask = scores >= high_score_threshold
    low_mask = scores < low_score_threshold

    # Calculate distances for each category
    hh_distances = distances[high_mask][:, high_mask].flatten()
    hl_distances = distances[high_mask][:, low_mask].flatten()
    ll_distances = distances[low_mask][:, low_mask].flatten()
    avg_distance = distances.mean()

    return hh_distances, hl_distances, ll_distances, avg_distance


# ── Kernel covariance heatmap helpers ─────────────────────────────────────────

def _subsample_kernel(x: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    N = len(x)
    if N <= n:
        return x
    gen = torch.Generator()
    gen.manual_seed(seed)
    idx, _ = torch.randperm(N, generator=gen)[:n].sort()
    return x[idx]


def _stride_select(x: torch.Tensor, n: int) -> torch.Tensor:
    """Select n evenly-spaced rows from x using a fixed stride."""
    N = len(x)
    if N <= n:
        return x
    idx = torch.linspace(0, N - 1, n).long()
    return x[idx]


def _get_kernel_embeddings(model, x: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Extract features: projects through finetuning_model (DeepGP) or returns x (GP)."""
    device = next(model.covar_module.parameters()).device
    is_deep_gp = hasattr(model, "finetuning_model")
    parts = []
    with torch.no_grad():
        for chunk in x.split(chunk_size):
            chunk_dev = chunk.to(device=device, dtype=torch.float64)
            if is_deep_gp:
                emb = model.finetuning_model(chunk_dev)
                if model.scale_embeddings:
                    emb = model.scale_to_bounds(emb)
            else:
                emb = chunk_dev
            parts.append(emb.cpu())
            del chunk_dev, emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0).to(torch.float64)


def _compute_cross_kernel_matrix(
    model,
    emb_rows: torch.Tensor,
    emb_cols: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    """Row-batched cross-covariance K(emb_rows, emb_cols). Returns (N_rows, N_cols) float32 array."""
    device = next(model.covar_module.parameters()).device
    N_rows = len(emb_rows)
    N_cols = len(emb_cols)
    K = np.empty((N_rows, N_cols), dtype=np.float32)
    cols_gpu = emb_cols.to(device=device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, N_rows, batch_size):
            end = min(start + batch_size, N_rows)
            rows_chunk = emb_rows[start:end].to(device=device, dtype=torch.float64)
            K_row = model.covar_module(rows_chunk, cols_gpu).evaluate()
            K[start:end, :] = K_row.float().cpu().numpy()
            del rows_chunk, K_row
    del cols_gpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return K


def _mean_order(K: np.ndarray, axis: int) -> np.ndarray:
    """Return sort index that places highest-mean rows (axis=1) or cols (axis=0) first."""
    return np.argsort(K.mean(axis=axis))[::-1]


def _render_cross_kernel_panel(ax_heat, ax_hist, K: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    N_rows, N_cols = K.shape

    # Sort rows (train) and cols (test) independently by mean covariance — descending
    row_ord = _mean_order(K, axis=1)
    col_ord = _mean_order(K, axis=0)
    K_ord = K[np.ix_(row_ord, col_ord)]

    k_mean = float(K.mean())
    k_std  = float(K.std())

    # Heatmap
    vmin = float(np.percentile(K, 1))
    vmax = float(np.percentile(K, 99))
    im = ax_heat.imshow(K_ord, aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax, origin="upper")
    ax_heat.set_title(title, fontsize=15, pad=4)
    ax_heat.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label("k(x, x')", fontsize=18)
    cb.ax.tick_params(labelsize=10)
    ax_heat.text(
        0.02, 0.97,
        f"mean={k_mean:.4f}\nstd={k_std:.4f}",
        transform=ax_heat.transAxes, fontsize=18, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    # Distribution of all cross-covariance values
    vals = K.ravel()
    rng = np.random.default_rng(0)
    sample = rng.choice(vals, size=min(50_000, len(vals)), replace=False)
    ax_hist.hist(sample, bins=60, density=True, color="steelblue", alpha=0.85)
    ax_hist.set_xlabel("k(x, x')", fontsize=18)
    ax_hist.set_ylabel("Density", fontsize=18)
    ax_hist.set_title("Cross-covariance distribution", fontsize=18, pad=4)
    ax_hist.tick_params(labelsize=10)


def plot_kernel_heatmap(
    model,
    train_x: torch.Tensor,
    splits: List[Tuple[str, torch.Tensor]],
    n_subsample: int = 256,
    kernel_batch_size: int = 256,
    embed_chunk_size: int = 128,
    save_path: Optional[str] = None,
    wandb_key: str = "final/kernel_heatmap",
) -> None:
    """Compute and log train-vs-split kernel cross-covariance heatmaps to wandb.

    Selects n_subsample points from train and from each split using a fixed stride
    (evenly spaced), then plots K(train_sub, split_sub). A collapsed kernel
    → uniformly bright heatmap, narrow histogram. A finetuned kernel → visible
    contrast, broader distribution.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import wandb

    if wandb.run is None:
        print("[kernel_heatmap] No active wandb run — skipping.")
        return

    model.eval()
    if hasattr(model, "finetuning_model"):
        model.finetuning_model.eval()

    train_sub = _stride_select(train_x, n=n_subsample)
    try:
        train_emb = _get_kernel_embeddings(model, train_sub, chunk_size=embed_chunk_size)
    except Exception as exc:
        print(f"[kernel_heatmap] Failed to embed train sequences: {exc}")
        return

    panels: List[Tuple[np.ndarray, str]] = []
    for split_name, x in splits:
        try:
            x_sub = _stride_select(x, n=n_subsample)
            split_emb = _get_kernel_embeddings(model, x_sub, chunk_size=embed_chunk_size)
            K = _compute_cross_kernel_matrix(model, train_emb, split_emb, batch_size=kernel_batch_size)
            panels.append((K, f"K(train, {split_name})"))
            del split_emb, x_sub
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[kernel_heatmap] Failed on split '{split_name}': {exc}")

    del train_emb, train_sub
    gc.collect()

    if not panels:
        print("[kernel_heatmap] No panels to render — skipping.")
        return

    n_panels = len(panels)
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
    fig = plt.figure(figsize=(6.5 * n_panels + 0.5, 9.0), dpi=150)
    gs = gridspec.GridSpec(
        2, n_panels, hspace=0.35, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )
    # fig.suptitle("GP Kernel Cross-Covariance ", fontsize=10)

    for i, (K, title) in enumerate(panels):
        ax_heat = fig.add_subplot(gs[0, i])
        ax_hist = fig.add_subplot(gs[1, i])
        _render_cross_kernel_panel(ax_heat, ax_hist, K, title)

    from PIL import Image as PILImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    wandb.log({wandb_key: wandb.Image(PILImage.open(buf), caption="GP kernel covariance — final surrogate")})
    buf.close()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"[kernel_heatmap] Saved PDF to {save_path}")

    plt.close(fig)
    print(f"[kernel_heatmap] Logged '{wandb_key}' to wandb.")
