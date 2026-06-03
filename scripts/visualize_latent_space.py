#!/usr/bin/env python3
"""
UMAP visualization of ESM2 latent space before and after LoRA finetuning.

Produces a 3-panel figure:
  Panel 1 – UMAP of original ESM2 1280-dim embeddings (no structure expected)
  Panel 2 – UMAP of finetuned 64-dim projected embeddings (GP feature space)
  Panel 3 – Per-sequence L2 embedding drift (||ft_emb - orig_emb||)
             to show which fitness regions were moved most by finetuning

Usage (defaults to alpha-amylase/one_to_many):
    python visualize_latent_space.py

Custom split / model dir:
    python visualize_latent_space.py \
        --model_dir /iopsstor/scratch/cscs/ssaumya/gollum_models/esm2_t33_650M_UR50D_seed1 \
        --train_csv data/flip2/alpha-amylase/one_to_many_train.csv \
        --test_csv  data/flip2/alpha-amylase/one_to_many_test.csv \
        --out       plots/latent_space_esm2/umap_latent.png
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from plot_style import apply as _apply_style
_apply_style()
# Okabe-Ito color-blind safe palette (Wong 2011, Nature Methods)
_CB = ['#0072B2', '#E69F00', '#009E73', '#56B4E9', '#D55E00', '#CC79A7', '#F0E442', '#000000']
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_CB)
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface")

from gollum.featurization.text import get_tokens
from gollum.featurization.deep import LLMFeaturizer

MODEL_NAME   = "facebook/esm2_t33_650M_UR50D"  # "esm2_t33_650M_UR50D"
EMBED_DIM    = 1280  # 1280 for ESM2-650M, 768 for T5-base
PROJ_DIM     = 64
TARGET_RATIO = 0.25


def get_raw_embeddings(featurizer, tokens_np, batch_size=32):
    """1280-dim average-pooled ESM2 output (before projector)."""
    tokens_t = torch.from_numpy(tokens_np).float().cuda()
    featurizer.eval()
    with torch.no_grad():
        emb = featurizer.get_embeddings(tokens_t, batch_size=batch_size)
    return emb.cpu().float().numpy()


def get_projected_embeddings(featurizer, tokens_np, batch_size=32):
    """64-dim GP feature space (after projector)."""
    tokens_t = torch.from_numpy(tokens_np).float().cuda()
    featurizer.eval()
    with torch.no_grad():
        emb = featurizer(tokens_t)
    return emb.cpu().float().numpy()


def scatter_panel(ax, coords_2d, fitness, mask_tr, mask_te, title, vmin, vmax, cmap):
    ax.scatter(
        coords_2d[mask_tr, 0], coords_2d[mask_tr, 1],
        c=fitness[mask_tr], cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.30, s=10, marker="o", rasterized=True,
    )
    ax.scatter(
        coords_2d[mask_te, 0], coords_2d[mask_te, 1],
        c=fitness[mask_te], cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.95, s=60, marker="^", edgecolors="black", linewidths=0.4,
        zorder=5,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               alpha=0.6, markersize=7, label=f"train ({mask_tr.sum()})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markersize=9, markeredgecolor="black", markeredgewidth=0.4,
               label=f"test ({mask_te.sum()})"),
    ]
    ax.legend(handles=handles, framealpha=0.7)


def plot_manuscript_figure(
    orig_2d, iter_projs, all_fitness, mask_tr, mask_te,
    out_path, vmin=None, vmax=None, dpi=300,
):
    """1×5 manuscript figure: original + iter0/3/6/9 UMAP panels with shared colorbar.

    Args:
        orig_2d:      (N, 2) UMAP coords for original ESM2 embeddings.
        iter_projs:   list of (label, umap_2d, ft_emb) from main(); must contain
                      iter_0, iter_3, iter_6, iter_9.
        all_fitness:  (N,) fitness array aligned with orig_2d / umap_2d rows.
        mask_tr:      boolean mask for training sequences.
        mask_te:      boolean mask for test sequences.
        out_path:     output file path (PDF or PNG).
        vmin/vmax:    colorbar range; default to data min/max.
        dpi:          rasterisation DPI.
    """
    target_labels = ["iter_0", "iter_3", "iter_6", "iter_9"]
    label_to_2d   = {label: proj_2d for label, proj_2d, _ in iter_projs}

    selected = []
    for tgt in target_labels:
        if tgt not in label_to_2d:
            print(f"[warn] {tgt} not found in iter_projs; skipping panel.")
            continue
        n = tgt.split("_")[1]
        selected.append((f"BO Iter {n}", label_to_2d[tgt]))

    panels = [("Original", orig_2d)] + selected

    vmin = all_fitness.min() if vmin is None else vmin
    vmax = all_fitness.max() if vmax is None else vmax
    cmap = plt.cm.viridis

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)

    for ax, (title, coords_2d) in zip(axes[0], panels):
        scatter_panel(ax, coords_2d, all_fitness, mask_tr, mask_te,
                      title, vmin, vmax, cmap)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.tight_layout(rect=[0, 0, 0.93, 1.0])
    cbar_ax = fig.add_axes([0.935, 0.08, 0.012, 0.82])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Fitness (target)", fontsize=12)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved manuscript figure → {out_path}")


def _discover_checkpoints(model_dir, iters=None):
    """Return sorted list of (iter_n, label, ckpt_path) found under model_dir.

    Looks for:
      - iter_{n}/finetuned_model.pt  (per-iteration saves from train.py)
      - final/finetuned_model.pt     (final save from train.py)
      - finetuned_model.pt           (legacy flat save)
    """
    results = []
    iter_re = re.compile(r"^iter_(\d+)$")
    if os.path.isdir(model_dir):
        # Numeric sort so iter_10 comes after iter_9, not after iter_1
        entries = sorted(
            [e for e in os.listdir(model_dir) if iter_re.match(e)],
            key=lambda e: int(iter_re.match(e).group(1)),
        )
        for entry in entries:
            ckpt = os.path.join(model_dir, entry, "finetuned_model.pt")
            if os.path.exists(ckpt):
                n = int(iter_re.match(entry).group(1))
                results.append((n, f"iter_{n}", ckpt))

    final_ckpt = os.path.join(model_dir, "final", "finetuned_model.pt")
    if os.path.exists(final_ckpt):
        n = results[-1][0] + 1 if results else 0
        results.append((n, "final", final_ckpt))

    if not results:
        flat_ckpt = os.path.join(model_dir, "finetuned_model.pt")
        if os.path.exists(flat_ckpt):
            results.append((0, "finetuned", flat_ckpt))

    if iters is not None:
        results = [(n, label, p) for n, label, p in results if n in iters]

    return results


def _load_ft_projected_emb(ckpt_path, ft_feat_kwargs, tokens_np, batch_size=32):
    """Load a finetuned LLMFeaturizer checkpoint and return projected embeddings."""
    ft_feat = LLMFeaturizer(**ft_feat_kwargs)
    state_dict = torch.load(ckpt_path, map_location="cuda")
    missing, unexpected = ft_feat.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    ft_feat.projector.to(torch.float64)
    emb = get_projected_embeddings(ft_feat, tokens_np, batch_size)
    del ft_feat
    torch.cuda.empty_cache()
    return emb


def _compute_distance_distributions(embeddings, fitness, low_thr, high_thr):
    """Compute upper-triangle pairwise L2 distances for HH, HL, LL sequence pairs.

    Uses upper triangle only (no self-pairs, no double-counting).
    High  = fitness >= high_thr (Q80 by default)
    Low   = fitness <= low_thr  (Q20 by default)
    """
    emb_t = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    fit_t = torch.from_numpy(np.asarray(fitness,    dtype=np.float32))

    dmat = torch.cdist(emb_t, emb_t, p=2)   # (N, N)
    N    = dmat.shape[0]

    high_mask = fit_t >= high_thr
    low_mask  = fit_t <= low_thr

    # Upper-triangle indices (offset=1 excludes diagonal)
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    d = dmat[idx_i, idx_j]

    hi_i, lo_i = high_mask[idx_i], low_mask[idx_i]
    hi_j, lo_j = high_mask[idx_j], low_mask[idx_j]

    return {
        "hh": d[hi_i & hi_j].numpy(),
        "hl": d[(hi_i & lo_j) | (lo_i & hi_j)].numpy(),
        "ll": d[lo_i & lo_j].numpy(),
    }


def _plot_distance_evolution(iter_dists, out_path, low_q, high_q, dpi=300):
    """KDE plot of pairwise L2 distance distributions across BO iterations.

    3 panels (HH / HL / LL), one KDE curve per iteration coloured by a
    sequential colormap so earlier → later iterations are visually distinct.
    """
    from scipy.stats import gaussian_kde

    n_iters = len(iter_dists)
    iter_colors = plt.cm.plasma(np.linspace(0.1, 0.88, n_iters))

    group_keys   = ["hh",           "hl",           "ll"]
    group_titles = ["High–High",    "High–Low",     "Low–Low"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, key, title in zip(axes, group_keys, group_titles):
        for (label, dists), color in zip(iter_dists, iter_colors):
            vals = dists[key]
            if len(vals) < 10:
                continue
            kde  = gaussian_kde(vals, bw_method="scott")
            xlo, xhi = np.percentile(vals, [1, 99])
            x = np.linspace(xlo, xhi, 400)
            ax.plot(x, kde(x), color=color, lw=1.8, alpha=0.85, label=label)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Pairwise L2 distance", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.grid(True, alpha=0.2)

    # Iteration legend on the first panel only
    axes[0].legend(fontsize=8, framealpha=0.7,
                   title="iteration", title_fontsize=9,
                   loc="upper right")

    fig.suptitle(
        f"Pairwise distance distributions in GP feature space  "
        f"(thresholds: Q{int(low_q*100)} / Q{int(high_q*100)})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default="/iopsstor/scratch/cscs/ssaumya/gollum_models/t5-base_seed1_10iters")
    parser.add_argument("--train_csv", default="data/flip2/alpha-amylase/one_to_many_train.csv")
    parser.add_argument("--test_csv",  default="data/flip2/alpha-amylase/one_to_many_test.csv")
    parser.add_argument("--n_train",   type=int, default=2575)
    parser.add_argument("--n_test",    type=int, default=488)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       default="plots/latent_space_t5/umap_latent.pdf")
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--dpi",        type=int, default=300,
                        help="DPI for rasterized elements (relevant for PDF/PNG output)")
    parser.add_argument("--iters",      type=str, default=None,
                        help="Comma-separated iter indices to visualise (e.g. '0,2,4'); "
                             "default: all detected checkpoints")
    parser.add_argument("--split",      type=str, default="both",
                        choices=["train", "test", "both"],
                        help="Which sequences to show in scatter plots and use for "
                             "distance distributions (default: both)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load sequences + fitness ─────────────────────────────────────────────
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    n_train = min(args.n_train, len(train_df))
    idx_tr  = rng.choice(len(train_df), size=n_train, replace=False)

    train_seqs    = train_df["sequence"].iloc[idx_tr].tolist()
    train_fitness = train_df["target"].iloc[idx_tr].values.astype(float)
    test_seqs     = test_df["sequence"].iloc[:args.n_test].tolist()
    test_fitness  = test_df["target"].iloc[:args.n_test].values.astype(float)

    all_seqs    = train_seqs + test_seqs
    all_fitness = np.concatenate([train_fitness, test_fitness])
    labels      = np.array(["train"] * len(train_seqs) + ["test"] * len(test_seqs))
    mask_tr     = labels == "train"
    mask_te     = labels == "test"

    if args.split == "train":
        mask_te = np.zeros_like(mask_te)
    elif args.split == "test":
        mask_tr = np.zeros_like(mask_tr)

    print(f"Sequences — train: {len(train_seqs)}, test: {len(test_seqs)}  "
          f"(showing: {args.split})")

    # ── Tokenize once ────────────────────────────────────────────────────────
    print("Tokenizing…")
    tokens_np = get_tokens(all_seqs, model_name=MODEL_NAME)

    # ── Original T5 → 1280-dim embeddings ─────────────────────────────────
    print("Loading original T5 (no LoRA)…")
    orig_feat = LLMFeaturizer(
        model_name=MODEL_NAME, input_dim=EMBED_DIM,
        projection_dim=None, pooling_method="average", trainable=False,
    )
    print("Computing original 1280-dim embeddings…")
    orig_emb = get_raw_embeddings(orig_feat, tokens_np, args.batch_size)
    del orig_feat
    torch.cuda.empty_cache()

    # ── Discover & load finetuned checkpoints ────────────────────────────────
    ft_feat_kwargs = dict(
        model_name=MODEL_NAME, input_dim=EMBED_DIM,
        projection_dim=PROJ_DIM, pooling_method="average",
        trainable=True, target_ratio=TARGET_RATIO, from_top=True,
    )
    iters_filter = [int(x) for x in args.iters.split(",")] if args.iters else None
    iter_ckpts = _discover_checkpoints(args.model_dir, iters_filter)
    if not iter_ckpts:
        print(f"No checkpoints found under {args.model_dir}; exiting.")
        return

    # ── PC1 fitness correlation on original embeddings ───────────────────────
    # from sklearn.decomposition import PCA
    # from scipy.stats import spearmanr

    # pca_orig = PCA(n_components=1).fit(orig_emb)
    # pc1_orig = pca_orig.transform(orig_emb).ravel()
    # rho_orig_tr, _ = spearmanr(pc1_orig[mask_tr], all_fitness[mask_tr])
    # rho_orig_te, _ = spearmanr(pc1_orig[mask_te], all_fitness[mask_te])
    # print(f"PC1-fitness Spearman — original: train={rho_orig_tr:.3f} test={rho_orig_te:.3f}")

    # ── UMAP on original embeddings ──────────────────────────────────────────
    umap_kw  = dict(n_neighbors=30, min_dist=0.1, random_state=args.seed, metric="cosine")
    vmin, vmax = all_fitness.min(), all_fitness.max()
    cmap       = plt.cm.viridis
    out_dir    = os.path.dirname(args.out) or "."

    print("UMAP on original 1280-dim embeddings…")
    orig_2d = umap.UMAP(n_components=2, **umap_kw).fit_transform(orig_emb)
    del orig_emb

    # ── Per-iteration finetuned embeddings + UMAP (independent per iter) ─────
    iter_projs = []  # (label, umap_2d, ft_emb)
    for _n, label, ckpt_path in iter_ckpts:
        print(f"Loading {label} checkpoint from {ckpt_path}…")
        ft_emb = _load_ft_projected_emb(ckpt_path, ft_feat_kwargs, tokens_np, args.batch_size)
        print(f"UMAP on {label} (64-dim projected)…")
        ft_2d = umap.UMAP(n_components=2, **umap_kw).fit_transform(ft_emb)
        iter_projs.append((label, ft_2d, ft_emb))

        # ── Individual file for this iteration ────────────────────────────────
        fig_i, ax_i = plt.subplots(figsize=(7, 6))
        scatter_panel(ax_i, ft_2d, all_fitness, mask_tr, mask_te,
                      f"Finetuned — {label}\n(64-dim GP feature space)",
                      vmin, vmax, cmap)
        sm_i = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm_i.set_array([])
        fig_i.colorbar(sm_i, ax=ax_i).set_label("Fitness (target)", fontsize=11)
        fig_i.tight_layout()
        indiv_path = os.path.join(out_dir, f"{label}_umap.pdf")
        fig_i.savefig(indiv_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  Saved → {indiv_path}")

    # ---- Manuscript figure with selected iterations (e.g. iter_0, iter_3, iter_6, iter_9) ──
    # manuscript_path = os.path.join(out_dir, "manuscript_figure.pdf")
    # plot_manuscript_figure(
    #     orig_2d, iter_projs, all_fitness, mask_tr, mask_te,
    #     manuscript_path, vmin, vmax, args.dpi,
    # )
    # sys.exit(0)

    # ── Plot combined figure ──────────────────────────────────────────────────

    n_panels = 1 + len(iter_projs)
    ncols = min(n_panels, 4)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5.5 * nrows),
                              squeeze=False)
    axes_flat = axes.ravel()

    scatter_panel(axes_flat[0], orig_2d, all_fitness, mask_tr, mask_te,
                  "Original T5\n(1280-dim, no finetuning)", vmin, vmax, cmap)

    for idx, (label, proj_2d, _) in enumerate(iter_projs, start=1):
        scatter_panel(axes_flat[idx], proj_2d, all_fitness, mask_tr, mask_te,
                      f"Finetuned — {label}\n(64-dim GP feature space)",
                      vmin, vmax, cmap)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.suptitle("T5 Latent Space Evolution", fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.06, 0.018, 0.86])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Fitness (target)", fontsize=13)

    out_stem, out_ext = os.path.splitext(args.out)
    out_path = args.out if len(iter_projs) <= 1 else f"{out_stem}_evolution{out_ext}"
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # ── Pairwise distance evolution ───────────────────────────────────────────
    # Thresholds: bottom Q20 = "low fitness", top Q20 (≥Q80) = "high fitness".
    # Equal-sized groups (20 % each) give a balanced number of HH / HL / LL pairs
    # while clearly separating the extremes of the fitness distribution.
    _LOW_Q, _HIGH_Q = 0.2, 0.8
    low_thr  = float(np.quantile(all_fitness, _LOW_Q))
    high_thr = float(np.quantile(all_fitness, _HIGH_Q))
    n_high = int((all_fitness >= high_thr).sum())
    n_low  = int((all_fitness <= low_thr).sum())
    print(f"\nDistance thresholds — low ≤ {low_thr:.4f} (Q{int(_LOW_Q*100)}, n={n_low})  "
          f"high ≥ {high_thr:.4f} (Q{int(_HIGH_Q*100)}, n={n_high})")

    active = mask_tr | mask_te  # respects --split
    iter_dists = []
    for label, _ft_2d, ft_emb in iter_projs:
        dists = _compute_distance_distributions(
            ft_emb[active], all_fitness[active], low_thr, high_thr,
        )
        print(f"  {label}: hh={len(dists['hh'])} hl={len(dists['hl'])} ll={len(dists['ll'])} pairs")
        iter_dists.append((label, dists))

    dist_path = f"{out_stem}_distances{out_ext}"
    _plot_distance_evolution(iter_dists, dist_path, _LOW_Q, _HIGH_Q, dpi=args.dpi)


if __name__ == "__main__":
    main()
