"""KDE of train vs test fitness for each FLIP2 split, on shared axes.

Looking for: test fitness reachable (overlaps train) but shifted enough that a GP
fit on train-like sequences won't trivially generalise. One PDF per split.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SPLITS = {
    "alpha-amylase": ["by_mutation", "close_to_far", "far_to_close", "one_to_many"],
    "ired":          ["two_to_many"],
    "nucB":          ["two_to_many"],
    "trpB":          ["by_position", "one_to_many", "two_to_many"],
}
OUT = "plots/fitness_kde"
C_TRAIN, C_TEST = "#3b6fb0", "#c0203f"  # train blue, test crimson


def kde_curve(vals, grid):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2 or np.std(vals) < 1e-9:
        return None
    return gaussian_kde(vals)(grid)


def plot_split(dataset, split):
    df = pd.read_csv(f"data/flip2/{dataset}/{split}.csv").dropna(subset=["target"])
    tr = df.loc[df["set"] == "train", "target"].values.astype(float)
    te = df.loc[df["set"] == "test", "target"].values.astype(float)
    allv = np.concatenate([tr, te])
    lo, hi = allv.min(), allv.max()
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    for vals, c, label in [(tr, C_TRAIN, "train"), (te, C_TEST, "test")]:
        d = kde_curve(vals, grid)
        if d is None:
            continue
        ax.plot(grid, d, color=c, lw=2, label=f"{label} (n={len(vals)})")
        ax.fill_between(grid, d, color=c, alpha=0.15)
        ax.axvline(np.median(vals), color=c, ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("fitness (target)")
    ax.set_ylabel("density")
    ax.set_title(f"{dataset} / {split}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = f"{OUT}/{dataset}_{split}.pdf"
    fig.savefig(path)
    plt.close(fig)
    # quick numeric read on the overlap/shift pattern
    ks_shift = abs(np.median(tr) - np.median(te))
    print(f"{dataset:14s} {split:16s} train[{tr.min():.2f},{tr.max():.2f}] "
          f"test[{te.min():.2f},{te.max():.2f}] median|shift|={ks_shift:.2f} -> {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for ds, splits in SPLITS.items():
        for sp in splits:
            plot_split(ds, sp)
            n += 1
    print(f"\nSaved {n} PDFs to {OUT}/")


if __name__ == "__main__":
    main()
