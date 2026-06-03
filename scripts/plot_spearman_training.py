"""
Plot Spearman rho vs epoch from a WandB CSV export with SEM error bars.

SEM is estimated from the per-group min/max using the expected range of a
normal distribution for N_SEEDS seeds: σ ≈ (max−min) / d2(n), SEM = σ/√n,
where d2(5) ≈ 2.326.

Usage:
    python scripts/plot_spearman_training.py \
        --csv wandb_export_2026-04-29T03_36_49.603+02_00.csv \
        --out figures/spearman_training.pdf
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# d2 constant (expected range / σ) for n=5 seeds; used to recover σ from range
_N_SEEDS = 5
_D2 = 2.326  # d2(5) from statistical process control tables

from plot_style import apply as _apply_style
_apply_style()

# Tableau 10 — standard in ICML/NeurIPS/ICLR figures
_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

# Map raw column prefix → display label
_LABELS = {
    "facebook/esm2_t33_650M_UR50D": "ESM2-650M",
    "t5-base": "T5-base",
}


def _col(model_key: str, suffix: str) -> str:
    return f"data.init_args.featurizer.init_args.model_name: {model_key} - {suffix}"


def main(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))  # single-column ICML width

    for color, (model_key, label) in zip(_COLORS, _LABELS.items()):
        mean   = df[_col(model_key, "test/spearman_rho")].values
        lo     = df[_col(model_key, "test/spearman_rho__MIN")].values
        hi     = df[_col(model_key, "test/spearman_rho__MAX")].values
        epochs = df["epoch"].values

        # σ ≈ range / d2(n), SEM = σ / √n
        sem = (hi - lo) / (_D2 * math.sqrt(_N_SEEDS))

        ax.errorbar(
            epochs, mean,
            yerr=sem,
            color=color,
            linewidth=1.2,
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.8,
            label=label,
        )

    ax.set_title("Spearman $\\rho$ on test split v/s BO epochs", pad=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Spearman $\rho$ (test)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="wandb_export_2026-04-29T03_36_49.603+02_00.csv")
    parser.add_argument("--out", default="figures/spearman_training.pdf")
    args = parser.parse_args()
    main(args.csv, args.out)
