"""Plot fitness (target) distribution for each FLIP2 landscape."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATASETS = {
    "alpha-amylase": "data/flip2/alpha-amylase/one_to_many.csv",
    "ired":          "data/flip2/ired/two_to_many.csv",
    "nucB":          "data/flip2/nucB/two_to_many.csv",
    "trpB":          "data/flip2/trpB/by_position.csv",
}
outdir = Path("plots/fitness_distributions"); outdir.mkdir(parents=True, exist_ok=True)

def draw(ax, y, title):
    ax.hist(y, bins=80, color="steelblue", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("fitness (target)"); ax.set_ylabel("count")
    ax.axvline(y.mean(), color="crimson", lw=1, ls="--", label=f"mean={y.mean():.2g}")
    ax.axvline(y.max(), color="darkgreen", lw=1, ls="-", label=f"max={y.max():.2g}")
    ax.legend(fontsize=8)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (name, path) in zip(axes.ravel(), DATASETS.items()):
    y = pd.read_csv(path, usecols=["target"])["target"].dropna()
    draw(ax, y, f"{name}  (n={len(y):,})")
    # also save individual
    ig, ia = plt.subplots(figsize=(6, 4))
    draw(ia, y, f"{name} fitness distribution (n={len(y):,})")
    ig.tight_layout(); ig.savefig(outdir / f"{name}.pdf"); plt.close(ig)

fig.tight_layout()
fig.savefig(outdir / "all_landscapes.pdf")
print("wrote plots to", outdir)
