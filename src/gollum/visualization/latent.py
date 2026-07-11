"""Latent diagnostics for DeepGP finetuning (opt-in via train.py flags).

Once per BO epoch, embeds the whole train+heldout pool with that epoch's just-fit
finetuning model. At the end of the run it plots the latent-space evolution (one
UMAP per epoch) and the d_hh/d_ll/d_hl fitness-group distances, and can save the
finetuned model of each epoch.
"""
import os

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from umap import UMAP


def pairwise_distances(emb, y):
    """Raw pairwise latent distances between fitness groups split at the 80/20
    target quantiles: within-high (hh), within-low (ll), cross (hl)."""
    hi = emb[y >= torch.quantile(y, 0.8)]
    lo = emb[y <= torch.quantile(y, 0.2)]
    hl = torch.cdist(hi, lo).flatten() if hi.numel() and lo.numel() else hi.new_empty(0)
    return {"hh": torch.pdist(hi), "ll": torch.pdist(lo), "hl": hl}


def group_distances(emb, y):
    """Mean of each pairwise-distance group (d_hh, d_ll, d_hl)."""
    return {f"d_{k}": (v.mean().item() if v.numel() else float("nan"))
            for k, v in pairwise_distances(emb, y).items()}


class LatentDiagnostics:
    def __init__(self, out_dir, viz_x, viz_y, save_models, viz_latent, plot_dist):
        self.dir = out_dir
        self.viz_x = viz_x
        self.viz_y = viz_y.squeeze().cpu()
        self.save_models = save_models
        self.viz_latent = viz_latent
        self.plot_dist = plot_dist
        self.records = []

    def record(self, model, epoch):
        if self.save_models:
            os.makedirs(self.dir, exist_ok=True)
            torch.save(model.finetuning_model.state_dict(),
                       os.path.join(self.dir, f"epoch{epoch}.pt"))
        if not (self.viz_latent or self.plot_dist):
            return
        model.finetuning_model.eval()
        with torch.no_grad():
            emb = model.embed_eval(self.viz_x).detach().cpu()
        pw = pairwise_distances(emb, self.viz_y)
        dists = {f"d_{k}": (v.mean().item() if v.numel() else float("nan"))
                 for k, v in pw.items()}
        wandb.log({**{f"latent/{k}": v for k, v in dists.items()}, "epoch": epoch})
        rec = {"epoch": epoch, **dists}
        if self.viz_latent:
            rec["emb"] = emb
        self.records.append(rec)
        if self.plot_dist:
            os.makedirs(self.dir, exist_ok=True)
            # Raw pairwise distances for exact offline re-plotting (one npz/epoch).
            # ponytail: fine up to ~20k-point pools; beyond that hl (|hi|*|lo|)
            # balloons — subsample there.
            np.savez_compressed(
                os.path.join(self.dir, f"dist_epoch{epoch}.npz"),
                **{k: v.to(torch.float32).numpy() for k, v in pw.items()},
            )

    def finalize(self):
        if not self.records:
            return
        os.makedirs(self.dir, exist_ok=True)
        if self.plot_dist:
            self.plot_distances()
        if self.viz_latent:
            self.plot_evolution()

    def plot_distances(self):
        epochs = [r["epoch"] for r in self.records]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, k, title in zip(axes, ["d_hh", "d_ll", "d_hl"],
                                ["d_hh (high–high)", "d_ll (low–low)", "d_hl (high–low)"]):
            ax.plot(epochs, [r[k] for r in self.records], marker="o")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel("mean latent distance")
        fig.suptitle("Latent group distances")
        fig.tight_layout()
        self.save_fig(fig, "distances")

    def plot_evolution(self):
        y = self.viz_y.numpy()
        n = len(self.records)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        flat = axes.ravel()
        for ax_i, rec in enumerate(self.records):
            xy = UMAP(n_components=2, random_state=42).fit_transform(rec["emb"].numpy())
            sc = flat[ax_i].scatter(xy[:, 0], xy[:, 1], c=y, cmap="viridis", s=8, alpha=0.7)
            flat[ax_i].set_title(f"epoch {rec['epoch']}")
            flat[ax_i].set_xticks([])
            flat[ax_i].set_yticks([])
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="target", shrink=0.6)
        fig.suptitle("Latent space evolution (UMAP)")
        self.save_fig(fig, "latent_evolution")

    def save_fig(self, fig, name):
        if wandb.run is not None:
            wandb.log({f"latent/{name}": wandb.Image(fig)})
        fig.savefig(os.path.join(self.dir, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
