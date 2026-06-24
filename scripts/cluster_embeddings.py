"""Standalone: embed an FLIP2 split with ESM2, project to 2D, color by split set.

Usage:
    python cluster_embeddings.py \
        --train data/flip2/alpha-amylase/far_to_close_train.csv \
        --test  data/flip2/alpha-amylase/far_to_close_test.csv \
        --model facebook/esm2_t33_650M_UR50D \
        --out plots/aa_far_to_close_esm2.png

Reads {sequence,target} CSVs, mean-pools per-residue ESM2 embeddings,
PCA->t-SNE to 2D, KMeans clusters, and writes a 2-panel scatter.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def embed(seqs, model_name, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state  # (B, L, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # mean-pool over real residues (drop padding); keeps special tokens but mask handles pad
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.cpu().float().numpy())
        print(f"  embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", end="\r")
    print()
    return np.concatenate(out, axis=0)


def peek(train_csv, test_csv, model_name):
    """Embed one train seq + one test seq and print the vectors / stats."""
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    pair = [tr["sequence"].iloc[0], te["sequence"].iloc[0]]
    X = embed(pair, model_name, batch_size=2)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    for name, vec, seq in zip(["TRAIN[0]", "TEST[0]"], X, pair):
        print(f"\n=== {name}  (len={len(seq)}, dim={vec.shape[0]}) ===")
        print(f"  seq[:40]   : {seq[:40]}")
        print(f"  mean={vec.mean():.4f}  std={vec.std():.4f}  min={vec.min():.4f}  max={vec.max():.4f}  L2={np.linalg.norm(vec):.4f}")
        print(f"  first 20   : {vec[:20]}")
    cos = float(X[0] @ X[1] / (np.linalg.norm(X[0]) * np.linalg.norm(X[1])))
    print(f"\ncosine(train[0], test[0]) = {cos:.4f}   L2 dist = {np.linalg.norm(X[0] - X[1]):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/flip2/alpha-amylase/far_to_close_train.csv")
    p.add_argument("--test", default="data/flip2/alpha-amylase/far_to_close_test.csv")
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--out", default="plots/aa_far_to_close_esm2.png")
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cache", default=None, help="optional .npz to cache embeddings")
    p.add_argument("--peek", action="store_true", help="just print embeddings for one train + one test seq, then exit")
    args = p.parse_args()

    if args.peek:
        peek(args.train, args.test, args.model)
        return

    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)
    seqs = tr["sequence"].tolist() + te["sequence"].tolist()
    split = np.array(["train"] * len(tr) + ["test"] * len(te))
    print(f"{len(tr)} train + {len(te)} test = {len(seqs)} sequences")

    if args.cache and os.path.exists(args.cache):
        print(f"loading cached embeddings from {args.cache}")
        X = np.load(args.cache)["X"]
    else:
        print(f"embedding with {args.model} ...")
        X = embed(seqs, args.model, batch_size=args.batch_size)
        if args.cache:
            np.savez_compressed(args.cache, X=X)

    Xs = StandardScaler().fit_transform(X)
    n_pca = min(50, Xs.shape[1], Xs.shape[0] - 1)
    Xp = PCA(n_components=n_pca, random_state=0).fit_transform(Xs)
    print(f"PCA -> {n_pca} dims, t-SNE -> 2D ...")
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Xp)

    clusters = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=0).fit_predict(Xp)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel 1: colored by split set
    for name, color in [("train", "#1f77b4"), ("test", "#d62728")]:
        m = split == name
        ax1.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, c=color, label=f"{name} (n={m.sum()})")
    ax1.set_title("ESM2 embedding space — colored by split")
    ax1.legend(markerscale=2)

    # Panel 2: colored by KMeans cluster, marker by split
    for name, marker in [("train", "o"), ("test", "x")]:
        m = split == name
        ax2.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6, c=clusters[m], cmap="tab10",
                    marker=marker, label=name, vmin=0, vmax=9)
    ax2.set_title(f"KMeans (k={args.n_clusters}) — o=train  x=test")
    ax2.legend(markerscale=1.5)

    for ax in (ax1, ax2):
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    fig.suptitle(os.path.basename(args.train).replace("_train.csv", ""), fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
