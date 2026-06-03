#!/usr/bin/env python3
"""
KDE plot of ALL pairwise (train × test) Euclidean distances for FLIP2 splits,
with optional Matérn-5/2 kernel value overlay.

Every (train_i, test_j) pair contributes one distance — no train×train or
test×test distances are included.

All datasets and feature spaces are overlaid on a single axes.
  Color   → dataset
  Linestyle → feature space (one-hot / static embedding / finetuned embedding)

Feature spaces
--------------
  one-hot   : always computed (raw 20-AA one-hot)
  static    : pretrained backbone embeddings, no finetuning (--static_model_name)
  finetuned : finetuned + projected embeddings (--model_dir)

Kernel value mode
-----------------
Pass --lengthscale_* (and optionally --outputscale_*) to transform the x-axis
from raw Euclidean distances to outputscale × Matérn-5/2(d / ℓ).  This makes
kernel collapse directly visible: if most of the distribution sits near 0,
the GP kernel is collapsed for that feature space.

Usage
-----
# one-hot only (raw distances)
python scripts/plot_pairwise_distances.py

# one-hot + static ESM2 + finetuned embedding (raw distances)
python scripts/plot_pairwise_distances.py \\
    --static_model_name facebook/esm2_t33_650M_UR50D \\
    --static_input_dim 1280 \\
    --model_dir /iopsstor/scratch/cscs/ssaumya/gollum_models/esm2_t33_650M_UR50D_seed1_10iters/ \\
    --model_name facebook/esm2_t33_650M_UR50D \\
    --input_dim 1280 \\
    --out figures/pairwise_distances.pdf

# kernel value mode — supply GP hyperparameters from wandb
python scripts/plot_pairwise_distances.py \\
    --static_model_name facebook/esm2_t33_650M_UR50D \\
    --static_input_dim 1280 \\
    --model_dir /path/to/finetuned/ \\
    --model_name facebook/esm2_t33_650M_UR50D \\
    --input_dim 1280 \\
    --lengthscale_onehot 1.0  --outputscale_onehot 1.0 \\
    --lengthscale_static 0.1  --outputscale_static 0.001 \\
    --lengthscale_finetuned 0.5 --outputscale_finetuned 2.0 \\
    --out figures/kernel_values.pdf
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import apply as _apply_style
_apply_style()
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}
ROOT = Path(__file__).resolve().parent.parent


# ── kernel helpers ────────────────────────────────────────────────────────────

def matern52(d: np.ndarray, ls: float) -> np.ndarray:
    """Matérn-5/2 base kernel values for distances d and scalar lengthscale ls."""
    r = np.sqrt(5.0) * d / ls
    return (1.0 + r + r ** 2 / 3.0) * np.exp(-r)


def kernel_values(d: np.ndarray, ls: float, outputscale: float) -> np.ndarray:
    """Full ScaleKernel(Matérn-5/2) values: outputscale × Matérn(d / ls)."""
    return outputscale * matern52(d, ls)


# ── one-hot encoding ──────────────────────────────────────────────────────────

def one_hot(sequences: list[str]) -> np.ndarray:
    L = len(sequences[0])
    X = np.zeros((len(sequences), L * len(AA)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            idx = AA_IDX.get(aa)
            if idx is not None:
                X[i, j * len(AA) + idx] = 1.0
    return X


def cross_dists_onehot(seqs_a: list[str], seqs_b: list[str]) -> np.ndarray:
    return cdist(one_hot(seqs_a), one_hot(seqs_b), metric="euclidean").ravel()


# ── embedding models ──────────────────────────────────────────────────────────

def _setup_gollum_path():
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault(
        "HF_HOME", "/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface"
    )


def _discover_checkpoint(model_dir: str, ckpt_iter: int | None) -> str | None:
    iter_re = re.compile(r"^iter_(\d+)$")
    iters = sorted(
        [e for e in os.listdir(model_dir) if iter_re.match(e)],
        key=lambda e: int(iter_re.match(e).group(1)),
    )
    if ckpt_iter is not None:
        candidate = os.path.join(model_dir, f"iter_{ckpt_iter}", "finetuned_model.pt")
        if os.path.exists(candidate):
            return candidate
        print(f"[warn] iter_{ckpt_iter} not found, falling back to latest", file=sys.stderr)

    final = os.path.join(model_dir, "final", "finetuned_model.pt")
    if os.path.exists(final):
        return final

    for entry in reversed(iters):
        p = os.path.join(model_dir, entry, "finetuned_model.pt")
        if os.path.exists(p):
            return p

    flat = os.path.join(model_dir, "finetuned_model.pt")
    if os.path.exists(flat):
        return flat

    return None


def load_finetuned_featurizer(args):
    """Finetuned LLMFeaturizer with LoRA + projection head."""
    _setup_gollum_path()
    from gollum.featurization.deep import LLMFeaturizer

    ckpt_path = _discover_checkpoint(args.model_dir, args.ckpt_iter)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found under {args.model_dir}")
    print(f"Loading finetuned checkpoint: {ckpt_path}", flush=True)

    featurizer = LLMFeaturizer(
        model_name=args.model_name,
        input_dim=args.input_dim,
        projection_dim=args.projection_dim,
        pooling_method=args.pooling_method,
        trainable=True,
        target_ratio=args.target_ratio,
        from_top=args.from_top,
    ).cuda()

    state = torch.load(ckpt_path, map_location="cuda")
    missing, unexpected = featurizer.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:3]}", file=sys.stderr)
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:3]}", file=sys.stderr)
    if hasattr(featurizer, "projector"):
        featurizer.projector.to(torch.float64)
    featurizer.eval()
    return featurizer


def load_static_featurizer(args):
    """Pretrained backbone only — no LoRA, no projection (identity)."""
    _setup_gollum_path()
    from gollum.featurization.deep import LLMFeaturizer

    print(f"Loading static model: {args.static_model_name}", flush=True)
    featurizer = LLMFeaturizer(
        model_name=args.static_model_name,
        input_dim=args.static_input_dim,
        projection_dim=None,       # identity projector → raw backbone output
        pooling_method=args.pooling_method,
        trainable=False,           # frozen pretrained weights, no LoRA
    ).cuda()
    featurizer.eval()
    return featurizer


@torch.no_grad()
def embed_sequences(featurizer, sequences: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from gollum.featurization.text import get_tokens

    tokens_np = get_tokens(sequences, model_name=model_name)
    tokens_t  = torch.from_numpy(tokens_np).float().cuda()
    chunks = []
    for i in range(0, len(sequences), batch_size):
        chunk = featurizer(tokens_t[i : i + batch_size])
        chunks.append(chunk.cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


def cross_dists_embedding(
    featurizer, seqs_a: list[str], seqs_b: list[str], model_name: str, batch_size: int
) -> np.ndarray:
    Xa = embed_sequences(featurizer, seqs_a, model_name, batch_size)
    Xb = embed_sequences(featurizer, seqs_b, model_name, batch_size)
    return cdist(Xa, Xb, metric="euclidean").ravel()


# ── data helpers ──────────────────────────────────────────────────────────────

def load_split(dataset: str) -> tuple[list[str], list[str]]:
    base = ROOT / "data" / "flip2" / dataset
    train_df = pd.read_csv(base.parent / (base.name + "_train.csv"))
    test_df  = pd.read_csv(base.parent / (base.name + "_test.csv"))
    return train_df["sequence"].tolist(), test_df["sequence"].tolist()


def subsample(seqs: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if len(seqs) <= n:
        return seqs
    return [seqs[i] for i in rng.choice(len(seqs), n, replace=False)]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="alpha-amylase/by_mutation",
                        help="Single FLIP2 dataset, e.g. alpha-amylase/by_mutation")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="figures/pairwise_distances.png")

    # ── static (pretrained) embedding ─────────────────────────────────────────
    stat = parser.add_argument_group("static embedding (optional)")
    stat.add_argument("--static_model_name", default=None,
                      help="HF model name for static (un-finetuned) embeddings, "
                           "e.g. facebook/esm2_t33_650M_UR50D")
    stat.add_argument("--static_input_dim", type=int, default=1280,
                      help="Hidden dim of the static model.")

    # ── finetuned embedding ────────────────────────────────────────────────────
    ft = parser.add_argument_group("finetuned embedding (optional)")
    ft.add_argument("--model_dir",  default=None)
    ft.add_argument("--model_name", default="facebook/esm2_t33_650M_UR50D")
    ft.add_argument("--input_dim",  type=int, default=1280)
    ft.add_argument("--projection_dim", type=int, default=64)
    ft.add_argument("--pooling_method", default="average")
    ft.add_argument("--target_ratio", type=float, default=0.25)
    ft.add_argument("--from_top", action="store_true", default=True)
    ft.add_argument("--ckpt_iter", type=int, default=None)
    ft.add_argument("--batch_size", type=int, default=32)

    # ── GP hyperparameters for kernel value mode (optional) ───────────────────
    kv = parser.add_argument_group(
        "kernel value mode (optional) — supply GP hyperparameters from wandb "
        "to plot outputscale × Matérn(d/ℓ) instead of raw distances"
    )
    kv.add_argument("--lengthscale_onehot",   type=float, default=None,
                    help="GP lengthscale for one-hot feature space")
    kv.add_argument("--outputscale_onehot",   type=float, default=1.0,
                    help="GP outputscale for one-hot feature space (default 1.0)")
    kv.add_argument("--lengthscale_static",   type=float, default=None,
                    help="GP lengthscale for static embedding feature space")
    kv.add_argument("--outputscale_static",   type=float, default=1.0,
                    help="GP outputscale for static embedding (default 1.0)")
    kv.add_argument("--lengthscale_finetuned", type=float, default=None,
                    help="GP lengthscale for finetuned embedding feature space")
    kv.add_argument("--outputscale_finetuned", type=float, default=1.0,
                    help="GP outputscale for finetuned embedding (default 1.0)")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    use_static    = args.static_model_name is not None
    use_finetuned = args.model_dir is not None

    # kernel value mode: active for any space where a lengthscale was supplied
    kv_mode = any([
        args.lengthscale_onehot is not None,
        args.lengthscale_static is not None,
        args.lengthscale_finetuned is not None,
    ])

    static_feat    = load_static_featurizer(args)    if use_static    else None
    finetuned_feat = load_finetuned_featurizer(args) if use_finetuned else None

    ds = args.dataset
    train_path = (ROOT / "data" / "flip2" / ds).parent / (Path(ds).name + "_train.csv")
    if not train_path.exists():
        sys.exit(f"Dataset not found: {train_path}")

    sns.set_theme(
        style="ticks", font_scale=1.05,
        rc={"font.family": "sans-serif"},
    )

    _p = plt.cm.plasma
    LS    = {"onehot": "-",  "static": "--", "finetuned": ":"}
    COLOR = {"onehot": _p(0.88), "static": _p(0.55), "finetuned": _p(0.25)}

    print(f"\nProcessing {ds} …", flush=True)
    train_seqs, test_seqs = load_split(ds)
    train_sub = subsample(train_seqs, args.max_samples, rng)
    test_sub  = subsample(test_seqs,  args.max_samples, rng)
    print(f"  {len(train_sub)} train × {len(test_sub)} test = "
          f"{len(train_sub)*len(test_sub):,} pairs", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    def _plot(values: np.ndarray, space: str, label: str) -> None:
        """Transform distances → kernel values if in kv_mode, then KDE-plot."""
        ls = getattr(args, f"lengthscale_{space}")
        os_ = getattr(args, f"outputscale_{space}")
        if kv_mode:
            if ls is None:
                # no hyperparameters for this space: use unit lengthscale so
                # the d/ℓ ratio is still meaningful relative to the others
                ls = 1.0
                os_ = 1.0
            values = kernel_values(values, ls, os_)
        sns.kdeplot(
            values,
            ax=ax, color=COLOR[space], linestyle=LS[space],
            fill=True, alpha=0.2, linewidth=1.8,
            clip=(0, None), label=label,
        )

    print("  one-hot …", flush=True)
    _plot(cross_dists_onehot(train_sub, test_sub), "onehot", "one-hot")

    if use_static:
        print("  static embedding …", flush=True)
        _plot(
            cross_dists_embedding(
                static_feat, train_sub, test_sub,
                args.static_model_name, args.batch_size),
            "static",
            f"static ({args.static_model_name.split('/')[-1]})",
        )

    if use_finetuned:
        print("  finetuned embedding …", flush=True)
        _plot(
            cross_dists_embedding(
                finetuned_feat, train_sub, test_sub,
                args.model_name, args.batch_size),
            "finetuned",
            "finetuned",
        )

    dataset_name, split_name = ds.split("/")
    ax.set_title(f"{dataset_name} · {split_name.replace('_', ' ')}", fontsize=11)
    if kv_mode:
        ax.set_xlabel("GP kernel value $\\sigma^2 \\cdot k(x, x')$ (train × test pairs)")
        ax.set_xlim(0, None)
    else:
        ax.set_xlabel("Euclidean distance (train × test pairs)")
        ax.set_xlim(left=0)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False, title="feature space", title_fontsize=8)
    sns.despine(ax=ax)

    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
