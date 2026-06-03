"""
Plot gp/lengthscale_mean / distances/avg as a function of BO epoch,
aggregated across seeds (up to MAX_SEEDS per setup), grouped by dataset split.

Only runs whose featurizer representation == "get_tokens" are included.
Different model setups appear as separate curves on each split's subplot.

Usage:
    python scripts/plot_lengthscale_ratio.py \
        --project gollum-flip2 \
        --splits trpB/one_to_many trpB/by_position alpha-amylase/by_mutation \
        --out figures/lengthscale_ratio.pdf
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from plot_style import apply as _apply_style
_apply_style()

# Tableau 10
_T10 = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

MAX_SEEDS = 10

_METRIC_LS  = "gp/lengthscale_mean"
_METRIC_AVG = "distances/avg"


# ── config helpers ────────────────────────────────────────────────────────────

def _cfg(run, key, default=None):
    return run.config.get(key, default)


def _setup_label(run):
    model_name = _cfg(run, "data.init_args.featurizer.init_args.model_name", "")
    surrogate  = _cfg(run, "surrogate_model.class_path", "")
    is_deepgp  = "DeepGP" in surrogate

    mn = model_name.lower()
    # if "esmc" in mn:
    #     base = "ESMC"
    if "esm" in mn:
        base = "ESM2"
    # elif "prot_t5" in mn:
    #     base = "ProtT5"
    # elif "t5" in mn:
    #     base = "T5"
    else:
        return None

    return f"{base}_{'ft' if is_deepgp else 'frozen'}"


# ── per-run history ───────────────────────────────────────────────────────────

def _fetch_curves(run):
    """Return (epochs, ls, avg, ratio) arrays, or None on failure."""
    try:
        hist = run.history(keys=["epoch", _METRIC_LS, _METRIC_AVG], pandas=True)
    except Exception as exc:
        warnings.warn(f"Run {run.name}: history fetch failed ({exc}) — skipping.")
        return None

    if hist.empty:
        warnings.warn(f"Run {run.name}: empty history — skipping.")
        return None

    missing = [c for c in ["epoch", _METRIC_LS, _METRIC_AVG] if c not in hist.columns]
    if missing:
        warnings.warn(f"Run {run.name}: missing columns {missing} — skipping.")
        return None

    hist = hist.dropna(subset=["epoch", _METRIC_LS, _METRIC_AVG])
    if hist.empty:
        return None

    hist = hist.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")

    epochs = hist["epoch"].to_numpy(dtype=float)
    ls     = hist[_METRIC_LS].to_numpy(dtype=float)
    avg    = hist[_METRIC_AVG].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(avg != 0, ls / avg, np.nan)

    return epochs, ls, avg, ratio


# ── data collection ───────────────────────────────────────────────────────────

def collect(project: str, split_substr: str, groups=None):
    """
    Returns dict: setup_label → list of (epochs_array, ratio_array)
    """
    api = wandb.Api()
    filters = {"state": "finished"}
    if groups:
        filters["group"] = {"$in": groups}

    runs = api.runs(project, filters=filters)

    raw = {}
    skipped = 0

    for run in runs:
        # Filter 1: split
        data_path = _cfg(run, "data.init_args.data_path", "")
        if split_substr not in data_path:
            skipped += 1
            continue

        # Filter 2: representation must be get_tokens
        rep = _cfg(run, "data.init_args.featurizer.init_args.representation", "")
        if rep != "get_tokens":
            skipped += 1
            continue

        label = _setup_label(run)
        if label is None:
            skipped += 1
            continue

        result = _fetch_curves(run)
        if result is None:
            continue

        epochs, ls, avg, ratio = result
        raw.setdefault(label, []).append((run.created_at, epochs, ls, avg, ratio))

    # Keep MAX_SEEDS most-recent per setup
    data = {}
    for label, entries in raw.items():
        entries.sort(key=lambda t: t[0], reverse=True)
        data[label] = [(ep, ls, avg, r) for _, ep, ls, avg, r in entries[:MAX_SEEDS]]

    n_kept = sum(len(v) for v in data.values())
    print(f"  [{split_substr}] {n_kept} runs across {len(data)} setups "
          f"(skipped {skipped}, capped at {MAX_SEEDS} seeds per setup).")
    for k, v in sorted(data.items()):
        print(f"    {k:<25}  n={len(v)}")

    return data


# ── aggregation ───────────────────────────────────────────────────────────────

def _to_grid(curves, grid):
    """Interpolate each (epochs, values) curve onto a common integer grid."""
    out = []
    for epochs, values in curves:
        interp = np.interp(grid, epochs, values,
                           left=values[0] if len(values) else np.nan,
                           right=values[-1] if len(values) else np.nan)
        out.append(interp)
    return np.array(out)  # (n_runs, n_grid)


# ── style ─────────────────────────────────────────────────────────────────────

_SETUP_ORDER = [
    # ("ESM2_frozen",    "ESM2",   "--"),
    ("ESM2_ft",        "ESM2",   "-"),
    # ("ESMC_frozen",    "ESMC",   "--"),
    # ("ESMC_ft",        "ESMC",   "-"),
    # ("ProtT5_frozen",  "ProtT5", "--"),
    # ("ProtT5_ft",      "ProtT5", "-"),
    # ("T5_frozen",      "T5",     "--"),
    # ("T5_ft",          "T5",     "-"),
]

_MODEL_COLOR = {
    "ESM2":   _T10[0],
    # "ESMC":   _T10[2],
    # "ProtT5": _T10[1],
    "T5":     _T10[3],
}


# ── per-subplot plotting ──────────────────────────────────────────────────────

def _agg(curves_1d, grid):
    """Interpolate a list of (epochs, values) onto grid, return mean ± sem."""
    mat = _to_grid(curves_1d, grid)
    valid = ~np.all(np.isnan(mat), axis=0)
    x   = grid[valid]
    mat = mat[:, valid]
    n   = np.sum(~np.isnan(mat), axis=0)
    mean = np.nanmean(mat, axis=0)
    sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return x, mean, sem


def _plot_split(ax, data, split_title, mode="separate"):
    if not data:
        ax.set_visible(False)
        return []

    max_epoch = max(
        int(ep[-1]) for curves in data.values() for ep, *_ in curves if len(ep)
    )
    grid = np.arange(0, max_epoch + 1)

    handles = []
    all_means, all_sems = [], []
    for key, model_label, linestyle in _SETUP_ORDER:
        if key not in data:
            continue
        curves = data[key]   # list of (epochs, ls, avg, ratio)
        color  = _MODEL_COLOR[model_label]
        suffix = "ft" if linestyle == "-" else "frozen"

        if mode == "separate":
            # Plot ℓ (solid) and d̄ (dashed) on a shared log y-axis
            x_ls,  m_ls,  s_ls  = _agg([(ep, ls)  for ep, ls, avg, _ in curves], grid)
            x_avg, m_avg, s_avg = _agg([(ep, avg) for ep, ls, avg, _ in curves], grid)

            l1, = ax.plot(x_ls,  m_ls,  color=color, linestyle="-",  linewidth=1.4,
                          label=rf"{model_label} {suffix} $\ell$")
            ax.fill_between(x_ls,  m_ls  - s_ls,  m_ls  + s_ls,
                            color=color, alpha=0.15, linewidth=0)
            l2, = ax.plot(x_avg, m_avg, color=color, linestyle="--", linewidth=1.4,
                          label=rf"{model_label} {suffix} $\bar{{d}}$")
            ax.fill_between(x_avg, m_avg - s_avg, m_avg + s_avg,
                            color=color, alpha=0.08, linewidth=0)
            handles.extend([l1, l2])

        else:
            # ratio or log_ratio: plot ℓ/d̄
            x, mean, sem = _agg([(ep, r) for ep, _, _, r in curves], grid)
            label = f"{model_label} {suffix}"
            line, = ax.plot(x, mean, color=color, linestyle=linestyle,
                            linewidth=1.4, label=label)
            ax.fill_between(x, mean - sem, mean + sem,
                            color=color, alpha=0.15, linewidth=0)
            handles.append(line)
            all_means.append(mean)
            all_sems.append(sem)

    if mode == "separate":
        ax.set_yscale("log")
        ax.set_ylabel("value (log scale)")
    elif mode == "log_ratio":
        ax.set_yscale("log")
        ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_ylabel(r"$\ell\,/\,\bar{d}$ (log scale)")
    else:
        ax.set_ylabel(r"$\ell$ / $\bar{d}$")
        if all_means:
            vals_lo = np.concatenate([m - s for m, s in zip(all_means, all_sems)])
            vals_hi = np.concatenate([m + s for m, s in zip(all_means, all_sems)])
            lo, hi = np.nanmin(vals_lo), np.nanmax(vals_hi)
            pad = max((hi - lo) * 0.15, hi * 0.05)
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(split_title, pad=4)
    ax.set_xlabel("BO epoch")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    return handles


# ── figure assembly ───────────────────────────────────────────────────────────

_SUPTITLES = {
    "separate":  r"GP lengthscale $\ell$ and avg pairwise distance $\bar{d}$  (log scale)",
    "log_ratio": r"GP lengthscale / avg distance  ($\ell\,/\,\bar{d}$, log scale)",
    "ratio":     r"GP lengthscale / avg distance  ($\ell\,/\,\bar{d}$)",
}


def plot(split_data_map, out_path, mode="separate"):
    n_splits = len(split_data_map)
    if n_splits == 0:
        print("Nothing to plot.")
        return

    ncols = min(n_splits, 3)
    nrows = math.ceil(n_splits / ncols)
    fig_w = min(3.25 * ncols, 6.75)
    fig_h = 2.6 * nrows

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False,
                             sharey=False)

    all_handles = []
    for idx, (title, data) in enumerate(split_data_map.items()):
        row, col = divmod(idx, ncols)
        handles = _plot_split(axes[row][col], data, title, mode=mode)
        if len(handles) > len(all_handles):
            all_handles = handles

    for idx in range(n_splits, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    ncol_legend = min(len(all_handles), 4 if mode != "separate" else 2)
    if all_handles:
        fig.legend(handles=all_handles,
                   loc="lower center",
                   ncol=ncol_legend,
                   fontsize=8,
                   frameon=False,
                   bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(_SUPTITLES[mode], fontsize=9, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="gollum-flip2")
    parser.add_argument("--splits", nargs="+", default=[
        "trpB/one_to_many",
        "trpB/by_position",
        "trpB/two_to_many",
        "alpha-amylase/by_mutation",
        "alpha-amylase/close_to_far",
        "alpha-amylase/far_to_close",
        "alpha-amylase/one_to_many",
        "ired/two_to_many",
        "nucB/two_to_many",
    ])
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--out", default="figures/lengthscale_ratio.pdf")
    parser.add_argument(
        "--mode", default="separate",
        choices=["separate", "log_ratio", "ratio"],
        help=(
            "separate  – ℓ and d̄ as two lines on a shared log y-axis (default); "
            "log_ratio – ℓ/d̄ on a log y-axis; "
            "ratio     – ℓ/d̄ on a linear y-axis (original behaviour)"
        ),
    )
    args = parser.parse_args()

    split_data_map = {}
    for split in args.splits:
        print(f"\n{'='*60}\nSplit: {split}\n{'='*60}")
        data = collect(args.project, split, args.groups)
        if not data:
            print(f"  No data for '{split}' — skipping.")
            continue
        parts = split.replace("_", " ").split("/")
        title = " / ".join(p.strip() for p in parts)
        split_data_map[title] = data

    plot(split_data_map, args.out, mode=args.mode)
