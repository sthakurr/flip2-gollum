"""
Plot log-regret curves from best-so-far values logged to WandB BO runs.

Log regret at epoch t is defined as:
    log_regret(t) = log(y* - best_so_far(t))
where y* = target_stat_max (the global maximum in the full dataset), stored
in each run's WandB summary by log_data_stats().

Usage (one split):
    python scripts/plot_log_regret.py \
        --project gollum_flip2 \
        --splits trpB/one_to_many \
        --out figures/log_regret.pdf

Usage (multiple splits → subplots in one figure):
    python scripts/plot_log_regret.py \
        --project gollum_flip2 \
        --splits trpB/one_to_many alpha-amylase/by_mutation nucB/two_to_many \
        --out figures/log_regret_multi.pdf

The shaded band is mean ± 1 SEM across seeds.
Runs are matched to the most recent MAX_RUNS seeds per (split, setup) pair.
"""

import argparse
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wandb

from plot_style import apply as _apply_style
_apply_style()

# Tableau 10 palette — standard in NeurIPS/ICML/ICLR figures
_T10 = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

# Max seeds to keep per setup (most-recent first)
MAX_RUNS = 5

# Small epsilon to guard against log(0) when best_so_far == global_max
_EPS = 1e-8


# ── setup labelling ─────────────────────

def _cfg(run, key, default=None):
    return run.config.get(key, default)


def _setup_label(run):
    if _cfg(run, "surrogate_model.class_path") is None:
        return "random"

    representation = _cfg(run, "data.init_args.featurizer.init_args.representation", "")
    if representation == "protein_one_hot":
        return "one_hot"

    model_name = _cfg(run, "data.init_args.featurizer.init_args.model_name", "")
    surrogate  = _cfg(run, "surrogate_model.class_path", "")
    is_deepgp  = "DeepGP" in surrogate

    mn = model_name.lower()
    if "esmc" in mn:
        base = "ESMC"
    elif "esm" in mn:
        base = "ESM2"
    elif "prot_t5" in mn:
        base = "ProtT5"
    elif "t5" in mn:
        base = "T5"
    else:
        return None

    return f"{base}_{'finetuned' if is_deepgp else 'frozen'}"


# ── per-run history fetching ──────────────────────────────────────────────────

def _log_regret_curve(run):
    """Return (epochs, log_regrets) arrays for a run, or (None, None) on failure."""
    global_max = run.summary.get("target_stat_max")
    if global_max is None:
        warnings.warn(f"Run {run.name}: no target_stat_max in summary — skipping.")
        return None, None

    try:
        hist = run.history(keys=["epoch", "train/best_so_far"], pandas=True)
    except Exception as exc:
        warnings.warn(f"Run {run.name}: history fetch failed ({exc}) — skipping.")
        return None, None

    if hist.empty or "train/best_so_far" not in hist.columns:
        warnings.warn(f"Run {run.name}: no best_so_far history — skipping.")
        return None, None

    hist = hist.dropna(subset=["epoch", "train/best_so_far"])
    if hist.empty:
        return None, None

    # Keep one value per epoch (last logged if duplicates)
    hist = hist.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")

    epochs = hist["epoch"].to_numpy(dtype=float)
    best   = hist["train/best_so_far"].to_numpy(dtype=float)

    regret = global_max - best
    # Clamp to avoid log(<=0)
    regret = np.where(regret <= 0, _EPS, regret)
    log_regret = np.log(regret)

    return epochs, log_regret


# ── data collection ───────────────────────────────────────────────────────────

def collect(project, split_substr, groups=None):
    """
    Returns:
        data  : dict  setup_label → list of (epochs_array, log_regret_array)
        n_epochs_max : int, maximum epoch seen across all runs
    """
    api = wandb.Api()
    filters = {"state": "finished"}
    if groups:
        filters["group"] = {"$in": groups}

    runs = api.runs(project, filters=filters)

    raw = {}   # setup_label → list of (created_at, epochs, log_regret)
    skipped = 0

    for run in runs:
        data_path = _cfg(run, "data.init_args.data_path", "")
        if split_substr not in data_path:
            skipped += 1
            continue

        label = _setup_label(run)
        if label is None:
            skipped += 1
            continue

        epochs, lr = _log_regret_curve(run)
        if epochs is None:
            continue

        raw.setdefault(label, []).append((run.created_at, epochs, lr))

    # Trim to MAX_RUNS most-recent per setup
    data = {}
    for label, entries in raw.items():
        entries.sort(key=lambda t: t[0], reverse=True)
        data[label] = [(ep, lr) for _, ep, lr in entries[:MAX_RUNS]]

    n_kept = sum(len(v) for v in data.values())
    n_epochs_max = 3
    print(f"  Collected {n_kept} runs across {len(data)} setups "
          f"(skipped {skipped}, capped at {MAX_RUNS} most recent per setup).")
    for k, v in sorted(data.items()):
        lengths = [len(ep) for ep, _ in v]
        print(f"    {k:<25}  n={len(v)}  epoch_lengths={lengths}")

    return data, n_epochs_max


# ── aggregation helpers ───────────────────────────────────────────────────────

def _interpolate_to_grid(curves, grid):
    """Interpolate each (epochs, values) curve onto a common integer grid."""
    out = []
    for epochs, values in curves:
        # np.interp needs sorted epochs (already sorted from collection)
        interp = np.interp(grid, epochs, values,
                           left=values[0] if len(values) else np.nan,
                           right=values[-1] if len(values) else np.nan)
        out.append(interp)
    return np.array(out)  # shape (n_runs, n_epochs)


# ── plotting ──────────────────────────────────────────────────────────────────

# Display order and style
_SETUP_ORDER = [
    ("T5_frozen",      "T5",      "frozen"),
    ("T5_finetuned",   "T5",      "finetuned"),
    ("ProtT5_frozen",  "ProtT5",  "frozen"),
    ("ProtT5_finetuned","ProtT5", "finetuned"),
    ("ESM2_frozen",    "ESM2",    "frozen"),
    ("ESM2_finetuned", "ESM2",    "finetuned"),
    ("ESMC_frozen",    "ESMC",    "frozen"),
    ("ESMC_finetuned", "ESMC",    "finetuned"),
    ("one_hot",        "One-hot", "one_hot"),
    ("random",         "Random",  "random"),
]

# Color per model family; linestyle distinguishes frozen/finetuned
_MODEL_COLOR = {
    "T5":      _T10[3],   # teal
    "ProtT5":  _T10[1],   # orange
    "ESM2":    _T10[0],   # blue
    "ESMC":    _T10[2],   # red
    "One-hot": _T10[4],   # green
    "Random":  "#999999",
}
_STYLE = {
    "frozen":   {"linestyle": "--", "linewidth": 1.4},
    "finetuned":{"linestyle": "-",  "linewidth": 1.6},
    "one_hot":  {"linestyle": "-",  "linewidth": 1.4},
    "random":   {"linestyle": ":",  "linewidth": 1.4},
}


def _plot_split(ax, data, n_epochs_max, split_title):
    grid = np.arange(0, n_epochs_max + 1)

    legend_handles = []
    for key, model_label, style_key in _SETUP_ORDER:
        if key not in data:
            continue
        curves = data[key]
        mat = _interpolate_to_grid(curves, grid)   # (n_runs, n_grid)

        # Drop columns that are all-NaN (epochs beyond all runs)
        valid_cols = ~np.all(np.isnan(mat), axis=0)
        x   = grid[valid_cols]
        mat = mat[:, valid_cols]

        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

        color = _MODEL_COLOR[model_label]
        style = _STYLE[style_key]

        line, = ax.plot(x, mean, color=color, label=f"{model_label} {'ft' if style_key == 'finetuned' else style_key}",
                        **style)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15)
        legend_handles.append(line)

    ax.set_xlabel("BO epoch", fontsize=9)
    ax.set_ylabel("Log regret", fontsize=9)
    ax.set_title(split_title, fontsize=9, pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    return legend_handles


def plot(split_data_map, out_path):
    """
    split_data_map : OrderedDict  split_label → (data_dict, n_epochs_max)
    """
    n_splits = len(split_data_map)
    if n_splits == 0:
        print("Nothing to plot.")
        return

    # ICML column widths: 3.25" single, 6.75" double
    # One subplot per split, arranged in a single row (wrap at 3)
    ncols = min(n_splits, 3)
    nrows = math.ceil(n_splits / ncols)
    fig_w = min(3.25 * ncols, 6.75)
    fig_h = 2.8 * nrows

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False,
                             sharey=False)

    all_handles = []
    for idx, (split_label, (data, n_emax)) in enumerate(split_data_map.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        handles = _plot_split(ax, data, n_emax, split_label)
        if len(handles) > len(all_handles):
            all_handles = handles

    # Hide unused axes
    for idx in range(n_splits, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared legend below the figure
    if all_handles:
        fig.legend(handles=all_handles,
                   loc="lower center",
                   ncol=min(len(all_handles), 5),
                   fontsize=8,
                   frameon=False,
                   bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot log-regret curves from WandB BO runs."
    )
    parser.add_argument("--project", default="gollum_flip2",
                        help="WandB project (entity/project or just project)")
    parser.add_argument("--splits",  nargs="+",
                        default=[
                            "trpB/one_to_many",
                            "trpB/by_position",
                            "trpB/two_to_many",
                            "alpha-amylase/by_mutation",
                            "alpha-amylase/close_to_far",
                            "alpha-amylase/far_to_close",
                            "alpha-amylase/one_to_many",
                            "ired/two_to_many",
                            "nucB/two_to_many",
                        ],
                        help="Substrings to match in data.init_args.data_path; "
                             "each produces one subplot in the output figure")
    parser.add_argument("--groups",  nargs="*", default=None,
                        help="Optional WandB group names to restrict search")
    parser.add_argument("--out",     default="figures/log-regrets/log_regret_combined.pdf",
                        help="Output file path (.pdf or .png)")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    split_data_map = {}
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")
        data, n_emax = collect(args.project, split, args.groups)
        if not data:
            print(f"  No data found for split '{split}' — skipping subplot.")
            continue
        # Use the last component as the title (e.g. "trpB / one_to_many")
        parts = split.replace("_", " ").split("/")
        title = " / ".join(p.strip() for p in parts)
        split_data_map[title] = (data, n_emax)

    if not split_data_map:
        print("No data collected for any split. Exiting.")
    else:
        plot(split_data_map, args.out)
