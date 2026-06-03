"""
Plot top-5% coverage bar chart at BO epoch N from existing WandB runs.

Usage:
    python scripts/plot_top5_coverage.py \
        --project gollum_flip2 \
        --epoch 3 \
        --splits trpB/one_to_many_train.csv alpha-amylase/by_mutation_train.csv \
        --out figures/top5_coverage_epoch3.pdf

One plot is saved per split; the split name is embedded in the output filename.

The script identifies each run's setup from its wandb config:
  - featurizer representation == protein_one_hot  →  one-hot
  - surrogate_model.class_path contains DeepGP    →  finetuned
  - otherwise                                     →  frozen
"""

import argparse
import math
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import wandb

from plot_style import apply as _apply_style
_apply_style()
# Sequential palette: deep indigo → purple → magenta → salmon → orange → gold
_PALETTE = ["#3B0CA6", "#7B01D1", "#BE2D86", "#DC5F6A", "#E07530", "#F0B030"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _cfg(run, key, default=None):
    """Access a flattened wandb config key (dot-separated)."""
    return run.config.get(key, default)


def _setup_label(run):
    """Return a canonical setup string for a run, or None to skip."""
    # Random baseline: no surrogate model in config
    if _cfg(run, "surrogate_model.class_path") is None:
        return "random"

    representation = _cfg(run, "data.init_args.featurizer.init_args.representation", "")
    if representation == "protein_one_hot":
        return "one_hot"

    model_name = _cfg(run, "data.init_args.featurizer.init_args.model_name", "")
    surrogate  = _cfg(run, "surrogate_model.class_path", "")
    is_deepgp  = "DeepGP" in surrogate

    model_name_lower = model_name.lower()
    if "esmc" in model_name_lower:           # check before generic "esm"
        base = "ESMC"
    elif "esm" in model_name_lower:
        base = "ESM2"
    elif "prot_t5" in model_name_lower:
        base = "ProtT5"
    elif "t5" in model_name_lower:
        base = "T5"
    else:
        return None   # unknown model — skip

    return f"{base}_{'finetuned' if is_deepgp else 'frozen'}"


def _coverage_at_epoch(run, target_epoch):
    """Return top5pct_coverage value at the given epoch, or NaN."""
    try:
        hist = run.history(keys=["epoch", "top5pct_coverage"], pandas=True)
    except Exception:
        return math.nan
    if hist.empty:
        return math.nan
    rows = hist[hist["epoch"] == target_epoch]
    if rows.empty:
        return math.nan
    return float(rows["top5pct_coverage"].iloc[-1])


# ── main ─────────────────────────────────────────────────────────────────────

def collect(project, split_substr, target_epoch, groups=None):
    api = wandb.Api()
    filters = {"state": "finished"}
    if groups:
        filters["group"] = {"$in": groups}

    runs = api.runs(project, filters=filters)

    data = {}   # setup_label → list of (created_at, coverage)
    skipped = 0
    for run in runs:
        # Dataset filter
        data_path = _cfg(run, "data.init_args.data_path", "")
        if split_substr not in data_path:
            skipped += 1
            continue

        label = _setup_label(run)
        if label is None:
            skipped += 1
            continue

        cov = _coverage_at_epoch(run, target_epoch)
        if math.isnan(cov):
            warnings.warn(f"Run {run.name} ({run.id}): no top5pct_coverage at epoch={target_epoch}")
            continue

        data.setdefault(label, []).append((run.created_at, cov))

    # Keep only the 5 most recent runs per setup
    MAX_RUNS = 10
    trimmed = {}
    for label, entries in data.items():
        entries.sort(key=lambda t: t[0], reverse=True)   # newest first (ISO timestamps sort lexicographically)
        trimmed[label] = [cov for _, cov in entries[:MAX_RUNS]]

    print(f"Collected {sum(len(v) for v in trimmed.values())} values across {len(trimmed)} setups "
          f"(skipped {skipped} runs, capped at {MAX_RUNS} most recent per setup).")
    for k, v in sorted(trimmed.items()):
        print(f"  {k:<25}  n={len(v)}  mean={np.mean(v):.3f}  std={np.std(v):.3f}  values={[round(x,4) for x in v]}")
    return trimmed


_GROUPS = [
    ("T5",      ["T5_frozen",     "T5_finetuned"]),
    ("ProtT5",  ["ProtT5_frozen", "ProtT5_finetuned"]),
    ("ESM2",    ["ESM2_frozen",   "ESM2_finetuned"]),
    ("ESMC",    ["ESMC_frozen",   "ESMC_finetuned"]),
    ("One-hot", ["one_hot"]),
]
# One color per model family; frozen bars get hatching to distinguish from finetuned
_MODEL_COLOR = {
    "T5":      _PALETTE[0],
    "ProtT5":  _PALETTE[1],
    "ESM2":    _PALETTE[2],
    "ESMC":    _PALETTE[3],
    "one_hot": _PALETTE[4],
    "random":  _PALETTE[5],
}


def _bar_color(key):
    for model in ("ESMC", "ESM2", "ProtT5", "T5", "one_hot"):
        if key.startswith(model):
            return _MODEL_COLOR[model]
    return _MODEL_COLOR["random"]


def _bar_hatch(key):
    return "//" if "frozen" in key else None


def _draw_bars(ax, data, target_epoch, *, show_ylabel=True, show_legend=True, title=None):
    bar_width = 0.32
    group_gap = 0.25

    positions, bar_meta = [], []
    x_cursor = 0.0
    x_ticks, x_labels = [], []

    # Always allocate space for every group so x layout is identical across all panels.
    for g_label, keys in _GROUPS:
        n_keys = len(keys)
        offsets = np.linspace(-(n_keys - 1) * bar_width / 2, (n_keys - 1) * bar_width / 2, n_keys)
        center = x_cursor + (n_keys - 1) * bar_width / 2
        for k, off in zip(keys, offsets):
            if k in data:
                positions.append(center + off)
                bar_meta.append(k)
        x_ticks.append(center)
        x_labels.append(g_label)
        x_cursor = center + (n_keys - 1) * bar_width / 2 + bar_width + group_gap

    x_right = x_cursor - group_gap  # right edge of last group

    for pos, key in zip(positions, bar_meta):
        vals = np.array(data[key])
        mean, se = vals.mean(), vals.std(ddof=1) / math.sqrt(len(vals))
        ax.bar(pos, mean, width=bar_width * 0.9,
               color=_bar_color(key), hatch=_bar_hatch(key),
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(pos, mean, yerr=se,
                    fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)

    if "random" in data:
        rand_vals = np.array(data["random"])
        rand_mean = rand_vals.mean()
        rand_std  = rand_vals.std(ddof=1) if len(rand_vals) > 1 else 0.0
        rc = _MODEL_COLOR["random"]
        ax.axhline(rand_mean, color=rc, linestyle="--", linewidth=1.5, zorder=5)
        ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
                   color=rc, alpha=0.15, zorder=2)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=15)
    ax.set_xlim(-bar_width, x_right + bar_width)  # consistent padding for all panels
    if show_ylabel:
        ax.set_ylabel(f"Top-5% coverage  (epoch {target_epoch})", fontsize=15)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=18)

    if show_legend:
        from matplotlib.lines import Line2D
        # Color = model family; hatch = frozen
        handles = [
            mpatches.Patch(color=_MODEL_COLOR[m], label=m.replace("_", "-"))
            for m in ("T5", "ProtT5", "ESM2", "ESMC", "one_hot")
        ] + [
            mpatches.Patch(facecolor="white", edgecolor="#555555",
                           hatch="//", label="Frozen"),
            mpatches.Patch(facecolor="white", edgecolor="#555555",
                           label="Finetuned"),
        ]
        if "random" in data:
            handles.append(Line2D([0], [0], color=_MODEL_COLOR["random"],
                                  linestyle="--", linewidth=1.5, label="Random"))
        ax.legend(handles=handles, frameon=False, fontsize=12, loc="upper right",
                  ncol=2)


def plot(data, target_epoch, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_bars(ax, data, target_epoch)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_combined(all_data, splits, target_epoch, out_path):
    """1 × N combined figure with one subplot per split."""
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for i, (ax, split) in enumerate(zip(axes, splits)):
        title = split.split("/")[-1].replace("_", " ")
        _draw_bars(ax, all_data[split], target_epoch,
                   show_ylabel=(i == 0),
                   show_legend=(i == n - 1),
                   title=title)
        if i > 0:
            ax.tick_params(labelleft=False)

    dataset = splits[0].split("/")[0].replace("-", " ").title()
    # fig.suptitle(f"{dataset} — top-5% coverage (epoch {target_epoch})", fontsize=18)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved (combined) → {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",  default="gollum_flip2",
                        help="WandB project (entity/project or just project)")
    parser.add_argument("--splits",   nargs="+",
                        default=None,
                        help="One or more substrings to match in data_path config field; "
                             "one plot is generated per split")
    parser.add_argument("--epoch",    type=int, default=3,
                        help="BO epoch to read coverage from (default 3)")
    parser.add_argument("--groups",   nargs="*", default=None,
                        help="Optional WandB group names to restrict search")
    parser.add_argument("--out",      default="figures/top5%/top5_coverage_epoch3.pdf")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    base, ext = os.path.splitext(args.out)
    splits = args.splits or [
        "alpha-amylase/by_mutation",
        "alpha-amylase/close_to_far",
        "alpha-amylase/far_to_close",
        "alpha-amylase/one_to_many",
        "nucB/two_to_many",
        "ired/two_to_many",
        "trpB/two_to_many",
    ]

    all_data = {}
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")
        slug = split.replace("/", "_").replace(".", "_")
        out_path = f"{base}_{slug}{ext}"

        data = collect(args.project, split, args.epoch, args.groups)
        if not data:
            print(f"No data found for split '{split}' — skipping.")
        else:
            all_data[split] = data
            plot(data, args.epoch, out_path)

    # Combined plot: one panel per split, grouped by dataset prefix
    from itertools import groupby
    collected_splits = list(all_data.keys())
    for dataset, group in groupby(collected_splits, key=lambda s: s.split("/")[0]):
        group_splits = list(group)
        if len(group_splits) < 2:
            continue
        slug = dataset.replace("-", "_")
        combined_out = f"{base}_{slug}_combined{ext}"
        plot_combined(all_data, group_splits, args.epoch, combined_out)
