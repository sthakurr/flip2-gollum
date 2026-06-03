#!/usr/bin/env python3
"""
Summarize final Spearman ρ across splits and configurations.

Reads final_test/spearman_rho and final_heldout/spearman_rho from wandb run
summaries, groups by (split × config), and reports mean ± stderr over up to
--max_seeds most-recent runs with distinct names (name encodes the seed).

Configurations
--------------
  one_hot      protein_one_hot representation
  T5_frozen    get_huggingface_embeddings + t5 model
  T5_ft        get_tokens + t5 model
  ESM2_frozen  get_huggingface_embeddings + esm model
  ESM2_ft      get_tokens + esm model

Usage
-----
  conda run -n gollum python scripts/summarize_spearman.py
  conda run -n gollum python scripts/summarize_spearman.py \\
      --project gollum-flip2 --max_seeds 10 --out results/spearman.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# ── config classification ─────────────────────────────────────────────────────

CONFIGS = [
    # (key,          display label,   accepted representations,         model substr or None)
    ("one_hot",      "One-Hot",       ("protein_one_hot",),             None),
    ("T5_frozen",    "T5 frozen",     ("get_huggingface_embeddings",),  "t5"),
    ("T5_ft",        "T5 ft",         ("get_tokens",),                  "t5"),
    ("ESM2_frozen",  "ESM2 frozen",   ("get_huggingface_embeddings",),  "esm"),
    ("ESM2_ft",      "ESM2 ft",       ("get_tokens",),                  "esm"),
]
CONFIG_KEYS   = [c[0] for c in CONFIGS]
CONFIG_LABELS = {c[0]: c[1] for c in CONFIGS}

METRIC_TEST    = "final_test/spearman_rho"
METRIC_HELDOUT = "final_heldout/spearman_rho"


def classify_run(run) -> str | None:
    rep = run.config.get("data.init_args.featurizer.init_args.representation") or ""
    mn  = (run.config.get("data.init_args.featurizer.init_args.model_name") or "").lower()
    for key, _label, reps, model_substr in CONFIGS:
        if rep not in reps:
            continue
        if model_substr is not None and model_substr not in mn:
            continue
        return key
    return None


def split_from_path(data_path: str) -> str:
    """'data/flip2/trpB/by_position_train.csv' → 'trpB/by_position'"""
    p    = Path(data_path)
    stem = p.stem.replace("_train", "").replace("_test", "")
    return f"{p.parent.name}/{stem}"


def _nan(v) -> bool:
    if v is None:
        return True
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ── data collection ───────────────────────────────────────────────────────────

def collect(project: str, max_seeds: int, min_epoch: int | None):
    """
    Returns nested dict:
        split → config_key → list of {test: float|nan, heldout: float|nan}
    Up to max_seeds entries per cell, selected as the most-recent distinct names.
    """
    api  = wandb.Api()
    runs = api.runs(project, filters={"state": "finished"})

    # split → config_key → name → (created_at, test_val, heldout_val)
    raw: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    skipped = 0

    for run in runs:
        data_path = run.config.get("data.init_args.data_path") or ""
        if not data_path:
            skipped += 1
            continue

        # Optional epoch gate (useful to exclude crashed/short runs)
        if min_epoch is not None:
            epoch = run.summary.get("epoch")
            if epoch is None or int(epoch) < min_epoch:
                skipped += 1
                continue

        cfg_key = classify_run(run)
        if cfg_key is None:
            skipped += 1
            continue

        test_val    = run.summary.get(METRIC_TEST)
        heldout_val = run.summary.get(METRIC_HELDOUT)

        # Skip if both metrics are absent/nan
        if _nan(test_val) and _nan(heldout_val):
            skipped += 1
            continue

        split    = split_from_path(data_path)
        name     = run.name
        existing = raw[split][cfg_key]
        if name not in existing or run.created_at > existing[name][0]:
            existing[name] = (run.created_at, test_val, heldout_val)

    print(f"Skipped {skipped} runs (no data_path / unrecognised config / below min epoch / no metrics).\n")

    # Trim to max_seeds most-recent per (split, config)
    data: dict[str, dict[str, list]] = defaultdict(dict)
    for split, cfg_map in raw.items():
        for cfg_key, name_map in cfg_map.items():
            entries = sorted(name_map.values(), key=lambda t: t[0], reverse=True)
            data[split][cfg_key] = [
                {"test": t, "heldout": h} for _, t, h in entries[:max_seeds]
            ]
    return data


# ── aggregation ───────────────────────────────────────────────────────────────

def _mean_sem(values) -> tuple[str, str, int]:
    vals = [float(v) for v in values if not _nan(v)]
    n    = len(vals)
    if n == 0:
        return "—", "", 0
    mean = np.mean(vals)
    sem  = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return f"{mean:.3f}", f"{sem:.3f}", n


def build_table(data: dict, metric: str) -> pd.DataFrame:
    rows = []
    for split in sorted(data.keys()):
        row = {"split": split}
        for cfg_key in CONFIG_KEYS:
            entries  = data[split].get(cfg_key, [])
            vals     = [e[metric] for e in entries]
            mean, sem, n = _mean_sem(vals)
            if mean == "—":
                row[CONFIG_LABELS[cfg_key]] = "—"
            else:
                row[CONFIG_LABELS[cfg_key]] = f"{mean} ± {sem}  (n={n})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def build_count_table(data: dict) -> pd.DataFrame:
    rows = []
    for split in sorted(data.keys()):
        row = {"split": split}
        for cfg_key in CONFIG_KEYS:
            row[CONFIG_LABELS[cfg_key]] = len(data[split].get(cfg_key, []))
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project",   default="gollum-flip2",
                        help="W&B project name.")
    parser.add_argument("--max_seeds", type=int, default=10,
                        help="Max runs (distinct names) per split × config cell.")
    parser.add_argument("--min_epoch", type=int, default=None,
                        help="Skip runs whose summary epoch < this value (e.g. 3 to drop crashed runs).")
    parser.add_argument("--out",       default=None,
                        help="Base path for CSV output (e.g. results/spearman.csv). "
                             "Two files are written: <base>_test.csv and <base>_heldout.csv.")
    args = parser.parse_args()

    print(f"Fetching finished runs from '{args.project}'  "
          f"(max_seeds={args.max_seeds}, min_epoch={args.min_epoch}) …\n")

    data = collect(args.project, args.max_seeds, args.min_epoch)
    if not data:
        sys.exit("No data found — check project name.")

    for metric_key, metric_label in [("test", "test"), ("heldout", "heldout")]:
        df = build_table(data, metric_key)
        print(f"\n{'='*90}")
        print(f"  final_{metric_label}/spearman_rho   (mean ± stderr,  up to {args.max_seeds} seeds per cell)")
        print(f"{'='*90}")
        pd.set_option("display.max_colwidth", 24)
        pd.set_option("display.width", 200)
        print(df.to_string())

        if args.out:
            base = Path(args.out)
            out  = base.parent / f"{base.stem}_{metric_key}{base.suffix or '.csv'}"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out)
            print(f"  → saved {out}")

    print(f"\n{'='*90}")
    print("  Run counts per split × config")
    print(f"{'='*90}")
    print(build_count_table(data).to_string())


if __name__ == "__main__":
    main()
