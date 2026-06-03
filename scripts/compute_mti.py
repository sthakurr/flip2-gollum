#!/usr/bin/env python3
"""Compute Maximum Training Identity (MTI) for all FLIP2 splits.

For each test sequence, MTI = max over training sequences of
(1 - normalised Hamming distance), i.e. the fraction of positions
that match the nearest training sequence.

All sequences within a split are same-length variants of the same protein,
so identity = (L - hamming_distance) / L.

Usage
-----
    python scripts/compute_mti.py
    python scripts/compute_mti.py --data-dir data/flip2 --output mti_results.csv
    python scripts/compute_mti.py --chunk-size 500   # lower if memory is tight
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def seqs_to_uint8(sequences: list[str]) -> np.ndarray:
    """Pack a list of equal-length strings into an (N, L) uint8 array."""
    L = len(sequences[0])
    for i, s in enumerate(sequences):
        if len(s) != L:
            raise ValueError(f"Sequence {i} has length {len(s)}, expected {L}")
    buf = b"".join(s.encode("ascii") for s in sequences)
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(sequences), L)


def compute_mti(train_seqs: list[str], test_seqs: list[str], chunk_size: int = 2000) -> np.ndarray:
    """Return MTI value for each test sequence.

    Uses scipy cdist (hamming metric) chunked over test sequences so the
    (chunk x N_train) output matrix stays in memory rather than an
    (N_test x N_train x L) intermediate.

    Args:
        train_seqs: Training sequences (all same length).
        test_seqs:  Test sequences (same length as train).
        chunk_size: Number of test sequences processed per cdist call.

    Returns:
        np.ndarray of shape (N_test,) with values in [0, 1].
    """
    train_arr = seqs_to_uint8(train_seqs)
    test_arr = seqs_to_uint8(test_seqs)

    N_test = len(test_seqs)
    mti = np.empty(N_test, dtype=np.float32)

    for start in range(0, N_test, chunk_size):
        end = min(start + chunk_size, N_test)
        chunk = test_arr[start:end]
        # cdist hamming = proportion of differing positions = hamming_dist / L
        ham = cdist(chunk, train_arr, metric="hamming")   # (chunk, N_train) float64
        mti[start:end] = 1.0 - ham.min(axis=1)

    return mti


def mti_stats(mti: np.ndarray) -> dict:
    return {
        "mean":      float(np.mean(mti)),
        "median":    float(np.median(mti)),
        "min":       float(np.min(mti)),
        "max":       float(np.max(mti)),
        "pct_gt90":  float(np.mean(mti > 0.90)),
        "pct_gt95":  float(np.mean(mti > 0.95)),
        "pct_gt99":  float(np.mean(mti > 0.99)),
    }


def find_splits(data_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted list of (train_csv, test_csv) path pairs."""
    pairs = []
    for train_path in sorted(data_dir.rglob("*_train.csv")):
        test_path = train_path.with_name(train_path.name.replace("_train", "_test"))
        if test_path.exists():
            pairs.append((train_path, test_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", default="data/flip2",
        help="Root directory containing FLIP2 protein subdirectories (default: data/flip2)",
    )
    parser.add_argument(
        "--output", default="mti_results.csv",
        help="Path to write the results CSV (default: mti_results.csv)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000,
        help="Test sequences per cdist call — reduce if memory is tight (default: 2000)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Error: data directory '{data_dir}' not found.")

    splits = find_splits(data_dir)
    if not splits:
        sys.exit(f"Error: no *_train.csv / *_test.csv pairs found under '{data_dir}'.")

    print(f"Found {len(splits)} splits under {data_dir}\n")

    rows = []
    for train_path, test_path in splits:
        protein = train_path.parent.name
        split_name = train_path.stem.replace("_train", "")
        label = f"{protein}/{split_name}"

        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        train_seqs = train_df["sequence"].tolist()
        test_seqs  = test_df["sequence"].tolist()

        print(f"{label:40s}  n_train={len(train_seqs):>7,}  n_test={len(test_seqs):>8,}", end="  ", flush=True)

        try:
            mti = compute_mti(train_seqs, test_seqs, chunk_size=args.chunk_size)
            stats = mti_stats(mti)
            print(f"mean={stats['mean']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        rows.append({
            "protein":   protein,
            "split":     split_name,
            "n_train":   len(train_seqs),
            "n_test":    len(test_seqs),
            **{k: round(v, 6) for k, v in stats.items()},
        })

    if not rows:
        sys.exit("No results to write.")

    df = pd.DataFrame(rows)

    col_fmt = {
        "protein":  "{}",
        "split":    "{}",
        "n_train":  "{:>8,}",
        "n_test":   "{:>8,}",
        "mean":     "{:.4f}",
        "median":   "{:.4f}",
        "min":      "{:.4f}",
        "max":      "{:.4f}",
        "pct_gt90": "{:.1%}",
        "pct_gt95": "{:.1%}",
        "pct_gt99": "{:.1%}",
    }

    print("\n" + "=" * 100)
    print("MTI Summary (Maximum Training Identity — higher = test seqs more similar to train)")
    print("=" * 100)
    header_row = (
        f"{'protein':<16} {'split':<20} {'n_train':>8} {'n_test':>8}  "
        f"{'mean':>7} {'median':>7} {'min':>7} {'max':>7}  "
        f"{'%>90':>7} {'%>95':>7} {'%>99':>7}"
    )
    print(header_row)
    print("-" * 100)
    for _, row in df.iterrows():
        print(
            f"{row['protein']:<16} {row['split']:<20} {row['n_train']:>8,} {row['n_test']:>8,}  "
            f"{row['mean']:>7.4f} {row['median']:>7.4f} {row['min']:>7.4f} {row['max']:>7.4f}  "
            f"{row['pct_gt90']:>7.1%} {row['pct_gt95']:>7.1%} {row['pct_gt99']:>7.1%}"
        )
    print("=" * 100)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
