"""Add the 'nlmut_1' representation column to a FLIP2 dataset CSV.

Each variant is described by its substitutions ONLY (no legend, no wild-type),
with each mutation written as space-separated tokens so the position is one token
and the residues are separate tokens (A42G -> "A 42 G"). Mutations are joined by
" , " (space-comma-space). The ``nlmut_1`` column is added IN PLACE to the given
CSV (which must contain a ``sequence`` column). Zero-mutation (wild-type) rows get
an empty string.

The consensus wild-type is computed from the ``set=="train"`` rows ONLY, so the
reference frame never sees the test split. Usage:

    python build_nlmut1_representation.py data/flip2/alpha-amylase/one_to_many.csv
"""
import argparse

import pandas as pd

from gollum.featurization.mutation import _DESC, _consensus


def _codes(seq, consensus):
    """Substitutions vs consensus as space-split codes like 'A 42 G' (1-based)."""
    return [
        f"{consensus[p]} {p + 1} {seq[p]}"
        for p in range(min(len(seq), len(consensus)))
        if seq[p] != consensus[p] and consensus[p] in _DESC and seq[p] in _DESC
    ]


def _describe(seq, consensus):
    """Render one variant: ' , '-joined space-split mutation codes."""
    return " , ".join(_codes(seq, consensus))


def build(in_path, out_path, sequence_column, split_column):
    df = pd.read_csv(in_path)
    if sequence_column not in df.columns:
        raise ValueError(
            f"Column '{sequence_column}' not in {in_path}. Columns: {list(df.columns)}"
        )

    # Consensus from train rows only (leak-free); fall back to all rows if no split.
    if split_column in df.columns:
        train_seqs = df.loc[df[split_column] == "train", sequence_column].tolist()
        if not train_seqs:
            raise ValueError(f"No {split_column}=='train' rows in {in_path}.")
    else:
        print(f"warning: no '{split_column}' column; using ALL rows for consensus.")
        train_seqs = df[sequence_column].tolist()

    consensus = _consensus(train_seqs)
    df["nlmut_1"] = [_describe(s, consensus) for s in df[sequence_column]]

    df.to_csv(out_path, index=False)
    n_zero = sum(s == consensus for s in df[sequence_column])
    print(f"consensus length: {len(consensus)} (from {len(train_seqs)} train rows)")
    print(f"wrote {len(df)} rows ({n_zero} zero-mutation) -> {out_path}")
    print("example nlmut_1:", repr(df["nlmut_1"].iloc[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_path", help="CSV to augment (must have a sequence column).")
    ap.add_argument("--out", default=None, help="Output CSV (default: in place).")
    ap.add_argument("--sequence-column", default="sequence")
    ap.add_argument("--split-column", default="set")
    args = ap.parse_args()

    # Default: write back to the same file, adding the nlmut_1 column in place.
    build(args.in_path, args.out or args.in_path, args.sequence_column, args.split_column)


if __name__ == "__main__":
    main()
