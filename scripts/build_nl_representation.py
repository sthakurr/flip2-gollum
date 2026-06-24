"""Build a natural-language mutation representation for FLIP2 datasets.

Each variant in a FLIP2 library differs from a per-dataset consensus wild-type by
only a handful of substitutions. This script describes each variant in plain
English — it spells out the substitutions and appends the full consensus WT
sequence — and writes the result to a new ``nl_mutation`` column. That column can
then be fed to the existing DeepGP + T5 pipeline (``representation: get_tokens``)
by pointing ``input_column`` at it; T5 embeds the prose and the deep kernel GP
learns from it.

The consensus is computed from the ``set=="train"`` rows ONLY, so the reference
frame never sees the test split (matching the leak-free behaviour of the
``mutation_context`` featurizer). The same train consensus is used to describe
both train and test variants.

Usage:
    python build_nl_representation.py data/flip2/alpha-amylase/one_to_many.csv
    python build_nl_representation.py <in.csv> [--out <out.csv>] \
        [--sequence-column sequence] [--split-column set]
"""
import argparse
import os

import pandas as pd

from gollum.featurization.mutation import _DESC, _consensus


def _mutations(seq, consensus):
    """Per-position substitutions of ``seq`` vs ``consensus`` as (pos1, wt, var)."""
    muts = []
    for p in range(min(len(seq), len(consensus))):
        wt, var = consensus[p], seq[p]
        if wt != var and wt in _DESC and var in _DESC:
            muts.append((p + 1, wt, var))  # 1-based position for readability
    return muts


def _describe(seq, consensus):
    """Render one variant as a natural-language description.

    Mutations come first (the discriminative part), the full WT sequence last.
    """
    muts = _mutations(seq, consensus)
    if not muts:
        return f"The wild-type sequence with no substitutions. The wild-type sequence is: {consensus}"
    clauses = "; ".join(
        f"{wt} at position {pos} is replaced by {var}" for pos, wt, var in muts
    )
    n = len(muts)
    return (
        f"Variant of the wild-type carrying {n} substitution(s): {clauses}. "
        f"The wild-type sequence is: {consensus}"
    )


def build(in_path, out_path, sequence_column, split_column):
    df = pd.read_csv(in_path)
    if sequence_column not in df.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in {in_path}. Columns: {list(df.columns)}"
        )

    if split_column in df.columns:
        train_seqs = df.loc[df[split_column] == "train", sequence_column].tolist()
        if not train_seqs:
            raise ValueError(
                f"No rows with {split_column}=='train' in {in_path}; cannot build a "
                "leak-free consensus."
            )
    else:
        # No split column: fall back to the whole set (e.g. a *_train.csv file).
        print(
            f"warning: no '{split_column}' column in {in_path}; computing consensus "
            "over ALL rows."
        )
        train_seqs = df[sequence_column].tolist()

    consensus = _consensus(train_seqs)
    df["nl_mutation"] = [_describe(s, consensus) for s in df[sequence_column]]

    df.to_csv(out_path, index=False)
    n_zero = sum(s == consensus for s in df[sequence_column])
    print(f"consensus length: {len(consensus)} (from {len(train_seqs)} train rows)")
    print(f"wrote {len(df)} rows ({n_zero} zero-mutation) -> {out_path}")
    print("example nl_mutation:")
    print(" ", df["nl_mutation"].iloc[0][:300], "...")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_path", help="Input CSV (with sequence/target/set columns).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <in>_nl.csv alongside the input).",
    )
    ap.add_argument("--sequence-column", default="sequence")
    ap.add_argument("--split-column", default="set")
    args = ap.parse_args()

    out_path = args.out
    if out_path is None:
        root, ext = os.path.splitext(args.in_path)
        out_path = f"{root}_nl{ext}"

    build(args.in_path, out_path, args.sequence_column, args.split_column)


if __name__ == "__main__":
    main()
