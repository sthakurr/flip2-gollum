"""Protein-sequence featurizers that don't require a neural model."""

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def one_hot_encode_sequences(sequences):
    """Convert protein sequences to a one-hot tensor ``(N, L, 20)``.

    Each amino acid becomes a binary 20-vector; sequences are right-padded to the
    longest sequence. Unknown characters are left as all-zero.
    """
    aa_to_index = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    seq_length = max(len(seq) for seq in sequences)
    one_hot = np.zeros((len(sequences), seq_length, len(AMINO_ACIDS)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:seq_length]):
            if aa in aa_to_index:
                one_hot[i, j, aa_to_index[aa]] = 1.0
    return one_hot


def one_hot_matrix(sequences, drop_constant=True):
    """Flatten one-hot encoded sequences into a 2D feature matrix ``(N, L*20)``.

    When ``drop_constant`` is True, columns with zero variance (positions/amino
    acids that never vary across the input — which dominate combinatorial
    libraries) are removed. This avoids feeding constant features to the kernel
    and prevents division-by-zero if the matrix is later standardized.
    """
    sequences = sequences.tolist() if hasattr(sequences, "tolist") else list(sequences)
    flat = one_hot_encode_sequences(sequences).reshape(len(sequences), -1)
    if drop_constant:
        flat = flat[:, flat.std(axis=0) > 0]
    return flat.astype(np.float32)
