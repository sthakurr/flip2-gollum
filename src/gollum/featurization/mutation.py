"""Structured mutation-context featurizer.

For libraries that are mostly identical (1-8 mutations vs a reference), a
mean-pooled full-sequence embedding dilutes the mutation signal. This featurizer
encodes each variant *relative to a per-dataset consensus wild-type* as a
fixed-size, order-invariant vector built from physicochemical descriptors of the
substitutions — no neural model required (deterministic; feed a plain GP).

Per mutation (pos p, wild-type a -> variant b):
  [ sinusoidal(p) | desc(a) | desc(b) | desc(b)-desc(a) ]
summed over the (1-8) mutations (permutation-invariant), with the mutation count
appended. Optionally concatenate the ESM2 Delta-embedding
(embed(variant) - embed(consensus)) by passing ``model_name``.
"""
import math
from collections import Counter

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy, residue volume (A^3), charge @pH7, polar, aromatic.
_HYDRO = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
          "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
          "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
          "W": -0.9, "Y": -1.3}
_VOL = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
        "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
        "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
        "W": 227.8, "Y": 193.6}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
_POLAR = set("RNDCEQHKSTY")
_AROM = set("FWYH")
DESC_DIM = 5
POS_DIM = 8
PER_MUT_DIM = POS_DIM + 3 * DESC_DIM  # pos | from | to | delta


def _aa_desc(a):
    # Hydropathy/volume scaled to comparable magnitudes (per-dim standardization
    # downstream handles the rest).
    return np.array(
        [_HYDRO[a] / 5.0, _VOL[a] / 100.0, _CHARGE.get(a, 0.0),
         1.0 if a in _POLAR else 0.0, 1.0 if a in _AROM else 0.0],
        dtype=np.float64,
    )


_DESC = {a: _aa_desc(a) for a in AA}


def _pos_enc(p, dim=POS_DIM):
    pe = np.zeros(dim)
    for i in range(dim // 2):
        freq = 1.0 / (10000 ** (2 * i / dim))
        pe[2 * i] = math.sin(p * freq)
        pe[2 * i + 1] = math.cos(p * freq)
    return pe


def _consensus(seqs):
    """Per-position most-frequent amino acid across the dataset (the reference)."""
    length = max(len(s) for s in seqs)
    cons = []
    for p in range(length):
        col = Counter(s[p] for s in seqs if p < len(s) and s[p] in _DESC)
        cons.append(col.most_common(1)[0][0] if col else "A")
    return "".join(cons)


def _structured(seqs, consensus):
    feats = np.zeros((len(seqs), PER_MUT_DIM + 1), dtype=np.float64)  # +1 = count
    for i, s in enumerate(seqs):
        acc = np.zeros(PER_MUT_DIM)
        n_mut = 0
        for p in range(min(len(s), len(consensus))):
            a, b = consensus[p], s[p]
            if a != b and a in _DESC and b in _DESC:
                da, db = _DESC[a], _DESC[b]
                acc += np.concatenate([_pos_enc(p), da, db, db - da])
                n_mut += 1
        feats[i, :PER_MUT_DIM] = acc
        feats[i, -1] = n_mut
    return feats


def get_mutation_context_features(texts, model_name=None, pooling_method="average"):
    """Structured mutation-context features vs a per-dataset consensus wild-type.

    If ``model_name`` is given, the ESM2/PLM Delta-embedding
    (embed(variant) - embed(consensus)) is concatenated; otherwise the
    structured descriptors are returned alone (plain-GP ready).
    """
    seqs = texts.tolist() if hasattr(texts, "tolist") else list(texts)
    consensus = _consensus(seqs)
    structured = _structured(seqs, consensus)

    if model_name:
        from gollum.featurization.text import get_huggingface_embeddings

        emb = get_huggingface_embeddings(
            seqs, model_name=model_name, pooling_method=pooling_method
        )
        cons_emb = get_huggingface_embeddings(
            [consensus], model_name=model_name, pooling_method=pooling_method
        )
        delta = emb - cons_emb  # (N, D) - (1, D) broadcast
        feats = np.concatenate([structured, delta], axis=1)
    else:
        feats = structured

    print(
        f"mutation_context: structured dim={structured.shape[1]}, "
        f"total dim={feats.shape[1]} (model_name={model_name})"
    )
    return feats.astype(np.float32)
