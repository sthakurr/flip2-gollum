"""One-time ESM3 structural annotation of a wild-type sequence, giving the
reasoning LLM real per-residue structural context (secondary structure,
solvent exposure, prediction confidence) instead of asking it to guess.

Folding happens once per wild-type and is cached to disk -- a BO run acquires
many candidates that are all near-identical single/few-point mutants of the
same wild-type, so the wild-type's fold is used as the structural reference
for every mutated position rather than re-folding every candidate.
"""
import hashlib
import json
import os

STRUCTURE_CACHE_DIR = os.environ.get("GOLLUM_STRUCTURE_CACHE", "structure_cache")

# 8-state DSSP-style codes ESM3's secondary_structure track uses.
_SS_NAMES = {
    "H": "helix", "G": "helix", "I": "helix",
    "E": "strand", "B": "strand",
    "T": "turn", "S": "bend",
    "C": "coil",
}


def _cache_path(wild_type, model_name):
    h = hashlib.md5(f"{model_name}:{wild_type}".encode()).hexdigest()[:16]
    os.makedirs(STRUCTURE_CACHE_DIR, exist_ok=True)
    return os.path.join(STRUCTURE_CACHE_DIR, f"esm3_{h}.json")


def fold_wild_type(wild_type: str, model_name: str = "esm3_sm_open_v1", num_steps: int = 8) -> dict:
    """Predict per-residue secondary structure, solvent accessibility, and
    pLDDT confidence for `wild_type` via ESM3, caching the result to disk
    keyed by sequence hash. SASA is ESM3's raw per-residue value (absolute
    Ų, not normalized by residue type); `annotate_position` buckets it into
    buried/intermediate/exposed using this protein's own SASA distribution
    rather than a fixed physical cutoff.
    """
    cache = _cache_path(wild_type, model_name)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3.from_pretrained(model_name, device=device)

    protein = ESMProtein(sequence=wild_type)
    protein = model.generate(protein, GenerationConfig(track="structure", num_steps=num_steps))
    protein = model.generate(protein, GenerationConfig(track="secondary_structure", num_steps=1))
    protein = model.generate(protein, GenerationConfig(track="sasa", num_steps=1))

    sasa = [float(v) if v is not None else None for v in protein.sasa]
    sasa_values = sorted(v for v in sasa if v is not None)
    q33 = sasa_values[len(sasa_values) // 3]
    q66 = sasa_values[2 * len(sasa_values) // 3]

    structure = {
        "secondary_structure": protein.secondary_structure,
        "sasa": sasa,
        "sasa_q33": q33,
        "sasa_q66": q66,
        "plddt": protein.plddt.tolist() if protein.plddt is not None else None,
    }
    with open(cache, "w") as f:
        json.dump(structure, f)
    return structure


def annotate_position(position: int, structure: dict) -> str:
    """Human-readable structural context at a 1-indexed sequence position, or
    None if out of range / unavailable."""
    idx = position - 1
    ss = structure.get("secondary_structure")
    if ss is None or not (0 <= idx < len(ss)):
        return None
    parts = [_SS_NAMES.get(ss[idx], ss[idx])]

    sasa = structure.get("sasa")
    if sasa and idx < len(sasa) and sasa[idx] is not None:
        if sasa[idx] < structure["sasa_q33"]:
            parts.append("buried")
        elif sasa[idx] > structure["sasa_q66"]:
            parts.append("exposed")
        else:
            parts.append("intermediate exposure")

    plddt = structure.get("plddt")
    if plddt and idx < len(plddt):
        parts.append(f"pLDDT {plddt[idx] * 100:.0f}%")

    return ", ".join(parts)
