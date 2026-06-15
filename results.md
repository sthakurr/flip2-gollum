# FLIP2 BO — experiments & findings (running log)

## Setup
- **Task:** two-phase Bayesian optimization on FLIP2. Phase 1 collects points by BO over
  the **train** split; Phase 2 runs BO over the held-aside **test** split, seeded by the
  Phase-1-collected points. GP refit each iteration on the accumulated set.
- **Primary metric:** `phase2/coverage_top5` = fraction of the test top-5% (≥ 95th pct of
  test target) acquired during Phase 2. Reported at the final Phase-2 epoch, 5 seeds.
- **Datasets so far:** `alpha-amylase/far_to_close`, `ired/two_to_many` (also `one_to_many`).
- **Phase-2 seed-source baselines** wired (`phase1_bo` | `random_train` | `all_train` |
  `none`) to isolate the value of BO collection — **runs pending**.

## Representations implemented
one-hot; ESM2 dense / PCA; ESM-C dense / PCA / SAE (scaffolded); t5; DeepGP/pllm variants;
**mutation_context** (structured substitution descriptors vs per-dataset consensus, optional
ESM2 Δ-embedding). Kernel selector: `matern_plain | matern_hvarfner | matern_stuyver`.

## Dataset diagnostics (consensus & mutation depth)
Consensus = per-position majority residue. **Split-invariant** (train==test==all, 0 diffs)
for every split checked; sequences fixed-length (aa 425, ired 290) → positions align.

| dataset / split | mut-depth vs consensus (train) | (test) | regime |
|---|---|---|---|
| aa / one_to_many | 0–8 (med 3) | 0–8 | balanced depth |
| ired / two_to_many | **0–2** | **3–15** (med 4) | **depth-extrapolation** |

## Key results — mutation context

**Δ-embedding + structured MC vs plain ESM2** (the real ablation; `coverage_top5`, 5 seeds):

| dataset | `esm2_dense` (no MC) | `esm2_dense_mc` (Δ + structured) | Δ |
|---|---|---|---|
| aa / far_to_close | 0.62 | **0.76** | **+0.14** (helps, no CI overlap) |
| ired / two_to_many | **0.23** | 0.17 | −0.06 (hurts, no CI overlap) |

**Aggregation ablation on ired** (`esm2_dense_mc`): `mean` 0.177 vs `sum` 0.171 →
**indistinguishable** (overlapping CIs). Aggregation is *not* the lever.

`fixed_mc` (structured-only, 24-dim) ≪ `esm2_dense_mc` on both datasets — i.e. ESM2 ≫ 24
hand-crafted descriptors (expected; not the question of interest).

## Current interpretation
The win/loss tracks **train→test mutation-depth shift**, and the likely cause is the
**reference-relative framing**: both MC channels (structured descriptors *and* the ESM2
Δ-embedding) are defined against the shallow consensus.
- **aa (balanced depth):** reference-relative *concentrates* the mutation signal → helps.
- **ired (deep test):** reference-relative features must *extrapolate against a far anchor*
  → hurts; the **absolute** ESM2 embedding has no reference and stays robust.
- Aggregation (sum/mean) doesn't matter because both remain reference-relative.

**Tentative finding:** *reference-relative mutation features help on shallow/balanced
libraries but degrade on depth-extrapolation splits, where absolute embeddings win — a
representation × split-type interaction.* (n = 2 datasets; do not over-generalize.)

## Pending / next
1. **2×2 decomposition on ired** to confirm the cause (`mc_structured` × `mc_delta`):

   | arm | mc_structured | mc_delta | channels |
   |---|---|---|---|
   | esm2_dense (0.23) | — | — | plain ESM2 |
   | **Δ-only** (decisive) | false | true | ESM2 Δ |
   | plain + structured | true | false | plain ESM2 ‖ structured |
   | Δ + structured (0.17) | true | true | ESM2 Δ ‖ structured |

   Δ-only ≈ 0.17 → reference-relative framing confirmed as culprit; ≈ 0.23 → blame structured.
2. Seed-source baselines (`random_train` / `all_train` / `none`) — does Phase-1 *BO collection*
   beat random/all train data as a Phase-2 seed?
3. Per-seed CIs reported for all arms; confirm aa gain and ired loss hold across seeds.

## Notes / caveats
- Phase-2 batch acquisition is **greedy top-k** (no diversity — the kriging fantasy is inert).
- Consensus computed over the full file = train-only (split-invariant) → no leakage.
- ESM-C SAE arm (TopK on ESM-C 6B) scaffolded but not validated (needs the real loader).
