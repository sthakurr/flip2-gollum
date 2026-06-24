# Representation comparison for two-phase BO on FLIP2

This doc captures **why** we built the representation-comparison harness, the
reasoning behind the design choices, **what** was added to the codebase, and
**how** to run it.

---

## 1. Background — the question

We're planning a two-phase Bayesian Optimization experiment on FLIP2
alpha-amylase (`data/flip2/alpha-amylase/one_to_many.csv`):

- **Phase 1** — start from 96 train points (`exclude_top=True`) and run BO for 3
  iterations × 96, **acquiring from the remaining train pool** → 96 + 3×96 = 384
  collected train points. Fit the GP on those and **evaluate model metrics on the
  held-aside test split** (the gate).
- **Phase 2** (future) — use that GP to discover the high-performing enzymes in
  the **test** split.

The question: **one-hot encoding vs ESM2 embeddings — which representation
transfers better from train → test?** (with ESM-C dense and ESM-C SAE features
added as further arms.)

### What the distance plots told us

We compared pairwise Euclidean-distance distributions (train-train, test-test,
train-test) in both spaces:

- **ESM2** — all three distributions are continuous, smooth, and overlap at the
  same scale. The test set sits *in-distribution* relative to train, so a GP
  with a smooth kernel can interpolate; test-test distances retain a real spread
  (≈0–0.35) → discriminative resolution within the test set.
- **One-hot** — test-test distances **collapse to a single spike at ~2.0**
  (≈ all test sequences differ by ~2 positions), i.e. almost **no within-test
  resolution**, and train-test distances are larger/quantized → the test set is
  a distinct cluster across a gap from train.

**Conclusion:** ESM2 is expected to transfer better. One-hot's collapsed
test-test geometry means *no lengthscale choice can give within-test
resolution* — the surrogate can't rank test candidates, which is the whole point
of Phase 2. The harness is built to **quantify** this rather than assume it.

---

## 2. GP design reasoning (for the surrogate)

Decisions baked into the configs, from our discussion:

- **Standardize the outcomes (`y`)** — always, regardless of representation.
  Makes the outputscale/noise priors meaningful and the marginal likelihood
  well-conditioned. (`standardize: true` on every arm.)
- **Per-dimension input normalisation** for embeddings — the precondition that
  makes BoTorch's **dimension-scaled lengthscale prior** valid. The prior depends
  only on `d` *because* it assumes each feature is on a unit scale; min-max scaling
  each feature to [0, 1] (over the complete candidate set) is what makes "typical
  pairwise distance ≈ √(2d)" true.
  (`normalize_input: per_dim_normalisation`.)
- **PCA to the effective dimension** for ESM2 — embeddings have effective
  dimensionality ≪ 1280 (highly correlated dims), so the `√d` prior on the
  nominal 1280 over-estimates the lengthscale. PCA gives the prior an honest `d`
  and avoids overfitting 1280-dim ARD on ~288 points. (`reduce_dim: 128`.)
- **One-hot** — drop zero-variance columns first (constant positions dominate a
  combinatorial library and would break standardization), then per-dim
  standardize.
- **ESM-C SAE arm** — keep only **active & variable** SAE features and use a
  **sparse-ARD GP** (Gamma lengthscale prior, ARD over the feature axes) so the
  few biologically meaningful features dominate. This is "SAASBO-lite": MAP
  fitting via `fit_gpytorch_mll`, keeping the standard posterior interface that
  the acquisition functions and metrics expect (full NUTS SAASBO would break it).

> SAE caveat: "captures biology" (interpretability) ≠ "better predictor of *this*
> property". The SAE arm is the most experimental and is gated behind trained SAE
> weights; treat it as a hypothesis to test, not a default.

---

## 3. What was added

**New**
- `src/gollum/metrics/ranking.py` — `spearman`, `kendall`, `top_k_recovery`,
  `precision_at_k`, `calculate_ranking_metrics`, and `log_surrogate_eval`
  (combines these with the existing NLPD/MSLL/QCE/R² and logs to W&B).
- `src/gollum/featurization/protein.py` — `one_hot_encode_sequences` and
  `one_hot_matrix` (flatten + drop zero-variance columns). Single source of truth.
- `configs/flip2_arms/{onehot,esm2_dense,esm2_pca,esmc_dense,esmc_sae}.yaml` —
  one config per arm.
- `tests/test_ranking.py` — unit tests for the ranking metrics.
- `jobs/run_rep_comparison.sh` — SLURM sweep (5 arms × 5 seeds).

**Modified**
- `src/gollum/data/module.py`
  - `respect_split` / `split_column` — Phase-1 sample drawn only from `set==train`,
    held-out design space = `set==test` (`_split_by_set`). Falls back to the old
    behavior when unset.
  - `normalize_data` — added `per_dim_normalisation` (per-dim min-max to [0,1] using
    bounds over the **complete candidate set**) and optional PCA (`reduce_dim`, fit on
    **train rows only** to avoid test leakage).
  - `test_subsample` — optional cap on huge test design spaces.
- `src/gollum/featurization/base.py` — registered `onehot`, `get_esmc_embeddings`,
  `get_esmc_sae_features`; added `sae_weights_path` passthrough.
- `src/gollum/featurization/text.py` — `get_esmc_embeddings` and
  `get_esmc_sae_features` (ESM-C via the EvolutionaryScale `esm` SDK, on-disk
  caching, graceful ImportError/missing-weights errors).
- `src/gollum/surrogate_models/gp.py` — `SparseArdGP` (sparse-ARD GP for the SAE arm).
- `train.py` — `--mode {gate,bo,both}`; refactored the BO loop into `run_bo`;
  added `run_gate` (fit on Phase-1 train, rank the test split).
- `src/gollum/utils/analysis.py` — re-exports one-hot helpers from the new module.

### Data split (respect_split)
- Initial sample drawn from `set==train`.
- **BO design space = the *remaining* train rows** (Phase-1 collects from train).
- `set==test` rows are **held aside** (`dm.test_*`), used only for the gate eval.

### The three modes
- **`bo`** — run Phase-1 BO over the train pool (96 init + `n_iters`×`batch_size`
  collected), logging best-found over iterations. No test eval.
- **`gate`** — fit the surrogate on the current train set and evaluate how well it
  ranks the **held-aside test split** (Spearman/Kendall/NLPD/MSLL/QCE + top-k
  recovery / precision@k). On its own (no BO) it fits on just the 96 init points.
- **`both`** — the full intended experiment: run Phase-1 BO over train to collect
  384 points, **then** fit on those and evaluate on the test split.

---

## 4. How to run

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### Step 1 — unit tests (pure scipy/numpy, fastest)
```bash
python -m pytest tests/test_ranking.py -q
```

### Step 2 — first end-to-end test: the one-hot gate (no BO)
Lightest end-to-end path — no model downloads, exercises set-aware split,
per-dim standardize, GP fit, and ranking metrics on just the 96 init points.
```bash
export WANDB_MODE=offline      # avoid needing a wandb login for the smoke test
python train.py --config configs/flip2_arms/onehot.yaml --mode gate --seed 0
```
Expect:
- `respect_split: 96 initial train points, 3122 train design-space, 488 held-aside test points`
- `Gate: fitting on 96 train points, evaluating on 488 test points`
- `Gate metrics (train->test):` with `test/spearman`, `test/nlpd`,
  `test/recovery@{1,3,5,10}`, etc.

### Step 2b — the full Phase-1 experiment (BO collect → test metrics)
Collects 96 + 3×96 = 384 train points via BO, then evaluates on the test split:
```bash
python train.py --config configs/flip2_arms/onehot.yaml --mode both --seed 0
```

### Step 3 — compare against ESM2 (the hypothesis)
First run downloads + caches the ESM2 model to `embeddings_cache/`.
```bash
python train.py --config configs/flip2_arms/esm2_dense.yaml --mode gate --seed 0
python train.py --config configs/flip2_arms/esm2_pca.yaml   --mode gate --seed 0
```
Expectation: ESM2 arms beat one-hot on Spearman / NLPD.

### Step 4 — the seed sweep (all arms, mode=both)
```bash
sbatch jobs/run_rep_comparison.sh     # 5 arms × 5 seeds on the cluster (MODE=both)
```

### Notes
- `--mode` on the CLI overrides the `mode:` in each config (configs default to `gate`).
- `esmc_dense` / `esmc_sae` need the EvolutionaryScale `esm` package (and SAE
  weights via `sae_weights_path` for the SAE arm); otherwise they error clearly
  and the sweep skips them.
- Tune `initializer.n_clusters` (Phase-1 budget, default 288) and `bo.batch_size`
  / `n_iters` (Phase-2 discovery budget) per experiment.
- For very large test splits (e.g. TrpB ~228k) set `data.init_args.test_subsample`
  to cap the design space.
