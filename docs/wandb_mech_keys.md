# W&B keys: DeepGP optimization diagnostics

Reference for the diagnostic panels logged by `train.py` and `src/gollum/surrogate_models/gp.py`.

## X-axes

| axis | meaning | set where |
|---|---|---|
| `train_step` | global optimizer step, counted across all DeepGP fits (never resets) | `_TRAIN_STEP` in `gp.py` |
| `epoch` | BO iteration `i`; `-1` = campaign start (before any optimization) | `train.py` loop |

Axes are bound in `train.py` right after `wandb.init` via `run.define_metric`.
`pdists/` and `ls/` contain keys on **both** axes, so their keys are listed explicitly (no globs);
a new key under those prefixes must be added to the right list by name, otherwise it falls back to W&B's `_step`.

## Sources

- **per-step** keys come from `gp_closure` in `DeepGP.fit`, on the **training points**, using the
  **training-mode** forward pass (LoRA dropout active). Values are read *before* that step's `optimizer.step()`.
- **per-epoch** keys come from `train.py` after each `suggest_next_experiments`, using the just-fitted model.
  Embeddings come from `DeepGP.embed(dm.x)`: eval mode, after `scale_to_bounds`, row-aligned with `dm.x`.
  "train" = `dm.train_indexes` (points the model was fit on), "test" = `dm.heldout_indices` (candidates).
- Distances are Euclidean (what RBF/Matérn use). Lengthscale = mean over dims (exact for isotropic kernels,
  rough for ARD). Kernel values use the base kernel only, i.e. `k / outputscale`, in [0, 1].

## Per step (`train_step`)

| key | definition | read as |
|---|---|---|
| `grads/{llm,gp}_max`, `_min`, `_mean` | max / min / mean of all gradient entries in the group | `llm` = finetuning model (LoRA + projector); `gp` = kernel, mean, likelihood |
| `grads/{llm,gp}_abs_mean` | mean of \|grad\| | typical gradient size (signed mean ≈ 0 for big groups) |
| `grads/{llm,gp}_norm` | L2 norm of the group's gradient | exploding / vanishing gradients |
| `lr/llm_lr`, `lr/gp_lr` | current learning rate of param group 0 / 1 | StepLR γ = 0.95 **per step**: lr ≈ 0.6% of initial after 100 steps |
| `lr/noise` | likelihood noise σ² (standardized-y units) | ≈ 1: everything explained as noise; ≈ floor (1e-4): interpolating |
| `lr/outputscale` | kernel outputscale s (standardized-y units) | signal variance; noise / s = noise-to-signal |
| `ls/lengthscale` | kernel lengthscale ℓ | absolute value is meaningless alone (see caveats) |
| `ls/ratio_train` | ℓ / `pdists/median` | ~1 sane; ≫1 kernel flat over train set; →0 points decorrelated |
| `embed/train_min`, `embed/train_max` | min / max entry of the train embeddings | range of the feature space the kernel sees |
| `pdists/mean`, `pdists/median` | mean / median pairwise distance between train embeddings | shrinking within a fit = fine-tuning compresses the training points |
| `pdists/hist` | image: histogram of those distances, mean (red) + median (blue dashed) | shape (e.g. bimodality) the median hides |

## Per epoch (`epoch`)

### End-of-fit curves (in `pdists/`, `ls/`)

| key | definition | notes |
|---|---|---|
| `pdists/mean_by_epoch`, `pdists/median_by_epoch` | train pairwise distances after the fit (eval-mode embeddings) | epoch -1 = first step of first fit (pretrained embeddings). Can differ slightly from the last per-step value (dropout vs eval) |
| `ls/lengthscale_by_epoch` | fitted ℓ | epoch -1 = initial ℓ |
| `ls/ratio_train_by_epoch` | ℓ / train median distance | epoch -1 available |
| `ls/ratio_test_by_epoch` | ℓ / candidate-candidate median distance | no epoch -1 (candidates are never embedded before the first fit) |
| `ls/ratio_by_max_pdist` | ℓ / max pairwise distance over the whole set | ≥ 1: even the farthest pair keeps k ≥ 0.52. Driven by a single pair (outlier-sensitive) |

### Whole set (`embed_all/`, `pdists_all/`, `ls_all/`)

| key | definition |
|---|---|
| `embed_all/min`, `embed_all/max` | min / max entry of all embeddings (train + candidates) |
| `pdists_all/mean`, `pdists_all/median`, `pdists_all/hist` | pairwise distances over the whole set (all N(N-1)/2 pairs, no subsampling) |
| `ls_all/ratio` | ℓ / whole-set median distance. Mixes train-train, train-test, test-test pairs; prefer the per-block keys |

### Kernel (`kernel/`)

Blocks: `train-train` and `test-test` (off-diagonal, upper triangle), `train-test` (all pairs).

| key | definition | read as |
|---|---|---|
| `kernel/{block}_mean`, `kernel/{block}_var` | mean / variance of k / outputscale over the block | mean near 1 + small var = near-flat kernel |
| `kernel/hist` | image: overlaid density histograms of the three blocks, means dashed. Lower x-limit = smallest observed value (changes per iteration) | spike at k ≈ 1 = near-duplicate points |
| `kernel/{block}_dy_spearman` | Spearman(k, \|y_i − y_j\|) over the block's pairs; test uses true yields (diagnostic only) | < 0: kernel-similar pairs have similar yields; ≈ 0: kernel is yield-blind. Train ≈ −1 with test ≈ 0 = learned metric doesn't generalize (overfitting). Ties broken arbitrarily → slight pull toward 0 |

### Candidate posterior (`post/`)

The GP prediction over all candidates (`dm.heldout_x`), latent f without observation noise (what analytic EI uses),
in yield units (outcome transform undone). Computed via `model.posterior` in chunks of 512.

| key | definition | read as |
|---|---|---|
| `post/std_median`, `post/std_min`, `post/std_max` | posterior σ(x\*) over candidates | uncertainty contrast available to acquisition |
| `post/std_rel_median` | median σ / prior σ, with prior σ = √outputscale · std(train y) | 1 = model knows nothing at candidates; → 0 = thinks it knows everything |
| `post/mean_std` | std of the posterior mean μ(x\*) across candidates | ≈ 0 = flat mean, candidates nearly tied |
| `post/mean_y_spearman` | Spearman(μ, true y) over candidates (diagnostic only) | does the model rank candidates correctly; compare with `kernel/train-train_dy_spearman` |
| `post/ei_z_chosen` | z = (μ − f\*) / σ at the chosen candidate, f\* = best train y | z > 0: chosen because mean beats incumbent (exploitation); z ≪ 0: chosen on uncertainty (exploration) |

`ei_z_chosen` replaces the planned "explore fraction" σφ(z) / EI: that ratio exceeds 1 whenever μ < f\*
(the mean term (μ − f\*)Φ(z) is negative), so it is not a fraction. z carries the same information.

### Candidate calibration (`cand/`)

Same candidate posterior as `post/`, but **with observation noise** and scored against the true held-out
yields, via `calculate_model_fit_metrics`. Built as a *diagonal* MultivariateNormal: the pool is embedded
in chunks so only marginals exist, and per-point predictive density is the quantity that answers
"is sigma honest" — a joint NLPD would fold in the correlation structure too.

| key | definition | read as |
|---|---|---|
| `cand/mse`, `cand/mae`, `cand/r2` | point-prediction error of μ over candidates | depends on μ only; blind to σ |
| `cand/nlpd` | mean per-point negative log predictive density | the σ-sensitive one. Same μ with σ 8× too small moved it 3.5 → 33.6 in a synthetic check |
| `cand/msll` | the same, minus the loss of a trivial predict-the-training-mean model | < 0 beats that baseline; ≈ 0 means the model adds nothing; ≫ 0 is actively worse than guessing the mean |
| `cand/qce` | \|fraction of y inside the 95% interval − 0.95\| | 0 = honest intervals. Grows toward 0.95 as σ collapses; a *too-wide* σ saturates near 0.05 |

`r2` and `nlpd` here are the quantities the gollum supplementary table reports per benchmark, so these
curves are directly comparable to it. Note `r2` can stay flat while `nlpd` blows up — that split is the
whole point of logging both.

### Other

| key | definition |
|---|---|
| `tsne/map` | image: t-SNE of whole-set embeddings, coloured by y; circles = train, red star = suggested. Warm-started from the previous map (rescaled to std 1e-4) so layouts stay comparable; `random_state = seed`. Neighbourhoods only, distances are not metric |
| `top5pct_coverage`, `train/best_so_far`, `top_{n}_count`, `quantile_{q}_count`, `evaluated_suggestions` | pre-existing BO metrics (`gollum/metrics/data_metrics.py`) |
| `distances/{avg,hh_mean,hl_mean,ll_mean}` | pre-existing: distances between high/low-scoring train points on the **raw input features** `dm.train_x`, not the learned embeddings |

## Reading aids

Matérn-5/2 correlation k(r) at r = d / ℓ:

| r | 0.2 | 0.4 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| k | 0.97 | 0.88 | 0.52 | 0.14 | 0.03 |

Caveats:
- **Scale is not identifiable**: with a learned projector, shrinking all embeddings by c ≡ multiplying ℓ by 1/c.
  Absolute distances and ℓ alone mean little; use ratios and kernel values.
- **LR schedule**: plateaus in per-step curves are mostly the per-step StepLR decay, not convergence.
  `gp_step_lr` is stored but unused (γ = 0.95 hardcoded in `DeepGP.fit`).
- **Frozen control**: `trainable: false` alone still trains the projector when `projection_dim` is set;
  use `projection_dim: null` for a fully frozen feature extractor.
- Image panels (`*/hist`, `tsne/map`) have their own step slider and may not follow the custom x-axis.
