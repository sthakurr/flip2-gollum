# from __future__ import annotations
from gollum.featurization.deep import BaseNNFeaturizer
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import GammaPrior
from torch.optim.lr_scheduler import StepLR

from botorch.optim.fit import fit_gpytorch_mll_torch
import wandb
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from abc import ABC, abstractmethod
from gpytorch.means.mean import Mean
from gpytorch.module import Module

from typing import Union
import numpy as np
import torch
import gpytorch
import io
import itertools
import math
import os
import PIL.Image



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# global optimizer-step counter across all DeepGP fits (x-axis for grads/, lr/, loss/ panels)
_TRAIN_STEP = itertools.count()


def _fig_to_image(fig):
    # render via png buffer: wandb.Image(fig) assumes pyplot was imported
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return wandb.Image(PIL.Image.open(buf))


def pdist_hist(d, title):
    """Histogram of pairwise distances with mean/median marked, as a wandb.Image."""
    from matplotlib.figure import Figure  # no pyplot: headless-safe, no global figure state

    d = d.cpu().numpy()
    mean, median = d.mean(), np.median(d)
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    ax.hist(d, bins=50, color="0.7")
    ax.axvline(mean, color="tab:red", label=f"mean {mean:.3g}")
    ax.axvline(median, color="tab:blue", linestyle="--", label=f"median {median:.3g}")
    ax.set_xlabel("pairwise L2 distance")
    ax.set_title(title)
    ax.legend()
    return _fig_to_image(fig)


@torch.no_grad()
def kernel_blocks(base_kernel, train_emb, test_emb):
    """Base-kernel correlations (outputscale excluded, so in [0, 1]) per block; diagonals of square blocks dropped."""
    def off_diag(k):
        i, j = torch.triu_indices(*k.shape, offset=1)
        return k[i, j]

    k = lambda a, b: base_kernel(a, b).to_dense().cpu()
    return {
        "train-train": off_diag(k(train_emb, train_emb)),
        "train-test": k(train_emb, test_emb).flatten(),
        "test-test": off_diag(k(test_emb, test_emb)),
    }


def select_hypers(module, hypers):
    """Keep the hypers this module actually has.

    gpytorch registers constrained hypers under a `raw_` name, so `named_parameters()`
    reports `...raw_lengthscale` while `initialize()` takes `...lengthscale`. Matching the
    plain name against the parameter names silently drops every entry.
    """
    names = {name for name, _ in module.named_parameters()}

    def raw(key):
        prefix, _, attr = key.rpartition(".")
        return f"{prefix}.raw_{attr}" if prefix else f"raw_{attr}"

    return {
        k: torch.as_tensor(v)
        for k, v in hypers.items()
        if v is not None and (k in names or raw(k) in names)
    }


class ScalePerDimToBounds(torch.nn.Module):
    """Min-max scale each embedding dimension separately, where ScaleToBounds scales globally.

    gpytorch's ScaleToBounds divides the whole tensor by one scalar taken from its global min and
    max, so a few wide dimensions set the scale and the rest barely move the Euclidean distance the
    kernel sees. Scaling per dimension gives every dimension equal weight, which raises the typical
    pairwise distance and lowers lengthscale / distance.

    Mirrors the "local" branch of Chen/Fleck/Stuyver's chemical-BO code, which min-max normalizes
    each representation column. Two deliberate differences: theirs normalizes over the whole library
    in one pass, while this follows ScaleToBounds and fits on the training points, reusing those
    bounds at eval; and the bounds default to the (-1, 1) already used by the global scaler so that
    an A/B against it differs only in the axis. Theirs targets [0, 1], which halves every distance
    and so doubles the starting lengthscale / distance unless the lengthscale is data-anchored.

    Unlike ScaleToBounds this does not clamp at eval: a candidate outside the training range stays
    outside instead of being folded onto the boundary.
    """

    def __init__(self, lower_bound=-1.0, upper_bound=1.0, eps=1e-6):
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.eps = eps
        self.register_buffer("min_val", torch.zeros(1))
        self.register_buffer("max_val", torch.ones(1))

    def forward(self, x):
        if self.training:
            # botorch hands acquisition a 3-D batch, so reduce over every leading dim, not just 0
            flat = x.reshape(-1, x.shape[-1])
            self.min_val.data = flat.min(dim=0).values
            self.max_val.data = flat.max(dim=0).values
        # a dimension that is constant across the training points would divide by zero
        span = (self.max_val - self.min_val).clamp_min(self.eps)
        return (x - self.min_val) * (0.95 * (self.upper_bound - self.lower_bound) / span) + 0.95 * self.lower_bound


def spearman(a, b):
    """Rank correlation of two 1D tensors (ties broken arbitrarily)."""
    ra, rb = (x.argsort().argsort().double() for x in (a, b))
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def kernel_hist(blocks, title):
    """Overlaid density histograms of kernel values per block, means marked. Returns wandb.Image."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    # lower edge from data, not 0: in the near-flat regime all mass sits in a few [0, 1] bins
    bins = np.linspace(min(v.min().item() for v in blocks.values()), 1, 51)
    for (name, v), c in zip(blocks.items(), ["tab:blue", "tab:orange", "tab:green"]):
        v = v.numpy()
        # density: block sizes differ by orders of magnitude
        ax.hist(v, bins=bins, density=True, histtype="step", color=c, label=f"{name} (mean {v.mean():.3g})")
        ax.axvline(v.mean(), color=c, linestyle="--")
    ax.set_xlabel("k(x, x') / outputscale")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _fig_to_image(fig)


def tsne_plot(z, y, train_idx, new_idx, title):
    """2D t-SNE map coloured by y; train points and new suggestions marked. Returns wandb.Image."""
    from matplotlib.figure import Figure

    y = np.asarray(y).ravel()
    # indices may arrive as torch tensors / lists; a 1-elem tensor would index numpy as a scalar
    train_idx, new_idx = np.asarray(train_idx).ravel(), np.asarray(new_idx).ravel()
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, s=4, cmap="viridis", alpha=0.6, linewidths=0)
    ax.scatter(z[train_idx, 0], z[train_idx, 1], s=18, facecolors="none", edgecolors="k", linewidths=0.6, label="train")
    ax.scatter(z[new_idx, 0], z[new_idx, 1], s=80, marker="*", c="tab:red", edgecolors="k", linewidths=0.5, label="suggested")
    fig.colorbar(sc, ax=ax, label="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return _fig_to_image(fig)


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass



class GP(SurrogateModel, SingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 1.0,
        initial_lengthscale_val: float = 1.0,
        gp_lr: float = 0.2,
    ) -> None:

        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        hypers_to_use = select_hypers(self, hypers)

        self.initialize(**hypers_to_use)
        self.gp_lr = gp_lr
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        mll = mll.to(self.train_x)
        

        try:
            
            fit_gpytorch_mll(
                mll
            )

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(
        self, x, observation_noise=False, return_var=True, return_posterior=False
    ):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)
        return (
            posterior
            if return_posterior
            else (posterior.mean, posterior.variance) if return_var else posterior.mean
        )


class DeepGP(SurrogateModel, SingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 2.0,
        initial_lengthscale_val: float = 5,
        ft_lr: float = 0.002,
        gp_lr: float = 0.02,
        gp_step_lr: float = 0.95,
        wd: float = 1e-3,
        wd_llm: float = 1e-3,
        scale_embeddings: bool = False,
        embedding_norm: str = "global",
        embedding_norm_bounds: tuple = (-1.0, 1.0),
        dimension_aware_prior: bool = False,
        dimension_aware_prefactor: float = 0.4,
        dimension_aware_offset: float = 4.0,
        train_mll_additionally: bool = False,
        median_heuristic_init: bool = False,
        finetuning_model: Union[None, BaseNNFeaturizer] = None,
    ) -> None:

        tkwargs = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        }
        train_x = train_x.to(torch.float64)
        train_y = train_y.to(**tkwargs)

        # TODO device handling
        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.mean_module = mean_module
        self.covar_module = covar_module

        self.finetuning_model = finetuning_model
        self.finetuning_model = self.finetuning_model.to(**tkwargs)
        # Restore LLM weights to bfloat16 — float64 is only needed for the GP
        # kernel and projector, not the language model itself.
        if hasattr(self.finetuning_model, "llm") and hasattr(self.finetuning_model, "_llm_dtype"):
            self.finetuning_model.llm = self.finetuning_model.llm.to(dtype=self.finetuning_model._llm_dtype)
        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        hypers_to_use = select_hypers(self, hypers)
        self.initialize(**hypers_to_use)
        self.train_x = self.train_x.to(**tkwargs)
        self.train_y = self.train_y.to(**tkwargs)
        lo, hi = embedding_norm_bounds
        if embedding_norm == "global":
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(lo, hi)
        elif embedding_norm == "per_dim":
            self.scale_to_bounds = ScalePerDimToBounds(lo, hi)
        else:
            raise ValueError(f"embedding_norm must be 'global' or 'per_dim', got {embedding_norm!r}")
        self.embedding_norm = embedding_norm

        self.ft_lr = ft_lr
        self.gp_lr = gp_lr
        self.gp_step_lr = gp_step_lr
        self.wd = wd
        self.wd_llm = wd_llm
        self.scale_embeddings = scale_embeddings
        self.train_mll_additionally = train_mll_additionally
        self._step_stats = {}
        self._first_log = None  # first per-step log = pre-optimization state

        self.to_gpu()

        if dimension_aware_prior:
            self.apply_dimension_aware_prior(dimension_aware_prefactor, dimension_aware_offset)
        if median_heuristic_init:
            self.init_lengthscale_from_median()

    @torch.no_grad()
    def apply_dimension_aware_prior(self, prefactor=0.4, offset=4.0):
        """Gamma hyperpriors on lengthscale and outputscale, scaled by the embedding dimension.

        Form taken from Chen, Fleck & Stuyver's chemical-BO kernel factory:

            l_mean = prefactor * sqrt(D) + offset
            lengthscale ~ Gamma(2 * l_mean, 2.0)    # mean l_mean
            outputscale ~ Gamma(l_mean, 1.0)        # mean l_mean

        both initialized at l_mean. D is the width of what the kernel actually sees: the projector
        output when projection_dim is set, otherwise the featurizer's input_dim. Their D is instead
        the number of computational-representation columns of the search space, post-PCA, for a
        fixed featurization, and their prefactor and offset were fitted for that pipeline. Treat the
        defaults as a starting point rather than a transferable law.

        A prior only bites if the loss sees it: ExactMarginalLogLikelihood adds the log-prob of every
        registered prior, and DeepGP.fit builds its closure from that same mll, so it does.
        """
        projection_dim = getattr(self.finetuning_model, "projection_dim", 64)
        D = projection_dim or getattr(self.finetuning_model, "input_dim", 768)
        if not D:
            print("dimension-aware prior skipped: no embedding dimension on the finetuning model")
            return
        l_mean = prefactor * math.sqrt(D) + offset

        base_kernel = getattr(self.covar_module, "base_kernel", self.covar_module)
        if getattr(base_kernel, "has_lengthscale", False):
            base_kernel.register_prior(
                "lengthscale_prior", GammaPrior(2.0 * l_mean, 2.0),
                lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v),
            )
            base_kernel.initialize(lengthscale=l_mean)
        if hasattr(self.covar_module, "outputscale"):
            self.covar_module.register_prior(
                "outputscale_prior", GammaPrior(1.0 * l_mean, 1.0),
                lambda m: m.outputscale, lambda m, v: m._set_outputscale(v),
            )
            self.covar_module.initialize(outputscale=l_mean)
        print(f"dimension-aware prior: D={D}, l_mean={l_mean:.4g}")

    @torch.no_grad()
    def init_lengthscale_from_median(self):
        """Median heuristic: set the lengthscale to the median pairwise distance of the train embeddings.

        A fixed initial lengthscale is not comparable across datasets — the embedding point cloud is
        several times tighter on some than on others, so the same value starts the kernel in a different
        regime each time. Anchoring it to the data puts every dataset at lengthscale / distance = 1
        before the first optimizer step. Re-run on every DeepGP construction, i.e. once per BO iteration,
        which is also when the previously fitted lengthscale is discarded anyway.
        """
        if self.train_x.shape[0] < 2:
            return

        was_training = self.finetuning_model.training
        self.finetuning_model.eval()  # no LoRA dropout: this estimate should not be noisy
        self.scale_to_bounds.train()  # fills the min/max buffers, as the first fit forward would
        embeddings = self.finetuning_model(self.train_x)
        if self.scale_embeddings:
            embeddings = self.scale_to_bounds(embeddings)
        d = torch.pdist(embeddings).median()
        self.finetuning_model.train(was_training)

        if not torch.isfinite(d) or d <= 0:
            print(f"median-heuristic init skipped: median pairwise distance = {d}")
            return

        base_kernel = getattr(self.covar_module, "base_kernel", self.covar_module)
        base_kernel.initialize(lengthscale=d)
        print(f"median-heuristic lengthscale init: {d.item():.4g}")

    def forward(self, x):
        finetuned = self.finetuning_model(x)

        if self.scale_embeddings:
            finetuned = self.scale_to_bounds(finetuned)
        # Removed self.finetuned assignment — storing as instance attr caused
        # the tensor to accumulate on GPU across iterations (2C fix).

        if self.training and wandb.run is not None:
            # train-point embedding stats, picked up by gp_closure for per-step logging
            lo, hi = torch.aminmax(finetuned.detach())
            d = torch.pdist(finetuned.detach())  # euclidean, as seen by RBF/Matern
            self._step_stats = {
                "embed/train_min": lo.item(),
                "embed/train_max": hi.item(),
                "pdists/mean": d.mean().item(),
                "pdists/median": d.median().item(),
                "pdists/hist": pdist_hist(d, "train points"),
            }

        mean_x = self.mean_module(finetuned)
        covar_x = self.covar_module(finetuned)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def to_gpu(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.likelihood.to(device)
        self.finetuning_model.to(device)
        self.train_x = self.train_x.to(device)
        self.train_targets = self.train_targets.to(device)
        self.train_y = self.train_y.to(device)

    def fit(self):
        self.train()
        self.likelihood.train()
        self.finetuning_model.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}")
        mll.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        def gp_closure():
            self.optimizer.zero_grad()
            output = self(self.train_x)
            mll_loss = -mll(output, self.train_targets.squeeze())
            mll_loss.backward()
            grads = [p.grad for p in self.parameters() if p.requires_grad]
            if wandb.run is not None:
                # split llm vs gp: gp hypers are few scalars, would be drowned by llm grads
                groups = {
                    "llm": self.finetuning_model.parameters(),
                    "gp": [*self.covar_module.parameters(), *self.mean_module.parameters(), *self.likelihood.parameters()],
                }
                log = {
                    "train_step": next(_TRAIN_STEP),
                    **self._step_stats,
                    "lr/llm_lr": self.optimizer.param_groups[0]["lr"],
                    "lr/gp_lr": self.optimizer.param_groups[1]["lr"],
                    "lr/noise": self.likelihood.noise.mean().item(),
                    "lr/outputscale": getattr(self.covar_module, "outputscale", torch.tensor(1.0)).mean().item(),
                    # mean over dims so ARD kernels also reduce to one global value
                    "ls/lengthscale": getattr(self.covar_module, "base_kernel", self.covar_module).lengthscale.mean().item(),
                }
                if "pdists/median" in log:
                    # median heuristic check vs train pairs (what the mll sees): ~1 sane, ->0 overfit, >>1 flat
                    log["ls/ratio_train"] = log["ls/lengthscale"] / log["pdists/median"]
                for name, params in groups.items():
                    g = [p.grad.detach().flatten() for p in params if p.requires_grad and p.grad is not None]
                    if g:
                        g = torch.cat(g)
                        log[f"grads/{name}_max"] = g.max().item()
                        log[f"grads/{name}_min"] = g.min().item()
                        log[f"grads/{name}_mean"] = g.mean().item()
                        log[f"grads/{name}_abs_mean"] = g.abs().mean().item()
                        log[f"grads/{name}_norm"] = g.norm().item()
                wandb.log(log)
                if self._first_log is None:
                    self._first_log = log
            return mll_loss, grads

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": (
                        p for p in self.finetuning_model.parameters() if p.requires_grad
                    ),
                    "lr": self.ft_lr,
                    "weight_decay": self.wd_llm,
                },
                {"params": self.covar_module.parameters()},
                {"params": self.mean_module.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.gp_lr,
            weight_decay=self.wd,
        )
        
        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Total number of parameters: {total_params}")
        
        
        fit_gpytorch_mll(
            mll,
            closure=gp_closure,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"optimizer": self.optimizer, "scheduler": scheduler},
            
        )

        if self.train_mll_additionally:
            for param in self.finetuning_model.parameters():
                param.requires_grad = False
            fit_gpytorch_mll(mll)

    @torch.no_grad()
    def embed(self, x, chunk_size=512):
        """Embeddings as the kernel sees them (eval mode, post scale_to_bounds), on CPU, row-aligned with x."""
        self.eval()
        out = []
        for start in range(0, x.size(0), chunk_size):
            e = self.finetuning_model(x[start : start + chunk_size].to(self.train_x.device))
            if self.scale_embeddings:
                e = self.scale_to_bounds(e)
            out.append(e.float().cpu())
        return torch.cat(out)


    def predict(
        self, x, observation_noise=True, return_var=True, return_posterior=False
    ):
        torch.cuda.empty_cache()
        for param in self.finetuning_model.parameters():
            param.requires_grad = False

        self.eval()
        self.finetuning_model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)

        return (
            posterior
            if return_posterior
            else (posterior.mean, posterior.variance) if return_var else posterior.mean
        )


