# from __future__ import annotations
import sys

from gollum.featurization.deep import BaseNNFeaturizer
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ConstantMean
from torch.optim.lr_scheduler import StepLR

from botorch.optim.fit import fit_gpytorch_mll_torch
import wandb
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from abc import ABC, abstractmethod
from gpytorch.means.mean import Mean
from gpytorch.module import Module
from gollum.surrogate_models.kernels import build_covar_module

from typing import Union
import numpy as np
import torch
import gpytorch
import os



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
        kernel: Union[str, None] = None,
        kernel_apply_prior: bool = False,
        kernel_init_large: bool = False,
    ) -> None:

        if covar_module is None and kernel is not None:
            covar_module = build_covar_module(
                kernel, dim=train_x.shape[-1],
                apply_prior=kernel_apply_prior, init_large=kernel_init_large,
            )

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

        hypers = {"likelihood.noise_covar.noise": torch.tensor(initial_noise_val)}
        if kernel is None:
            hypers["covar_module.base_kernel.lengthscale"] = torch.tensor(initial_lengthscale_val)
            hypers["covar_module.outputscale"] = torch.tensor(initial_outputscale_val)

        existing_parameters = {name for name, _ in self.named_parameters()}
        print(f"Existing parameters in the model: {existing_parameters}")
        hypers_to_use = {
            k: torch.tensor(v)
            for k, v in hypers.items()
            if k in existing_parameters and v is not None
        }

        self.initialize(**hypers_to_use)
        self.gp_lr = gp_lr
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        mll = mll.to(self.train_x)

        if wandb.run is not None:
            with torch.no_grad():
                kx = self.transform_inputs(self.train_x)
                wandb.log({
                    "embed/median_pairwise_dist": torch.pdist(kx).median().item(),
                    "fit_step": 0,
                })

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


class SparseArdGP(GP):
    """GP with a sparse axis-aligned (ARD) Matérn kernel, sized to the input
    dimension at construction time. Intended for the ESM-C SAE arm: a
    sparsity-promoting Gamma lengthscale prior pushes most per-dimension
    lengthscales large (irrelevant features) so the few biologically meaningful
    SAE features dominate — the SAASBO idea, but using MAP fitting via
    ``fit_gpytorch_mll`` so it keeps the standard posterior interface that the
    acquisition functions and ranking metrics expect.
    """

    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 1.0,
        initial_lengthscale_val: float = 1.0,
        gp_lr: float = 0.2,
        nu: float = 2.5,
        lengthscale_prior_concentration: float = 3.0,
        lengthscale_prior_rate: float = 6.0,
    ) -> None:
        from gpytorch.kernels import ScaleKernel, MaternKernel
        from gpytorch.priors import GammaPrior

        ard_num_dims = train_x.shape[-1]
        base_kernel = MaternKernel(
            nu=nu,
            ard_num_dims=ard_num_dims,
            lengthscale_prior=GammaPrior(
                lengthscale_prior_concentration, lengthscale_prior_rate
            ),
        )
        covar_module = ScaleKernel(base_kernel)
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            standardize=standardize,
            normalize=normalize,
            initial_noise_val=initial_noise_val,
            noise_constraint=noise_constraint,
            initial_outputscale_val=initial_outputscale_val,
            initial_lengthscale_val=initial_lengthscale_val,
            gp_lr=gp_lr,
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
        normalise_embeddings: bool = False,
        scale_embeddings: bool = False,
        minmax_embeddings: bool = False,
        train_mll_additionally: bool = False,
        finetuning_model: Union[None, BaseNNFeaturizer] = None,
        embedding_norm: str = "scale_to_bounds",
        kernel: Union[str, None] = None,
        kernel_apply_prior: bool = True,
        kernel_init_large: bool = True,
        max_fit_iter: int = 1000,
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

        ft_out_dim = (
            finetuning_model.projection_dim
            if getattr(finetuning_model, "projection_dim", None)
            else finetuning_model.input_dim
        )

        if covar_module is None:
            covar_module = build_covar_module(
                kernel or "matern_hvarfner", dim=ft_out_dim,
                apply_prior=kernel_apply_prior, init_large=kernel_init_large,
            )
        if mean_module is None:
            mean_module = ConstantMean()

        self.mean_module = mean_module
        self.covar_module = covar_module

        self.normalise_embeddings = normalise_embeddings
        self.minmax_embeddings = minmax_embeddings
        self.register_buffer("_embed_mean", None)
        self.register_buffer("_embed_std", None)
        self.register_buffer("_embed_min", None)
        self.register_buffer("_embed_range", None)

        self.finetuning_model = finetuning_model
        self.finetuning_model = self.finetuning_model.to(**tkwargs)
        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        # When a kernel string is provided, build_covar_module already sets lengthscale/outputscale to their dimension-aware values.
        hypers = {"likelihood.noise_covar.noise": torch.tensor(initial_noise_val)}
        if kernel is None:
            hypers["covar_module.base_kernel.lengthscale"] = torch.tensor(initial_lengthscale_val)
            hypers["covar_module.outputscale"] = torch.tensor(initial_outputscale_val)

        existing_parameters = {name for name, _ in self.named_parameters()}
        hypers_to_use = {
            k: torch.tensor(v)
            for k, v in hypers.items()
            if k in existing_parameters and v is not None
        }
        self.initialize(**hypers_to_use)
        self.train_x = self.train_x.to(**tkwargs)
        self.train_y = self.train_y.to(**tkwargs)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        self.ft_lr = ft_lr
        self.gp_lr = gp_lr
        self.gp_step_lr = gp_step_lr
        self.wd = wd
        self.wd_llm = wd_llm
        self.scale_embeddings = scale_embeddings
        self.train_mll_additionally = train_mll_additionally
        self.max_fit_iter = max_fit_iter

        if not (self.normalise_embeddings or self.scale_embeddings or self.minmax_embeddings):
            if (kernel or "matern_hvarfner") in ("matern_stuyver", "matern_hvarfner"):
                self.minmax_embeddings = True
            else:
                self.scale_embeddings = True

        self.to_gpu()

    def forward(self, x):
        finetuned = self.finetuning_model(x)
        if self.minmax_embeddings:
            # Per-dim min-max to [0,1]^d (unit hypercube) — the input scale the
            # dimension-aware priors (stuyver/hvarfner) assume. Fit on the train
            # batch, stored, reused at eval.
            if self.training:
                mn = finetuned.min(dim=0, keepdim=True).values
                rng = (finetuned.max(dim=0, keepdim=True).values - mn).clamp_min(1e-8)
                self._embed_min = mn.detach()
                self._embed_range = rng.detach()
            else:
                mn, rng = self._embed_min, self._embed_range
            finetuned = (finetuned - mn) / rng
        elif self.normalise_embeddings:
            if self.training:
                mean = finetuned.mean(dim=0, keepdim=True)
                std = finetuned.std(dim=0, keepdim=True).clamp_min(1e-8)
                self._embed_mean = mean.detach()
                self._embed_std = std.detach()
            else:
                mean, std = self._embed_mean, self._embed_std
            finetuned = (finetuned - mean) / std
        elif self.scale_embeddings:
            finetuned = self.scale_to_bounds(finetuned)
        self.finetuned = finetuned

        mean_x = self.mean_module(self.finetuned)
        covar_x = self.covar_module(self.finetuned)
        
        wandb.log({"lr/llm_lr": self.optimizer.param_groups[0]["lr"]})
        wandb.log({"lr/gp_lr": self.optimizer.param_groups[1]["lr"]})

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _kernel_input(self):
        """Projected + normalised train embeddings — the space the kernel sees."""
        emb = self.finetuning_model(self.train_x)
        if self.minmax_embeddings:
            mn = emb.min(dim=0, keepdim=True).values
            rng = (emb.max(dim=0, keepdim=True).values - mn).clamp_min(1e-8)
            emb = (emb - mn) / rng
        elif self.normalise_embeddings:
            mean = emb.mean(dim=0, keepdim=True)
            std = emb.std(dim=0, keepdim=True).clamp_min(1e-8)
            emb = (emb - mean) / std
        elif self.scale_embeddings:
            emb = self.scale_to_bounds(emb)
        return emb

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
        mll.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # --- DEBUG timing: per-step loss + wall-clock. Comment out when done. ---
        import time as _time
        _dbg = {"step": 0, "t0": _time.perf_counter(), "t_prev": _time.perf_counter()}

        def gp_closure():
            self.optimizer.zero_grad()
            output = self(self.train_x)
            mll_loss = -mll(output, self.train_targets.squeeze())
            mll_loss.backward()
            grads = [p.grad for p in self.parameters() if p.requires_grad]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _now = _time.perf_counter()
            _dbg["step"] += 1
            print(
                f"[fit] step {_dbg['step']:3d}  loss={mll_loss.item():.6f}  "
                f"dt={_now - _dbg['t_prev']:.3f}s  total={_now - _dbg['t0']:.2f}s",
                flush=True,
            )
            _dbg["t_prev"] = _now
            # Embedding spread this step (reuse forward's output); shrinking
            # across steps ⇒ feature collapse (Ober DKL pathology).
            # Lengthscale/outputscale track kernel saturation: collapse onset
            # should coincide with the lengthscale entering the large regime.
            base = getattr(self.covar_module, "base_kernel", self.covar_module)
            log = {
                "embed/median_pairwise_dist": torch.pdist(self.finetuned.detach()).median().item(),
                "kernel/lengthscale_mean": base.lengthscale.detach().mean().item(),
                "fit_step": _dbg["step"],
            }
            if hasattr(self.covar_module, "outputscale"):
                log["kernel/outputscale"] = self.covar_module.outputscale.detach().mean().item()
            wandb.log(log)
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

        # Baseline embedding spread before any training step (fit_step=0).
        with torch.no_grad():
            wandb.log({
                "embed/median_pairwise_dist": torch.pdist(self._kernel_input()).median().item(),
                "fit_step": 0,
            })

        # total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Total number of parameters: {total_params}")
        
        
        fit_gpytorch_mll(
            mll,
            closure=gp_closure,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={
                "optimizer": self.optimizer,
                "scheduler": scheduler,
                "step_limit": self.max_fit_iter,
            },
        )
        print(
            f"[fit] DONE  {_dbg['step']} steps in {_time.perf_counter() - _dbg['t0']:.2f}s "
            f"({(_time.perf_counter() - _dbg['t0']) / max(_dbg['step'], 1):.3f}s/step)  "
            f"trainable_params={total_params}",
            flush=True,
        )

        if self.train_mll_additionally:
            for param in self.finetuning_model.parameters():
                param.requires_grad = False
            fit_gpytorch_mll(mll)

        

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


