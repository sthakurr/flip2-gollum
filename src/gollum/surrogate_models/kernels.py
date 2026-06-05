"""Covariance-kernel factory for the GP surrogates.

A single entry point, :func:`build_covar_module`, builds one of three Matérn-5/2
kernel/prior regimes selected by a one-word config string. The kernel is always
sized to the *actual* kernel-input dimension ``dim`` (for DeepGP this is the
projection/embedding output dim, NOT the token input), so the dimension-aware
priors use the correct ``d``.

Options
-------
- ``"matern_plain"``     : ScaleKernel(Matérn-5/2), single lengthscale, NO ARD,
                           no dimension-aware prior (gpytorch defaults).
- ``"matern_hvarfner"``  : ARD Matérn-5/2 with the BoTorch dimension-scaled
                           LogNormal lengthscale prior (loc=√2+½·ln d, scale=√3),
                           NO outputscale (relies on Standardize(y)).
- ``"matern_stuyver"``   : ARD Matérn-5/2 with a dimension-aware Gamma hyperprior
                           (Chen/Fleck/Stuyver, ChemRxiv 10.26434/chemrxiv.10001986):
                           a single shared GammaPrior on the lengthscale across all
                           dims, plus a Gamma outputscale prior, with
                           l0 = o0 = 0.4·√d + 4.0. Hyperparameters are MAP-fit under
                           these fixed priors (no separate hyperprior learning).
"""
import math

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
)

VALID_KERNELS = ("matern_plain", "matern_hvarfner", "matern_stuyver")


def build_covar_module(kernel: str, dim: int):
    """Build a covariance module for the given one-word ``kernel`` and input ``dim``.

    Args:
        kernel: one of ``matern_plain`` | ``matern_hvarfner`` | ``matern_stuyver``.
        dim: dimensionality of the kernel inputs (``train_x.shape[-1]`` for a plain
            GP; the finetuning model's output dim ``ft_out_dim`` for a DeepGP).
    """
    if kernel == "matern_plain":
        # Single shared lengthscale (no ARD), no dimension-aware prior.
        return ScaleKernel(MaternKernel(nu=2.5))

    if kernel == "matern_hvarfner":
        # ARD + LogNormal dimension-scaled lengthscale prior, no outputscale.
        return get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim, use_rbf_kernel=False
        )

    if kernel == "matern_stuyver":
        # Dimension-aware Gamma hyperprior, shared across ARD lengthscale dims.
        # https://pubs.acs.org/doi/full/10.1021/acs.jctc.6c00251
        # taken from the repo https://github.com/chimie-paristech-CTM/HSF-ChemBO/blob/main/base/kernels.py
        l0 = 0.4 * math.sqrt(dim) + 4.0  # prior mean for both lengthscale and outputscale
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_prior=GammaPrior(concentration=2.0 * l0, rate=2.0),
        )
        base_kernel.lengthscale = l0  # initialize at the prior mean
        covar_module = ScaleKernel(
            base_kernel, outputscale_prior=GammaPrior(concentration=l0, rate=1.0)
        )
        covar_module.outputscale = l0  # initialize at the prior mean (o0 = l0)
        return covar_module

    raise ValueError(
        f"Unknown kernel '{kernel}'; choose one of {VALID_KERNELS}."
    )
