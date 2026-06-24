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


def _strip_priors(covar_module):
    """Remove all registered priors so they don't contribute to the MLL."""
    for m in covar_module.modules():
        if hasattr(m, "_priors"):
            m._priors.clear()


def _reset_to_small_init(covar_module, value: float = 1.0):
    """Reset lengthscale (and outputscale) to a small default, undoing the
    dimension-scaled init while leaving the kernel structure intact."""
    base = getattr(covar_module, "base_kernel", covar_module)
    base.lengthscale = value
    if hasattr(covar_module, "outputscale"):
        covar_module.outputscale = value


def build_covar_module(
    kernel: str, dim: int, apply_prior: bool = True, init_large: bool = True
):
    """Build a covariance module for the given one-word ``kernel`` and input ``dim``.

    Args:
        kernel: one of ``matern_plain`` | ``matern_hvarfner`` | ``matern_stuyver``.
        dim: dimensionality of the kernel inputs (``train_x.shape[-1]`` for a plain
            GP; the finetuning model's output dim ``ft_out_dim`` for a DeepGP).
        apply_prior: keep the dimension-aware prior (True) or strip it (False).
        init_large: keep the dimension-scaled large init (True) or reset to a
            small lengthscale (False). Together these two flags give the 2x2
            init-vs-prior ablation; defaults (True, True) = original behavior.
    """
    if kernel == "matern_plain":
        # Single shared lengthscale (no ARD), no dimension-aware prior.
        covar_module = ScaleKernel(MaternKernel(nu=2.5, initial_lengthscale=1.0, ard_num_dims=None))

    elif kernel == "matern_hvarfner":
        # ARD + LogNormal dimension-scaled lengthscale prior, no outputscale.
        covar_module = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim, use_rbf_kernel=False
        )

    elif kernel == "matern_stuyver":
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

    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'; choose one of {VALID_KERNELS}."
        )

    # Ablation toggles: disentangle the dimension-aware prior from its large init.
    if not apply_prior:
        _strip_priors(covar_module)
    if not init_large:
        _reset_to_small_init(covar_module)
    return covar_module
