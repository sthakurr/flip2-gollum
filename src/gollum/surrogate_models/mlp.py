"""MLP surrogates: an MLP head on the (frozen) FM embedding, used either as a
single greedy regressor or as a deep ensemble whose spread across heads is the
predictive uncertainty.

The BO loop instantiates the surrogate from its config once per round
(``BotorchOptimizer.train_surrogate_model``), so the heads, the AdamW moments and
any LR state are dropped and rebuilt at every round by construction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch.nn import init

from gollum.surrogate_models.gp import SurrogateModel

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
MIN_VAR = 1e-9  # keeps the posterior MVN valid when the heads agree exactly


def _make_head(input_dim, hidden_features, num_layers, activation):
    """num_layers hidden blocks + a scalar output layer (so num_layers=1 is a
    2-layer MLP)."""
    layers = []
    d = input_dim
    for _ in range(num_layers):
        layers += [nn.Linear(d, hidden_features), ACTIVATIONS[activation]()]
        d = hidden_features
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _init_head(head, seed=None):
    """Kaiming-uniform init of the head's linear weights (zero bias). With a
    ``seed`` the draw happens under a forked CPU RNG, so per-head diversity is
    reproducible and the global RNG stream (batch shuffling) is untouched."""
    def _init():
        for m in head.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
                init.zeros_(m.bias)

    if seed is None:
        _init()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _init()


class MLP(SurrogateModel, Model):
    """MLP head(s) on the raw embedding, trained by minibatch AdamW on MSE
    against standardized targets.

    ``n_heads=1`` is the greedy arm (use with PosteriorMean); ``n_heads=5`` with
    ``seed_heads=True`` is the deep ensemble, whose heads are trained jointly
    with a summed MSE loss on the same labeled set (no bootstrapping) and differ
    only through their initialization.
    """

    @property
    def num_outputs(self) -> int:
        return 1

    def __init__(
        self,
        train_x: torch.Tensor = None,
        train_y: torch.Tensor = None,
        hidden_features: int = 128,
        num_layers: int = 1,
        activation: str = "elu",
        n_heads: int = 1,
        seed_heads: bool = False,
        head_seed_stride: int = 1337,
        lr: float = 3e-4,
        wd: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 3000,
        patience: int = 100,
        clip_grad: float = 1.0,
    ) -> None:
        super().__init__()
        tkwargs = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        }
        self.train_x = train_x.to(**tkwargs)
        self.train_y = train_y.to(**tkwargs)

        # Targets are standardized for training; predictions are de-standardized
        # back to the raw scale the acquisition function (and best_f) live in.
        self.y_mean = self.train_y.mean(dim=0)
        self.y_std = self.train_y.std(dim=0).clamp_min(1e-6)

        self.heads = nn.ModuleList(
            [
                _make_head(
                    self.train_x.shape[-1], hidden_features, num_layers, activation
                )
                for _ in range(n_heads)
            ]
        )
        for k, head in enumerate(self.heads):
            _init_head(head, seed=(k + 1) * head_seed_stride if seed_heads else None)

        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.clip_grad = clip_grad

        self.to(**tkwargs)

    def fit(self):
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        targets = (self.train_y - self.y_mean) / self.y_std
        n = self.train_x.shape[0]
        best_loss, n_bad, epoch = float("inf"), 0, 0

        for epoch in range(self.max_epochs):
            perm = torch.randperm(n, device=self.train_x.device)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = self.train_x[idx], targets[idx]
                optimizer.zero_grad()
                loss = sum(F.mse_loss(head(xb), yb) for head in self.heads)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
                optimizer.step()
                epoch_loss += loss.item() * idx.numel()
            epoch_loss /= n

            if epoch_loss < best_loss:
                best_loss, n_bad = epoch_loss, 0
            else:
                n_bad += 1
                if n_bad >= self.patience:
                    break

        self.eval()
        print(
            f"[fit] MLP({len(self.heads)} heads) stopped at epoch {epoch + 1}, "
            f"train loss {epoch_loss:.6f} (best {best_loss:.6f})",
            flush=True,
        )
        if wandb.run is not None:
            wandb.log({"mlp/train_loss": epoch_loss, "mlp/epochs": epoch + 1})

    def head_means(self, x):
        """Per-head predictive means on the raw target scale, shape (n_heads, *x
        batch shape)."""
        preds = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=0)
        return preds * self.y_std + self.y_mean

    def posterior(
        self,
        X: torch.Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
    ):
        preds = self.head_means(X)
        mean = preds.mean(dim=0)
        var = (
            preds.var(dim=0).clamp_min(MIN_VAR)
            if len(self.heads) > 1
            else torch.full_like(mean, MIN_VAR)
        )
        posterior = GPyTorchPosterior(MultivariateNormal(mean, DiagLinearOperator(var)))
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def predict(
        self, x, observation_noise=False, return_var=True, return_posterior=False
    ):
        self.eval()
        with torch.no_grad():
            posterior = self.posterior(x)
        return (
            posterior
            if return_posterior
            else (posterior.mean, posterior.variance) if return_var else posterior.mean
        )


class EnsembleThompsonSampling(AcquisitionFunction):
    """Thompson sampling over the ensemble heads.

    Each batch slot draws one head index uniformly and takes the candidate with
    the largest predicted mean under that head; already-selected candidates are
    excluded so a batch of k yields k distinct points. The batch is selected in
    ``select_batch`` (BotorchOptimizer dispatches to it); ``forward`` is the
    single-point fallback, one head draw per call.
    """

    def __init__(self, model: MLP) -> None:
        super().__init__(model=model)

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(len(self.model.heads), (1,)).item())
        return self.model.head_means(X)[k].squeeze(-1)

    def select_batch(self, design_space: torch.Tensor, batch_size: int):
        means = self.model.head_means(design_space)  # (n_heads, N)
        n = means.shape[-1]
        heads = torch.randint(len(self.model.heads), (min(batch_size, n),))
        taken = torch.zeros(n, dtype=torch.bool, device=means.device)
        indices, values = [], []
        for k in heads.tolist():
            masked = means[k].masked_fill(taken, -float("inf"))
            i = int(masked.argmax())
            taken[i] = True
            indices.append(i)
            values.append(means[k, i])
        return indices, torch.stack(values)
