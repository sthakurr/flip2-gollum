from typing import Any, Dict, Optional
from gollum.data.utils import torch_delete_rows
from gollum.utils.config import instantiate_class
from torch import Tensor
import torch
import warnings
import time
import logging

logger = logging.getLogger(__name__)

class BotorchOptimizer:
    def __init__(
        self,
        design_space: Optional[Tensor] = None,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
        batch_strategy: str = "kriging",
        batch_size: int = 1,
        tkwargs: Optional[Dict[str, Any]] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
    ):

        self.design_space = design_space
        self.surrogate_model_config = (
            surrogate_model_config or BotorchOptimizer.default_surrogate_model_config()
        )
        self.acq_function_config = (
            acq_function_config or BotorchOptimizer.default_acq_function_config()
        )
        self.acquisition_function = None
        self.batch_strategy = batch_strategy
        self.batch_size = batch_size

        self.tkwargs = tkwargs
        print("Using device:", self.tkwargs["device"])


    def lie_to_me(self, candidate, train_y, strategy="kriging"):
        supported_strategies = ["cl_min", "cl_mean", "cl_max", "kriging"]
        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of "
                + str(supported_strategies)
                + ", "
                + "got %s" % strategy
            )

        if strategy == "cl_min":
            y_lie = (
                torch.min(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-min lie
        elif strategy == "cl_mean":
            y_lie = (
                torch.mean(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-mean lie
        elif strategy == "cl_max":
            y_lie = (
                torch.max(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-max lie
        else:
            y_lie, _ = self.surrogate_model.predict(candidate)
        return y_lie

    def train_surrogate_model(self, train_x, train_y, epoch: int = 0):
        with warnings.catch_warnings():
            self.surrogate_model = instantiate_class(
                self.surrogate_model_config,
                train_x=train_x,
                train_y=train_y,
            )
            self.surrogate_model.fit(epoch=epoch)

    def suggest_next_experiments(
        self,
        train_x,
        train_y,
        design_space,
        epoch: int = 0,
    ):
        t_gp = time.perf_counter()
        self.train_surrogate_model(train_x, train_y, epoch=epoch)
        gp_time = time.perf_counter() - t_gp
        print(f"[BO] GP training: {gp_time:.1f}s (n_train={train_x.shape[0]})")

        t_acq = time.perf_counter()
        additional_acq_function_params = self.update_acquisition_function_params(
            train_y
        )
        self.acquisition_function = instantiate_class(
            self.acq_function_config, **additional_acq_function_params
        )
        print(f"[BO] Acq setup: {time.perf_counter() - t_acq:.1f}s")

        t_opt = time.perf_counter()
        if self.batch_size == 1:
            best_point, best_indices, _ = self.optimize_acquisition_function(design_space)
            result = [best_point], [best_indices.item()]
        else:
            candidates, candidate_indices, _ = self.optimize_acquisition_function_batch(
                train_x,
                train_y,
                design_space,
            )
            result = candidates, candidate_indices
        print(f"[BO] Acq optimization (batch={self.batch_size}, design_space={design_space.shape[0]}): {time.perf_counter() - t_opt:.1f}s")
        print(f"[BO] Total suggest_next: {time.perf_counter() - t_gp:.1f}s")
        return result

    def optimize_acquisition_function(
        self,
        design_space,
        chunk_size: int = 512,
    ):
        if design_space.size(0) == 0:
            raise ValueError("design_space is empty — all candidates have been evaluated.")

        # GPyTorch sq_dist broadcasts x2 - adjustment to (chunk, n_train, d).
        # For high-dim inputs (e.g. protein one-hot: d=8500) this can OOM at the
        # default chunk_size=512. Cap it so the intermediate stays under 4 GB.
        try:
            n_train = self.surrogate_model.train_inputs[0].shape[0]
            d = design_space.shape[-1]
            bytes_per_elem = design_space.element_size()
            safe_chunk = max(1, int(4 * 1024 ** 3 / (n_train * d * bytes_per_elem)))
            chunk_size = min(chunk_size, safe_chunk)
        except Exception:
            pass

        acq_chunks = []
        with torch.no_grad():
            for start in range(0, design_space.size(0), chunk_size):
                chunk = design_space[start : start + chunk_size].unsqueeze(-2)
                acq_chunks.append(self.acquisition_function(chunk).reshape(-1).cpu())
        acq_values = torch.cat(acq_chunks, dim=0)
        best_indices = acq_values.topk(1)[1]
        best_point = design_space[best_indices].squeeze(1)
        return best_point, best_indices, acq_values

    def optimize_acquisition_function_batch(self, train_x, train_y, design_space):

        if self.batch_strategy in ["kriging", "cl_min", "cl_mean", "cl_max"]:
            n_to_pick = min(self.batch_size, design_space.size(0))
            if n_to_pick < self.batch_size:
                print(
                    f"[BO] Warning: design space has {design_space.size(0)} candidates, "
                    f"fewer than batch_size={self.batch_size}. "
                    f"Picking {n_to_pick} instead."
                )

            # Evaluate the acquisition function once and take top-k.
            # Proper KB would retrain the GP with each hallucinated observation
            # to shift the posterior and encourage diversity, but retraining
            # batch_size times is prohibitively expensive with LLM-based GPs.
            # Since the surrogate model is never updated between picks in the
            # sequential loop, the loop is equivalent to top-k from a single
            # pass — so we do that directly.
            #
            # To restore sequential KB (with GP retraining per pick):
            #   for _ in range(n_to_pick):
            #       best_point, best_idx, acq_values = self.optimize_acquisition_function(design_space)
            #       y_lie = self.lie_to_me(best_point, train_y, strategy=self.batch_strategy)
            #       train_x = torch.cat([train_x, best_point])
            #       train_y = torch.cat([train_y, y_lie])
            #       self.train_surrogate_model(train_x, train_y)           # <-- missing step
            #       self.acquisition_function = instantiate_class(...)     # <-- rebuild acq
            #       original_pos = original_positions[best_idx].item()
            #       design_space = torch_delete_rows(design_space, best_idx)
            #       original_positions = torch_delete_rows(original_positions, best_idx)
            _, _, acq_values = self.optimize_acquisition_function(design_space)
            topk_indices = acq_values.topk(n_to_pick).indices

            original_positions = torch.arange(design_space.size(0))
            candidates = [design_space[idx].unsqueeze(0) for idx in topk_indices]
            candidate_indices = [original_positions[idx].item() for idx in topk_indices]
            candidate_acq_values = [acq_values]

        return candidates, candidate_indices, candidate_acq_values

    @staticmethod
    def default_surrogate_model_config():
        # return default surrogate model config
        return {
            "class_path": "gollum.surrogate_models.gp.GP",
            "init_args": {
                "covar_module": {
                    "class_path": "gpytorch.kernels.ScaleKernel",
                    "init_args": {
                        "base_kernel": {
                            "class_path": "gpytorch.kernels.MaternKernel",
                            "init_args": {"nu": 2.5},
                        }
                    },
                },
                "likelihood": {
                    "class_path": "gpytorch.likelihoods.GaussianLikelihood",
                    "init_args": {"noise": 1e-4},
                },
                "normalize": False,
                "initial_noise_val": 1.0e-4,
                "noise_constraint": 1.0e-05,
                "initial_outputscale_val": 1.0,
                "initial_lengthscale_val": 1.0,
            },
        }

    @staticmethod
    def default_acq_function_config():
        # return default acquisition function config
        return {
            "class_path": "botorch.acquisition.UpperConfidenceBound",
            "init_args": {"beta": 2.0, "maximize": True},
        }

    def update_acquisition_function_params(self, train_y):
        params = {"model": self.surrogate_model}
        if "ExpectedImprovement" in self.acq_function_config["class_path"]:
            params["best_f"] = train_y.max().item()
           

        return params
