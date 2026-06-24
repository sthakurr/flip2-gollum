from typing import Any, Dict, Optional
from gollum.data.utils import torch_delete_rows
from gollum.utils.config import instantiate_class
import torch
import warnings

class BotorchOptimizer:
    def __init__(
        self,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
        batch_strategy: str = "kriging",
        batch_size: int = 1,
        tkwargs: Optional[Dict[str, Any]] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
    ):

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

    def train_surrogate_model(self, train_x, train_y):
        with warnings.catch_warnings():
            self.surrogate_model = instantiate_class(
                self.surrogate_model_config,
                train_x=train_x,
                train_y=train_y,
            )
            self.surrogate_model.fit()

    def suggest_next_experiments(
        self,
        train_x,
        train_y,
        design_space,
    ):
        self.train_surrogate_model(train_x, train_y)

        additional_acq_function_params = self.update_acquisition_function_params(
            train_y
        )
        self.acquisition_function = instantiate_class(
            self.acq_function_config, **additional_acq_function_params
        )

        if self.batch_size == 1:
            best_point, _best_indices, _acq_values = self.optimize_acquisition_function(
                design_space
            )
            return [best_point]
        else:
            candidates, _indices, _acq_values = self.optimize_acquisition_function_batch(
                design_space,
            )
            return candidates

    def optimize_acquisition_function(
        self,
        design_space,
        chunk_size: int = 256,
    ):
        with torch.no_grad():
            X = design_space.unsqueeze(-2)
            acq_chunks = []
            for start in range(0, X.shape[0], chunk_size):
                chunk = X[start : start + chunk_size]
                acq_chunks.append(self.acquisition_function(chunk).reshape(-1))
            acq_values = torch.cat(acq_chunks, dim=0)
        best_indices = acq_values.topk(1)[1]
        best_point = X[best_indices].squeeze(1)
        return best_point, best_indices, acq_values

    def optimize_acquisition_function_batch(self, design_space):
        _, _, acq_values = self.optimize_acquisition_function(design_space)
        k = min(self.batch_size, acq_values.shape[0])
        top = acq_values.topk(k)
        X = design_space.unsqueeze(-2)
        candidates = [X[i] for i in top.indices]
        candidate_indices = top.indices.tolist()
        candidate_acq_values = top.values
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
