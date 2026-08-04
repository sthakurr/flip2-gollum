"""Runtime setup for the training entry point: warning/log suppression.

``configure_runtime`` must be called before the libraries that emit these
warnings do any work, so entry points call it before their heavy imports.
"""
import logging
import re
import warnings

import torch
from botorch.exceptions import InputDataWarning

pl_logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")


class IgnoreDeviceFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


def configure_runtime():
    warnings.filterwarnings("ignore", category=InputDataWarning)
    warnings.filterwarnings(
        "ignore",
        message="ExpectedImprovement has known numerical issues that lead to suboptimal optimization performance"
    )

    pl_logger.addFilter(IgnoreDeviceFilter())

    warnings.filterwarnings(
        "ignore",
        message=re.escape(
            "You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. "
            "To properly utilize them, you should set "
            "`torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision "
            "for performance. For more details, read "
            "https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html"
        ),
        category=UserWarning,
        module="torch",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=UserWarning,
        module="pytorch_lightning.trainer.connectors.data_connector",
    )

    torch.set_float32_matmul_precision("high")
