import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from typing import Optional, Union, List

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from gollum.data.dataset import SingleSampleDataset
from gollum.data.utils import torch_delete_rows
from gollum.initialization.initializers import BOInitializer
from gollum.data.utils import find_duplicates, find_nan_rows
from gollum.featurization.base import Featurizer
from sklearn.decomposition import PCA
from abc import ABC
import os
os.environ["OMP_NUM_THREADS"] = "28"


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    DataModule for BO on tabular reaction datasets. Expects a CSV with at least
    one column of reaction representations (e.g. SMILES) and one column of target
    values (e.g. yield). The initial BO sample is drawn from the whole dataset by 
    default, but optionally the initial sample can be drawn only from rows where
    a specified split column has the value "train" (and the held-out design space 
    is the remaining "train" rows, while the "test" rows are held aside for evaluation 
    of the fitted GP). The initial sample can optionally exclude the top performers, 
    to make the optimization more challenging. The data is featurized using a 
    user-specified featurizer (default: identity), and the input features can be 
    normalized using a variety of options. Optional PCA reduction is also supported. 
    The module provides train and validation dataloaders for the initial sample and 
    the held-out design space, respectively; the BO loop should call ``update_results`` 
    after each batch of experiments to update the training data with new observations.

    Arguments:
        data_path: Path to the input CSV file.
        input_column: Name(s) of the column(s) containing the reaction representations.
        target_column: Name of the column containing the target values.
        maximize: Whether the optimization objective is to maximize (True) or minimize (False) the target.
        init_sample_size: Number of initial samples to draw for BO.
        featurizer: Featurizer object to convert reaction representations into feature vectors.
        initializer: BOInitializer object to select the initial sample; if None, defaults to random sampling.
        exclude_top: Whether to exclude the top-performing reactions from the initial sample.
        normalize_input: Method for normalizing input features; options include "standard_scaling", "standard_scaling_per_dim", "l2_max_scaling", "l2_normalize", or "original" (no normalization).
        respect_split: Whether to respect a train/test split defined in the data; if True, the initial sample is drawn only from "train" rows and "test" rows are held aside for evaluation.
        split_column: Name of the column defining the train/test split; only used if respect_split is True.
        reduce_dim: Optional PCA reduction (int n_components or float variance fraction), fit on train rows to avoid test leakage; only applied if not None.
        test_subsample: Optional cap on the number of test points for evaluation when respecting the train/test split; only applied if respect_split is True and the test split is larger than this number.
    """
    def __init__(
        self,
        data_path: str,
        input_column: Union[str, List[str]] = "input",
        target_column: str = "target",
        maximize: bool = True,
        init_sample_size: int = 10,
        featurizer: Featurizer = Featurizer(),
        initializer: BOInitializer = None,
        exclude_top: bool = False,
        normalize_input: str = "standard_scaling",
        respect_split: bool = False,
        split_column: str = "set",
        reduce_dim: Optional[Union[int, float]] = None,
        test_subsample: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.target_column = target_column
        self.input_column = input_column
        self.init_sample_size = init_sample_size
        self.featurizer = featurizer
        self.initializer = (
            initializer
            if initializer is not None
            else BOInitializer(method="true_random", n_clusters=init_sample_size)
        )
        self.exclude_top = exclude_top
        self.normalize_input = normalize_input
        # Two-phase (train->test) options: when respect_split is True the initial
        # (Phase-1) sample is drawn only from rows where split_column=="train" and
        # the held-out design space (Phase-2 candidates) is the "test" rows.
        self.respect_split = respect_split
        self.split_column = split_column
        # Optional PCA reduction (int n_components or float variance fraction),
        # fit on train rows to avoid test leakage.
        self.reduce_dim = reduce_dim
        print(f"reduce_dim: {self.reduce_dim}")
        # Optional cap on the test design space for very large test splits.
        self.test_subsample = test_subsample
        self.maximize = maximize
        self.setup()

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        if not self.maximize:
            self.data[self.target_column] = -self.data[self.target_column]

    def featurize_data(self):
        x = self.featurizer.featurize(self.data[self.input_column])
        y = self.data[self.target_column].values

        self.x = torch.from_numpy(x).to(torch.float64)
        self.y = torch.from_numpy(y).to(torch.float64).unsqueeze(-1)

    def preprocess_data(self):
        nan_rows = find_nan_rows(self.x)
        self.x = torch_delete_rows(self.x, nan_rows)
        self.y = torch_delete_rows(self.y, nan_rows)
        self.data = self.data.drop(self.data.index[nan_rows]).reset_index(drop=True)

        duplicates = find_duplicates(self.x)
        self.x = torch_delete_rows(self.x, duplicates)
        self.y = torch_delete_rows(self.y, duplicates)
        self.data = self.data.drop(self.data.index[duplicates]).reset_index(drop=True)

        self.original_indices = np.arange(len(self.x))

        print(f"Removed {len(nan_rows)} nan rows")
        print(f"Removed {len(duplicates)} duplicate rows")

    def split_data(self):
        # Held-aside evaluation split (the test set); only populated when
        # respecting the train/test split. Used by the gate, never by Phase-1 BO.
        self.test_x = None
        self.test_y = None
        self.test_indices = None

        if self.respect_split:
            init_indexes, heldout_positions = self._split_by_set()
        else:
            init_indexes, heldout_positions = self._split_by_initializer()

        print(f"Selected reactions: {init_indexes}")

        self.train_indexes = np.asarray(init_indexes)
        self.train_x = self.x[init_indexes]
        self.train_y = self.y[init_indexes]

        self.heldout_x = self.x[heldout_positions]
        self.heldout_y = self.y[heldout_positions]
        self.heldout_indices = torch.tensor(heldout_positions)

        sorted_indices = torch.argsort(self.heldout_y.squeeze())

        self.heldout_x = self.heldout_x[sorted_indices]
        self.heldout_y = self.heldout_y[sorted_indices]
        self.heldout_indices = self.heldout_indices[sorted_indices]

        train_df = self.data.loc[list(init_indexes)].copy()
        heldout_df = self.data.loc[self.heldout_indices.tolist()].copy()
        self.data = pd.concat([train_df, heldout_df])

    def _split_by_initializer(self):
        """Original behavior: draw the initial sample from the whole pool; the
        held-out design space is everything else."""
        if self.exclude_top:
            indices = torch.arange(len(self.y))
            median_value = self.y.median()
            condition = self.y > median_value
            self.exclude = indices[condition.squeeze()].tolist()
        else:
            self.exclude = None

        init_indexes, _ = self.initializer.fit(self.x, exclude=self.exclude)
        heldout_positions = sorted(
            set(self.original_indices.tolist()) - set(init_indexes)
        )
        return list(init_indexes), heldout_positions

    def _split_by_set(self):
        """Two-phase split. The initial sample is drawn only from ``train`` rows;
        the BO design space (Phase-1 candidates) is the *remaining* ``train`` rows
        — Phase-1 collects from the train distribution. The ``test`` rows are held
        aside in ``self.test_*`` for evaluating the fitted GP (the gate)."""
        set_labels = np.asarray(self.data[self.split_column])
        train_pool = np.where(set_labels == "train")[0]
        test_pool = np.where(set_labels == "test")[0]
        if len(train_pool) == 0 or len(test_pool) == 0:
            raise ValueError(
                f"respect_split=True but column '{self.split_column}' has "
                f"{len(train_pool)} train / {len(test_pool)} test rows."
            )

        # Initial (seed) sample drawn from the train pool, optionally excluding
        # the top performers so BO has to discover them.
        if self.exclude_top:
            sub_y = self.y[train_pool].squeeze()
            median_value = sub_y.median()
            exclude_local = torch.arange(len(train_pool))[sub_y > median_value].tolist()
        else:
            exclude_local = None

        local_init, _ = self.initializer.fit(self.x[train_pool], exclude=exclude_local)
        init_indexes = train_pool[np.asarray(local_init, dtype=int)].tolist()

        # Phase-1 BO design space = remaining train rows.
        heldout_positions = sorted(set(train_pool.tolist()) - set(init_indexes))

        # Held-aside test split used only for evaluation (gate), optionally capped.
        test_positions = test_pool.tolist()
        if self.test_subsample and len(test_positions) > self.test_subsample:
            rng = np.random.default_rng(0)
            test_positions = sorted(
                rng.choice(
                    test_positions, size=self.test_subsample, replace=False
                ).tolist()
            )
            print(
                f"Subsampled test eval set from {len(test_pool)} to "
                f"{self.test_subsample} points"
            )
        self.test_x = self.x[test_positions]
        self.test_y = self.y[test_positions]
        self.test_indices = torch.tensor(test_positions)

        print(
            f"respect_split: {len(init_indexes)} initial train points, "
            f"{len(heldout_positions)} train design-space, "
            f"{len(test_positions)} held-aside test points"
        )
        return init_indexes, heldout_positions
        

    def update_results(self, experiment_results, experiment_indices):
        if self.target_column not in self.data.columns:
            self.data[self.target_column] = 0.0

        
        self.y[experiment_indices] = experiment_results

        self.train_y = torch.cat([self.train_y, experiment_results], dim=0)
        self.train_x = torch.cat(
            [self.train_x, self.heldout_x[experiment_indices]], dim=0
        )

        self.heldout_x = torch_delete_rows(self.heldout_x, experiment_indices)
        self.heldout_y = torch_delete_rows(self.heldout_y, experiment_indices)

        self.train_indexes = torch.cat([self.train_indexes, experiment_indices], dim=0)

    def _fit_rows(self):
        """Row mask used to fit normalization statistics. When respecting the
        train/test split we fit on train rows only, to avoid test leakage."""
        if self.respect_split and self.split_column in self.data:
            mask = np.asarray(self.data[self.split_column]) == "train"
        else:
            mask = np.ones(len(self.x), dtype=bool)
        return torch.from_numpy(np.where(mask)[0])

    def normalize_data(self):
        def standard_scaling(X):
            return (X - X.mean()) / X.std()

        def l2_max_scaling(X):
            return X / torch.norm(X, dim=1).max()

        def l2_normalize(X):
            return X / torch.norm(X, dim=1, keepdim=True)

        fit_idx = self._fit_rows()

        if self.normalize_input == "standard_scaling":
            self.x = standard_scaling(self.x)
        elif self.normalize_input == "standard_scaling_per_dim":
            # Per-feature mean/std (the precondition for the dimension-scaled
            # lengthscale prior), fit on the train rows only.
            mean = self.x[fit_idx].mean(dim=0, keepdim=True)
            std = self.x[fit_idx].std(dim=0, keepdim=True).clamp_min(1e-8)
            self.x = (self.x - mean) / std
        elif self.normalize_input == "l2_max_scaling":
            self.x = l2_max_scaling(self.x)
        elif self.normalize_input == "l2_normalize":
            self.x = l2_normalize(self.x)
        elif self.normalize_input == "original":
            pass

        # Optional PCA reduction to the effective dimension (fit on train rows).
        if self.reduce_dim:
            pca = PCA(n_components=self.reduce_dim, random_state=0)
            pca.fit(self.x[fit_idx].cpu().numpy())
            reduced = pca.transform(self.x.cpu().numpy())
            self.x = torch.from_numpy(reduced).to(self.x.dtype)
            print(
                f"PCA reduced features to {self.x.shape[-1]} dims "
                f"({pca.explained_variance_ratio_.sum():.3f} variance retained)"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data()
        self.featurize_data()
        self.preprocess_data()
        self.normalize_data()
        self.split_data()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = SingleSampleDataset(self.train_x, self.train_y)
        return DataLoader(train_dataset, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_dataset = SingleSampleDataset(self.heldout_x, self.heldout_y)
        return DataLoader(valid_dataset, num_workers=4)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = [item.to(device) for item in batch]
        return batch
