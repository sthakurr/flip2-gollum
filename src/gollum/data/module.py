import pandas as pd
import numpy as np
import torch
from typing import Optional, Union, List

from torch.utils.data import DataLoader

from gollum.data.dataset import SingleSampleDataset
from gollum.data.utils import torch_delete_rows
from gollum.initialization.initializers import BOInitializer
from gollum.data.utils import find_duplicates, find_nan_rows
from gollum.featurization.base import Featurizer
from abc import ABC
import os
os.environ["OMP_NUM_THREADS"] = "28"


class BaseDataModule(ABC):
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
        test_data_path: Optional[str] = None,
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
        self.maximize = maximize
        self.test_data_path = test_data_path
        self.test_x = None
        self.test_y = None
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
        if self.exclude_top:
            indices = torch.arange(len(self.y))
            median_value = self.y.median()
            condition = self.y > median_value
            self.exclude = indices[condition.squeeze()].tolist()
        else:
            self.exclude = None

        init_indexes, _ = self.initializer.fit(self.x, exclude=self.exclude)
        print(f"Selected reactions: {init_indexes}")

        train_df_indices = self.original_indices[init_indexes].tolist()
        self.train_indexes = (
            init_indexes  
        )

        all_index_set = set(self.original_indices.tolist())
        train_index_set = set(train_df_indices)
        heldout_df_indices = list(all_index_set - train_index_set)

       
        self.train_x = self.x[init_indexes]
        self.train_y = self.y[init_indexes]

        index_to_position = {
            idx.item(): pos for pos, idx in enumerate(self.original_indices)
        }
        heldout_positions = [index_to_position[idx] for idx in heldout_df_indices]

        self.heldout_x = self.x[heldout_positions]
        self.heldout_y = self.y[heldout_positions]
        self.heldout_indices = torch.tensor(heldout_df_indices)

        sorted_indices = torch.argsort(self.heldout_y.squeeze())

        self.heldout_x = self.heldout_x[sorted_indices]
        self.heldout_y = self.heldout_y[sorted_indices]
        self.heldout_indices = self.heldout_indices[sorted_indices]

        train_df = self.data.loc[train_df_indices].copy()
        heldout_df = self.data.loc[self.heldout_indices.tolist()].copy()
        self.data = pd.concat([train_df, heldout_df])
        

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

    def normalize_data(self):
        self._norm_mean: Optional[torch.Tensor] = None
        self._norm_std:  Optional[torch.Tensor] = None

        if self.normalize_input == "standard_scaling":
            self._norm_mean = self.x.mean()
            self._norm_std  = self.x.std()
            self.x = (self.x - self._norm_mean) / self._norm_std
        elif self.normalize_input == "l2_max_scaling":
            self.x = self.x / torch.norm(self.x, dim=1).max()
        elif self.normalize_input == "l2_normalize":
            self.x = self.x / torch.norm(self.x, dim=1, keepdim=True)
        elif self.normalize_input == "original":
            pass

    def load_test_data(self):
        test_df = pd.read_csv(self.test_data_path)
        x = self.featurizer.featurize(test_df[self.input_column])
        y = test_df[self.target_column].values
        self.test_x = torch.from_numpy(x).to(torch.float64)
        self.test_y = torch.from_numpy(y).to(torch.float64).unsqueeze(-1)
        if not self.maximize:
            self.test_y = -self.test_y
        # Apply the same normalization that was fitted on training features.
        if self._norm_mean is not None and self._norm_std is not None:
            self.test_x = (self.test_x - self._norm_mean) / self._norm_std
        print(f"Loaded {len(self.test_x)} test sequences from {self.test_data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data()
        self.featurize_data()
        self.preprocess_data()
        self.normalize_data()
        self.split_data()
        if self.test_data_path is not None:
            self.load_test_data()

    def train_dataloader(self) -> DataLoader:
        train_dataset = SingleSampleDataset(self.train_x, self.train_y)
        return DataLoader(train_dataset, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        valid_dataset = SingleSampleDataset(self.heldout_x, self.heldout_y)
        return DataLoader(valid_dataset, num_workers=4)


class SplitDataModule(BaseDataModule):
    """
    Redesigned data module for train/test-split evaluation:
      - Seed training set : n_train examples randomly drawn from train_path
      - Candidate space   : all sequences from test_path

    dm.x              = cat([sampled_train_features, all_test_features])
    dm.train_indexes  = indices 0..n_train-1  (into dm.x)
    dm.heldout_indices = indices n_train..n_train+n_test-1 (into dm.x)
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        n_train: int,
        input_column: Union[str, List[str]] = "sequence",
        target_column: str = "target",
        maximize: bool = True,
        featurizer: Featurizer = None,
        normalize_input: str = "original",
        # Accept but ignore BaseDataModule args that bleed through from bochemian.yaml defaults
        data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        exclude_top: bool = False,
        init_sample_size: int = 10,
        initializer: Optional[BOInitializer] = None,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.n_train = n_train
        self.input_column = input_column
        self.target_column = target_column
        self.maximize = maximize
        self.featurizer = featurizer if featurizer is not None else Featurizer()
        self.normalize_input = normalize_input
        self.test_x = None
        self.test_y = None
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # --- load and sample from training split ---
        train_df = pd.read_csv(self.train_path)
        if not self.maximize:
            train_df[self.target_column] = -train_df[self.target_column]

        n_train_all = len(train_df)
        sampled_idx = np.sort(
            np.random.choice(n_train_all, size=self.n_train, replace=False)
        )
        sampled_df = train_df.iloc[sampled_idx].reset_index(drop=True)

        # --- load test split (full candidate space) ---
        test_df = pd.read_csv(self.test_path)
        if not self.maximize:
            test_df[self.target_column] = -test_df[self.target_column]
        n_test = len(test_df)

        # --- featurize ---
        sampled_x = torch.from_numpy(
            self.featurizer.featurize(sampled_df[self.input_column])
        ).to(torch.float64)
        test_x = torch.from_numpy(
            self.featurizer.featurize(test_df[self.input_column])
        ).to(torch.float64)

        sampled_y = (
            torch.from_numpy(sampled_df[self.target_column].values)
            .to(torch.float64)
            .unsqueeze(-1)
        )
        test_y = (
            torch.from_numpy(test_df[self.target_column].values)
            .to(torch.float64)
            .unsqueeze(-1)
        )

        # --- combine into unified dm.x / dm.y / dm.data with sequential index ---
        self.x = torch.cat([sampled_x, test_x], dim=0)
        self.y = torch.cat([sampled_y, test_y], dim=0)

        test_df_indexed = test_df.reset_index(drop=True)
        test_df_indexed.index += self.n_train
        self.data = pd.concat([sampled_df, test_df_indexed])

        self.original_indices = np.arange(len(self.x))

        # --- train / heldout split ---
        self.train_indexes = np.arange(self.n_train)
        heldout_pos = np.arange(self.n_train, self.n_train + n_test)
        self.heldout_indices = torch.tensor(heldout_pos)

        self.train_x = self.x[self.train_indexes]
        self.train_y = self.y[self.train_indexes]
        self.heldout_x = self.x[heldout_pos]
        self.heldout_y = self.y[heldout_pos]

        print(
            f"[SplitDataModule] n_train={self.n_train} from {self.train_path} "
            f"| candidate space={n_test} from {self.test_path}"
        )
