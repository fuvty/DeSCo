import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from typing import Optional

from torch_geometric.loader import DataLoader
import torch.utils.data as torch_data

from subgraph_counting.workload import Workload
from subgraph_counting.LRP_dataset import (
    LRP_Dataset,
    collate_lrp_dgl_light_index_form_wrapper,
)


class LightningDataLoader_LRP(LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[LRP_Dataset] = None,
        test_dataset: Optional[LRP_Dataset] = None,
        val_dataset: Optional[LRP_Dataset] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_lrp_dgl_light_index_form_wrapper(4),
        )

    def val_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_lrp_dgl_light_index_form_wrapper(4),
        )

    def test_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_lrp_dgl_light_index_form_wrapper(4),
        )


class LightningDataLoader(LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[Workload] = None,
        test_dataset: Optional[Workload] = None,
        val_dataset: Optional[Workload] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
