import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from typing import Optional

from torch_geometric.loader import DataLoader

from subgraph_counting.workload import Workload


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
