"""Synthetic dataset and DataLoader utilities."""

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic classification dataset with reproducible separable features."""

    def __init__(
        self,
        num_samples: int = 100,
        in_features: int = 4,
        num_classes: int = 2,
    ) -> None:
        generator = torch.Generator().manual_seed(42)
        self.x = torch.randn(num_samples, in_features, generator=generator)
        weights = torch.ones(in_features, 1)
        scores = self.x @ weights
        self.y = (scores.squeeze(1) > 0).long() % num_classes

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create PyTorch DataLoader with given batch size."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)