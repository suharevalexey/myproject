"""Synthetic dataset and DataLoader utilities (stubs for TDD Red phase)."""

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        num_samples: int = 100,
        in_features: int = 4,
        num_classes: int = 2,
    ) -> None:
        pass

    def __len__(self) -> int:
        # Заглушка: возвращает неверную длину 0
        return 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1), torch.zeros(1)


def create_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)