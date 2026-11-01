"""Trainer routines (stubs for TDD Red phase)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, lr: float = 0.01) -> None:
        self.model = model

    def fit(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 1,
        fast_dev_run: bool = False,
    ) -> list[dict[str, float]] | dict[str, float]:
        # Заглушка: пустой словарь или пустой список
        return {} if fast_dev_run else []

    def train_epoch_with_checks(
        self, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[float, float]:
        # Заглушка: loss не убывает (1.0 не меньше 1.0)
        return 1.0, 1.0

    def overfit_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor], steps: int = 200
    ) -> float:
        # Заглушка: высокий loss
        return 10.0