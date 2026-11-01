"""Neural network layers and models (stubs for TDD Red phase)."""

import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Заглушка: возвращает тензор заведомо неверной формы [1]
        return torch.zeros(1)


class SimpleClassifier(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, hidden_dim: int = 16
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Заглушка: возвращает тензор заведомо неверной формы [1]
        return torch.zeros(1)