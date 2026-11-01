"""Neural network layers and models."""

import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    """Linear block: Linear layer + ReLU activation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class SimpleClassifier(nn.Module):
    """Multilayer classifier built using LinearBlock components."""

    def __init__(
        self, in_features: int, num_classes: int, hidden_dim: int = 16
    ) -> None:
        super().__init__()
        self.block = LinearBlock(in_features=in_features, out_features=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))