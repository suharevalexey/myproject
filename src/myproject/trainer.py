"""Model training, smoke checks, and overfit routines."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Trainer handling forward, backward, optimizer step, and sanity checks."""

    def __init__(self, model: nn.Module, lr: float = 0.01) -> None:
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def fit(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 1,
        fast_dev_run: bool = False,
    ) -> list[dict[str, float]] | dict[str, float]:
        """Train model across epochs or run 1 dry-run batch (fast_dev_run)."""
        if fast_dev_run:
            self.model.train()
            for x, y in dataloader:
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                return {"loss": float(loss.item())}
            return {"loss": 0.0}

        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            count = 0
            for x, y in dataloader:
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item()) * len(x)
                count += len(x)

            epoch_loss = total_loss / max(count, 1)
            history.append({"epoch": float(epoch), "loss": epoch_loss})

        return history

    def train_epoch_with_checks(
        self, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[float, float]:
        """Smoke test run: computes initial loss, trains 1 epoch, records gradients."""
        self.model.train()
        first_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        initial_loss: float = 0.0

        for x, y in dataloader:
            first_batch = (x, y)
            with torch.no_grad():
                initial_loss = float(self.criterion(self.model(x), y).item())
            break

        for x, y in dataloader:
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

        assert first_batch is not None
        fx, fy = first_batch
        with torch.no_grad():
            final_loss = float(self.criterion(self.model(fx), fy).item())

        return initial_loss, final_loss

    def overfit_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor], steps: int = 200
    ) -> float:
        """Overfit single batch until convergence."""
        x, y = batch
        self.model.train()
        for _ in range(steps):
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

        return float(loss.item())