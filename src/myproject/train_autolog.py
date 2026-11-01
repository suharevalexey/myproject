"""Training script with MLflow PyTorch autologging."""

import sys

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from myproject.dataset import SyntheticDataset
from myproject.models import SimpleClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8")  # ty: ignore[unresolved-attribute]


class LightningClassifier(pl.LightningModule):
    """Lightning wrapper for SimpleClassifier."""

    def __init__(
        self,
        in_features: int = 8,
        num_classes: int = 2,
        hidden_dim: int = 32,
        lr: float = 0.01,
    ):  # noqa: E501
        super().__init__()
        # Сохраняем гиперпараметры (MLflow залогирует их автоматически)
        self.save_hyperparameters()
        self.model = SimpleClassifier(
            in_features=in_features, num_classes=num_classes, hidden_dim=hidden_dim
        )  # noqa: E501
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:  # noqa: E501
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        # Логируем loss в Lightning — MLflow перехватит его сам
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def run_autolog_training(lr: float = 0.01, max_epochs: int = 5) -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("pytorch_autolog_experiment")

    # Включаем автоматическое логирование MLflow!
    mlflow.pytorch.autolog(log_models=True)

    dataset = SyntheticDataset(num_samples=200, in_features=8)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LightningClassifier(in_features=8, num_classes=2, hidden_dim=32, lr=lr)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        logger=False,  # autolog сам перехватывает события тренера
    )

    with mlflow.start_run(run_name=f"autolog_lr_{lr}"):
        trainer.fit(model, dataloader)

    print(f"Автологирование для lr={lr} успешно завершено!")


if __name__ == "__main__":
    print("--- Запуск autologging: lr=0.02 ---")
    run_autolog_training(lr=0.02, max_epochs=5)

    print("--- Запуск autologging: lr=0.002 ---")
    run_autolog_training(lr=0.002, max_epochs=5)
