"""Training script with MLflow tracking."""

import os
import sys

import mlflow
import mlflow.pytorch
import torch

from myproject.dataset import SyntheticDataset, create_dataloader
from myproject.models import SimpleClassifier
from myproject.trainer import Trainer

# Фикс кодировки для эмодзи MLflow на Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8")  # ty: ignore[unresolved-attribute]


def run_training(
    lr: float = 0.01,
    epochs: int = 10,
    batch_size: int = 16,
    hidden_dim: int = 32,
    seed: int = 42,
) -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("pytorch_training")

    run_name = f"run_lr_{lr}_epochs_{epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_type": "SimpleClassifier",
                "framework": "PyTorch",
                "task": "classification",
            }
        )

        mlflow.log_params(
            {
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "seed": seed,
            }
        )

        torch.manual_seed(seed)
        dataset = SyntheticDataset(num_samples=200, in_features=8)
        loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)

        model = SimpleClassifier(in_features=8, num_classes=2, hidden_dim=hidden_dim)
        trainer = Trainer(model=model, lr=lr)

        for epoch in range(epochs):
            history = trainer.fit(loader, epochs=1)
            epoch_loss = (
                history[0]["loss"] if isinstance(history, list) else history["loss"]
            )
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"[{run_name}] Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}")

        os.makedirs("output", exist_ok=True)
        model_path = os.path.join("output", "model.pt")
        torch.save(model.state_dict(), model_path)

        mlflow.log_artifact(model_path, artifact_path="weights")

        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

        print(f"Запуск {run_name} успешно завершён!\n")


if __name__ == "__main__":
    print("--- Запуск 1: lr=0.05 ---")
    run_training(lr=0.05, epochs=10)

    print("--- Запуск 2: lr=0.005 ---")
    run_training(lr=0.005, epochs=10)
