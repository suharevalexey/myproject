import pytest
import torch

from myproject.dataset import SyntheticDataset, create_dataloader
from myproject.models import SimpleClassifier
from myproject.trainer import Trainer


@pytest.mark.e2e
def test_full_training_pipeline_e2e() -> None:
    """E2E: полный пайплайн от подготовки данных до завершения обучения и инференса."""
    # 1. Arrange
    dataset = SyntheticDataset(num_samples=50, in_features=4)
    loader = create_dataloader(dataset, batch_size=8, shuffle=True)
    model = SimpleClassifier(in_features=4, num_classes=2)
    trainer = Trainer(model=model, lr=0.02)

    # 2. Act
    history = trainer.fit(loader, epochs=3)

    # 3. Assert
    assert len(history) == 3
    assert history[-1]["loss"] < history[0]["loss"]

    # Проверяем инференс
    test_input = torch.randn(1, 4)
    model.eval()
    with torch.no_grad():
        preds = model(test_input)
    assert preds.shape == (1, 2)