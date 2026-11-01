import pytest
import torch

from myproject.dataset import SyntheticDataset, create_dataloader
from myproject.models import SimpleClassifier
from myproject.trainer import Trainer


@pytest.mark.smoke
def test_dry_run_fast_dev_run() -> None:
    """Dry run (fast_dev_run): 1-2 батча без падений forward/loss/backward/step."""
    ds = SyntheticDataset(num_samples=8, in_features=4)
    loader = create_dataloader(ds, batch_size=2, shuffle=False)
    model = SimpleClassifier(in_features=4, num_classes=2)
    trainer = Trainer(model=model, lr=0.01)

    metrics = trainer.fit(loader, epochs=1, fast_dev_run=True)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


@pytest.mark.smoke
def test_smoke_training_step_and_gradients() -> None:
    """Smoke: 1 эпоха, 8 сэмплов, батч 2.

    3 ассерта:
    1) loss конечен
    2) loss убывает относительно начала
    3) у каждого параметра градиент не None
    """
    ds = SyntheticDataset(num_samples=8, in_features=4)
    loader = create_dataloader(ds, batch_size=2, shuffle=False)
    model = SimpleClassifier(in_features=4, num_classes=2)
    trainer = Trainer(model=model, lr=0.05)

    initial_loss, final_loss = trainer.train_epoch_with_checks(loader)

    # 1. Loss конечен
    assert torch.isfinite(torch.tensor(final_loss))
    # 2. Loss убывает относительно начального
    assert final_loss < initial_loss
    # 3. У всех параметров есть градиент (None означает, что параметр не участвует)
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Параметр {name} не получил градиент!"


@pytest.mark.overfit
def test_overfit_single_batch(dummy_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Overfit: модель должна запомнить один батч до почти нулевого loss (< 0.05)."""
    model = SimpleClassifier(in_features=4, num_classes=2)
    trainer = Trainer(model=model, lr=0.1)
    final_loss = trainer.overfit_batch(dummy_batch, steps=200)
    assert final_loss < 0.05