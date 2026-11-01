import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def sample_features() -> int:
    return 4


@pytest.fixture
def sample_classes() -> int:
    return 2


@pytest.fixture
def dummy_batch(
    sample_features: int, sample_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Фикстура одного батча: (X: [batch_size, in_features], y: [batch_size])."""
    x = torch.randn(8, sample_features)
    y = torch.randint(0, sample_classes, (8,))
    return x, y