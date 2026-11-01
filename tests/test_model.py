import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from myproject.models import SimpleClassifier


@pytest.mark.unit
def test_classifier_output_dimensions(sample_features: int, sample_classes: int) -> None:
    """Unit: модель возвращает правильное количество классов."""
    model = SimpleClassifier(in_features=sample_features, num_classes=sample_classes)
    x = torch.randn(4, sample_features)
    logits = model(x)
    assert logits.shape == (4, sample_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.property
@given(batch_size=st.integers(min_value=1, max_value=16))
def test_classifier_batch_invariance(batch_size: int) -> None:
    """Property: инвариант батча — работа с любым размером пакета без сбоев."""
    model = SimpleClassifier(in_features=4, num_classes=2)
    x = torch.randn(batch_size, 4)
    logits = model(x)
    assert logits.shape == (batch_size, 2)