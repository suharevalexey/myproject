import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from myproject.models import LinearBlock


@pytest.mark.unit
def test_linear_block_forward_shape() -> None:
    """Unit: проверяем размерность выхода блока."""
    block = LinearBlock(in_features=4, out_features=8)
    x = torch.randn(2, 4)
    out = block(x)
    assert out.shape == (2, 8)


@pytest.mark.property
@given(
    batch_size=st.integers(min_value=1, max_value=32),
    in_features=st.integers(min_value=2, max_value=16),
    out_features=st.integers(min_value=2, max_value=16),
)
def test_linear_block_shape_invariant(
    batch_size: int, in_features: int, out_features: int
) -> None:
    """Property: закон сохранения формы для любых случайных размерностей."""
    block = LinearBlock(in_features=in_features, out_features=out_features)
    x = torch.randn(batch_size, in_features)
    out = block(x)
    assert out.shape == (batch_size, out_features)
    assert torch.isfinite(out).all()