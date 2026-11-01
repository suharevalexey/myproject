import pytest

from myproject.dataset import SyntheticDataset, create_dataloader


@pytest.mark.unit
def test_dataset_len_and_item() -> None:
    """Unit: датасет возвращает корректную длину и форму сэмпла."""
    ds = SyntheticDataset(num_samples=20, in_features=4)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (4,)
    assert isinstance(y.item(), int)


@pytest.mark.unit
def test_dataloader_batching() -> None:
    """Unit: даталоадер собирает корректные батчи."""
    ds = SyntheticDataset(num_samples=10, in_features=4)
    loader = create_dataloader(ds, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3  # 4 + 4 + 2
    first_x, first_y = batches[0]
    assert first_x.shape == (4, 4)
    assert first_y.shape == (4,)