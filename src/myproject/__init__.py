def main() -> None:
    print("Hello from myproject!")

"""MLOps homework project."""

from myproject.dataset import SyntheticDataset, create_dataloader
from myproject.models import LinearBlock, SimpleClassifier
from myproject.trainer import Trainer

__all__ = [
    "LinearBlock",
    "SimpleClassifier",
    "SyntheticDataset",
    "create_dataloader",
    "Trainer",
]