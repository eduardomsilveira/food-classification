import datasets
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from .reproducibility import DEFAULT_SEED


def split_food101(
    dataset_dict: datasets.DatasetDict,
    validation_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> datasets.DatasetDict:
    train_validation = dataset_dict["train"].train_test_split(
        test_size=validation_fraction,
        stratify_by_column="label",
        seed=seed,
    )
    return datasets.DatasetDict({
        "train": train_validation["train"],
        "validation": train_validation["test"],
        "test": dataset_dict["validation"],
    })


class FoodDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[index]

        match item:
            case {"image": image, "label": label}:
                image = self.transform(image)
                return (image, label)
            case _:
                raise ValueError(f"Unexpected format: {item.keys()}")
