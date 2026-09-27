import datasets
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


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
