import os
from dataclasses import dataclass

import datasets
import torch
import torchvision.transforms.v2 as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0

from .food_dataset import FoodDataset
from .utils import plot, train


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 32
    shuffle: bool = True


def make_loader(
    dataset: datasets.Dataset,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = dataset.select(range(100))
    food_dataset = FoodDataset(dataset=dataset, transform=transform)
    loader = DataLoader(dataset=food_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def main():
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        "ethz/food101",
        token=os.environ.get("HF_TOKEN"),
    )

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(
            size=256,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(size=224),
        transforms.ToDtype(Config.dtype, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_loader = make_loader(
        dataset=dataset_dict["train"],
        transform=transform,
        batch_size=Config.batch_size,
        shuffle=Config.shuffle,
    )
    validation_loader = make_loader(
        dataset=dataset_dict["validation"],
        transform=transform,
        batch_size=Config.batch_size,
        shuffle=False,
    )

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
    for param in model.features.parameters():
        param.requires_grad = False
    model.to(device=Config.device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_scores, validation_losses, validation_scores = train(
        n_epochs=20,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=Config.device,
    )

    plot(
        train_losses=train_losses,
        train_scores=train_scores,
        validation_losses=validation_losses,
        validation_scores=validation_scores,
    )


if __name__ == "__main__":
    main()
