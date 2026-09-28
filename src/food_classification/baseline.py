import json
import os
from dataclasses import dataclass
from pathlib import Path

import datasets
import torch
import torchvision.transforms.v2 as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0

from .food_dataset import FoodDataset, split_food101
from .utils import plot, train, validate_epoch


@dataclass
class Config:
    output_dir: Path = Path("outputs/baseline")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 32
    shuffle: bool = True
    validation_fraction: float = 0.2
    split_seed: int = 42


def make_loader(
    dataset: datasets.Dataset,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    food_dataset = FoodDataset(dataset=dataset, transform=transform)
    loader = DataLoader(dataset=food_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def main():
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        "ethz/food101",
        token=os.environ.get("HF_TOKEN"),
    )
    dataset_dict = split_food101(
        dataset_dict,
        validation_fraction=Config.validation_fraction,
        seed=Config.split_seed,
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

    results = train(
        n_epochs=20,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=Config.device,
    )

    checkpoint_path = Config.output_dir / "baseline.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": "baseline",
            "architecture": "efficientnet_b0",
            "num_classes": model.classifier[1].out_features,
            "class_names": dataset_dict["train"].features["label"].names,
            "epochs": len(results.train_losses),
            "dataset_split": {
                "validation_fraction": Config.validation_fraction,
                "seed": Config.split_seed,
                "stratify_by_column": "label",
                "test_source": "ethz/food101:validation",
                "sizes": {name: len(split) for name, split in dataset_dict.items()},
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    results.save_json(Config.output_dir / "training_results.json")

    test_loader = make_loader(
        dataset=dataset_dict["test"],
        transform=transform,
        batch_size=Config.batch_size,
        shuffle=False,
    )
    test_loss, test_score = validate_epoch(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=Config.device,
    )
    (Config.output_dir / "test_results.json").write_text(
        json.dumps(
            {"loss": test_loss, "f1_weighted": test_score}, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"test-loss: {test_loss}, test-score: {test_score}")

    plot(results, output_dir=Config.output_dir)


if __name__ == "__main__":
    main()
