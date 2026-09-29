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
from .reproducibility import (
    DEFAULT_SEED,
    reproducibility_metadata,
    seed_everything,
    seed_worker,
)
from .training_results import TrainingResults
from .utils import evaluate, plot, plot_confusion_matrix, train


@dataclass
class Config:
    output_dir: Path = Path("outputs/stochastic_depth")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    stochastic_depth_prob: float = 0.2
    batch_size: int = 32
    shuffle: bool = True
    validation_fraction: float = 0.2
    seed: int = DEFAULT_SEED
    split_seed: int = DEFAULT_SEED
    num_workers: int = 0


def make_loader(
    dataset: datasets.Dataset,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
) -> DataLoader:
    food_dataset = FoodDataset(dataset=dataset, transform=transform)
    loader = DataLoader(
        dataset=food_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
        num_workers=num_workers,
    )
    return loader


def main():
    seed_everything(Config.seed)

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
        seed=Config.seed,
        num_workers=Config.num_workers,
    )
    validation_loader = make_loader(
        dataset=dataset_dict["validation"],
        transform=transform,
        batch_size=Config.batch_size,
        shuffle=False,
        seed=Config.seed + 1,
        num_workers=Config.num_workers,
    )

    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
        stochastic_depth_prob=Config.stochastic_depth_prob,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
    for param in model.features.parameters():
        param.requires_grad = False
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
    model.to(device=Config.device)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_path = Config.output_dir / "stochastic_depth.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {
        "experiment": "stochastic_depth",
        "architecture": "efficientnet_b0",
        "stochastic_depth_prob": Config.stochastic_depth_prob,
        "num_classes": model.classifier[1].out_features,
        "class_names": dataset_dict["train"].features["label"].names,
        "reproducibility": {
            **reproducibility_metadata(Config.seed, Config.device),
            "num_workers": Config.num_workers,
            "loader_seeds": {
                "train": Config.seed,
                "validation": Config.seed + 1,
                "test": Config.seed + 2,
            },
        },
        "dataset_split": {
            "validation_fraction": Config.validation_fraction,
            "seed": Config.split_seed,
            "stratify_by_column": "label",
            "test_source": "ethz/food101:validation",
            "sizes": {name: len(split) for name, split in dataset_dict.items()},
        },
    }

    def save_epoch(results: TrainingResults) -> None:
        epoch = len(results.train_losses)
        checkpoint = {
            **checkpoint_metadata,
            "epochs": epoch,
            "model_state_dict": model.state_dict(),
        }
        epoch_path = Config.output_dir / f"stochastic_depth_epoch_{epoch:03d}.pt"
        for path in (epoch_path, checkpoint_path):
            temporary_path = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(path)
        results.save_json(Config.output_dir / "training_results.json")

    results = train(
        n_epochs=10,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=Config.device,
        on_epoch_end=save_epoch,
        update_batchnorm_stats=True,
        enable_stochastic_depth=True,
    )

    test_loader = make_loader(
        dataset=dataset_dict["test"],
        transform=transform,
        batch_size=Config.batch_size,
        shuffle=False,
        seed=Config.seed + 2,
        num_workers=Config.num_workers,
    )
    test_results = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=Config.device,
        class_names=dataset_dict["train"].features["label"].names,
    )
    test_results.save_json(Config.output_dir / "test_results.json")
    print(
        f"test: loss={test_results.loss:.4f}, "
        f"accuracy-top1={test_results.accuracy_top1:.4f}, "
        f"accuracy-top5={test_results.accuracy_top5:.4f}"
    )

    plot(results, output_dir=Config.output_dir)
    plot_confusion_matrix(test_results, output_dir=Config.output_dir)


if __name__ == "__main__":
    main()
