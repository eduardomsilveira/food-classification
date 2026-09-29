from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import StochasticDepth

from .training_results import (
    ClassMetrics,
    EpochMetrics,
    EvaluationResults,
    TrainingResults,
)


@dataclass
class _MetricAccumulator:
    loss_sum: float = 0.0
    samples: int = 0
    correct_top1: int = 0
    correct_top5: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        logits = logits.detach()
        batch_size = labels.size(0)
        self.loss_sum += loss.detach().item() * batch_size
        self.samples += batch_size
        self.correct_top1 += (logits.argmax(dim=1) == labels).sum().item()
        top5 = logits.topk(k=min(5, logits.size(1)), dim=1).indices
        self.correct_top5 += (top5 == labels[:, None]).any(dim=1).sum().item()

    def compute(self) -> EpochMetrics:
        if self.samples == 0:
            raise ValueError("Cannot evaluate an empty loader")
        return EpochMetrics(
            loss=self.loss_sum / self.samples,
            accuracy_top1=self.correct_top1 / self.samples,
            accuracy_top5=self.correct_top5 / self.samples,
        )


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: str,
    update_batchnorm_stats: bool = False,
    enable_stochastic_depth: bool = False,
) -> EpochMetrics:
    model.train()
    model.features.eval()
    if update_batchnorm_stats:
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
    if enable_stochastic_depth:
        for module in model.features.modules():
            if isinstance(module, StochasticDepth):
                module.train()
    metrics = _MetricAccumulator()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        metrics.update(logits, labels, loss)

    return metrics.compute()


@torch.no_grad()
def _evaluation_pass(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    num_classes: int | None = None,
) -> tuple[EpochMetrics, torch.Tensor | None]:
    model.eval()
    metrics = _MetricAccumulator()
    confusion = (
        torch.zeros((num_classes, num_classes), dtype=torch.int64)
        if num_classes is not None
        else None
    )
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if num_classes is not None and logits.size(1) != num_classes:
            raise ValueError("Class names must match the model output classes")
        loss = loss_fn(logits, labels)
        metrics.update(logits, labels, loss)
        if confusion is not None:
            predictions = logits.argmax(dim=1)
            # Accumulate on CPU to also support strict deterministic CUDA runs.
            indices = labels.cpu() * num_classes + predictions.cpu()
            confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
                num_classes, num_classes
            )
    return metrics.compute(), confusion


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> EpochMetrics:
    metrics, _ = _evaluation_pass(model, loader, loss_fn, device)
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    class_names: list[str],
) -> EvaluationResults:
    metrics, confusion = _evaluation_pass(
        model, loader, loss_fn, device, num_classes=len(class_names)
    )
    true_positives = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    per_class = [
        ClassMetrics(
            class_id=index,
            class_name=name,
            precision=(true_positives[index] / predicted[index].clamp_min(1)).item(),
            recall=(true_positives[index] / support[index].clamp_min(1)).item(),
            support=support[index].item(),
        )
        for index, name in enumerate(class_names)
    ]
    return EvaluationResults(
        **metrics.model_dump(),
        per_class=per_class,
        confusion_matrix=confusion.tolist(),
    )


def train(
    n_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: str,
    on_epoch_end: Callable[[TrainingResults], None] | None = None,
    update_batchnorm_stats: bool = False,
    enable_stochastic_depth: bool = False,
) -> TrainingResults:
    results = TrainingResults(
        train_losses=[],
        validation_losses=[],
        train_accuracy_top1=[],
        validation_accuracy_top1=[],
        train_accuracy_top5=[],
        validation_accuracy_top5=[],
    )
    for epoch in range(n_epochs):
        train_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=train_loader,
            device=device,
            update_batchnorm_stats=update_batchnorm_stats,
            enable_stochastic_depth=enable_stochastic_depth,
        )
        validation_metrics = validate_epoch(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        for split, metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
        ):
            getattr(results, f"{split}_losses").append(metrics.loss)
            getattr(results, f"{split}_accuracy_top1").append(metrics.accuracy_top1)
            getattr(results, f"{split}_accuracy_top5").append(metrics.accuracy_top5)
            print(
                f"epoch {epoch + 1}/{n_epochs} {split}: loss={metrics.loss:.4f}, "
                f"accuracy-top1={metrics.accuracy_top1:.4f}, "
                f"accuracy-top5={metrics.accuracy_top5:.4f}"
            )
        if on_epoch_end is not None:
            on_epoch_end(results)
    return results


def plot(results: TrainingResults, output_dir: str | Path = "docs") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(results.train_losses) + 1)

    style = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
    metrics = (
        (
            results.train_losses,
            results.validation_losses,
            "Loss",
            output_dir / "loss.pdf",
        ),
        (
            results.train_accuracy_top1,
            results.validation_accuracy_top1,
            "Accuracy top-1",
            output_dir / "accuracy_top1.pdf",
        ),
        (
            results.train_accuracy_top5,
            results.validation_accuracy_top5,
            "Accuracy top-5",
            output_dir / "accuracy_top5.pdf",
        ),
    )

    with plt.rc_context(style):
        for train_values, validation_values, ylabel, output in metrics:
            fig, ax = plt.subplots(figsize=(15 / 2.54, 8 / 2.54), layout="constrained")
            try:
                ax.plot(epochs, train_values, color="tab:blue", label="Treino")
                ax.plot(
                    epochs,
                    validation_values,
                    color="tab:orange",
                    linestyle="--",
                    label="Validação",
                )
                ax.set_xlabel("Época")
                ax.set_ylabel(ylabel)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(axis="y", color="0.85", linewidth=0.5)
                ax.legend()
                fig.savefig(output)
            finally:
                plt.close(fig)


def plot_confusion_matrix(
    results: EvaluationResults, output_dir: str | Path = "docs"
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = torch.tensor(results.confusion_matrix, dtype=torch.float64)
    normalized = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
    labels = [item.class_name.replace("_", " ") for item in results.per_class]
    size = max(8, len(labels) * 0.18)
    with plt.rc_context({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 12,
    }):
        fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
        try:
            heatmap = ax.imshow(normalized.numpy(), cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.tick_params(axis="both", labelsize=6 if len(labels) > 30 else 10)
            ax.set_xlabel("Classe predita")
            ax.set_ylabel("Classe real")
            fig.colorbar(heatmap, ax=ax, shrink=0.75, label="Fração da classe real")
            fig.savefig(output_dir / "confusion_matrix.pdf")
        finally:
            plt.close(fig)
