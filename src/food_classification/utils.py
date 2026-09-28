import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score

from .training_results import TrainingResults


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    model.train()

    loss_accum = MeanMetric(nan_strategy="error").to(device=device)
    score_accum = MulticlassF1Score(num_classes=101, average="weighted").to(
        device=device
    )

    for batch in loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)

        loss_accum.update(value=loss.item())
        score_accum.update(preds=predictions, target=labels)

    loss = loss_accum.compute().item()
    score = score_accum.compute().item()

    return loss, score


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()

    loss_accum = MeanMetric(nan_strategy="error").to(device=device)
    score_accum = MulticlassF1Score(num_classes=101, average="weighted").to(
        device=device
    )

    with torch.no_grad():
        for batch in loader:
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            predictions = logits.argmax(dim=1)

            loss_accum.update(loss.item())
            score_accum.update(preds=predictions, target=labels)

    loss = loss_accum.compute().item()
    score = score_accum.compute().item()

    return loss, score


def train(
    n_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: str,
) -> TrainingResults:
    train_losses = []
    train_scores = []

    validation_losses = []
    validation_scores = []

    for epoch in range(n_epochs):
        loss, score = train_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=train_loader,
            device=device,
        )
        train_losses.append(loss)
        train_scores.append(score)

        print(f"train-loss: {loss}, train-score: {score}")

        loss, score = validate_epoch(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        validation_losses.append(loss)
        validation_scores.append(score)

        print(f"validation-loss: {loss}, validation-score: {score}")

    return TrainingResults(
        train_losses=train_losses,
        validation_losses=validation_losses,
        train_scores=train_scores,
        validation_scores=validation_scores,
    )


def plot(results: TrainingResults) -> None:
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
        (results.train_losses, results.validation_losses, "Loss", "docs/loss.pdf"),
        (
            results.train_scores,
            results.validation_scores,
            "F1-Score Ponderado",
            "docs/score.pdf",
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
