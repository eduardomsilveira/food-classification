import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

dataset = load_dataset("ethz/food101")
dataset.set_transform(
    lambda batch: {
        "image": [preprocess(image.convert("RGB")) for image in batch["image"]],
        "label": batch["label"],
    }
)

model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
for param in model.features.parameters():
    param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

loader = DataLoader(dataset["train"], batch_size=64, shuffle=True, num_workers=8)
val_loader = DataLoader(dataset["validation"], batch_size=64, num_workers=8)

model.train()
for step, batch in enumerate(loader):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    loss = loss_fn(model(images), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, len(loader), loss.item())

accuracy = MulticlassAccuracy(num_classes=101).to(device)
f1 = MulticlassF1Score(num_classes=101, average="macro").to(device)

model.eval()
with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        preds = model(images).argmax(dim=1)
        accuracy.update(preds, labels)
        f1.update(preds, labels)

print("accuracy", accuracy.compute().item())
print("f1", f1.compute().item())
