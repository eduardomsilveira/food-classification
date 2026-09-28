from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict


class JsonResults(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class EpochMetrics(JsonResults):
    loss: float
    accuracy_top1: float
    accuracy_top5: float


class TrainingResults(JsonResults):
    train_losses: list[float]
    validation_losses: list[float]
    train_accuracy_top1: list[float]
    validation_accuracy_top1: list[float]
    train_accuracy_top5: list[float]
    validation_accuracy_top5: list[float]


class ClassMetrics(JsonResults):
    class_id: int
    class_name: str
    precision: float
    recall: float
    support: int


class EvaluationResults(EpochMetrics):
    per_class: list[ClassMetrics]
    # Rows are true classes; columns are predicted classes, in class_id order.
    confusion_matrix: list[list[int]]
