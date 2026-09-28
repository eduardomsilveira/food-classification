from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict


class TrainingResults(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    train_losses: list[float]
    validation_losses: list[float]
    train_scores: list[float]
    validation_scores: list[float]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
