import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetItem:
    question: str
    answer: str


class JsonDatasetLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[DatasetItem]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items: list[DatasetItem] = []
        for row in data:
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not q or not a:
                continue
            items.append(DatasetItem(question=q, answer=a))
        return items
