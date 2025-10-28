import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base_loader import DatasetLoader


@dataclass(frozen=True)
class SciBenchItem:
    """SciBench 데이터셋 아이템"""
    id: int
    problem_text: str
    answer_number: str
    answer_latex: str = ""
    unit: str = ""
    source: str = ""
    problemid: str = ""

    @property
    def question(self) -> str:
        return self.problem_text

    @property
    def answer(self) -> str:
        if self.unit:
            return f"{self.answer_number} {self.unit}"
        return self.answer_number


class SciBenchDatasetLoader(DatasetLoader):
    """SciBench 데이터셋을 위한 JSON 로더"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {self.path}")

    def load(self, limit: Optional[int] = None) -> list[SciBenchItem]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        items: list[SciBenchItem] = []

        for row in data:
            item_id = row.get("id", 0)
            problem_text = str(row.get("problem_text", "")).strip()
            answer_number = str(row.get("answer_number", "")).strip()
            answer_latex = str(row.get("answer_latex", "")).strip()
            unit = str(row.get("unit", "")).strip()
            source = str(row.get("source", "")).strip()
            problemid = str(row.get("problemid", "")).strip()

            if not problem_text or not answer_number:
                continue

            items.append(SciBenchItem(
                id=item_id,
                problem_text=problem_text,
                answer_number=answer_number,
                answer_latex=answer_latex,
                unit=unit,
                source=source,
                problemid=problemid
            ))

            if limit is not None and len(items) >= limit:
                break

        return items
