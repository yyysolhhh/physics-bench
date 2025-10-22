import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetItem:
    id: int
    question: str
    answer: str
    unit: str = ""
    source: str = ""
    problemid: str = ""


class JsonDatasetLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, limit: int | None = None) -> list[DatasetItem]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items: list[DatasetItem] = []
        for row in data:
            item_id = row.get("id")
            problem_text = str(row.get("problem_text", "")).strip()
            answer_number = str(row.get("answer_number", "")).strip()
            unit = str(row.get("unit", "")).strip()
            source = str(row.get("source", "")).strip()
            problemid = str(row.get("problemid", "")).strip()
            
            if not problem_text or not answer_number:
                continue
                
            # 답변과 단위를 결합
            if unit:
                answer = f"{answer_number} {unit}"
            else:
                answer = answer_number
                
            items.append(DatasetItem(
                id=item_id,
                question=problem_text,
                answer=answer,
                unit=unit,
                source=source,
                problemid=problemid
            ))
            if limit is not None and len(items) >= limit:
                break
        return items
