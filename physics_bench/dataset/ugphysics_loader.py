import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .base_loader import DatasetLoader


@dataclass(frozen=True)
class UGPhysicsItem:
    """UGPhysics 데이터셋 아이템"""
    index: int
    domain: str
    subject: str
    topic: str
    problem: str
    solution: str
    answers: str
    answer_type: str
    unit: Optional[str] = None
    is_multiple_answer: bool = False
    language: str = "EN"
    level: str = ""

    @property
    def question(self) -> str:
        return self.problem

    @property
    def answer(self) -> str:
        return self.answers

    @property
    def id(self) -> int:
        return self.index


class UGPhysicsDatasetLoader(DatasetLoader):
    """UGPhysics 데이터셋을 위한 JSONL 로더"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {self.path}")

    def _iter_lines(self) -> Iterator[dict]:
        """JSONL 파일의 각 줄을 순회"""
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류 (라인 건너뜀): {e}")
                    continue

    def load(self, limit: Optional[int] = None) -> list[UGPhysicsItem]:
        """UGPhysics JSONL 데이터셋을 로드합니다.
        
        Args:
            limit: 로드할 최대 아이템 수 (None이면 모든 아이템)
            
        Returns:
            UGPhysicsItem 리스트
        """
        items: list[UGPhysicsItem] = []

        for row in self._iter_lines():
            # 필수 필드 추출
            index = row.get("index", 0)
            domain = str(row.get("domain", "")).strip()
            subject = str(row.get("subject", "")).strip()
            topic = str(row.get("topic", "")).strip()
            problem = str(row.get("problem", "")).strip()
            solution = str(row.get("solution", "")).strip()
            answers = str(row.get("answers", "")).strip()
            answer_type = str(row.get("answer_type", "")).strip()
            unit = row.get("unit")
            is_multiple_answer = bool(row.get("is_multiple_answer", False))
            language = str(row.get("language", "EN")).strip()
            level = str(row.get("level", "")).strip()

            # 필수 필드 검증
            if not problem or not answers:
                continue

            items.append(UGPhysicsItem(
                index=index,
                domain=domain,
                subject=subject,
                topic=topic,
                problem=problem,
                solution=solution,
                answers=answers,
                answer_type=answer_type,
                unit=unit,
                is_multiple_answer=is_multiple_answer,
                language=language,
                level=level
            ))

            if limit is not None and len(items) >= limit:
                break

        return items


class UGPhysicsMultiSubjectLoader:
    """여러 UGPhysics 과목을 한번에 로드하는 헬퍼 클래스"""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

        # 사용 가능한 과목들
        self.subjects = [
            "AtomicPhysics",
            "ClassicalElectromagnetism",
            "ClassicalMechanics",
            "Electrodynamics",
            "GeometricalOptics",
            "QuantumMechanics",
            "Relativity",
            "SemiconductorPhysics",
            "Solid-StatePhysics",
            "StatisticalMechanics",
            "TheoreticalMechanics",
            "Thermodynamics",
            "WaveOptics"
        ]

    def load_subject(self, subject: str, language: str = "en", limit: Optional[int] = None) -> list[UGPhysicsItem]:
        """특정 과목의 데이터를 로드합니다.
        
        Args:
            subject: 과목명 (예: "ClassicalMechanics")
            language: 언어 ("en" 또는 "zh")
            limit: 로드할 최대 아이템 수
            
        Returns:
            UGPhysicsItem 리스트
        """
        if subject not in self.subjects:
            raise ValueError(f"지원하지 않는 과목입니다: {subject}. 사용 가능한 과목: {self.subjects}")

        file_path = self.base_path / subject / f"{language}.jsonl"
        loader = UGPhysicsDatasetLoader(file_path)
        return loader.load(limit)

    def load_all_subjects(self, language: str = "en", limit_per_subject: Optional[int] = None) -> dict[
        str, list[UGPhysicsItem]]:
        """모든 과목의 데이터를 로드합니다.
        
        Args:
            language: 언어 ("en" 또는 "zh")
            limit_per_subject: 과목당 최대 아이템 수
            
        Returns:
            과목명을 키로 하는 UGPhysicsItem 리스트 딕셔너리
        """
        results = {}
        for subject in self.subjects:
            try:
                results[subject] = self.load_subject(subject, language, limit_per_subject)
            except FileNotFoundError:
                print(f"경고: {subject} 파일을 찾을 수 없습니다.")
                results[subject] = []
        return results
