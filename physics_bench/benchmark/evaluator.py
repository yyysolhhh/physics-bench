from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class ExactMatchEvaluator:

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def evaluate(self, y_true: list[str], y_pred: list[str]) -> EvaluationResult:
        # TODO: correct 결정 기준
        assert len(y_true) == len(y_pred)
        total = len(y_true)
        correct = 0
        for t, p in zip(y_true, y_pred):
            if self._normalize(t) == self._normalize(p):
                correct += 1
        return EvaluationResult(total=total, correct=correct)