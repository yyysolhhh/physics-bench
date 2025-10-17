from dataclasses import dataclass
from typing import Optional

from rich.progress import Progress

from physics_bench.dataset.loader import DatasetItem
from physics_bench.llm.base import BaseLLMClient
from physics_bench.llm.openai_client import OpenAIClient
from physics_bench.llm.qwen_client import QwenClient
from .evaluator import ExactMatchEvaluator, EvaluationResult


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None


def _make_llm(spec: ModelSpec) -> BaseLLMClient:
    if spec.provider.lower() == "openai":
        return OpenAIClient(model_name=spec.model, temperature=spec.temperature, max_tokens=spec.max_tokens)
    elif spec.provider.lower() == "qwen":
        return QwenClient(model_name=spec.model, temperature=spec.temperature, max_tokens=spec.max_tokens)
    raise ValueError(f"지원하지 않는 provider: {spec.provider}")


class BenchmarkRunner:
    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec
        self.evaluator = ExactMatchEvaluator()

    def run_with_items(self, items: list[DatasetItem]) -> EvaluationResult:
        llm = _make_llm(self.model_spec)

        y_true: list[str] = []
        y_pred: list[str] = []

        with Progress() as progress:
            task = progress.add_task("질의 중...", total=len(items))
            for item in items:
                answer = llm.generate_answer(question=item.question)
                y_true.append(item.answer)
                y_pred.append(answer)
                progress.advance(task)

        return self.evaluator.evaluate(y_true=y_true, y_pred=y_pred)