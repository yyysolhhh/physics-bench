import re
from dataclasses import dataclass
from typing import Optional, Any

from rich.console import Console
from rich.progress import Progress

from physics_bench.llm import LLMRegistry
from physics_bench.llm.base import BaseLLMClient
from physics_bench.prompts import PHYSICS_USER_PROMPT
from .evaluator import PhysicsEvaluator, EvaluationResult


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None


def _make_llm(spec: ModelSpec) -> BaseLLMClient:
    return LLMRegistry.create_client(
        provider=spec.provider,
        model_name=spec.model,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens
    )


class BenchmarkRunner:
    def __init__(self, model_spec: ModelSpec, prompt_template, verbose: bool = False):
        self.model_spec = model_spec
        self.evaluator = PhysicsEvaluator()
        self.console = Console()
        self.verbose = verbose
        self.system_prompt = prompt_template
        self.user_prompt_template = PHYSICS_USER_PROMPT

    def run_with_items(self, items: list[Any]) -> EvaluationResult:
        llm = _make_llm(self.model_spec)

        y_true: list[str] = []
        y_pred: list[str] = []
        detailed_results = []

        with Progress() as progress:
            task = progress.add_task("진행중...", total=len(items))

            for i, item in enumerate(items):
                user_prompt = self.user_prompt_template.format(question=item.question)

                answer = llm.generate_answer(system_prompt=self.system_prompt, user_prompt=user_prompt)
                print(answer)
                cleaned_answer = self._clean_answer(answer)

                is_correct = self.evaluator.evaluate_single(cleaned_answer, item.answer)

                # 상세 결과 저장 (제네릭하게 처리)
                detailed_results.append({
                    'id': getattr(item, 'id', getattr(item, 'index', i)),
                    'problem_id': getattr(item, 'problemid', ''),
                    'source': getattr(item, 'source', getattr(item, 'subject', 'Unknown')),
                    'question': item.question[:100] + "..." if len(item.question) > 100 else item.question,
                    'ground_truth': item.answer,
                    'predicted': cleaned_answer,
                    'is_correct': is_correct
                })

                if self.verbose:
                    self.console.print(f"\n[bold blue]문제 {i + 1}/{len(items)}[/bold blue]")
                    source = getattr(item, 'source', getattr(item, 'subject', 'Unknown'))
                    problem_id = getattr(item, 'problemid', '')
                    self.console.print(f"[purple]Source:[/purple] {source}")
                    if problem_id:
                        self.console.print(f"[purple]Problem ID:[/purple] {problem_id}")
                    self.console.print(f"[green]정답:[/green] {item.answer}")
                    self.console.print(f"[blue]예측:[/blue] {cleaned_answer}")
                    self.console.print(
                        f"[{'green' if is_correct else 'red'}]결과:[/{'green' if is_correct else 'red'}] {'정답' if is_correct else '오답'}")
                    self.console.print("-" * 80)

                y_true.append(item.answer)
                y_pred.append(cleaned_answer)
                progress.advance(task)

        # 전체 결과 계산
        result = self.evaluator.evaluate(y_true=y_true, y_pred=y_pred)

        # 상세 통계 출력
        self._print_detailed_stats(result, detailed_results)

        return result

    def _clean_answer(self, answer: str) -> str:
        """LLM 답변에서 수치와 단위 추출하여 정리"""
        # 1. "답:" 패턴 찾기 (우선순위)
        answer_patterns = [
            r'답:\s*([^\n]+)',
            r'Answer:\s*([^\n]+)',
            r'최종\s*답:\s*([^\n]+)',
            r'정답:\s*([^\n]+)',
            r'결과:\s*([^\n]+)',
            r'Final\s*Answer:\s*([^\n]+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
                # 추가 정리
                cleaned = re.sub(r'^[-=*]\s*', '', cleaned)  # 앞의 기호 제거
                cleaned = re.sub(r'\s*[-=*]\s*$', '', cleaned)  # 뒤의 기호 제거
                return cleaned

        # 2. "답:" 없이 수치와 단위가 포함된 줄 찾기
        lines = answer.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # 숫자와 단위가 모두 포함된 줄 찾기
            if re.search(r'-?\d+\.?\d*\s*[a-zA-Z°]+', line):
                cleaned = re.sub(r'^[-=*]\s*', '', line)
                cleaned = re.sub(r'\s*[-=*]\s*$', '', cleaned)
                return cleaned

        # 3. 숫자만 포함된 줄 찾기
        for line in reversed(lines):
            line = line.strip()
            if re.search(r'-?\d+\.?\d*', line) and len(line) < 50:  # 너무 긴 줄 제외
                cleaned = re.sub(r'^[-=*]\s*', '', line)
                cleaned = re.sub(r'\s*[-=*]\s*$', '', cleaned)
                return cleaned

        # 4. 마지막으로 전체 답변에서 수치 추출
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[-1]  # 마지막 숫자 반환

        return answer.strip()

    def _print_detailed_stats(self, result: EvaluationResult, detailed_results: list):
        """상세 통계 출력"""
        self.console.print(f"\n[bold green]=== 벤치마크 결과 ===[/bold green]")
        self.console.print(f"총 문제 수: {result.total}")
        self.console.print(f"정답 수: {result.correct}")
        self.console.print(f"정확도: {result.accuracy:.2%}")

        # Source별 성능
        source_stats = {}
        for item in detailed_results:
            source = item['source']
            if source not in source_stats:
                source_stats[source] = {'total': 0, 'correct': 0}
            source_stats[source]['total'] += 1
            if item['is_correct']:
                source_stats[source]['correct'] += 1

        self.console.print(f"\n[bold yellow]Source별 성능:[/bold yellow]")
        for source, stats in source_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            self.console.print(f"  {source}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")
