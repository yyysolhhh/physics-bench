import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from rich.console import Console
from rich.progress import Progress

from physics_bench.llm import LLMRegistry
from physics_bench.llm.base import BaseLLMClient
from physics_bench.prompts import PHYSICS_USER_PROMPT
from physics_bench.utils.logging_config import setup_benchmark_logger
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
    def __init__(self, model_spec: ModelSpec, prompt_template, verbose: bool = False, log_file: Optional[str] = None,
                 concurrency: int = 8):
        self.model_spec = model_spec
        self.evaluator = PhysicsEvaluator()
        self.console = Console()
        self.verbose = verbose
        self.system_prompt = prompt_template
        self.user_prompt_template = PHYSICS_USER_PROMPT

        # 로깅 설정
        self.log_file = log_file
        self.logger = setup_benchmark_logger(log_file)
        self.start_time = None
        self.concurrency = max(1, int(concurrency))

    def run_with_items(self, items: list[Any]) -> tuple[EvaluationResult, list[dict]]:
        self.start_time = time.time()
        llm = _make_llm(self.model_spec)

        total_items = len(items)
        y_true: list[str] = [getattr(item, 'answer', '') for item in items]
        y_pred: list[str] = [""] * total_items
        detailed_results: list[dict] = [None] * total_items  # type: ignore[assignment]

        # 시작 메시지
        self.logger.info(f"벤치마크 시작 - 총 {total_items}개 문제")
        self.logger.info(f"모델: {self.model_spec.provider}/{self.model_spec.model}")

        async def _process_all() -> None:
            sem = asyncio.Semaphore(self.concurrency)

            async def _worker(i: int, item: Any) -> None:
                async with sem:
                    try:
                        user_prompt = self.user_prompt_template.format(question=item.question)
                        answer = await asyncio.to_thread(
                            llm.generate_answer,
                            self.system_prompt,
                            user_prompt,
                        )
                        cleaned_answer = self._clean_answer(answer)
                        is_correct = self.evaluator.evaluate_single(cleaned_answer, item.answer)
                        item_id = getattr(item, 'id', getattr(item, 'index', i + 1))
                        self.logger.info(
                            f"ID:{item_id} | 결과:{'정답' if is_correct else '오답'} | LLM답변:{cleaned_answer} | 정답:{item.answer} | 질문:{item.question}"
                        )
                        detailed_results[i] = {
                            'id': item_id,
                            'problem_id': getattr(item, 'problemid', ''),
                            'source': getattr(item, 'source', getattr(item, 'subject', 'Unknown')),
                            'question': item.question,
                            'ground_truth': item.answer,
                            'predicted': cleaned_answer,
                            'is_correct': is_correct,
                        }
                        y_pred[i] = cleaned_answer
                    except Exception as e:
                        item_id = getattr(item, 'id', getattr(item, 'index', i + 1))
                        error_msg = f"ID:{item_id} | 에러: {str(e)}"
                        self.logger.error(error_msg)
                        detailed_results[i] = {
                            'id': item_id,
                            'problem_id': getattr(item, 'problemid', ''),
                            'source': getattr(item, 'source', getattr(item, 'subject', 'Unknown')),
                            'question': item.question,
                            'ground_truth': item.answer,
                            'predicted': f"ERROR: {str(e)}",
                            'is_correct': False,
                        }
                        y_pred[i] = "ERROR"
                    finally:
                        progress.advance(task)

            async with asyncio.TaskGroup() as tg:
                for i, item in enumerate(items):
                    tg.create_task(_worker(i, item))

        with Progress() as progress:
            task = progress.add_task("진행중...", total=total_items)
            asyncio.run(_process_all())

        # 전체 결과 계산
        result = self.evaluator.evaluate(y_true=y_true, y_pred=y_pred)

        # 완료 메시지
        elapsed_time = time.time() - self.start_time

        # 토큰 사용량 정보
        usage_stats = llm.get_usage_stats()
        if usage_stats:
            self.logger.info(
                f"토큰 사용량 - Total:{usage_stats.get('total_tokens', 0)}, Prompt:{usage_stats.get('prompt_tokens', 0)}, Completion:{usage_stats.get('completion_tokens', 0)}"
            )

        self.logger.info(
            f"벤치마크 완료 - 정답:{result.correct}/{result.total} (정확도:{result.accuracy:.2%}) - 소요시간:{elapsed_time:.1f}초"
        )

        # 상세 통계 출력
        self._print_detailed_stats(result, [r for r in detailed_results if r is not None])

        # JSON 파일 저장
        if self.log_file:
            self._save_results_json(result, [r for r in detailed_results if r is not None], elapsed_time, usage_stats)

        return result, [r for r in detailed_results if r is not None]

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

        if len(source_stats) > 1:  # 여러 소스가 있을 때만 출력
            self.console.print(f"\n[bold red]Source별 성능:[/bold red]")
            for source, stats in source_stats.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                self.console.print(f"  {source}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")

    def _save_results_json(self, result: EvaluationResult, detailed_results: list, elapsed_time: float,
                           usage_stats: Optional[dict] = None):
        """결과를 JSON 파일로 저장"""
        if not self.log_file:
            return

        # 로그 파일 경로를 기반으로 JSON 파일 경로 생성
        log_path = Path(self.log_file)
        json_path = log_path.parent / "results.json"

        # 메타데이터 구성
        metadata = {
            'model_provider': self.model_spec.provider,
            'model_name': self.model_spec.model,
            'temperature': self.model_spec.temperature,
            'max_tokens': self.model_spec.max_tokens,
            'prompt_template': self.system_prompt[:100] + '...' if len(
                self.system_prompt) > 100 else self.system_prompt,
        }

        # 요약 통계
        summary = {
            'total': result.total,
            'correct': result.correct,
            'accuracy': result.accuracy,
            'elapsed_time_seconds': round(elapsed_time, 2)
        }

        # 토큰 사용량 정보 추가
        if usage_stats:
            summary['token_usage'] = usage_stats

        # 소스별 통계 계산
        source_stats = {}
        for item in detailed_results:
            source = item['source']
            if source not in source_stats:
                source_stats[source] = {'total': 0, 'correct': 0}
            source_stats[source]['total'] += 1
            if item['is_correct']:
                source_stats[source]['correct'] += 1

        # JSON 데이터 구성
        output_data = {
            'metadata': metadata,
            'summary': summary,
            'source_statistics': {
                source: {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0
                }
                for source, stats in source_stats.items()
            },
            'results': detailed_results
        }

        # JSON 파일 저장
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"결과 파일: {json_path}")
