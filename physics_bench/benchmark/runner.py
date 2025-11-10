import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

# from rich.console import Console
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
                 concurrency: int = 1, solution_prompt_template: Optional[str] = None):
        self.model_spec = model_spec
        self.evaluator = PhysicsEvaluator()
        # self.console = Console()  # 주석처리: Console 사용 안함
        self.verbose = verbose
        self.system_prompt = prompt_template
        self.solution_system_prompt = solution_prompt_template
        self.user_prompt_template = PHYSICS_USER_PROMPT

        self.log_file = log_file
        self.logger = setup_benchmark_logger(log_file)  # 파일 저장 로거
        self.start_time = None
        self.concurrency = max(1, int(concurrency))
        self.json_path = None
        self.file_lock = asyncio.Lock()  # 파일 쓰기 동기화용
        self.pending_results = []  # 배치 저장을 위한 대기 중인 결과
        self.save_batch_size = 10  # 10개마다 저장
        self.last_save_time = time.time()
        self.save_interval = 30.0  # 30초마다 저장
        if log_file:
            log_path = Path(log_file)
            self.json_path = log_path.parent / "results.json"

    def run_with_items(self, items: list[Any], existing_results_file: Optional[Path] = None) -> tuple[
        EvaluationResult, list[dict]]:
        self.start_time = time.time()
        llm = _make_llm(self.model_spec)

        total_items = len(items)
        detailed_results: list[Optional[dict]] = [None for _ in range(total_items)]

        self.logger.info(f"벤치마크 시작 - 총 {total_items}개 문제")
        self.logger.info(f"모델: {self.model_spec.provider}/{self.model_spec.model}")

        async def _process_all() -> None:
            sem = asyncio.Semaphore(self.concurrency)

            async def _worker(i: int, item: Any) -> None:
                async with sem:
                    try:
                        user_prompt = self.user_prompt_template.format(question=item.question)
                        # 1단계: 정답만 수집
                        answer_only = await asyncio.to_thread(
                            llm.generate_answer,
                            self.system_prompt,
                            user_prompt,
                        )
                        # 1차 자동판정 (정답만, 규칙 기반)
                        is_correct, pred_list, gt_list = self.evaluator.evaluate_first_stage(
                            answer_only, item.answer
                        )
                        # 1차에서 평가에 쓰이는 정답만 (\boxed{...} 형식)
                        answer_extracted = getattr(self.evaluator, 'last_extracted_pred', None)

                        solution_text = None
                        student_solution_for_eval = None  # 2차 평가에 사용된 student_solution (풀이만)
                        student_answers_for_eval = None  # 2차 평가에 사용된 student_answers (정답만)

                        # 필요 시 2단계: 풀이 요청 후 평가
                        if not is_correct:
                            # 참고 해설 추출 (UGPhysics 데이터셋의 solution 필드, 2차 판정에서만 사용)
                            reference_solution = getattr(item, 'solution', None) or ""

                            solution_text = await asyncio.to_thread(
                                llm.generate_answer,
                                self.solution_system_prompt,
                                user_prompt,
                            )
                            # 2차 답 전체를 콘솔에 임시 출력
                            # print(f"[DEBUG] 2차 답 전체: {solution_text}")

                            is_correct = self.evaluator.evaluate_second_stage(
                                question=item.question,
                                predicted_raw=answer_only,
                                ground_truth_raw=item.answer,
                                pred_list=pred_list,
                                gt_list=gt_list,
                                student_solution=solution_text,
                                reference_solution=reference_solution,
                            )
                            # 2차 평가에 사용된 student_solution과 student_answers 가져오기
                            student_solution_for_eval = getattr(self.evaluator, 'last_student_solution', None)
                            student_answers_for_eval = getattr(self.evaluator, 'last_student_answers', None)
                        item_id = getattr(item, 'id', getattr(item, 'index', i + 1))
                        self.logger.info(
                            f"ID:{item_id} | 결과:{'정답' if is_correct else '오답'} | 1차답:{answer_only} | 2차풀이:{'있음' if solution_text else '없음'} | 정답:{item.answer}"
                        )
                        result_dict = {
                            'id': item_id,
                            'problem_id': getattr(item, 'problemid', ''),
                            'source': getattr(item, 'source', getattr(item, 'subject', 'Unknown')),
                            'domain': getattr(item, 'domain', ''),
                            'subject': getattr(item, 'subject', ''),
                            'topic': getattr(item, 'topic', ''),
                            'answer_type': getattr(item, 'answer_type', ''),
                            'level': getattr(item, 'level', ''),
                            'question': item.question,
                            'ground_truth': item.answer,
                            'answer_only_raw': answer_only,  # 1차 LLM 답 전체
                            'answer_extracted': answer_extracted,  # 1차에서 평가에 쓰이는 정답만
                            'student_solution': student_solution_for_eval,  # 2차 평가에 사용된 student_solution (풀이만)
                            'student_answers': student_answers_for_eval,  # 2차 평가에 사용된 student_answers (정답만)
                            'model_judge_msg': getattr(self.evaluator, 'last_model_judge_msg', None),
                            'is_correct': is_correct,
                            'error': None,  # 에러 없음
                        }
                        detailed_results[i] = result_dict

                        # 배치 저장 (10개마다 또는 30초마다)
                        if self.json_path:
                            await self._queue_result_for_save(result_dict, existing_results_file)
                    except Exception as e:
                        item_id = getattr(item, 'id', getattr(item, 'index', i + 1))
                        error_msg = f"ID:{item_id} | 에러: {str(e)}"
                        self.logger.error(error_msg)

                        result_dict = {
                            'id': item_id,
                            'problem_id': getattr(item, 'problemid', ''),
                            'source': getattr(item, 'source', getattr(item, 'subject', 'Unknown')),
                            'domain': getattr(item, 'domain', ''),
                            'subject': getattr(item, 'subject', ''),
                            'topic': getattr(item, 'topic', ''),
                            'answer_type': getattr(item, 'answer_type', ''),
                            'level': getattr(item, 'level', ''),
                            'question': item.question,
                            'ground_truth': item.answer,
                            'answer_only_raw': None,
                            'answer_extracted': None,
                            'student_solution': None,
                            'student_answers': None,
                            'model_judge_msg': None,
                            'is_correct': False,
                            'error': str(e),  # 에러 메시지 저장
                        }
                        detailed_results[i] = result_dict

                        # 배치 저장 (10개마다 또는 30초마다)
                        if self.json_path:
                            await self._queue_result_for_save(result_dict, existing_results_file)

                        progress.advance(task)

            async with asyncio.TaskGroup() as tg:
                for i, item in enumerate(items):
                    tg.create_task(_worker(i, item))

        with Progress() as progress:
            task = progress.add_task("진행중...", total=total_items)
            asyncio.run(_process_all())

        completed_results = [r for r in detailed_results if r is not None]
        total_correct = sum(1 for r in completed_results if r.get('is_correct', False))
        result = EvaluationResult(total=len(completed_results), correct=total_correct)

        elapsed_time = time.time() - self.start_time

        usage_stats = llm.get_usage_stats()
        if usage_stats:
            self.logger.info(
                f"토큰 사용량 - Total:{usage_stats.get('total_tokens', 0)}, Prompt:{usage_stats.get('prompt_tokens', 0)}, Completion:{usage_stats.get('completion_tokens', 0)}"
            )

        self.logger.info(
            f"벤치마크 완료 - 정답:{result.correct}/{result.total} (정확도:{result.accuracy:.2%}) - 소요시간:{elapsed_time:.1f}초"
        )

        self._print_detailed_stats(result, [r for r in detailed_results if r is not None])

        # 남은 결과 모두 저장
        if self.json_path and self.pending_results:
            asyncio.run(self._flush_pending_results(existing_results_file))

        # 최종 통계 업데이트
        if self.log_file:
            self._update_results_json_summary([r for r in detailed_results if r is not None], elapsed_time, usage_stats,
                                              existing_results_file)

        return result, [r for r in detailed_results if r is not None]

    async def _queue_result_for_save(self, result_dict: dict, existing_results_file: Optional[Path] = None):
        """결과를 대기열에 추가하고, 배치 크기나 시간 간격에 도달하면 저장"""
        async with self.file_lock:
            self.pending_results.append(result_dict)

            current_time = time.time()
            should_save = (
                    len(self.pending_results) >= self.save_batch_size or
                    (current_time - self.last_save_time) >= self.save_interval
            )

            if should_save:
                await self._flush_pending_results_internal(existing_results_file)

    async def _flush_pending_results(self, existing_results_file: Optional[Path] = None):
        """대기 중인 모든 결과 저장"""
        async with self.file_lock:
            await self._flush_pending_results_internal(existing_results_file)

    async def _flush_pending_results_internal(self, existing_results_file: Optional[Path] = None):
        """내부 저장 로직 (락이 이미 획득된 상태에서 호출)"""
        if not self.pending_results:
            return

        # 기존 결과 읽기
        existing_results = []
        existing_ids = set()

        # 먼저 self.json_path에서 읽기 (이미 저장된 결과)
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if 'results' in existing_data:
                        existing_results = existing_data['results']
                        existing_ids = {str(r.get('id')) for r in existing_results if r.get('id') is not None}
            except Exception:
                pass

        # self.json_path에 없으면 existing_results_file에서 읽기 (초기 실행 시)
        if not existing_results and existing_results_file and existing_results_file.exists():
            try:
                with open(existing_results_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if 'results' in existing_data:
                        existing_results = existing_data['results']
                        existing_ids = {str(r.get('id')) for r in existing_results if r.get('id') is not None}
            except Exception:
                pass

        # 새 결과 추가 (중복 체크)
        new_count = 0
        for result_dict in self.pending_results:
            result_id = result_dict.get('id')
            if result_id is not None and str(result_id) not in existing_ids:
                existing_results.append(result_dict)
                existing_ids.add(str(result_id))
                new_count += 1

        if new_count > 0:
            # 통계 재계산
            total_correct = sum(1 for r in existing_results if r.get('is_correct', False))
            total_items = len(existing_results)
            overall_accuracy = total_correct / total_items if total_items > 0 else 0

            # 메타데이터 및 통계 업데이트
            metadata = {
                'model_provider': self.model_spec.provider,
                'model_name': self.model_spec.model,
                'temperature': self.model_spec.temperature,
                'max_tokens': self.model_spec.max_tokens,
                'prompt_template': self.system_prompt[:100] + '...' if len(
                    self.system_prompt) > 100 else self.system_prompt,
            }

            source_stats = {}
            for item in existing_results:
                source = item['source']
                if source not in source_stats:
                    source_stats[source] = {'total': 0, 'correct': 0}
                source_stats[source]['total'] += 1
                if item['is_correct']:
                    source_stats[source]['correct'] += 1

            output_data = {
                'metadata': metadata,
                'summary': {
                    'total': total_items,
                    'correct': total_correct,
                    'accuracy': overall_accuracy,
                },
                'source_statistics': {
                    source: {
                        'total': stats['total'],
                        'correct': stats['correct'],
                        'accuracy': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0
                    }
                    for source, stats in source_stats.items()
                },
                'results': existing_results
            }

            # 파일에 저장
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"결과 저장: {new_count}개 항목 추가 (총 {len(existing_results)}개)")

        # 대기열 비우기
        self.pending_results.clear()
        self.last_save_time = time.time()

    def _print_detailed_stats(self, result: EvaluationResult, detailed_results: list):
        """상세 통계 출력"""
        self.logger.info(f"\n=== 벤치마크 결과 ===")
        self.logger.info(f"총 문제 수: {result.total}")
        self.logger.info(f"정답 수: {result.correct}")
        self.logger.info(f"정확도: {result.accuracy:.2%}")

        source_stats = {}
        for item in detailed_results:
            source = item['source']
            if source not in source_stats:
                source_stats[source] = {'total': 0, 'correct': 0}
            source_stats[source]['total'] += 1
            if item['is_correct']:
                source_stats[source]['correct'] += 1

        if len(source_stats) > 1:
            self.logger.info(f"\nSource별 성능:")
            for source, stats in source_stats.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                self.logger.info(f"  {source}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")

    def _update_results_json_summary(self, detailed_results: list, elapsed_time: float,
                                     usage_stats: Optional[dict] = None, existing_results_file: Optional[Path] = None):
        """결과 JSON 파일의 통계만 업데이트 (각 항목은 이미 저장됨)"""
        if not self.json_path or not self.json_path.exists():
            return

        try:
            # 기존 파일 읽기
            with open(self.json_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            # 통계 재계산
            all_results = output_data.get('results', [])
            total_correct = sum(1 for r in all_results if r.get('is_correct', False))
            total_items = len(all_results)
            overall_accuracy = total_correct / total_items if total_items > 0 else 0

            # summary 업데이트
            output_data['summary'] = {
                'total': total_items,
                'correct': total_correct,
                'accuracy': overall_accuracy,
                'elapsed_time_seconds': round(elapsed_time, 2)
            }

            if usage_stats:
                output_data['summary']['token_usage'] = usage_stats

            # source_statistics 재계산
            source_stats = {}
            for item in all_results:
                source = item['source']
                if source not in source_stats:
                    source_stats[source] = {'total': 0, 'correct': 0}
                source_stats[source]['total'] += 1
                if item['is_correct']:
                    source_stats[source]['correct'] += 1

            output_data['source_statistics'] = {
                source: {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0
                }
                for source, stats in source_stats.items()
            }

            # 파일에 저장
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"결과 파일 업데이트: {self.json_path} (총 {len(all_results)}개 결과)")
        except Exception as e:
            self.logger.warning(f"결과 파일 통계 업데이트 실패: {e}")
