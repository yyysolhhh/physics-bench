import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import print

from physics_bench.benchmark import ModelSpec, BenchmarkRunner
from physics_bench.dataset import (
    UGPhysicsMultiSubjectLoader,
    download_huggingface_dataset
)
from physics_bench.llm import LLMRegistry
from physics_bench.prompts import (
    PHYSICS_SYSTEM_PROMPT_KO,
    PHYSICS_SYSTEM_PROMPT_EN,
    PHYSICS_SOLUTION_PROMPT_KO,
    PHYSICS_SOLUTION_PROMPT_EN,
    PHYSICS_USER_PROMPT
)
from physics_bench.utils.config import get_env
from physics_bench.utils.logging_config import generate_log_filename

app = typer.Typer()


@app.command("run")
def run(
        provider: str = typer.Option("gemini", "--provider", help=f"모델 제공자 ({', '.join(LLMRegistry.get_providers())})"),
        model: str = typer.Option(None, "--model", help="모델 이름 (Ollama 사용 시 필수, 예: qwen2.5-math:7b)"),
        temperature: float = typer.Option(0.0, "--temperature", help="샘플링 온도"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="최대 토큰(미지정 시 모델 기본값)"),
        limit: Optional[int] = typer.Option(None, "--limit", help="각 과목마다 상위 N개로 제한"),
        verbose: bool = typer.Option(True, "--verbose", "-v", help="상세한 출력 표시"),
        lang: str = typer.Option("ko", "--lang", help="프롬프트 언어 (ko/en)"),
):
    if provider not in LLMRegistry.get_providers():
        available = ", ".join(LLMRegistry.get_providers())
        raise typer.BadParameter(f"지원하지 않는 provider: {provider}. 사용 가능: {available}")

    lang = lang.lower()
    if lang not in ("ko", "en"):
        raise typer.BadParameter("--lang 은 ko 또는 en 이어야 합니다.")
    system_prompt = PHYSICS_SYSTEM_PROMPT_KO if lang == "ko" else PHYSICS_SYSTEM_PROMPT_EN
    solution_prompt = PHYSICS_SOLUTION_PROMPT_KO if lang == "ko" else PHYSICS_SOLUTION_PROMPT_EN

    # Ollama는 모델 이름을 직접 지정, 다른 provider는 환경변수 사용
    if provider == "ollama":
        if not model:
            raise typer.BadParameter("Ollama 사용 시 --model 옵션이 필요합니다. 예: --model qwen2.5-math:7b")
        model_name = model
    else:
        model_env_var = LLMRegistry.get_model_env_var(provider)
        model_name = get_env(model_env_var)
        if not model_name:
            raise typer.BadParameter(f"{model_env_var} 환경변수가 설정되지 않았습니다.")

    spec = ModelSpec(provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

    base_log_file = generate_log_filename(model_name=model_name)
    base_output_dir = Path(base_log_file).parent

    multi_loader = UGPhysicsMultiSubjectLoader(Path("dataset") / "ugphysics")
    all_subjects_data = multi_loader.load_all_subjects(language="en", limit_per_subject=limit)

    print(f"UGPhysics 로드 완료: {len(all_subjects_data)}개 과목")

    subject_reports = {}
    all_detailed_results = []  # 모든 과목의 상세 결과 수집

    # 각 과목별로 실행
    for subject_name, subject_items in all_subjects_data.items():
        print(f"\n[bold cyan]=== {subject_name} 실행 중 ===[/bold cyan]")

        run_timestamp = base_output_dir.name  # YYYYMMDD_HHMMSS
        subject_log_file = str(base_output_dir / subject_name / f"{subject_name}_{run_timestamp}.log")

        runner = BenchmarkRunner(
            model_spec=spec,
            prompt_template=system_prompt,
            verbose=verbose,
            log_file=subject_log_file,
            solution_prompt_template=solution_prompt,
        )
        report, detailed_results = runner.run_with_items(subject_items)

        subject_reports[subject_name] = report
        all_detailed_results.extend(detailed_results)

    # 전체 결과 계산
    total_correct = sum(r.correct for r in subject_reports.values())
    total_items = sum(r.total for r in subject_reports.values())
    overall_accuracy = total_correct / total_items if total_items > 0 else 0

    # 카테고리별 정답률 계산
    def calculate_accuracy_by_key(key: str) -> dict:
        """특정 키(domain, subject, topic, answer_type, level)별 정답률 계산"""
        stats = {}
        for result in all_detailed_results:
            value = result.get(key, 'Unknown')
            if value not in stats:
                stats[value] = {'total': 0, 'correct': 0}
            stats[value]['total'] += 1
            if result.get('is_correct', False):
                stats[value]['correct'] += 1
        
        return {
            k: {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': round(v['correct'] / v['total'], 4) if v['total'] > 0 else 0.0
            }
            for k, v in stats.items()
        }

    overall_json_path = base_output_dir / "overall_results.json"
    overall_data = {
        'metadata': {
            'model_provider': spec.provider,
            'model_name': spec.model,
            'temperature': spec.temperature,
            'max_tokens': spec.max_tokens,
            'lang': lang,
            'timestamp': datetime.now().isoformat()
        },
        'summary': {
            'total_problems': total_items,
            'total_correct': total_correct,
            'overall_accuracy': round(overall_accuracy, 4),
            'subject_count': len(subject_reports)
        },
        'subject_statistics': {
            subject: {
                'total': report.total,
                'correct': report.correct,
                'accuracy': round(report.accuracy, 4)
            }
            for subject, report in subject_reports.items()
        },
        'domain_statistics': calculate_accuracy_by_key('domain'),
        'topic_statistics': calculate_accuracy_by_key('topic'),
        'answer_type_statistics': calculate_accuracy_by_key('answer_type'),
        'level_statistics': calculate_accuracy_by_key('level'),
    }

    with open(overall_json_path, 'w', encoding='utf-8') as f:
        json.dump(overall_data, f, ensure_ascii=False, indent=2)

    print(f"\n[bold green]=== 전체 결과 ===[/bold green]")
    print(f"[bold]전체 결과 파일: {overall_json_path}[/bold]")
    print(f"[bold]최종 결과: {total_correct}/{total_items} (정확도 {overall_accuracy * 100:.2f}%)[/bold]")

    if overall_data['summary'].get('token_usage'):
        usage = overall_data['summary']['token_usage']
        print(
            f"[bold cyan]토큰 사용량: Total={usage.get('total_tokens', 0)}, Prompt={usage.get('prompt_tokens', 0)}, Completion={usage.get('completion_tokens', 0)}[/bold cyan]")


@app.command("ask")
def ask(
        provider: str = typer.Option("gemini", "--provider", help=f"모델 제공자 ({', '.join(LLMRegistry.get_providers())})"),
        model: str = typer.Option(None, "--model", help="모델 이름 (Ollama 사용 시 필수, 예: qwen2.5-math:7b)"),
        temperature: float = typer.Option(0.0, "--temperature", help="샘플링 온도"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="최대 토큰(미지정 시 모델 기본값)"),
        lang: str = typer.Option("ko", "--lang", help="프롬프트 언어 (ko/en)"),
        with_solution: bool = typer.Option(False, "--with-solution", help="풀이 과정 포함 여부"),
        question: Optional[str] = typer.Option(None, "--question", "-q", help="질문 (미지정 시 입력 요청)"),
):
    """한 문제를 입력받아 LLM의 답변을 받습니다."""
    if provider not in LLMRegistry.get_providers():
        available = ", ".join(LLMRegistry.get_providers())
        raise typer.BadParameter(f"지원하지 않는 provider: {provider}. 사용 가능: {available}")

    lang = lang.lower()
    if lang not in ("ko", "en"):
        raise typer.BadParameter("--lang 은 ko 또는 en 이어야 합니다.")

    # Ollama는 모델 이름을 직접 지정, 다른 provider는 환경변수 사용
    if provider == "ollama":
        if not model:
            raise typer.BadParameter("Ollama 사용 시 --model 옵션이 필요합니다. 예: --model qwen2.5-math:7b")
        model_name = model
    else:
        model_env_var = LLMRegistry.get_model_env_var(provider)
        model_name = get_env(model_env_var)
        if not model_name:
            raise typer.BadParameter(f"{model_env_var} 환경변수가 설정되지 않았습니다.")

    spec = ModelSpec(provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

    # LLM 클라이언트 생성
    try:
        llm_client = LLMRegistry.create_client(
            provider=spec.provider,
            model_name=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens
        )
        print(f"[bold]Provider: {spec.provider}, Model: {spec.model}[/bold]\n")
    except Exception as e:
        print(f"[bold red]❌ LLM 클라이언트 생성 실패: {e}[/bold red]")
        raise typer.Exit(1)

    # 질문 입력
    if not question:
        question = typer.prompt("질문을 입력하세요")

    if not question.strip():
        print("[bold red]❌ 질문이 비어있습니다.[/bold red]")
        raise typer.Exit(1)

    # 프롬프트 구성
    if with_solution:
        system_prompt = PHYSICS_SOLUTION_PROMPT_KO if lang == "ko" else PHYSICS_SOLUTION_PROMPT_EN
        print("[bold cyan]풀이 과정 포함 모드[/bold cyan]")
    else:
        system_prompt = PHYSICS_SYSTEM_PROMPT_KO if lang == "ko" else PHYSICS_SYSTEM_PROMPT_EN
        print("[bold cyan]정답만 모드[/bold cyan]")
    
    user_prompt = PHYSICS_USER_PROMPT.format(question=question)

    # LLM 호출
    try:
        print("[bold cyan]답변 생성 중...[/bold cyan]")
        response = llm_client.generate_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # 답변 출력
        print("\n" + "=" * 50)
        print("[bold green]답변:[/bold green]")
        print(response)
        print("=" * 50)

        # 사용량 통계 출력
        usage_stats = llm_client.get_usage_stats()
        if usage_stats:
            print(f"\n[bold cyan]토큰 사용량:[/bold cyan] {usage_stats}")

    except Exception as e:
        print(f"[bold red]❌ 오류 발생: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("download")
def download(
        dataset_name: str = typer.Argument(..., help="Hugging Face 데이터셋 이름"),
        split: str = typer.Argument(..., help="데이터셋 split (train/test/validation)"),
        output: str = typer.Option(None, "--output", "-o", help="출력 파일명"),
        limit: Optional[int] = typer.Option(None, "--limit", help="다운로드할 항목 수 제한"),
):
    """Hugging Face에서 데이터셋을 다운로드하고 JSON으로 저장"""
    try:
        download_huggingface_dataset(
            dataset_name=dataset_name,
            output_file=output,
            split=split,
            limit=limit
        )
    except Exception as e:
        print(f"다운로드 실패: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
