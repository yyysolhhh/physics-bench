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
    PHYSICS_BENCHMARK_PROMPT,
    PHYSICS_SIMPLE_PROMPT,
    PHYSICS_DETAILED_PROMPT,
    PHYSICS_NUMERICAL_PROMPT
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
        prompt_style: str = typer.Option("numerical", "--prompt",
                                         help="프롬프트 스타일 (simple/benchmark/detailed/numerical)"),
):
    if provider not in LLMRegistry.get_providers():
        available = ", ".join(LLMRegistry.get_providers())
        raise typer.BadParameter(f"지원하지 않는 provider: {provider}. 사용 가능: {available}")

    prompt_templates = {
        "simple": PHYSICS_SIMPLE_PROMPT,
        "benchmark": PHYSICS_BENCHMARK_PROMPT,
        "detailed": PHYSICS_DETAILED_PROMPT,
        "numerical": PHYSICS_NUMERICAL_PROMPT,
    }

    if prompt_style not in prompt_templates:
        available = ", ".join(prompt_templates.keys())
        raise typer.BadParameter(f"지원하지 않는 prompt_style: {prompt_style}. 사용 가능: {available}")

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

    # 로그 파일 경로 설정 (기본 경로)
    base_log_file = generate_log_filename(model_name=model_name)
    base_output_dir = Path(base_log_file).parent

    multi_loader = UGPhysicsMultiSubjectLoader(Path("dataset") / "ugphysics")
    all_subjects_data = multi_loader.load_all_subjects(language="en", limit_per_subject=limit)

    print(f"UGPhysics 로드 완료: {len(all_subjects_data)}개 과목")

    all_results = []
    subject_reports = {}

    # 각 과목별로 실행
    for subject_name, subject_items in all_subjects_data.items():
        print(f"\n[bold cyan]=== {subject_name} 실행 중 ===[/bold cyan]")

        # 과목별 로그 파일 경로
        subject_log_file = str(base_output_dir / subject_name / "benchmark.log")

        # 과목별 runner 실행
        runner = BenchmarkRunner(
            model_spec=spec,
            prompt_template=prompt_templates[prompt_style],
            verbose=verbose,
            log_file=subject_log_file
        )
        report, detailed_results = runner.run_with_items(subject_items)

        # 과목별 결과 저장
        subject_reports[subject_name] = report

        # 전체 결과에 추가 (각 문제별 상세 결과에 subject 추가)
        for result in detailed_results:
            result['subject'] = subject_name
        all_results.extend(detailed_results)

    # 전체 결과 계산
    total_correct = sum(r.correct for r in subject_reports.values())
    total_items = sum(r.total for r in subject_reports.values())
    overall_accuracy = total_correct / total_items if total_items > 0 else 0

    # 전체 결과 JSON 저장 (날짜 폴더 바로 아래)
    overall_json_path = base_output_dir / "overall_results.json"
    overall_data = {
        'metadata': {
            'model_provider': spec.provider,
            'model_name': spec.model,
            'temperature': spec.temperature,
            'max_tokens': spec.max_tokens,
            'prompt_style': prompt_style,
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
        'all_problems': all_results
    }

    with open(overall_json_path, 'w', encoding='utf-8') as f:
        json.dump(overall_data, f, ensure_ascii=False, indent=2)

    print(f"\n[bold green]=== 전체 결과 ===[/bold green]")
    print(f"[bold]전체 결과 파일: {overall_json_path}[/bold]")
    print(f"[bold]최종 결과: {total_correct}/{total_items} (정확도 {overall_accuracy * 100:.2f}%)[/bold]")
    
    # 토큰 사용량 정보 출력
    if overall_data['summary'].get('token_usage'):
        usage = overall_data['summary']['token_usage']
        print(f"[bold cyan]토큰 사용량: Total={usage.get('total_tokens', 0)}, Prompt={usage.get('prompt_tokens', 0)}, Completion={usage.get('completion_tokens', 0)}[/bold cyan]")


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
