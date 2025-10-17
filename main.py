from typing import Optional

import typer
from rich import print

from physics_bench.benchmark import BenchmarkRunner, ModelSpec
from physics_bench.dataset import JsonDatasetLoader, download_huggingface_dataset
from physics_bench.utils.config import get_env

app = typer.Typer()


@app.command("run")
def run(
        dataset: str = typer.Option("dataset/dataset.json", "--dataset", help="JSON 데이터셋 경로"),
        provider: str = typer.Option("qwen", "--provider", help="모델 제공자 (openai, qwen)"),
        temperature: float = typer.Option(0.0, "--temperature", help="샘플링 온도"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="최대 토큰(미지정 시 모델 기본값)"),
        limit: Optional[int] = typer.Option(None, "--limit", help="데이터셋 상위 N개로 제한"),
):
    model_mapping = {
        "openai": get_env("OPENAI_MODEL"),
        "qwen": get_env("QWEN_MODEL"),
    }

    if provider not in model_mapping:
        raise typer.BadParameter(f"지원하지 않는 provider: {provider}. 사용 가능: {list(model_mapping.keys())}")

    model_name = model_mapping[provider]
    spec = ModelSpec(provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

    loader = JsonDatasetLoader(dataset)
    items = loader.load(limit=limit)

    runner = BenchmarkRunner(model_spec=spec)
    report = runner.run_with_items(items)

    print("\n[bold]결과 요약[/bold]")
    print(f"정답 수: {report.correct}/{report.total} (정확도 {report.accuracy * 100:.2f}%)")


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
