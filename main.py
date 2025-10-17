from typing import Optional

import typer
from rich import print

from physics_bench.benchmark import BenchmarkRunner, ModelSpec
from physics_bench.utils.config import get_env

app = typer.Typer(add_completion=False)


@app.command("run")
def run(
        dataset: str = typer.Option("dataset/dataset.json", "--dataset", help="JSON 데이터셋 경로"),
        provider: str = typer.Option("qwen", "--provider", help="모델 제공자 (openai, qwen)"),
        temperature: float = typer.Option(0.0, "--temperature", help="샘플링 온도"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="최대 토큰(미지정 시 모델 기본값)"),
        limit: Optional[int] = typer.Option(None, "--limit", help="데이터셋 상위 N개로 제한"),
):
    # 미리 정해둔 모델명 매핑
    model_mapping = {
        "openai": get_env("OPENAI_MODEL"),
        "qwen": get_env("QWEN_MODEL"),
    }

    if provider not in model_mapping:
        raise typer.BadParameter(f"지원하지 않는 provider: {provider}. 사용 가능: {list(model_mapping.keys())}")

    model_name = model_mapping[provider]
    spec = ModelSpec(provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)
    runner = BenchmarkRunner(model_spec=spec)
    report = runner.run(dataset_path=dataset, limit=limit)

    print("\n[bold]결과 요약[/bold]")
    print(f"정답 수: {report.correct}/{report.total} (정확도 {report.accuracy * 100:.2f}%)")


if __name__ == "__main__":
    app()
