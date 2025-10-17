# physics-bench

LLM 기반 물리 문제 QA 벤치마크 프레임워크.

## 설치

```zsh
uv sync
```

## 환경 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```zsh
# OpenAI 설정
OPENAI_API_KEY=
OPENAI_MODEL=

# Qwen 설정  
QWEN_API_KEY=dummy
QWEN_MODEL=
QWEN_BASE_URL=
```

## 실행 예시

```zsh
python main.py run
```

### 옵션
- `--dataset`: 데이터셋 경로 (기본값: dataset/dataset.json)
- `--provider`: 모델 제공자 (기본값: qwen, 선택: openai, qwen)
- `--limit`: 상위 N개로 제한 (선택)
- `--temperature`: 샘플링 온도 (기본값: 0.0)
- `--max-tokens`: 최대 토큰 (선택)

```zsh
# 예시
python main.py run --dataset dataset/dataset.json --provider qwen --limit 10
```


