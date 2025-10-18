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

# Anthropic 설정
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=

# Gemini 설정
GEMINI_API_KEY=
GEMINI_MODEL=
```

## 실행 예시

### 1. 데이터셋 다운로드
```zsh
# Hugging Face에서 데이터셋 다운로드
python main.py download {데이터셋 이름} {split}

# 출력 파일명 지정 (미지정 시 기본: {데이터셋}_{split}.json)
python main.py download {데이터셋 이름} test --output saved.json

# 개수 제한
python main.py download {데이터셋 이름} validation --limit 100
```

**download 명령어 인자/옵션:**
- `dataset_name` (인자, 필수): Hugging Face 데이터셋 이름
- `split` (인자, 필수): 다운로드할 split 이름 (예: train/test/validation)
- `--output` (옵션): 출력 파일명 (미지정 시 `{데이터셋}_{split}.json`)
- `--limit` (옵션): 다운로드할 항목 수 제한

### 2. 벤치마크 실행
```zsh
# 기본 실행
python main.py run

# 옵션 사용
python main.py run --dataset downloaded_dataset.json --provider qwen --limit 10
```

**run 명령어 옵션:**
- `--dataset`: 데이터셋 경로 (기본값: dataset/dataset.json)
- `--provider`: 모델 제공자 (기본값: qwen, 선택: openai, qwen, anthropic, gemini)
- `--limit`: 상위 N개로 제한 (선택)
- `--temperature`: 샘플링 온도 (기본값: 0.0)
- `--max-tokens`: 최대 토큰 (선택)
