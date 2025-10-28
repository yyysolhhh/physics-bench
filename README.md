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

# Ollama 설정 (로컬 기본 설정 사용 시 불필요)
# OLLAMA_BASE_URL=http://localhost:11434  # 다른 서버 사용 시만 필요

# LangSmith 설정 (선택사항 - 토큰 사용량 추적)
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=physics-bench
```

## 토큰 사용량 추적

모든 LLM 클라이언트에서 토큰 사용량이 자동으로 추적됩니다:

### 1. 자동 수집
- 각 클라이언트(OpenAI, Anthropic, Gemini, Qwen)가 토큰 사용량을 자동 추적
- 벤치마크 완료 시 로그에 출력
- 결과 JSON 파일의 `summary.token_usage`에 저장

### 2. LangSmith 연동 (OpenAI만 지원)

OpenAI 클라이언트를 사용할 때 LangSmith로 상세한 추적 및 분석이 가능합니다:

1. **LangSmith 가입 및 API 키 발급**
   - https://smith.langchain.com 에서 가입
   - Settings > API Keys에서 API 키 생성

2. **환경변수 설정**
   ```bash
   # .env 파일에 추가
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=physics-bench
   ```

3. **LangSmith에서 확인**
   - OpenAI 클라이언트 사용 시 자동으로 호출 내역 추적
   - 대시보드에서 각 호출별 상세 정보 확인
   - 호출 체인, 프롬프트, 응답, 지연시간 등 분석 가능

> **참고**: OpenAI와 Ollama는 LangChain을 사용하므로 LangSmith 자동 추적이 가능합니다. Qwen, Anthropic, Gemini는 네이티브 SDK를 사용하므로 LangSmith 자동 추적이 되지 않지만, 토큰 사용량은 기본적으로 추적됩니다.

## Ollama 사용

로컬에 Ollama로 설치한 모델을 사용할 수 있습니다:

### 1. Ollama 설치 및 모델 다운로드
```bash
# Ollama 설치 (https://ollama.ai)
# 모델 다운로드
ollama pull qwen2.5-math:7b
ollama pull qwen3:8b
ollama pull gemma3:4b
ollama pull llama3.1:8b
```

### 2. 벤치마크 실행
```bash
# 특정 모델로 실행
python main.py run --provider ollama --model qwen2.5-math:7b

# 다른 옵션과 함께
python main.py run --provider ollama --model llama3.1:8b --limit 5 --temperature 0.7
```

### 3. 설치된 모델 확인
```bash
ollama list
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
python main.py run --provider qwen --limit 10
python main.py run --provider openai --limit 10
python main.py run --provider anthropic --limit 10

# Ollama 모델 사용
python main.py run --provider ollama --model qwen2.5-math:7b --limit 5
python main.py run --provider ollama --model qwen3:8b
python main.py run --provider ollama --model gemma3:4b --limit 10
python main.py run --provider ollama --model llama3.1:8b
```

**run 명령어 옵션:**
- `--provider`: 모델 제공자 (기본값: gemini, 선택: openai, qwen, anthropic, gemini, ollama)
- `--model`: 모델 이름 (Ollama 사용 시 필수, 예: qwen2.5-math:7b)
- `--limit`: 각 과목마다 상위 N개로 제한 (선택)
- `--temperature`: 샘플링 온도 (기본값: 0.0)
- `--max-tokens`: 최대 토큰 (선택)
- `--prompt`: 프롬프트 스타일 (기본값: numerical, 선택: simple/benchmark/detailed/numerical)
- `--verbose`: 상세한 출력 표시 (기본값: true)

## 결과 파일

실행 후 `outputs/모델명/날짜시간/` 폴더에 다음 파일들이 생성됩니다:

- `overall_results.json`: 전체 벤치마크 결과 (토큰 사용량 포함)
- `과목별폴더/benchmark.log`: 과목별 로그 파일
- `과목별폴더/results.json`: 과목별 상세 결과
