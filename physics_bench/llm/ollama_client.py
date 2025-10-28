from typing import Optional, override

from langchain_ollama import ChatOllama

from .base import BaseLLMClient
from .registry import register_llm
from ..utils.config import get_env


@register_llm("ollama", "OLLAMA_MODEL")
class OllamaClient(BaseLLMClient):
    def __init__(
            self,
            model_name: str,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            base_url: Optional[str] = None,
    ) -> None:
        """
        Ollama 클라이언트 초기화
        
        Args:
            model_name: Ollama 모델 이름 (예: qwen2.5-math:7b)
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
            base_url: Ollama 서버 URL (기본값: http://localhost:11434)
        """
        # LangChain의 ChatOllama 기본 base_url은 http://localhost:11434
        # base_url이 제공되거나 환경변수가 있으면 사용, 없으면 None (기본값 사용)
        url = base_url or get_env("OLLAMA_BASE_URL")
        
        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        
        # base_url이 지정된 경우에만 추가
        if url:
            kwargs["base_url"] = url
        
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        
        self.llm = ChatOllama(**kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 토큰 사용량 추적용
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @override
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Ollama 모델로 답변 생성"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm.invoke(messages)
        
        # 사용량 통계 업데이트
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage']
            self.total_prompt_tokens += usage.get('prompt_tokens', 0)
            self.total_completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)
        
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()
    
    @override
    def get_usage_stats(self) -> dict[str, int]:
        """토큰 사용량 통계 반환"""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens
        }

