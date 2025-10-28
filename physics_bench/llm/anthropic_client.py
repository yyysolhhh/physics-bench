from typing import Optional, override, Any

from anthropic import Anthropic

from .base import BaseLLMClient
from .registry import register_llm
from ..prompts import PHYSICS_TUTOR_SYSTEM_PROMPT, PHYSICS_USER_PROMPT
from ..utils.config import get_env


@register_llm("anthropic", "ANTHROPIC_MODEL")
class AnthropicClient(BaseLLMClient):
    def __init__(
            self,
            model_name: str,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            api_key: Optional[str] = None,
    ) -> None:
        key = api_key or get_env("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY 가 필요합니다. .env 또는 환경변수로 설정하세요.")

        self.client = Anthropic(api_key=key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 토큰 사용량 추적용
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @override
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "system": system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 4096,
        }

        response = self.client.messages.create(**kwargs)
        
        # 사용량 통계 업데이트
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self.total_input_tokens += getattr(usage, 'input_tokens', 0)
            self.total_output_tokens += getattr(usage, 'output_tokens', 0)
            self.total_tokens += getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
        
        return response.content[0].text.strip()
    
    @override
    def get_usage_stats(self) -> dict:
        """토큰 사용량 통계 반환"""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.total_input_tokens,
            'completion_tokens': self.total_output_tokens
        }
