from typing import Optional, override, Any

from anthropic import Anthropic

from .base import BaseLLMClient
from .registry import register_llm
from ..prompts import PHYSICS_TUTOR_SYSTEM_PROMPT, PHYSICS_TUTOR_USER_PROMPT
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

    @override
    def generate_answer(self, question: str) -> str:
        messages = [
            {"role": "user", "content": PHYSICS_TUTOR_USER_PROMPT.format(question=question)}
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "system": PHYSICS_TUTOR_SYSTEM_PROMPT,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 4096,  # 기본값 설정
        }

        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def generate_answer_with_usage(self, question: str) -> dict[str, Any]:
        """토큰 사용량 정보와 함께 답변을 생성합니다."""
        messages = [
            {"role": "user", "content": PHYSICS_TUTOR_USER_PROMPT.format(question=question)}
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "system": PHYSICS_TUTOR_SYSTEM_PROMPT,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 4096,  # 기본값 설정
        }

        response = self.client.messages.create(**kwargs)

        return {
            "answer": response.content[0].text.strip(),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }
