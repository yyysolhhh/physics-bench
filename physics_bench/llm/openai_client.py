from typing import Optional, override

from langchain_openai import ChatOpenAI

from .base import BaseLLMClient
from .registry import register_llm
from ..utils.config import get_env


@register_llm("openai", "OPENAI_MODEL")
class OpenAIClient(BaseLLMClient):
    def __init__(
            self,
            model_name: str,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            api_key: Optional[str] = None,
    ) -> None:
        key = api_key or get_env("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY 가 필요합니다. .env 또는 환경변수로 설정하세요.")

        kwargs = {"model": model_name, "temperature": temperature, "api_key": key}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        self.llm = ChatOpenAI(**kwargs)

    @override
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()
