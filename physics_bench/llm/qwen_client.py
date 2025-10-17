from typing import Optional, override

from openai import OpenAI

from .base import BaseLLMClient
from ..prompts import PHYSICS_TUTOR_SYSTEM_PROMPT, PHYSICS_TUTOR_USER_PROMPT
from ..utils.config import get_env


class QwenClient(BaseLLMClient):
    def __init__(
            self,
            model_name: str,
            base_url: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            api_key: Optional[str] = None,
    ) -> None:
        key = api_key or get_env("QWEN_API_KEY")
        url = base_url or get_env("QWEN_BASE_URL")

        self.client = OpenAI(
            base_url=url,
            api_key=key,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @override
    def generate_answer(self, question: str) -> str:
        messages = [
            {"role": "system", "content": PHYSICS_TUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": PHYSICS_TUTOR_USER_PROMPT.format(question=question)}
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
