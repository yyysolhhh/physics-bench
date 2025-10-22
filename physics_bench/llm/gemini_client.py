from typing import Optional, override

import google.generativeai as genai

from .base import BaseLLMClient
from .registry import register_llm
from ..prompts import PHYSICS_TUTOR_SYSTEM_PROMPT, PHYSICS_USER_PROMPT
from ..utils.config import get_env


@register_llm("gemini", "GEMINI_MODEL")
class GeminiClient(BaseLLMClient):
    def __init__(
            self,
            model_name: str,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            api_key: Optional[str] = None,
    ) -> None:
        key = api_key or get_env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 가 필요합니다. .env 또는 환경변수로 설정하세요.")

        genai.configure(api_key=key)
        # self.model = genai.GenerativeModel(
        #     model_name=model_name,
        #     system_instruction=PHYSICS_TUTOR_SYSTEM_PROMPT
        # )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

    @override
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt
        )

        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            generation_config["max_output_tokens"] = self.max_tokens

        response = model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        return response.text.strip()
