from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_usage_stats(self) -> dict[str, Any]:
        """LLM 호출 사용량 통계 반환 (토큰 수, 비용 등)"""
        raise NotImplementedError
