from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
