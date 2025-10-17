from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(self, question: str) -> str:
        raise NotImplementedError
