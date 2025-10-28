from abc import ABC, abstractmethod
from typing import Optional, Any


class DatasetLoader(ABC):
    """데이터셋 로더의 기본 인터페이스"""

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> list[Any]:
        """
        Args:
            limit: 로드할 최대 아이템 수 (None이면 모든 아이템)
            
        Returns:
            데이터셋 아이템 리스트
        """
        pass
