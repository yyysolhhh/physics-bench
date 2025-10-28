from .base import BaseLLMClient


class LLMRegistry:
    """LLM 클라이언트 등록 및 생성 관리"""

    _providers: dict[str, type[BaseLLMClient]] = {}
    _model_env_vars: dict[str, str] = {}

    @classmethod
    def register(cls, provider_name: str, client_class: type[BaseLLMClient], model_env_var: str):
        cls._providers[provider_name.lower()] = client_class
        cls._model_env_vars[provider_name.lower()] = model_env_var

    @classmethod
    def get_providers(cls) -> list[str]:
        return list(cls._providers.keys())

    @classmethod
    def create_client(cls, provider: str, **kwargs) -> BaseLLMClient:
        provider = provider.lower()
        if provider not in cls._providers:
            available = ", ".join(cls.get_providers())
            raise ValueError(f"지원하지 않는 provider: {provider}. 사용 가능: {available}")

        client_class = cls._providers[provider]
        return client_class(**kwargs)

    @classmethod
    def get_model_env_var(cls, provider: str) -> str:
        return cls._model_env_vars.get(provider.lower(), "")


def register_llm(provider_name: str, model_env_var: str):
    """LLM 클라이언트를 자동으로 등록하는 데코레이터"""

    def decorator(client_class: type[BaseLLMClient]):
        LLMRegistry.register(provider_name, client_class, model_env_var)
        return client_class

    return decorator
