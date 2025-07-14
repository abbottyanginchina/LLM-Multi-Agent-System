from typing import Optional
from class_registry import ClassRegistry

from .llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name == "":
            model_name = "gpt-4o"

        if model_name == "mock":
            model = cls.registry.get(model_name)
        elif "deepseek" in model_name:
            model = cls.registry.get("deepseek", model_name)
        else:  # any version of ChatGPT like "gpt-4o"
            model = cls.registry.get("ChatGPT", model_name)

        return model
