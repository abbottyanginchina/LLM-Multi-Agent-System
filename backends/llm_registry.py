from class_registry import ClassRegistry
from typing import Optional

class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None):
        if 'deepseek' in model_name:
            model = cls.registry.get('openAIChat', model_name)

        return model