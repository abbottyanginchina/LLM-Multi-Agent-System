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
    def get(cls, role: Optional[str] = None) -> LLM:
        # 根据角色返回对应的LLM实例
        # 暂时统一使用deepseek-chat
        try:
            if role is None or role == "":
                # 默认模型
                return cls.registry.get("deepseek", "deepseek-chat")
            elif "Planner" in role:
                return cls.registry.get("deepseek", "deepseek-chat")
            elif "Copywriter" in role:
                return cls.registry.get("deepseek", "deepseek-chat")
            elif "Polisher" in role:
                return cls.registry.get("deepseek", "deepseek-chat")
            elif "Critic" in role:
                return cls.registry.get("deepseek", "deepseek-chat")
            else:
                return cls.registry.get("deepseek", "deepseek-chat")
        except Exception as e:
            print(f"Warning: Failed to get LLM for role {role}: {e}")
