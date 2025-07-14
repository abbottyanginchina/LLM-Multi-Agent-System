# 导入所有prompt模块以确保它们被注册
from .math_prompt_set import *
from .prompt_set_registry import PromptSetRegistry

__all__ = ["PromptSetRegistry"]
