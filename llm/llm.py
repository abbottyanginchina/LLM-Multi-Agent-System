from abc import ABC, abstractmethod
from typing import List, Union, Optional

from .format import Message


class LLM(ABC):
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.2
    DEFUALT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        pass

    # 添加便捷方法
    async def acall(self, prompt: str, **kwargs) -> str:
        """异步调用LLM的便捷方法"""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        result = await self.agen(messages, **kwargs)
        return result if isinstance(result, str) else result[0]

    def call(self, prompt: str, **kwargs) -> str:
        """同步调用LLM的便捷方法"""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        result = self.gen(messages, **kwargs)
        return result if isinstance(result, str) else result[0]
