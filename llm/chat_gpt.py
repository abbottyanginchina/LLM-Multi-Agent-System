from typing import List, Union, Optional, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

from .format import Message
from .llm import LLM
from .llm_registry import LLMRegistry

OPENAI_API_KEY = [""]
BASE_URL = ""

load_dotenv("template.env")  # 明确加载template.env文件
MINE_BASE_URL = os.getenv("MINE_BASE_URL")
MINE_API_KEY = os.getenv("MINE_API_KEYS")  # 注意这里改为MINE_API_KEYS


@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict]):
    client = AsyncOpenAI(
        base_url=MINE_BASE_URL,
        api_key=MINE_API_KEY,
    )
    chat_completion = await client.chat.completions.create(
        messages=msg,
        model=model,
    )
    response = chat_completion.choices[0].message.content
    return response


async def achat_deepseek(
    model: str,
    msg: List[Dict],
):
    client = AsyncOpenAI(
        base_url=MINE_BASE_URL,
        api_key=MINE_API_KEY,
    )
    chat_completion = await client.chat.completions.create(
        messages=msg,
        model=model,
    )
    response = chat_completion.choices[0].message.content
    return response


@LLMRegistry.register("ChatGPT")
class ChatGPT(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            # 转换Message对象为字典格式
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    formatted_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )
                elif isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:
                    formatted_messages.append({"role": "user", "content": str(msg)})
            messages = formatted_messages
        return await achat(self.model_name, messages)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # 同步版本 - 简单实现，实际项目中可能需要使用同步的OpenAI客户端
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.agen(messages, max_tokens, temperature, num_comps)
            )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            return asyncio.run(self.agen(messages, max_tokens, temperature, num_comps))


@LLMRegistry.register("deepseek")
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            # 转换Message对象为字典格式
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    formatted_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )
                elif isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:
                    formatted_messages.append({"role": "user", "content": str(msg)})
            messages = formatted_messages
        return await achat_deepseek(self.model_name, messages)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # 同步版本 - 简单实现
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.agen(messages, max_tokens, temperature, num_comps)
            )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            return asyncio.run(self.agen(messages, max_tokens, temperature, num_comps))
