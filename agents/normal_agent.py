from typing import List, Any, Dict
import re

from structure.node import Node
from .agent_registry import AgentRegistry
from llm.llm_registry import LLMRegistry
from prompt.math_prompt_set import PromptSetRegistry


def find_strings_between_pluses(text):
    return re.findall(r"\@(.*?)\@", text)


@AgentRegistry.register("NormalAgent")
class NormalAgent(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
    ):
        super().__init__(id, "NormalAgent")

        # 获取对应的PromptSet。NormalAgent默认使用通用提示词集
        self.prompt_set = PromptSetRegistry.get("normal")

        # 设置每一个role对应一个特定的llm
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm = LLMRegistry.get(self.role)

        self.constraint = self.prompt_set.get_analyze_constraint(self.role)

    def _process_inputs(
        self,
        raw_inputs: Any,
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ) -> tuple[str, str]:
        """To be overriden by the descendant class"""
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        # 提取任务信息
        if isinstance(raw_inputs, dict) and "task" in raw_inputs:
            task = raw_inputs["task"]
        else:
            task = str(raw_inputs)

        system_prompt = f"{self.constraint}"
        user_prompt = (
            f"The task is: {task}\n"
            if self.role != "Fake"
            else self.prompt_set.get_adversarial_answer_prompt(task)
        )
        spatial_str = ""
        temporal_str = ""

        for id, info in spatial_info.items():
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += (
            f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n"
            if len(spatial_str)
            else ""
        )
        user_prompt += (
            f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}"
            if len(temporal_str)
            else ""
        )
        return system_prompt, user_prompt

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(
            input, spatial_info, temporal_info
        )
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(
        self,
        input: Any,
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(
            input, spatial_info, temporal_info
        )
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await self.llm.agen(message)
        return response
