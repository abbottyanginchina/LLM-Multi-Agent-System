from typing import List, Any, Dict
import asyncio
import re
from structure.node import Node
from agents.agent_registry import AgentRegistry
from dataset.data_process import get_predict
from llm.llm_registry import LLMRegistry
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("FinalRefer")
class FinalRefer(Node):
    """
    最终决策代理，用于整合多个代理的结果并做出最终决策
    """

    def __init__(self, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(
            id=None, agent_name="FinalRefer", domain=domain, llm_name=llm_name
        )
        self.domain = domain
        self.llm = LLMRegistry.get(llm_name) if llm_name else None
        self.prompt_set = PromptSetRegistry.get(domain) if domain else None

    def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any] = None,
        **kwargs,
    ) -> List[Any]:
        """处理输入，构建决策prompt - 仿照AgentPrune-main的实现"""
        if not self.prompt_set:
            return ["Error: No prompt set available"]

        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{self.role}.\n {self.constraint}"

        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"{id}: {info.get('output', 'No output')}\n\n"

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        task_content = raw_inputs.get("task", str(raw_inputs))
        user_prompt = f"{decision_few_shot} The task is:\n\n {task_content}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str}"

        print(f"🔍 [调试] FinalRefer构建的prompt:")
        print(f"System: {system_prompt}")
        print(f"User: {user_prompt}")

        return system_prompt, user_prompt

    def extract_choice_answer(self, text: str) -> str:
        """从文本中提取选择题答案 A, B, C, D"""
        if not isinstance(text, str):
            text = str(text)

        # 多种方式提取选择题答案
        # 1. 查找单独的字母答案
        choice_pattern = r"\b[ABCD]\b"
        choice_matches = re.findall(choice_pattern, text)
        if choice_matches:
            return choice_matches[0]

        # 2. 查找行首的字母答案
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0] in "ABCD" and (len(line) == 1 or line[1] in ". ):"):
                return line[0]

        # 3. 查找 "答案是 X" 格式
        answer_patterns = [
            r"答案是\s*([ABCD])",
            r"answer is\s*([ABCD])",
            r"选择\s*([ABCD])",
            r"option\s*([ABCD])",
            r"正确答案是\s*([ABCD])",
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].upper()

        return None

    def _execute(self, input: Any, info: Dict[str, Any], **kwargs):
        """
        整合所有代理的输出并做出最终决策 - 仿照AgentPrune-main的实现
        """
        try:
            # 适配新的参数格式 - spatial_info对应info，temporal_info为空
            spatial_info = info if info else {}
            temporal_info = kwargs.get("temporal_info", {})

            print(
                f"🔍 [调试] FinalRefer开始处理，收到的spatial_info: {list(spatial_info.keys()) if spatial_info else 'None'}"
            )

            if not self.llm or not self.prompt_set:
                # 如果没有LLM或prompt_set，使用简单投票策略
                return self._simple_voting_strategy(spatial_info)

            # 使用LLM进行决策 - 仿照AgentPrune-main
            system_prompt, user_prompt = self._process_inputs(
                input, spatial_info, temporal_info
            )
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = self.llm.gen(message)
            print(f"🔍 [调试] FinalRefer收到LLM响应: {response}")

            return response

        except Exception as e:
            print(f"🔍 [调试] FinalRefer执行出错: {str(e)}")
            return f"Error in final decision: {str(e)}"

    async def _async_execute(self, input: Any, info: Dict[str, Any], **kwargs):
        """
        异步执行最终决策 - 仿照AgentPrune-main的实现
        """
        try:
            # 适配新的参数格式 - spatial_info对应info，temporal_info为空
            spatial_info = info if info else {}
            temporal_info = kwargs.get("temporal_info", {})

            print(
                f"🔍 [调试] FinalRefer异步开始处理，收到的spatial_info: {list(spatial_info.keys()) if spatial_info else 'None'}"
            )

            if not self.llm or not self.prompt_set:
                # 如果没有LLM或prompt_set，使用简单投票策略
                return self._simple_voting_strategy(spatial_info)

            # 使用LLM进行决策 - 仿照AgentPrune-main
            system_prompt, user_prompt = self._process_inputs(
                input, spatial_info, temporal_info
            )
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.llm.agen(message)
            print(f"🔍 [调试] FinalRefer收到LLM异步响应: {response}")

            return response

        except Exception as e:
            print(f"🔍 [调试] FinalRefer异步执行出错: {str(e)}")
            return f"Error in async final decision: {str(e)}"

    def _simple_voting_strategy(self, spatial_info: Dict[str, Any]) -> List[str]:
        """简单投票策略作为后备方案"""
        agent_answers = []

        for node_id, node_info in spatial_info.items():
            if "output" in node_info:
                output = node_info["output"]
                if isinstance(output, list):
                    for out in output:
                        choice = self.extract_choice_answer(str(out))
                        if choice:
                            agent_answers.append(choice)
                else:
                    choice = self.extract_choice_answer(str(output))
                    if choice:
                        agent_answers.append(choice)

        if not agent_answers:
            return ["No valid choice answers found"]

        # 简单投票
        from collections import Counter

        vote_counts = Counter(agent_answers)
        most_common = vote_counts.most_common(1)
        final_choice = most_common[0][0] if most_common else agent_answers[0]

        print(f"🔍 [调试] 简单投票结果: {dict(vote_counts)}, 最终选择: {final_choice}")
        return [final_choice]
