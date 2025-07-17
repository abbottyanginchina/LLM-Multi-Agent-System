from typing import List, Any, Dict
import asyncio
from structure.node import Node
from agents.agent_registry import AgentRegistry
from llm.llm_registry import LLMRegistry
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("MathSolver")
class MathSolver(Node):
    """
    数学求解代理，用于解决数学问题
    """

    def __init__(self, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(
            id=None, agent_name="MathSolver", domain=domain, llm_name=llm_name
        )
        self.role = "Math Solver"  # 修正角色名称

        # 调试断点1：显示MathSolver的角色分配
        print(f"🔍 [调试] MathSolver {self.id} 被分配角色: '{self.role}'")

        self.llm = LLMRegistry.get(llm_name) if llm_name else None
        self.prompt_set = (
            PromptSetRegistry.get("Math_nocot") if domain == "gsm8k" else None
        )

    def _execute(self, input: Any, info: Dict[str, Any], **kwargs):
        """
        执行数学问题求解
        """
        try:
            if isinstance(input, dict) and "task" in input:
                problem = input["task"]
            else:
                problem = str(input)

            # 构建提示词
            if self.prompt_set:
                prompt = self.prompt_set.get_answer_prompt(problem, self.role)
            else:
                prompt = f"Please solve this math problem step by step: {problem}"

            # 调试断点2：显示构建的完整prompt
            print(f"🔍 [调试] MathSolver {self.id} 角色 '{self.role}' 的完整prompt:")
            print(f"--- 开始 ---")
            print(prompt)
            print(f"--- 结束 ---")

            print(f"Prompt: {prompt}")

            # 如果有LLM，使用LLM求解，否则返回简单回复
            if self.llm:
                response = self.llm.call(prompt)
            else:
                response = f"Processing math problem: {problem}\nThis is a placeholder solution."

            return [response]
        except Exception as e:
            return [f"Error solving math problem: {str(e)}"]

    async def _async_execute(self, input: Any, info: Dict[str, Any], **kwargs):
        """
        异步执行数学问题求解
        """
        try:
            if isinstance(input, dict) and "task" in input:
                problem = input["task"]
            else:
                problem = str(input)

            # 构建提示词
            if self.prompt_set:
                prompt = self.prompt_set.get_answer_prompt(problem, self.role)
            else:
                prompt = f"Please solve this math problem step by step: {problem}"

            # 调试断点3：显示异步执行的完整prompt
            print(
                f"🔍 [调试] MathSolver {self.id} 异步执行 - 角色 '{self.role}' 的完整prompt:"
            )
            print(f"--- 开始 ---")
            print(prompt)
            print(f"--- 结束 ---")

            print(f"Async Prompt: {prompt}")

            # 如果有LLM，使用LLM求解，否则返回简单回复
            if self.llm:
                response = await self.llm.acall(prompt)
            else:
                # 模拟异步处理
                await asyncio.sleep(0.1)
                response = f"Processing math problem: {problem}\nThis is a placeholder solution."

            return [response]
        except Exception as e:
            return [f"Error solving math problem: {str(e)}"]

    def _process_inputs(
        self, raw_inputs: List[Any], info: Dict[str, Any], **kwargs
    ) -> List[Any]:
        """
        处理输入
        """
        processed_inputs = []
        for input_item in raw_inputs:
            if isinstance(input_item, dict):
                processed_inputs.append(input_item)
            else:
                processed_inputs.append({"task": str(input_item)})
        return processed_inputs
