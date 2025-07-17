from typing import List, Any, Dict
import re

from structure.node import Node
from agents.agent_registry import AgentRegistry
from llm.llm_registry import LLMRegistry
from prompt.prompt_set_registry import PromptSetRegistry
from tools.wiki import search_wiki_main


def find_strings_between_pluses(text):
    return re.findall(f"\@(.*?)\@", text)


@AgentRegistry.register("AnalyzeAgent")
class AnalyzeAgent(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
    ):
        super().__init__(id, "AnalyzeAgent", domain, llm_name)
        self.domain = domain  # 保存domain以便在其他方法中使用
        self.llm = LLMRegistry.get(llm_name)
        # 修复：根据domain获取正确的prompt_set
        if domain == "gsm8k":
            self.prompt_set = PromptSetRegistry.get("Math_nocot")
        elif domain == "mmlu":
            self.prompt_set = PromptSetRegistry.get("mmlu")
        else:
            self.prompt_set = PromptSetRegistry.get(domain)

        self.role = self.prompt_set.get_role() if role is None else role

        # 调试断点1：显示AnalyzeAgent的角色分配
        print(
            f"🔍 [调试] AnalyzeAgent {self.id} 被分配角色: '{self.role}' (传入role参数: {role})"
        )

        # 修复：根据不同的prompt_set类型调用正确的constraint方法
        if domain == "mmlu":
            self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        else:
            self.constraint = self.prompt_set.get_constraint(self.role)

        # 调试断点2：显示角色约束/constraint
        print(f"🔍 [调试] AnalyzeAgent {self.id} 的角色约束: '{self.constraint}'")

        self.wiki_summary = ""

    async def _process_inputs(
        self, raw_inputs: Dict[str, str], info: Dict[str, Dict], **kwargs
    ) -> List[Any]:
        """To be overridden by the descendant class."""
        """Process the raw_inputs(most of the time is a List[Dict])"""
        system_prompt = f"{self.constraint}"

        # 修复：适应AgentPrune-main的输入格式，使用task字段
        task_content = raw_inputs.get("task", str(raw_inputs))

        user_prompt = (
            f"The task is :{task_content}\n"
            if self.role != "Fake"
            else self.prompt_set.get_adversarial_answer_prompt(task_content)
        )
        string = ""
        for id, information in info.items():
            if (
                self.role == "Wiki Searcher"
                and information["role"] == "Knowlegable Expert"
            ):
                queries = find_strings_between_pluses(information["output"])
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            string += f"Agent {id}, role is {information.get('role', 'Unknown')}, output is:\n\n {information.get('output', 'No output')}\n\n"

        user_prompt += (
            f"At the same time, the outputs of other agents are as follows:\n\n{string} \n\n"
            if len(string)
            else ""
        )

        # 调试断点3：显示AnalyzeAgent的完整prompt
        print(f"🔍 [调试] AnalyzeAgent {self.id} 角色 '{self.role}' 的完整prompt:")
        print(f"--- System Prompt 开始 ---")
        print(system_prompt)
        print(f"--- System Prompt 结束 ---")
        print(f"--- User Prompt 开始 ---")
        print(user_prompt)
        print(f"--- User Prompt 结束 ---")

        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], info: Dict[str, Dict], **kwargs):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(
            self._process_inputs(input, info, **kwargs)
        )
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(
        self, input: Dict[str, str], info: Dict[str, Dict], **kwargs
    ):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(input, info)
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 调试断点4：显示AnalyzeAgent异步执行的完整消息
        print(
            f"🔍 [调试] AnalyzeAgent {self.id} 异步执行 - 角色 '{self.role}' 的LLM消息:"
        )
        print(f"--- 消息结构 ---")
        for i, msg in enumerate(message):
            print(f"Message {i+1} ({msg['role']}): {msg['content'][:200]}...")
        print(f"--- 消息结构结束 ---")

        response = await self.llm.agen(message)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        return response
