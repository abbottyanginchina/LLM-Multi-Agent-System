from typing import List, Any, Dict, Optional, Union, Tuple
import asyncio

from structure.node import Node
from agents.agent_registry import AgentRegistry
from llm.llm_registry import LLMRegistry


@AgentRegistry.register("FinalDecision")
class FinalDecision(Node):
    """
    FinalDecision Agent: Integrates story fragments from multiple agents into a coherent narrative.
    """

    def __init__(
        self,
        llm_name: str = "",
        agent_name: str = "FinalDecision",
        **kwargs,
    ):
        super().__init__(id=None, agent_name=agent_name)
        self.llm_name: str = llm_name
        self.llm = LLMRegistry.get(llm_name) if llm_name else None
        self.role: str = "Story Integrator"

    def _build_prompts(
        self, fragments: List[Dict[str, str]], task_info: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Constructs system and user prompts for the LLM based on fragments and task requirements.
        """
        # System prompt: role definition
        system_prompt = "You are a master storyteller tasked with weaving story fragments into a single, coherent narrative."

        # Task requirements
        reqs = [
            f"- {k.title()}: {task_info.get(k, '')}"
            for k in ["theme", "genre", "setting", "tone", "length"]
            if task_info.get(k)
        ]
        constraints = task_info.get("constraints", [])
        if constraints:
            reqs.append("- Constraints:")
            reqs.extend([f"  * {c}" for c in constraints])
        req_text = "\n".join(reqs)

        # Fragments text
        frag_lines = []
        for idx, frag in enumerate(fragments, 1):
            frag_lines.append(
                f"--- Fragment {idx} ({frag['role']}) ---\n{frag['content']}"
            )
        frag_text = "\n\n".join(frag_lines)

        user_prompt = (
            f"Please integrate the following story fragments into a cohesive narrative:\n\n"
            f"{req_text}\n\n{frag_text}\n\n"
            "Ensure a smooth flow and maintain the specified tone and genre."
        )
        return system_prompt, user_prompt

    def _collect_fragments(self, spatial_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extracts and cleans outputs from spatial_info into a list of fragments.
        """
        fragments = []
        for agent_id, info in spatial_info.items():
            content = info.get("output")
            if not content:
                continue

            # Handle list outputs
            if isinstance(content, list):
                if len(content) > 0:
                    content = content[0]
                else:
                    continue

            text = str(content).strip()
            if text:
                fragments.append(
                    {
                        "agent_id": agent_id,
                        "role": info.get("role", "Unknown"),
                        "content": text,
                    }
                )
        return fragments

    def _execute(
        self,
        input: Any,
        spatial_info: Dict[str, Any],
        temporal_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        # Ensure task dict
        if isinstance(input, dict):
            task_info = input.get("task", {})
            if not isinstance(task_info, dict):
                task_info = {"theme": str(task_info)} if task_info else {}
        else:
            task_info = {"theme": str(input)} if input else {}

        fragments = self._collect_fragments(spatial_info or {})

        if not fragments:
            return self._create_default_story(task_info)

        system_prompt, user_prompt = self._build_prompts(fragments, task_info)

        # LLM invocation
        if self.llm:
            try:
                response = self.llm.gen(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                return (
                    response.strip()
                    if isinstance(response, str)
                    else str(response).strip()
                )
            except Exception as e:
                self._log(f"LLM chat failed: {e}")

        # Fallback simple integration
        return self._simple_story_integration(fragments, task_info)

    async def _async_execute(
        self,
        input: Any,
        spatial_info: Dict[str, Any],
        temporal_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        # Ensure task dict
        if isinstance(input, dict):
            task_info = input.get("task", {})
            if not isinstance(task_info, dict):
                task_info = {"theme": str(task_info)} if task_info else {}
        else:
            task_info = {"theme": str(input)} if input else {}

        fragments = self._collect_fragments(spatial_info or {})

        if not fragments:
            return self._create_default_story(task_info)

        system_prompt, user_prompt = self._build_prompts(fragments, task_info)

        if self.llm:
            try:
                if hasattr(self.llm, "agen"):
                    response = await self.llm.agen(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )
                else:
                    response = self.llm.gen(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )
                return (
                    response.strip()
                    if isinstance(response, str)
                    else str(response).strip()
                )
            except Exception as e:
                self._log(f"Async LLM failed: {e}")

        return self._simple_story_integration(fragments, task_info)

    def _log(self, message: str) -> None:
        # Replace prints with proper logging if available
        print(f"[FinalDecision] {message}")

    def _simple_story_integration(
        self, fragments: List[Dict[str, str]], task_info: Dict[str, Any]
    ) -> str:
        """智能整合多个故事片段成为一个连贯的故事"""
        if not fragments:
            return self._create_default_story(task_info)

        # 分析片段类型和质量
        analyzed_fragments = []
        for frag in fragments:
            role = frag["role"]
            content = frag["content"]

            # 根据角色类型分析内容
            if "Planner" in role:
                # Planner通常提供故事大纲和结构
                analyzed_fragments.append(
                    {"type": "outline", "role": role, "content": content, "priority": 1}
                )
            elif "Copywriter" in role:
                # Copywriter通常提供完整的故事文本
                analyzed_fragments.append(
                    {"type": "story", "role": role, "content": content, "priority": 2}
                )
            elif "Polisher" in role:
                # Polisher通常提供精修后的版本
                analyzed_fragments.append(
                    {
                        "type": "polished",
                        "role": role,
                        "content": content,
                        "priority": 3,
                    }
                )
            elif "Critic" in role:
                # Critic通常提供评论和建议
                analyzed_fragments.append(
                    {
                        "type": "critique",
                        "role": role,
                        "content": content,
                        "priority": 0,
                    }
                )
            else:
                analyzed_fragments.append(
                    {"type": "general", "role": role, "content": content, "priority": 1}
                )

        # 选择最佳的故事版本（优先选择Polisher的输出，然后是Copywriter）
        best_story = None
        for frag in sorted(
            analyzed_fragments, key=lambda x: x["priority"], reverse=True
        ):
            if frag["type"] in ["polished", "story"] and self._is_complete_story(
                frag["content"]
            ):
                best_story = frag
                break

        if best_story:
            # 提取纯故事内容，去掉元信息
            story_content = self._extract_story_content(best_story["content"])

            # 添加简洁的标题，提取故事标题而不是任务描述
            if "title:" in story_content.lower() or "**title" in story_content.lower():
                # 如果内容中已有标题，提取它
                title_match = None
                for line in story_content.split("\n")[:5]:  # 检查前5行
                    if "title:" in line.lower() or "**title" in line.lower():
                        title_match = line.split(":", 1)[-1].strip()
                        title_match = title_match.replace("**", "").strip()
                        break

                if title_match:
                    title = f"# {title_match}"
                else:
                    title = "# The Light of Eldermere"  # 默认标题
            else:
                theme = task_info.get("theme", "A Tale of Adventure")
                genre = task_info.get("genre", "")
                if genre:
                    title = f"# A {genre} Tale: {theme}"
                else:
                    title = f"# {theme}"

            return f"{title}\n\n{story_content}"
        else:
            # 如果没有完整故事，尝试从大纲构建
            return self._build_story_from_fragments(analyzed_fragments, task_info)

    def _is_complete_story(self, content: str) -> bool:
        """判断内容是否是完整的故事"""
        # 简单的启发式判断
        content_lower = content.lower()

        # 检查是否包含故事元素
        has_characters = any(
            word in content_lower
            for word in ["kael", "lyria", "rory", "elian", "finn", "lyra"]
        )
        has_narrative = any(
            word in content_lower for word in ["chapter", "scene", "story", "tale"]
        )
        has_dialogue = '"' in content or '*"' in content

        # 检查长度（完整故事应该相对较长）
        is_substantial = len(content) > 500

        return (has_characters or has_dialogue) and has_narrative and is_substantial

    def _extract_story_content(self, content: str) -> str:
        """从内容中提取纯故事部分，去掉meta信息"""
        lines = content.split("\n")
        story_lines = []
        in_story = False

        for line in lines:
            line = line.strip()

            # 跳过明显的meta信息
            if any(
                skip_word in line.lower()
                for skip_word in [
                    "title:",
                    "genre:",
                    "theme:",
                    "word count:",
                    "recommendation:",
                    "critique:",
                    "**title:",
                    "### **",
                    "would you like",
                    "let me know",
                ]
            ):
                continue

            # 检测故事开始
            if any(
                start_word in line.lower()
                for start_word in [
                    "opening scene",
                    "the wind howled",
                    "the crimson sun",
                    "chapter 1:",
                    "the village of",
                    "once upon a time",
                ]
            ):
                in_story = True

            # 如果在故事中，保留内容
            if in_story and line:
                story_lines.append(line)

            # 检测故事结束标记
            if line.lower().strip() in ["*the end*.", "*the end*", "the end", "---"]:
                break

        # 如果没有找到明确的故事，返回处理后的全部内容
        if not story_lines:
            # 移除明显的meta标记
            filtered_lines = []
            for line in lines:
                if not any(
                    skip in line.lower()
                    for skip in [
                        "**word count**",
                        "**ending note**",
                        "would you like",
                        "let me know",
                        "### **",
                        "recommendation:",
                        "critique items:",
                    ]
                ):
                    filtered_lines.append(line)
            return "\n".join(filtered_lines).strip()

        return "\n".join(story_lines).strip()

    def _build_story_from_fragments(
        self, fragments: List[Dict], task_info: Dict[str, Any]
    ) -> str:
        """从片段构建简单故事"""
        theme = task_info.get("theme", "A Tale of Adventure")
        genre = task_info.get("genre", "Fantasy")

        title = f"# A {genre} Tale: {theme}"

        # 收集所有有用的内容
        story_parts = [title, "\n"]

        for frag in fragments:
            if frag["type"] != "critique":  # 跳过纯评论
                content = self._extract_story_content(frag["content"])
                if content and len(content) > 100:  # 只包含有意义的内容
                    story_parts.append(f"\n{content}\n")

        story_parts.append("\n*The End*")
        return "\n".join(story_parts)

    def _create_default_story(self, task_info: Any) -> str:
        # Fallback story when no fragments
        if isinstance(task_info, dict):
            theme = task_info.get("theme", "An Untold Story")
        else:
            theme = str(task_info) if task_info else "An Untold Story"
        return (
            f"# {theme}\n\nNo fragments provided. A new story awaits your imagination."
        )

    def _process_inputs(
        self,
        raw_inputs: Any,
        spatial_info: Dict[str, Any],
        temporal_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Process inputs for the FinalDecision agent.
        This method is required by the Node base class.
        """
        # FinalDecision uses the raw inputs directly
        return [raw_inputs] if not isinstance(raw_inputs, list) else raw_inputs
