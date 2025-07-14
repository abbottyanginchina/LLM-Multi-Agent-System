from typing import List, Any, Dict
import asyncio
from structure.node import Node
from agents.agent_registry import AgentRegistry
from dataset.data_process import get_predict


@AgentRegistry.register("FinalRefer")
class FinalRefer(Node):
    """
    最终决策代理，用于整合多个代理的结果并做出最终决策
    """
    
    def __init__(self, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(id=None, agent_name="FinalRefer", domain=domain, llm_name=llm_name)
        self.role = "Final Decision Maker"
    
    def _execute(self, input: Any, info: Dict[str, Any], **kwargs):
        """
        整合所有代理的输出并做出最终决策
        """
        try:
            # 收集所有前置节点的输出
            all_outputs = []
            if info:
                for node_id, node_info in info.items():
                    if "output" in node_info:
                        all_outputs.extend(node_info["output"] if isinstance(node_info["output"], list) else [node_info["output"]])
            
            if not all_outputs:
                return ["No valid outputs to process"]
            
            # 简单的决策策略：选择第一个有效的数值答案
            final_answer = "No answer found"
            for output in all_outputs:
                try:
                    # 尝试从输出中提取数值答案
                    predicted = get_predict(str(output))
                    if predicted is not None and predicted != "":
                        final_answer = str(output)
                        break
                except:
                    continue
            
            return [final_answer]
        except Exception as e:
            return [f"Error in final decision: {str(e)}"]
    
    async def _async_execute(self, input: Any, info: Dict[str, Any], **kwargs):
        """
        异步执行最终决策
        """
        try:
            # 收集所有前置节点的输出
            all_outputs = []
            if info:
                for node_id, node_info in info.items():
                    if "output" in node_info:
                        all_outputs.extend(node_info["output"] if isinstance(node_info["output"], list) else [node_info["output"]])
            
            if not all_outputs:
                return ["No valid outputs to process"]
            
            # 模拟异步处理
            await asyncio.sleep(0.1)
            
            # 简单的决策策略：选择第一个有效的数值答案
            final_answer = "No answer found"
            for output in all_outputs:
                try:
                    # 尝试从输出中提取数值答案
                    predicted = get_predict(str(output))
                    if predicted is not None and predicted != "":
                        final_answer = str(output)
                        break
                except:
                    continue
            
            return [final_answer]
        except Exception as e:
            return [f"Error in final decision: {str(e)}"]
    
    def _process_inputs(self, raw_inputs: List[Any], info: Dict[str, Any], **kwargs) -> List[Any]:
        """
        处理输入
        """
        # 对于决策节点，主要关注前置节点的输出而不是原始输入
        return raw_inputs if raw_inputs else [{}]
