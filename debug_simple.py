#!/usr/bin/env python3
"""
简化调试决策节点问题的脚本
"""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, ".")

# 导入必需的模块
import agents
import llm
import prompt
from structure.graph import Graph


async def debug_decision_node():
    """专门调试决策节点的问题"""
    print("=== 调试决策节点问题 ===")

    # 创建一个简单的图
    graph = Graph(
        domain="gsm8k",
        llm_name="deepseek-chat",
        agent_names=["MathSolver"],  # 只用一个节点
        decision_method="FinalRefer",
    )

    print(f"图中的节点数量: {len(graph.nodes)}")
    print(f"决策节点: {graph.decision_node}")

    # 手动执行节点
    test_input = {"task": "1+1等于多少？"}
    node_id = list(graph.nodes.keys())[0]
    node = graph.nodes[node_id]

    print(f"\n手动执行节点 {node_id}...")
    await node.async_execute(test_input)
    print(f"节点输出: {node.outputs}")

    # 手动连接决策节点
    print(f"\n手动连接决策节点...")
    node.add_successor(graph.decision_node)
    print(f"决策节点前驱: {[pred.id for pred in graph.decision_node.predecessors]}")

    # 获取决策节点的信息
    decision_info = graph.decision_node.get_info()
    print(f"决策节点获取的信息: {decision_info}")

    # 手动执行决策节点
    print(f"\n手动执行决策节点...")
    await graph.decision_node.async_execute(test_input)
    print(f"决策节点输出: {graph.decision_node.outputs}")


if __name__ == "__main__":
    asyncio.run(debug_decision_node())
