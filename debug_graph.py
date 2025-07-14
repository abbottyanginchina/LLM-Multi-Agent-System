#!/usr/bin/env python3
"""
调试图结构和节点执行的脚本
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
from tools.json_reader import JSONLReader
from dataset.data_process import data_process


async def debug_graph_execution():
    """调试图的执行过程"""
    print("=== 调试图结构和执行 ===")

    # 创建一个简单的图
    agent_names = ["MathSolver", "MathSolver"]  # 创建2个MathSolver节点
    decision_method = "FinalRefer"

    graph = Graph(
        domain="gsm8k",
        llm_name="deepseek-chat",
        agent_names=agent_names,
        decision_method=decision_method,
        node_kwargs=None,
    )

    print(f"图中的节点数量: {len(graph.nodes)}")
    print(f"节点ID列表: {list(graph.nodes.keys())}")
    print(f"决策节点: {graph.decision_node}")

    # 检查每个节点的连接情况
    for node_id, node in graph.nodes.items():
        print(f"\n节点 {node_id} ({node.agent_name}):")
        print(f"  前驱节点: {[pred.id for pred in node.predecessors]}")
        print(f"  后继节点: {[succ.id for succ in node.successors]}")
        print(f"  入度: {len(node.predecessors)}")
        print(f"  出度: {len(node.successors)}")

    # 测试连接构建
    print("\n=== 构建连接 ===")
    log_prob = graph.construct_connection()
    print(f"连接构建完成，log_prob: {log_prob}")

    # 再次检查连接情况
    for node_id, node in graph.nodes.items():
        print(f"\n连接后节点 {node_id}:")
        print(f"  前驱节点: {[pred.id for pred in node.predecessors]}")
        print(f"  后继节点: {[succ.id for succ in node.successors]}")
        print(f"  入度: {len(node.predecessors)}")

    # 计算入度为0的节点
    in_degree = {
        node_id: len(node.predecessors) for node_id, node in graph.nodes.items()
    }
    zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

    print(f"\n入度统计: {in_degree}")
    print(f"入度为0的节点: {zero_in_degree_queue}")

    # 测试图的异步运行
    print("\n=== 测试图运行 ===")
    test_input = {"task": "1+1等于多少？"}

    try:
        result, log_probs = await graph.arun(test_input, num_rounds=1)
        print(f"✅ 图运行成功")
        print(f"结果: {result}")
        print(f"Log probs: {log_probs}")

        # 检查每个节点的输出
        for node_id, node in graph.nodes.items():
            print(f"节点 {node_id} 输出: {node.outputs}")

        print(f"决策节点输出: {graph.decision_node.outputs}")

        # 检查决策节点的前驱和输入信息
        print(
            f"\n决策节点前驱: {[pred.id for pred in graph.decision_node.predecessors]}"
        )
        decision_info = graph.decision_node.get_info()
        print(f"决策节点获取的信息: {decision_info}")

    except Exception as e:
        print(f"❌ 图运行失败: {e}")
        import traceback

        traceback.print_exc()


async def debug_single_node():
    """调试单个节点的执行"""
    print("\n=== 调试单个节点执行 ===")

    try:
        from agents.agent_registry import AgentRegistry

        # 创建一个MathSolver节点
        math_solver = AgentRegistry.get(
            "MathSolver", domain="gsm8k", llm_name="deepseek-chat"
        )
        print(f"✅ MathSolver创建成功: {math_solver}")
        print(f"LLM: {math_solver.llm}")
        print(f"Prompt Set: {math_solver.prompt_set}")

        # 测试节点执行
        test_input = {"task": "1+1等于多少？"}
        result = await math_solver.async_execute(test_input)
        print(f"✅ 节点执行结果: {result}")

    except Exception as e:
        print(f"❌ 节点执行失败: {e}")
        import traceback

        traceback.print_exc()


async def debug_small_dataset():
    """使用少量数据测试完整流程"""
    print("\n=== 调试小数据集处理 ===")

    try:
        # 加载数据集
        ROOT = Path(".")
        dataset_path = ROOT / "dataset" / "gsm8k.jsonl"
        dataset = JSONLReader.parse_file(dataset_path)
        dataset = data_process(dataset)

        print(f"数据集大小: {len(dataset)}")
        print(f"第一条数据: {dataset[0]}")

        # 只处理第一条数据
        first_item = dataset[0]
        task = first_item["task"]
        answer = first_item["answer"]

        print(f"任务: {task}")
        print(f"答案: {answer}")

        # 创建图并处理
        graph = Graph(
            domain="gsm8k",
            llm_name="deepseek-chat",
            agent_names=["MathSolver"],
            decision_method="FinalRefer",
        )

        input_dict = {"task": task}
        result, log_probs = await graph.arun(input_dict, num_rounds=1)

        print(f"图处理结果: {result}")
        print(f"Log probs: {log_probs}")

        # 提取预测答案
        from dataset.data_process import get_predict

        if result and len(result) > 0:
            predict_answer = get_predict(result[0])
            print(f"预测答案: {predict_answer}")
            print(f"真实答案: {answer}")

            try:
                is_correct = (
                    float(predict_answer) == float(answer)
                    if predict_answer is not None
                    else False
                )
                print(f"是否正确: {is_correct}")
            except:
                print(f"无法比较答案")

    except Exception as e:
        print(f"❌ 小数据集测试失败: {e}")
        import traceback

        traceback.print_exc()


async def main():
    print("开始调试...")

    await debug_single_node()
    await debug_graph_execution()
    await debug_small_dataset()


if __name__ == "__main__":
    asyncio.run(main())
