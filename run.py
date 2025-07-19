import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List, Union, Literal
import random
from tools.json_reader import JSONLReader, JSONReader
from datetime import datetime
from structure.graph import Graph
from dataset.query import Query
import agents  # 导入agents模块以注册所有代理
import llm  # 导入llm模块以注册所有LLM
import prompt  # 导入prompt模块以注册所有提示集

sys.stdout.reconfigure(encoding="utf-8")

# 根目录路径  __file__ 是 run.py 的路径，.parent 获取根目录
ROOT = Path(__file__).parent


# 存储结果
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump([], file)

    with open(result_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# 数据表
def dataloader(data_list, batch_size, i_batch):
    # 截取从i_batch*batch_size到i_batch*batch_size+batch_size的数据
    # 截取batch_size个数据
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


# 加载配置文件
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# 可覆盖参数
# 缺少optimized_spatial和optimized_temporal
def parse_args():
    parser = argparse.ArgumentParser(description="arguement")

    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="")
    parser.add_argument("--dataset_json", type=str, default="dataset/story_task.json")

    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=[
            "DirectAnswer",
            "FullConnected",
            "Random",
            "Chain",
            "Debate",
            "Layered",
            "Star",
        ],
        help="Mode of operation. Default is 'FullConnected'.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of optimization/inference rounds for one query",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="The num of training iterations."
    )
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["NormalAgent"],
        help="Specify agent names as a list of strings",
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[4],
        help="Specify the number of agents for each name in agent_names",
    )
    parser.add_argument(
        "--decision_method",
        type=str,
        default="FinalDecision",
        help="The decison method of the agentprune",
    )
    args = parser.parse_args()

    result_path = ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    result_file = None
    # 数据集处理
    # 构建数据集文件的完整路径
    input_query = JSONReader.parse_file(args.dataset_json)

    # 直接使用我们的 Query 类处理数据

    # 创建一个临时的 Query 实例来处理数据转换
    if input_query:
        input_dict = Query.record_to_input(input_query)
    else:
        print("Failed to load data from JSON file.")
        return

    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 结果保存路径
    result_dir = Path(f"{ROOT}/result")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    agent_names = [
        name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)
    ]
    # 决定最终结果的方法
    decision_method = args.decision_method

    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(
        agent_names=agent_names,
        decision_method=decision_method,
        **kwargs,
    )

    print(80 * "-")
    start_ts = time.time()
    answer_log_probs = []

    if not input_dict or not input_dict.get("task"):
        print("No valid task available.")
        return

    realized_graph = copy.deepcopy(graph)
    task = input_dict["task"]

    answer_log_probs.append(
        asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds))
    )
    raw_results = await asyncio.gather(*answer_log_probs)
    raw_results = raw_results[0]  # 获取第一个任务的结果
    utilities: List[float] = []
    data = load_result(result_file)

    updated_item = {
        "Question": task,
        "Answer": raw_results,  # 添加结果到保存的数据中
        "Mode": args.mode,
        "Agent_Names": args.agent_names,
        "Agent_Nums": args.agent_nums,
        "Decision_Method": args.decision_method,
        "Timestamp": current_time,
    }
    data.append(updated_item)

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    # 同时保存一个纯文本版本（更易读）
    txt_file = result_file.with_suffix(".txt")
    with open(txt_file, "w", encoding="utf-8") as file:
        file.write(f"Question: {task}\n\n")
        file.write("=" * 60 + "\n")
        file.write("STORY OUTPUT:\n")
        file.write("=" * 60 + "\n\n")
        if isinstance(raw_results, list) and len(raw_results) > 0:
            file.write(raw_results[0])
        else:
            file.write(str(raw_results))
        file.write(f"\n\n{'=' * 60}\n")
        file.write(f"Configuration:\n")
        file.write(f"Mode: {args.mode}\n")
        file.write(f"Agents: {args.agent_names} (nums: {args.agent_nums})\n")
        file.write(f"Decision Method: {args.decision_method}\n")
        file.write(f"Timestamp: {current_time}\n")

    # 同时保存一个Markdown版本（格式更美观）
    md_file = result_file.with_suffix(".md")
    with open(md_file, "w", encoding="utf-8") as file:
        file.write(f"# Story Generation Result\n\n")
        file.write(f"**Generated on:** {current_time}\n\n")
        file.write(f"## Task\n\n{task}\n\n")
        file.write(f"## Generated Story\n\n")
        if isinstance(raw_results, list) and len(raw_results) > 0:
            file.write(raw_results[0])
        else:
            file.write(str(raw_results))
        file.write(f"\n\n## Configuration\n\n")
        file.write(f"- **Mode:** {args.mode}\n")
        file.write(f"- **Agents:** {args.agent_names} (nums: {args.agent_nums})\n")
        file.write(f"- **Decision Method:** {args.decision_method}\n")

    print(f"Results saved to: {result_file}")
    print(f"Text version saved to: {txt_file}")
    print(f"Markdown version saved to: {md_file}")
    print(f"\n{'='*60}")
    print("FINAL STORY OUTPUT:")
    print(f"{'='*60}")
    if isinstance(raw_results, list) and len(raw_results) > 0:
        print(raw_results[0])
    else:
        print(raw_results)
    print(f"{'='*60}")
    print(f"Batch time {time.time()-start_ts:.3f}")
    print("utilities:", utilities)


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
    ],
    N: int,
):
    node_kwargs = None

    # 层状图
    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    # 星型图
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    # 链式图
    def generate_chain_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(n - 1):
            matrix[i][i + 1] = 1
        return matrix

    # 随机图
    def generate_random_graph(n, edge_prob=0.3):
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < edge_prob:
                    matrix[i][j] = 1
        return matrix

    # 根据模式生成对应的邻接矩阵
    fixed_spatial_masks = None
    fixed_temporal_masks = None

    if mode == "FullConnected":
        # 全连接：所有节点互相连接（除了自己）
        fixed_spatial_masks = [[1 if i != j else 0 for j in range(N)] for i in range(N)]
    elif mode == "DirectAnswer":
        # 直接回答：没有节点间连接
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        # 链式连接
        fixed_spatial_masks = generate_chain_graph(N)
    elif mode == "Random":
        # 随机连接
        fixed_spatial_masks = generate_random_graph(N)
    elif mode == "Layered":
        # 层状连接
        fixed_spatial_masks = generate_layered_graph(N)
    elif mode == "Star":
        # 星型连接
        fixed_spatial_masks = generate_star_graph(N)
    elif mode == "Debate":
        # 辩论模式：所有节点互相连接
        fixed_spatial_masks = [[1 if i != j else 0 for j in range(N)] for i in range(N)]

    # 时间连接默认为全连接
    if fixed_temporal_masks is None:
        fixed_temporal_masks = [[1 for j in range(N)] for i in range(N)]

    return {
        "node_kwargs": node_kwargs,
        "fixed_spatial_masks": fixed_spatial_masks,
        "fixed_temporal_masks": fixed_temporal_masks,
    }


if __name__ == "__main__":
    asyncio.run(main())
