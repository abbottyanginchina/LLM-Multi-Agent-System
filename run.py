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
from dataset.data_process import data_process, get_predict
from datetime import datetime
from structure.graph import Graph
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
    parser.add_argument("--dataset_json", type=str, default="dataset.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="deepseek-chat")
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
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
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
        "--domain",
        type=str,
        default="gsm8k",
        help="Domain (the same as dataset name), default 'gsm8k'",
    )
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["MathSolver"],
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
        default="FinalRefer",
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
    dataset_path = ROOT / "dataset" / args.dataset_json
    dataset = JSONLReader.parse_file(dataset_path)
    dataset = data_process(dataset)
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 结果保存路径
    result_dir = Path(f"{ROOT}/result")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"

    agent_names = [
        name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)
    ]
    # 决定最终结果的方法
    decision_method = args.decision_method

    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        **kwargs,
    )
    # 优化器
    optimizer = torch.optim.Adam(
        [graph.spatial_logits, graph.temporal_logits], lr=args.lr
    )

    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = 0, 0

    for i_batch in range(num_batches):
        print(f"Batch {i_batch}", 80 * "-")
        start_ts = time.time()
        answer_log_probs = []
        answers = []

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            print("No more data available.")
            break

        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            task = record["task"]
            step = record["step"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_log_probs.append(
                asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds))
            )
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)

        for task, answer, log_prob, true_answer in zip(
            current_batch, raw_answers, log_probs, answers
        ):
            predict_answer = get_predict(answer[0])

            # 安全地比较答案，处理None值
            try:
                if predict_answer is not None and true_answer is not None:
                    is_solved = float(predict_answer) == float(true_answer)
                else:
                    is_solved = False
            except (ValueError, TypeError):
                is_solved = False

            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved / total_executed
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            updated_item = {
                "Question": task,
                "Answer": true_answer,
                "Step": step,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
            }
            data.append(updated_item)
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        total_loss = torch.mean(torch.stack(loss_list))

        print(f"Batch time {time.time()-start_ts:.3f}")
        print(f"Accuracy:{accuracy}")
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

    return {"node_kwargs": node_kwargs}


if __name__ == "__main__":
    asyncio.run(main())
