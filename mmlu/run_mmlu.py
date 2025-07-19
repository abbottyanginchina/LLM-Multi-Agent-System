import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import asyncio
from typing import Union, Literal, List
import argparse
import random

from structure.graph import Graph
from mmlu.train_mmlu import train
from pathlib import Path

AgentPrune_ROOT = Path(
    os.path.realpath(os.path.join(os.path.split(__file__)[0], "../.."))
)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

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
            "Mesh",
            "FakeFullConnected",
            "FakeRandom",
            "FakeChain",
            "FakeStar",
            "FakeMesh",
            "FakeAGRandom",
            "FakeAGFull",
        ],
        help="Mode of operation. Default is 'FullConnected'.",
    )
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["AnalyzeAgent"],
        help="Specify agent names as a list of strings",
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[5],
        help="Specify the number of agents for each name in agent_names",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of optimization/inference rounds for one query",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--decision_method",
        type=str,
        default="FinalRefer",
        help="the decision method of the final node",
    )
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args


async def main():
    args = parse_args()

    mode = args.mode
    decision_method = args.decision_method
    agent_names = [
        name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)
    ]
    kwargs = get_kwargs(mode, len(agent_names))

    graph = Graph(
        agent_names=agent_names,
        decision_method=decision_method,
        **kwargs,
    )

    data = {
        "question": "What is the embryological origin of the hyoid bone?",
        "choices": [
            "The first pharyngeal arch",
            "The first and second pharyngeal arches",
            "The second pharyngeal arch",
            "The second and third pharyngeal arches",
        ],
        "answer": "D",
    }

    # 将数据转换为AgentPrune-main期望的格式
    formatted_question = (
        f"{data['question']}\n"
        f"A. {data['choices'][0]}\n"
        f"B. {data['choices'][1]}\n"
        f"C. {data['choices'][2]}\n"
        f"D. {data['choices'][3]}\n"
    )

    # 转换为AgentPrune-main的输入格式
    input_data = {"task": formatted_question}

    await train(
        graph=graph,
        data=data,
        input_data=input_data,
        num_rounds=args.num_rounds,
        lr=args.lr,
    )


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
        Literal["FakeFullConnected"],
        Literal["FakeRandom"],
        Literal["FakeChain"],
        Literal["FakeStar"],
        Literal["FakeMesh"],
        Literal["FakeAGRandom"],
        Literal["FakeAGFull"],
    ],
    N: int,
):

    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0] * N for _ in range(N)]
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

    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i + 1, N):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal"}]
    elif mode == "FullConnected" or mode == "FakeFullConnected" or mode == "FakeAGFull":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random" or mode == "FakeRandom" or mode == "FakeAGRandom":
        fixed_spatial_masks = [
            [random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)
        ]
        fixed_temporal_masks = [
            [random.randint(0, 1) for _ in range(N)] for _ in range(N)
        ]
    elif mode == "Chain" or mode == "FakeChain":
        fixed_spatial_masks = [
            [1 if i == j + 1 else 0 for i in range(N)] for j in range(N)
        ]
        fixed_temporal_masks = [
            [1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)
        ]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == "Mesh" or mode == "FakeMesh":
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == "Star" or mode == "FakeStar":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]

    if "Fake" in mode and "AG" not in mode:
        node_kwargs = [
            {"role": "Fake"} if i % 2 == N % 2 else {"role": "Normal"} for i in range(N)
        ]
    elif "Fake" in mode and "AG" in mode:
        node_kwargs = [
            {"role": "Fake"} if i % 2 == N % 2 else {"role": None} for i in range(N)
        ]

    return {
        "node_kwargs": node_kwargs,
    }


if __name__ == "__main__":
    asyncio.run(main())
