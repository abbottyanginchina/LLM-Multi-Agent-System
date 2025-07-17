import torch
import copy

from structure.graph import Graph
from .accuracy import Accuracy


async def train(
    graph: Graph,
    data: dict,
    input_data: dict,
    num_rounds: int = 1,
) -> None:

    realized_graph = copy.deepcopy(graph)

    target = data["answer"]

    print(f"问题: {data['question']}")
    print(f"选项:")
    for i, choice in enumerate(data["choices"]):
        print(f"   {chr(65+i)}. {choice}")
    print(f"正确答案: {target}")
    print(f"\n{'='*50}")
    print(f"多智能体协作...")
    print(f"{'='*50}")

    # 使用转换后的输入格式
    raw_answer, log_probs = await realized_graph.arun(input_data, num_rounds)

    accuracy = Accuracy()
    accuracy.update(raw_answer, target)

    print(f"\n{'='*50}")
    print(f"最终结果:")
    print(f"{'='*50}")
    print(f"多智能体系统的答案: {raw_answer}")
    print(f"正确答案: {target}")

    # 判断答案是否正确
    if raw_answer and len(raw_answer) > 0:
        predicted = raw_answer[0] if isinstance(raw_answer, list) else raw_answer
        if str(predicted).strip().upper() == str(target).strip().upper():
            print(f"回答正确！")
        else:
            print(f"回答错误")
    else:
        print(f"⚠️  没有得到有效答案")

    print(f"{'='*50}\n")
