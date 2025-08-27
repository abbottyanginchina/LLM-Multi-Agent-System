import argparse

from structure.graph import Graph
from structure.structure_mode import get_structure_mode

def parse_args():
    parser = argparse.ArgumentParser(description="Run the multi-agent system.")
    parser.add_argument(
        "--llm_name",
        type=str,
        default="deepseek-ai/DeepSeek-V3",
        help="Name of the LLM to use (default: deepseek-ai/DeepSeek-V3)."
    )
    parser.add_argument(
        "--agent_names",
        type=str,
        nargs='+',
        default=["normalAgent"],
        help="List of agent names to initialize (default: ['normal_agent'])."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["Debate", "FullConnected"],
        default="Debate",
        help="Mode of operation for the agents (default: Debate)."
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds for the agents to interact (default: 3)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    fixed_spatial_masks, fixed_temporal_masks = get_structure_mode(args)

    graph = Graph(llm_name=args.llm_name,
                  agent_names=args.agent_names,
                  fixed_spatial_masks=fixed_spatial_masks,
                  fixed_temporal_masks=fixed_temporal_masks,
                  rounds=args.num_rounds
                  )

    # inputs = "Please expand the sentence: “A boy stands on a tall building and suddenly jumps down.”" #task1
    task = "Please help me to answer the following question: “How can I write a good essay?”" #task2
    # task = 'task2'
    graph.run(task, num_rounds=args.num_rounds)

if __name__ == "__main__":
    main()
