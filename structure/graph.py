from abc import ABC
from typing import Any, List, Optional, Dict
import shortuuid
import numpy as np
import torch

from structure.node import Node
from agents.agent_registry import AgentRegistry


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.

    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(
        self,
        domain: str,
        llm_name: Optional[str],
        agent_names: List[str],
        decision_method: str,
        node_kwargs: List[Dict] = None,
    ):
        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        self.decision_node: Node = AgentRegistry.get(
            decision_method, **{"domain": self.domain, "llm_name": self.llm_name}
        )
        self.nodes: Dict[str, Node] = {}
        self.node_kwargs = (
            node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        )

        # 初始化用于优化的logits参数
        num_agents = len(agent_names)
        self.spatial_logits = torch.nn.Parameter(torch.randn(num_agents, num_agents))
        self.temporal_logits = torch.nn.Parameter(torch.randn(num_agents))

        self.init_nodes()  # add nodes to the self.nodes

    @property
    def adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.successors)
        return num_edges

    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(
            f"Node not found: {id} among" f"{[node.id for node in self.nodes.values()]}"
        )

    def add_node(self, node: Node):
        node_id = (
            node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        )
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)

                # 调试断点5：显示Graph中创建的agent实例及其角色
                print(
                    f"🔍 [调试] Graph 创建 {agent_name} 实例 ID: {agent_instance.id}, 角色: {getattr(agent_instance, 'role', 'N/A')}"
                )

                self.add_node(agent_instance)

    def clear_connection(self):
        """
        clear all connections of the nodes in the graph
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].predecessors = []
            self.nodes[node_id].successors = []
        self.decision_node.predecessors = []
        self.decision_node.successors = []

    def connect_decision_node(self, last_node_id: str = None):
        for node_id in self.nodes.keys():
            if last_node_id is None:
                self.nodes[node_id].add_successor(self.decision_node)
            elif last_node_id == node_id:
                self.nodes[node_id].add_successor(self.decision_node)

    def construct_connection(self):
        """构建简单的图连接 - 目前不进行优化剪枝"""
        self.clear_connection()

        # 改进的连接策略：确保至少有一个节点入度为0
        node_ids = list(self.nodes.keys())
        num_nodes = len(node_ids)

        if num_nodes == 0:
            return torch.tensor(0.0)

        if num_nodes == 1:
            # 只有一个节点，不需要连接
            return torch.tensor(0.0)

        # 对于多个节点，创建链式连接：node0 -> node1 -> node2 -> ... -> nodeN
        # 这样node0的入度为0，可以开始执行
        for i in range(num_nodes - 1):
            out_node = self.nodes[node_ids[i]]
            in_node = self.nodes[node_ids[i + 1]]
            out_node.add_successor(in_node)

        return torch.tensor(0.0)  # 简化：不计算log_probs

    def run(
        self,
        inputs: Any,
        num_rounds: int = 3,
        max_tries: int = 3,
        aggregate_mode: str = "all connected",
    ) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = torch.tensor(0.0)
        current_node_id = None  # 初始化变量
        for round in range(num_rounds):
            log_probs += self.construct_connection()
            in_degree = {
                node_id: len(node.predecessors) for node_id, node in self.nodes.items()
            }
            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(
                            inputs
                        )  # output is saved in the node.output
                        break
                    except Exception as e:
                        print(f"Error executing node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer found")

        return final_answers, log_probs

    async def arun(
        self,
        input: Dict[str, str],
        num_rounds: int = 3,
        max_tries: int = 3,
        aggregate_mode: str = "all connected",
    ) -> List[Any]:
        log_probs = torch.tensor(0.0)
        current_node_id = None  # 初始化变量
        for round in range(num_rounds):
            log_probs += self.construct_connection()
            in_degree = {
                node_id: len(node.predecessors) for node_id, node in self.nodes.items()
            }
            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await self.nodes[current_node_id].async_execute(input)
                        break
                    except Exception as e:
                        print(f"Error executing node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()
        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer found")
        return final_answers, log_probs

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()
