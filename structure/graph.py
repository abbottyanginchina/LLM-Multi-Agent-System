import time
import torch
import shortuuid
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from agents.agent_registry import AgentRegistry
from agents.normal_agent import NormalAgent
from agents.malicious_agent import MaliciousAgent
from agents.final_decision import FinalRefer, FinalDirect, FinalMajorVote   
from structure.node import Node

class Graph(ABC):
    def __init__(self, 
                 agent_names: List[str],
                 llm_name: str,
                 rounds: int,
                 fixed_spatial_masks: List[List[int]],
                 fixed_temporal_masks: List[List[int]],
                 decision_agent: bool,
                 decision_method: str
                 ):
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        self.rounds = rounds
        self.decision_agent = decision_agent
        self.decision_node: Node = AgentRegistry.get(decision_method, **{"llm_name":self.llm_name})

        self.init_node()
        self.init_potential_edges()

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_node(self):
        """
        Initialize nodes(agent) in the graph.
        """
        for agent_name in self.agent_names:
            agent_instance = AgentRegistry.get(agent_name, llm_name=self.llm_name)
            self.add_node(agent_instance)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def init_potential_edges(self):
        """
        Creates all potential edges list to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []

    def construct_spatial_connection(self): 
        self.clear_spatial_connection()
        for potential_connection, edge_mask in zip(self.potential_spatial_edges, self.fixed_spatial_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])

            if edge_mask == 1.0:  
                if not self.check_cycle(in_node, {out_node}):  
                    out_node.add_successor(in_node, 'spatial')

    def construct_temporal_connection(self, round: int = 0): 
        self.clear_temporal_connection()
        if round == 0:
            return 

        for potential_connection, edge_mask in zip(self.potential_temporal_edges, self.fixed_temporal_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])

            if edge_mask == 1.0:   
                if not self.check_cycle(in_node, {out_node}):  
                    out_node.add_successor(in_node, 'temporal')

    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()

    def run(self, inputs: Any, num_rounds:int, max_tries: int = 1):
        for round in range(num_rounds):
            print(f"==== Round {round + 1} ====")
            self.construct_spatial_connection()
            self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs)  # Execute the node
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                        time.sleep(60)  # Wait before retrying
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()  # Update memory after each round

        if self.decision_agent:
            self.connect_decision_node()
            self.decision_node.execute(inputs)
            final_answers = self.decision_node.outputs
            if len(final_answers) == 0:
                final_answers.append("No answer of the decision node")
            else:
                print(f"Final Answer: {final_answers}")
