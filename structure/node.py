from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import asyncio
import shortuuid


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.

    Methods:
        add_predecessor(operation):
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation):
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs):
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs):
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(
        self,
        id: Optional[str],
        agent_name: str = "",
        domain: str = "",
        llm_name: str = "",
    ):
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name: str = agent_name
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.predecessors: List[Node] = []
        self.successors: List[Node] = []
        self.role = ""
        self.last_memory: Dict[str, List[Any]] = {
            "inputs": [],
            "outputs": [],
            "raw_inputs": [],
        }
        self.conversation_history: List[Dict] = (
            []
        )  # chat history of the whole conversation

    @property
    def node_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: "Node"):
        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.add_successor(self)

    def add_successor(self, operation: "Node"):
        if operation not in self.successors:
            self.successors.append(operation)
            operation.add_predecessor(self)

    def remove_predecessor(self, operation: "Node"):
        if operation in self.predecessors:
            self.predecessors.remove(operation)
            operation.remove_successor(self)

    def remove_successor(self, operation: "Node"):
        if operation in self.successors:
            self.successors.remove(operation)
            operation.remove_predecessor(self)

    def clear_connections(self):
        self.predecessors: List[Node] = []
        self.successors: List[Node] = []

    def update_memory(self):
        self.last_memory["inputs"] = self.inputs.copy()
        self.last_memory["outputs"] = self.outputs.copy()
        self.last_memory["raw_inputs"] = self.raw_inputs.copy()

    def get_info(self) -> Dict[str, Dict]:
        """Return a dict that maps id to info."""
        info = {}
        if self.predecessors is not None:
            for predecessor in self.predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif (
                    isinstance(predecessor_outputs, list)
                    and len(predecessor_outputs) == 0
                ):
                    continue
                else:
                    predecessor_output = predecessor_outputs
                info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": predecessor_output,
                }
        return info

    def execute(self, input: Any, **kwargs):
        self.outputs = []
        info: Dict[str, Dict] = self.get_info()
        results = [self._execute(input, info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    async def async_execute(self, input: Any, **kwargs):
        self.outputs = []
        info: Dict[str, Any] = self.get_info()
        tasks = [asyncio.create_task(self._async_execute(input, info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    @abstractmethod
    def _execute(self, input: List[Any], info: Dict[str, Any], **kwargs):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input: List[Any], info: Dict[str, Any], **kwargs):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(
        self, raw_inputs: List[Any], info: Dict[str, Any], **kwargs
    ) -> List[Any]:
        """To be overriden by the descendant class"""
        """ Process the raw_inputs(most of the time is a List[Dict]) """
