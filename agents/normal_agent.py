from agents.agent_registry import AgentRegistry
from backends.llm_registry import LLMRegistry
from structure.node import Node
from backends.llm_chat import openAIChat
from typing import Optional, List, Dict
from backends.prompts import PromptTemplates, role_description

@AgentRegistry.register('normalAgent')
class NormalAgent(Node):
    def __init__(self, 
                 id=None, 
                 agent_name: Optional[str] = None,
                 llm_name: Optional[str] = None
                 ):
        super().__init__(id=id, llm_name=llm_name, agent_name=agent_name)
        self.role = PromptTemplates.get_role()
        self.llm = LLMRegistry.get(llm_name)
        
    # Protected method to process inputs
    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = role_description[self.role] + PromptTemplates.get_constraint()
        user_prompt = f"The task is: {raw_inputs}"

        # Gather spatial information
        spatial_str = ""
        for id, info in spatial_info.items():
            if 'None.' in (info['output'] if isinstance(info['output'], list) else [info['output']]):
                continue
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""

        # Gather temporal information
        temporal_str = ""
        for id, info in temporal_info.items():
            if 'None.' in (info['output'] if isinstance(info['output'], list) else [info['output']]):
                continue
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        print(f"==== ID:{self.id} Role:{self.role} ====")
        print(f"User Prompt: {user_prompt}")
        return system_prompt, user_prompt

    def _execute(self, raw_inputs, spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """
        Execute the agent's action based on the provided inputs and context.
        """
        system_prompt, user_prompt = self._process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self.llm.generate(message)
        # import pdb; pdb.set_trace()  # Debugging line to inspect the response

        print(f"==== ID:{self.id} Role:{self.role} ====")
        print(f"Response: {response}")
        # print(user_prompt)
        return response
