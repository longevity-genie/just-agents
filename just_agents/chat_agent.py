from just_agents.llm_session import LLMSession
from typing import Any, Dict, List

class ChatAgent:

    def __init__(self, llm_options:Dict[str, Any], system:str = "You are helpfull agent.", character:str = "", functions:List[callable] = None):
        self.session:LLMSession = LLMSession(llm_options, system, functions)
        self.character:str = character


    def __call__(self, prompt:str):
        return self.session.query(prompt)
