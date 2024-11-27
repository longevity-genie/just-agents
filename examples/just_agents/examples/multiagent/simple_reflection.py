from typing import Any
from just_agents.patterns.reflection_agent import ReflectionAgent
from just_agents.simple.llm_session import LLMSession
from just_agents.llm_options import OPENAI_GPT4oMINI
from dotenv import load_dotenv
load_dotenv(override=True)

def author_message_handler(message: dict[str, Any]):
    print(f"Author:\n{message['content']}\n")

def critic_message_handler(message: dict[str, Any]):
    print(f"Critic:\n{message['content']}\n")


author = LLMSession(llm_options=OPENAI_GPT4oMINI) #, system_prompt="You are a professional poet.")
author.memory.add_on_assistant_message(author_message_handler)
critic = LLMSession(llm_options=OPENAI_GPT4oMINI) #, system_prompt="You are a professional literature critic.")
critic.memory.add_on_assistant_message(critic_message_handler)

agent = ReflectionAgent(author=author, critic=critic)
prompt = "Write hiku about recursion."
print("Prompt: ", prompt)
print("Result:\n", agent.query(prompt))
