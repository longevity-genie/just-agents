import loguru

from just_agents.chat_agent import ChatAgent
from dotenv import load_dotenv
from just_agents.llm_options import LLAMA3
load_dotenv()
from loguru import logger

customer: ChatAgent = ChatAgent(llm_options = LLAMA3, role = "customer at a shop",
                                goal = "Your goal is to order what you want, while speaking concisely and clearly", task="Find the best headphones!")
