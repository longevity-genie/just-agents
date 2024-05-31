from dotenv import load_dotenv

from just_agents.chat_agent import ChatAgent
from just_agents.llm_options import LLAMA3
from loguru import logger
load_dotenv()

customer: ChatAgent = ChatAgent(llm_options = LLAMA3, role = "customer at a shop",
                               goal = "Your goal is to order what you want, while speaking concisely and clearly",
                               task="Find the best headphones!")
storekeeper: ChatAgent = ChatAgent(llm_options = LLAMA3,
                                  role = "helpful storekeeper",
                                  goal="earn profit by selling what customers need",
                                  task="sell to the customer")


exchanges: int = 3
customer.memory.add_on_message(lambda m: logger.info(f"Customer: {m}") if m.role == "user" else logger.info(f"Storekeeper: {m}"))

customer_reply = "Hi."
for _ in range(exchanges):
    storekeeper_reply = storekeeper.query(customer_reply)
    customer_reply = customer.query(storekeeper_reply)


