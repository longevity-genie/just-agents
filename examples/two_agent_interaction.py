import loguru

from just_agents.chat_agent import ChatAgent
from dotenv import load_dotenv
from just_agents.llm_options import LLAMA3
load_dotenv()
from loguru import logger

customer: ChatAgent = ChatAgent(llm_options = LLAMA3, role = "customer at a shop",
                               goal = "Your goal is to order what you want, while speaking concisely and clearly", task="Find the best headphones!")
storekeeper: ChatAgent = ChatAgent(llm_options = LLAMA3,
                                  role = "helpful storekeeper", goal="earn profit by selling what customers need", task="sell to the customer")


customer_reply = "Hi."
exchanges: int = 3
customer.add_on_message(lambda dic: logger.info(f"CUSTOMER VIEW: {dic}"))
for _ in range(exchanges):
    storekeeper_reply = storekeeper.query(customer_reply)
    customer_reply = customer.query(storekeeper_reply)


