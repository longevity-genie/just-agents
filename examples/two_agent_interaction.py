from just_agents.chat_agent import ChatAgent
from dotenv import load_dotenv
from just_agents.llm_options import LLAMA3
load_dotenv()
customer:ChatAgent = ChatAgent(llm_options = LLAMA3, role = "customer at a shop",
                               goal = "Your goal is to order what you want, while speaking concisely and clearly", log_level = "INFO")
storekeeper:ChatAgent = ChatAgent(llm_options = LLAMA3,
                                  role = "helpful storekeeper", log_level = "INFO")

customer_reply = "Hi."
print(customer.role, ": ", customer_reply)

for _ in range(3):
    storekeeper_reply = storekeeper.query(customer_reply)
    print(storekeeper.role, ": ", storekeeper_reply)

    customer_reply = customer.query(storekeeper_reply)
    print(customer.role, ": ", customer_reply)


