from just_agents.chat_agent import ChatAgent
from dotenv import load_dotenv
load_dotenv()

llama3 = {
    "model": "groq/llama3-70b-8192",
    "temperature": 0.7,
    "api_base": "https://api.groq.com/openai/v1",
}

customer:ChatAgent = ChatAgent(llama3, "You are a customer at a shop. Speak concisely and clearly.", "customer")
storekeeper:ChatAgent = ChatAgent(llama3, "You are a helpful storekeeper.", "storekeeper")

customer_reply = "Hi."
print(customer.character,": ",customer_reply)

for _ in range(3):
    storekeeper_reply = storekeeper(customer_reply)
    print(storekeeper.character, ": ", storekeeper_reply)

    customer_reply = customer(storekeeper_reply)
    print(customer.character, ": ", customer_reply)


