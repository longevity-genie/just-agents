from dotenv import load_dotenv

from just_agents.simple.chat_agent import ChatAgent
from just_agents.llm_options import LLAMA3_2_VISION
import typer

app = typer.Typer(no_args_is_help=True)

"""
This example shows how two agents can be used to simulate a simple chat.
"""

@app.command()
def purchase_example():
    load_dotenv()

    customer: ChatAgent = ChatAgent(llm_options = LLAMA3_2_VISION, role = "customer at a shop",
                                   goal = "Your goal is to order what you want, while speaking concisely and clearly",
                                   task="Find the best headphones!")
    storekeeper: ChatAgent = ChatAgent(llm_options = LLAMA3_2_VISION,
                                      role = "helpful storekeeper",
                                      goal="earn profit by selling what customers need",
                                      task="sell to the customer")

    exchanges: int = 3
    customer.memory.add_on_message(lambda m: typer.echo(f"Customer: {m['content']}") if m["role"] == "user" else typer.echo(f"Storekeeper: {m['content']}"))
    customer_reply = "Hi."
    for _ in range(exchanges):
        storekeeper_reply = storekeeper.query(customer_reply)
        customer_reply = customer.query(storekeeper_reply)

if __name__ == "__main__":
    app()