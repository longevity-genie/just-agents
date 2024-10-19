from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
load_dotenv()

assistant: IAgent = build_agent("yaml_initialization_example.yaml")
print(assistant.query("Count the number of occurrences of the letter ’L’ in the word - ’LOLLAPALOOZA’."))