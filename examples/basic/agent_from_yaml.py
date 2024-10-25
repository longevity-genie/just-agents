from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
load_dotenv(override=True)

basic_examples_dir = Path(__file__).parent.absolute()

if __name__ == "__main__":
    assistant: IAgent = build_agent(basic_examples_dir / "agent_from_yaml.yaml")
    print(assistant.query("Count the number of occurrences of the letter ’L’ in the word - ’LOLLAPALOOZA’."))