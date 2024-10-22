from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
load_dotenv(override=True)

assistant: IAgent = build_agent("examples/simple_code_agent.yaml")
assistant.query("Get FGF2 human protein sequence with biopython from uniprot")