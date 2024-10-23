from dotenv import load_dotenv

from just_agents.interfaces.IAgent import build_agent, IAgent
load_dotenv(override=True)

assistant: IAgent = build_agent("code_agent.yaml")
result, thoughts = assistant.query("Get FGF2 human protein sequence with biopython from uniprot")
print("Thoughts: ", thoughts)
print("Result: ", result)