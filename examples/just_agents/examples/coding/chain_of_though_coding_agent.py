from dotenv import load_dotenv
from just_agents.examples.coding.mounts import coding_examples_dir
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
from just_agents import llm_options
import pprint
load_dotenv(override=True)

if __name__ == "__main__":
    ref="FLPMSAKS"

    # WARNING: TEMPORALLY NOT WORKING

    #here we use claude sonnet 3.5 as default mode, if you want to switch to another please edit yaml
    config_path = coding_examples_dir / "coding_agents.yaml"
    options =  llm_options.OPENAI_GPT4oMINI  #llm_options.ANTHROPIC_CLAUDE_3_5_SONNET
    agent: ChainOfThoughtAgent = ChainOfThoughtAgent.from_yaml("Bioinformatician", file_path=config_path)
    agent.memory.add_on_message(lambda message: print(message))
    prompt = "Get FGF2 human protein sequence from uniprot using biopython. As a result, return only the sequence"
    result, thoought = agent.think(prompt)
    print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    pprint.pprint(result)
    pprint.pprint(thoought)

    #agent.save_to_yaml("Bioinformatician", file_path=config_path)
    #agent: BaseAgent = BaseAgent.from_yaml("Bioinformatician", file_path=config_path)
    #result = agent.query("Get FGF2 human protein sequence from uniprot using biopython. As a result, return only the sequence")
    #assert amino_match_endswith(result, ref), f"Sequence ending doesn't match reference {ref}: {result}"