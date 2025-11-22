from dotenv import load_dotenv
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
import just_agents.llm_options

def count_letters(character: str, word: str) -> str:
    """ Returns the number of character occurrences in the word. """
    count:int = 0
    for char in word:
        if char == character:
            count += 1
    print("Function: ", character, " occurres in ", word, " ", count, " times.")
    return str(count)


def test_function_query():
    load_dotenv(override = True)
    from pathlib import Path
    # FOR LLAMA 4 THERE IS A BUG FROM GROQ SIDE ON COMBINED USE OF STRUCTURED OUTPUT AND TOOLS ON V4 MODELS
    # https://community.groq.com/discussion-forum-7/tool-use-failed-on-llama4-models-230
    # PPS: as of 15 aug completely broken for GROQ
    llm_options = just_agents.llm_options.GEMINI_2_5_FLASH
    prompt = "Count the number of occurrences of the letter 'L' in the word - 'LOLLAPALOOZA'."
    agent: ChainOfThoughtAgent = ChainOfThoughtAgent.from_yaml("ChainOfThoughtAgent", "agent_profiles",
                                          file_path=Path("profiles/agent_profiles.yaml"))
    assert len(agent.list_tools()) == 1
    agent.llm_options = just_agents.llm_options.GEMINI_2_5_FLASH
    result, thoughts = agent.think(prompt)
    assert thoughts
    assert result
    assert "4" in result.content

    agent: ChainOfThoughtAgent = ChainOfThoughtAgent(llm_options=llm_options, tools=[count_letters])
    result, thoughts = agent.think(prompt)
    print("Thoughts: ", thoughts)
    print("Results: ", result)
    assert "4" in result.content
