import asyncio
from dotenv import load_dotenv
from just_agents.simple.cot_agent import ChainOfThoughtAgent

import just_agents.llm_options

def count_letters(character:str, word:str) -> str:
    """ Returns the number of character occurrences in the word. """
    count:int = 0
    for char in word:
        if char == character:
            count += 1
    print("Function: ", character, " occurres in ", word, " ", count, " times.")
    return str(count)


def test_function_query():
    load_dotenv(override = True)

    opt = just_agents.llm_options.OPENAI_GPT4oMINI.copy()
    agent: ChainOfThoughtAgent = ChainOfThoughtAgent(opt, tools=[count_letters])
    result, thoughts = agent.query("Count the number of occurrences of the letter ’L’ in the word - ’LOLLAPALOOZA’.")
    print("Thoughts: ", thoughts)
    print("Results: ", result)
    assert "4" in result

async def process_stream(async_generator):
    for item in async_generator:
        print(item)


def test_stream_function_query():
    load_dotenv(override = True)

    opt = just_agents.llm_options.OPENAI_GPT4oMINI.copy()
    agent: ChainOfThoughtAgent = ChainOfThoughtAgent(opt, tools=[count_letters])
    stream = agent.stream("Count the number of occurrences of the letter ’L’ in the word - ’LOLLAPALOOZA’.")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_stream(stream))
    result = agent.last_message()
    print(result)
    assert "4" in str(result["content"])