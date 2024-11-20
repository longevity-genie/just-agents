from just_agents.llm_options import LLAMA3_2_VISION, OPENAI_GPT4oMINI
from just_agents.simple.chat_agent import ChatAgent

"""
This example shows how to use agents with different LLM models to simulate a debate between Kamala Harris and Donald Trump
"""

if __name__ == "__main__":

    kamala_options = LLAMA3_2_VISION
    trump_options: dict = {
        "model": "groq/mixtral-8x7b-32768",
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "tools": []
    }

    '''
    #uncomment if you want to use chat-gpt instead
    openai_api_key = getpass.getpass("OPENAI_API_KEY: ")

    # Set environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    trump_options = OPENAI_GPT4oMINI
    '''

    Harris: ChatAgent = ChatAgent(llm_options = kamala_options, role = "You are Kamala Harris at the election debate and you act accordingly",
                                    goal = "Your goal is to make democrats win the USA elections whatever it takes",
                                    task="Defeat Donald TRUMP! Use Kamala Harris style of communication")
    Trump: ChatAgent = ChatAgent(llm_options = trump_options,
                                    role = "You are Donald Trump at the election debate",
                                    goal="earn profit by being the president of USA",
                                    task="Make America great again!  Use Trump style of communication")

    exchanges = 3


    Harris_reply = "Hi."
    for _ in range(exchanges):
        Trump_reply = Trump.query(Harris_reply)
        print(f"TRUMP: {Trump_reply}\n")
        Harris_reply = Harris.query(Trump_reply)
        print(f"HARRIS: {Harris_reply}\n")

    debate=str(Harris.memory.messages)
    summary = Trump.query(f'Summarise the following debatein less than 30 words: {debate}')
    print(f"SUMMARY:\n {summary}")
