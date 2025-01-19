from typing import Any
from just_agents.base_agent import ChatAgent
from just_agents.llm_options import LLAMA3_3

"""
This example shows how to use agents with different LLM models to simulate a moderated debate between Kamala Harris and Donald Trump
"""

if __name__ == "__main__":

    kamala_options = LLAMA3_3
    trump_options = LLAMA3_3

    # Create a moderator with the same model as Kamala for consistency
    moderator_options: dict[str, Any] = {
        "model": "groq/mixtral-8x7b-32768",
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "tools": []
    }

    Harris: ChatAgent = ChatAgent(
        llm_options=kamala_options, 
        role="You are Kamala Harris in a presidential debate",
        goal="Win the debate with clear, concise responses",
        task="Respond briefly and effectively to debate questions"
    )

    Trump: ChatAgent = ChatAgent(
        llm_options=trump_options,
        role="You are Donald Trump in a presidential debate",
        goal="Win the debate with your signature style",
        task="Respond briefly and effectively to debate questions"
    )

    Moderator: ChatAgent = ChatAgent(
        llm_options=moderator_options,
        role="You are a neutral debate moderator",
        goal="Ensure a fair and focused debate",
        task="Generate clear, specific questions about key political issues"
    )

    exchanges = 2

    # Start with moderator generating the first question
    for _ in range(exchanges):
        # Moderator generates question
        question = Moderator.query("Generate a concise debate question about a current political issue.")
        print(f"\nMODERATOR: {question}\n")

        # Trump answers first
        trump_reply = Trump.query(question)
        print(f"TRUMP: {trump_reply}\n")

        # Harris responds
        harris_reply = Harris.query(f"Question: {question}\nTrump's response: {trump_reply}")
        print(f"HARRIS: {harris_reply}\n")

    debate = str(Harris.memory.messages)
    summary = Moderator.query(f'Summarise the following debate in less than 30 words: {debate}')
    print(f"SUMMARY:\n {summary}")
