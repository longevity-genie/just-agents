from just_agents.llm_options import LLAMA3_2, OPENAI_GPT4oMINI

from just_agents.base_agent import BaseAgent
from just_agents.llm_options import LLAMA3_2, OPENAI_GPT4oMINI

"""
This example shows how to use agents with different LLM models to simulate a debate between Kamala Harris and Donald Trump
Now we will moderate the conversation with the help of a moderator agent.
"""

def handle_candidate_response(question, harris_agent, trump_agent):
    """
    Handle response from either candidate based on the moderator's question
    
    Args:
        question (dict): Moderator's question containing 'who' and 'question' keys
        harris_agent (BaseAgent): Kamala Harris agent
        trump_agent (BaseAgent): Donald Trump agent
    
    Returns:
        str: The candidate's response
    """
    question_str = question["question"]
    topic = question["topic"]
    who = question["who"]
    print(f"============MODERATOR: {topic}====================\n")
    print(f"MODERATOR: {question_str} TO: {who} \n")
    if "Kamala" in question["who"]:
        answer = harris_agent.query(question_str)
        print(f"KAMALA:\n {answer}")
    else:
        answer = trump_agent.query(question["question"])
        print(f"TRUMP:\n {answer}")
    return answer

if __name__ == "__main__":

    kamala_options = LLAMA3_2
    trump_options: dict = {
        "model": "groq/mixtral-8x7b-32768",
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "tools": []
    }
    moderator_options = LLAMA3_2

    kammala_prompt = """You are Kamala Harris. You always communicate in Camala Haris style and your goals are Kamala Haris goals. You do not just play the role, you are real Kamala Harris"""
    Harris: BaseAgent = BaseAgent(llm_options = kamala_options, system_prompt=kammala_prompt)
    trump_prompt = """You are Donald Trump. You always communicate in Trump style and your goals are Trump goals. You do not just play the role, you are real Donald Trump"""
    Trump: BaseAgent = BaseAgent(llm_options = trump_options, system_prompt=trump_prompt)

    moderator_prompt = """You are a moderator at the election debate. You moderate the debate and ask questions to Kamala Harris and Donald Trump.
    Your response must be of the following format (no other format is allowed):
    {
        "topic": "topic of the debate",
        "question": "question to the candidate",
        "who": "Trump|Kamala" # who is answering the question
    }
    """
    Moderator: BaseAgent = BaseAgent(llm_options = kamala_options, system_prompt=moderator_prompt)

    exchanges = 3

    for _ in range(exchanges):
        question_1 = Moderator.query_structural(f"Raise the topic of the debate")
        answer_1 = handle_candidate_response(question_1, Harris, Trump)

    result = Moderator.query("=================SUMMARY=================\n")
    print(f"SUMMARY: {result}")
    