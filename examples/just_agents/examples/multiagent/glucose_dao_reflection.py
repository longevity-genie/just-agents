from dotenv import load_dotenv
from pathlib import Path

from just_agents import llm_options
#from just_agents_tools.search import get_semantic_paper
from just_agents.base_agent import ChatAgent
import typer

app = typer.Typer(no_args_is_help=True)

current_dir = Path(__file__).parent.absolute()
examples_dir = current_dir.parent
output_dir = examples_dir.parent  / "output" / current_dir.name
output_dir.mkdir(exist_ok=True, parents=True)

"""
This example shows how a multiagent system can be used to create a plan for a DAO or nonprofit.
Here we use the simple reflection mechanism to iteratively improve the plan by having a planner and a reviewer.
In real-world use-cases we can have more sophisticated configurations, when there are also agents that search in the internet, agents that do coding, etc.
"""

@app.command()
def planning_example():
    load_dotenv()

    # Initialize conversation file
    conversation_file = output_dir / "glucose_dao_conversation.txt"
    conversation_file.write_text("") # Clear/create file

    planner_options = llm_options.DEEPSEEK_R1
    
    planner: ChatAgent = ChatAgent(llm_options = planner_options , role = "You are a helpful adviser that helps people to create their NGOs and DAOs",
                                   goal = "Your goal is to help the user to make the best possible action plan to create her NGO or DAO.",
                                   task="Create the best actionable plan possible while being realististic knowing limitations of the time and resources of the founder and current state of the art, take into account the feedback and suggestions, improve it until it is perfect.")
    reviewer: ChatAgent = ChatAgent(llm_options = llm_options.OPENAI_GPT4oMINI,
                                      role = "you represent the interests of the DAO or nonprofit creator and provide feedback and suggestions for the plan which is generated for you",
                                      goal="provide the best feedback ever and ask for specific improvements",
                                      task="evaluate the plan and provide feedback and suggestions for the plan")

    exchanges: int = 2
    prompt = """
    I am a diabetic person with self-taught data-science, bioinformatics and machine learning skills and I want to create a nonprofit organization or DAO (let it be Glucose DAO) that will collect data from continuous glucose monitors (CGM) and use it to develop machine learning models for diabetes management.
    Current state of the art if the following:
    * Underutilized data from increasing CGM adoption: 463 million people have diabetes worldwide and millions of them already use CGMs for many years. Existing datasets have only dozens of people monitored at short intervals of several months [Renat Sergazinov, et. al, "GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks." OpenReview, 2023, see also https://github.com/IrinaStatsLab/Awesome-CGM). While acknowledging the endeavor of Irina Gaynanova lab of summarizing existing datasets, we believe that a novel, large longitudinal dataset must be created.
    * Most existing data are of clinical/academic nature: While useful for research and development, most existing datasets are usually of academic/clinical nature, have short timespans, have a small number of non-diabetic people, and do not include some important aspects (like data on sensor resets, for example).
    * Existing models are hard to deploy and use in real-world scenarios and are not user-friendly: Only a small part of the models is public and most of them are hard to deploy and use on an everyday basis for an average user. What is important for the end-user is that the model pre-trained on public datasets should be fine-tuned on your own data to predict your future values and your risks.
    * Existing data sharing platforms (like https://www.tidepool.org/bigdata) sell users' personal information to companies without compensating the data owners. We believe users should be fairly rewarded when their data is sold for commercial purposes. One potential solution is the development of smart contracts by the Glucose DAO.
    Please, create me a personal action plan to create and develop the dao. Hightlight 4 months and 2 years goals and KPIs
    """
    
    # Write initial prompt
    with conversation_file.open("a") as f:
        f.write(f"Initial Prompt:\n{prompt}\n\n")
    
    for i in range(exchanges):
        plan = planner.query(prompt)
        feedback = reviewer.query(plan)
        prompt = feedback
        
        # Write conversation round to file
        with conversation_file.open("a") as f:
            f.write(f"Round {i+1}\n================================\n")
            f.write(f"Plan:\n{plan}\n--------------------------------\n")
            f.write(f"Feedback:\n{feedback}\n************************************\n")

if __name__ == "__main__":
    app()