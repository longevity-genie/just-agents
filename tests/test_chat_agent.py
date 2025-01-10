import pprint
from dotenv import load_dotenv
from just_agents.base_agent import ChatAgent
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
from just_agents.llm_options import LLAMA3_3, OPENAI_GPT4oMINI
from pprint import pprint

from just_agents.tools.db import sqlite_query
import pytest
import requests
import os
from pathlib import Path
from pycomfort.logging import to_nice_stdout
from pydantic import BaseModel, Field
from typing import Optional

to_nice_stdout()
db_path = Path("tests/data/open_genes.sqlite")


@pytest.fixture(scope="session")
def open_genes_db() -> Path:
    """Downloads the open_genes.sqlite database if not present and returns the path"""
    # Define the database path
    db_path = Path("tests/data/open_genes.sqlite").absolute().resolve()

    
    # Create directories if they don't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download file if it doesn't exist
    if not db_path.exists():
        url = "https://github.com/longevity-genie/longevity_gpts/raw/main/open_genes/data/open_genes.sqlite"
        print(f"Downloading database from {url}")
        
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the file
        with open(db_path, "wb") as f:
            f.write(response.content)
        
        print(f"Database saved to {db_path}")
    
    return db_path



def test_quering(open_genes_db):
    load_dotenv(override = True)

    agent = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form comands to work with the {open_genes_db} sqlite database",
                    task="formulate appropriate comands to operate in the given database.",
                    tools=[sqlite_query],
                    llm_options=LLAMA3_3)

    
    response = agent.query("Show me all tables in the database")
    agent.memory.pretty_print_all_messages()
    assert response is not None, "Response should not be None"
    
    # Add specific table checks
    expected_tables = [
        'lifespan_change',
        'gene_criteria',
        'gene_hallmarks',
        'longevity_associations'
    ]
    
    for table in expected_tables:
        assert table.lower() in response.lower(), f"Expected table '{table}' not found in response"



class AgentResponse(BaseModel):
    user_question: str = Field(..., description="The original question asked by the user")
    answer: Optional[str] = Field(default="", description="Agent's initial answer or analysis")
    delegate_to: Optional[str] = Field(default="", description="Name of the agent to delegate to, empty if no delegation")
    question: Optional[str] = Field(default="", description="Question to delegate to another agent, empty if no delegation")
    final_answer: Optional[str] = Field(default="", description="Final answer to the user's question, empty if pending delegation")

    class Config:
        json_schema_extra = {
            "example": {
                "user_question": "What genes extend lifespan?",
                "answer": "This requires database access to analyze lifespan data",
                "delegate_to": "agent_db",
                "question": "SELECT * FROM lifespan_change WHERE species='mouse' ORDER BY lifespan_change DESC",
                "final_answer": ""
            }
        }

@pytest.mark.skip(reason="Temporarily skipping this test")
def test_delegation():
    
    agent_db = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form comands to work with the {db_path} sqlite database",
                    task="formulate appropriate comands to operate in the given database.",
                    tools=[sqlite_query],
                    llm_options=OPENAI_GPT4oMINI
                    )

    ponder_agent = ChatAgent(
        role="helpful agent which will only distribute the tasks to it's calling list and make minimal suggestions",
        goal="help users by guiding them to appropriate instances that will answer their questions",
        task=f"You are generalist agent which can delegate tasks to other agents, so far you only know agent_db which can run database queries on your behalf.",
        format="""all your answers you represent solely as JSON object without any text either beore or after it. The format is the following: 
        {
        "user_question": "question",
        "answer": "your answer", 
        "delegate_to": "agent_name", 
        "question": "question_to_delegate",
        "final_answer": "final answer to the question"
        } If you do not delegate then delegate_to and question should be empty. if you are ready to answer the question then final_answer should be your answer, if not it must be empty""",
        llm_options=OPENAI_GPT4oMINI
    )

    for i in range(3):
        result = ponder_agent.query_structural(
            "Interventions on which genes extended mice lifespan most of all? Search all the relevant tables in the open-genes sqlite and only for mouse", paarser=type[AgentResponse])
        if result["delegate_to"] == "agent_db":
            result = agent_db.query(result["question"])
        print("\nQuery Result:")  # Add a header for clarity
        pprint(result)
