import pprint
from time import sleep, time
from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import ChatAgent
from just_agents.data_classes import ImageContent, Message, Role, TextContent
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
from just_agents.llm_options import LLAMA3_3, OPENAI_GPT4oMINI, GEMINI_2_FLASH, GEMINI_2_FLASH_EXP, OPENAI_GPT4o
from pprint import pprint

from just_agents.tools.db import sqlite_query
import pytest
import requests
import os
from pathlib import Path
from pycomfort.logging import to_nice_stdout
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


to_nice_stdout()
db_path = Path("tests/data/open_genes.sqlite")
sleep_time = float(os.getenv("SLEEP_TIME", "20.0"))


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


def _test_database_tables(agent: ChatAgent, open_genes_db: Path):
    """Helper function to test database table queries with different agents"""
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
    
    # Check each expected table is in the agent's response
    for table in expected_tables:
        assert table.lower() in response.lower(), f"Expected table '{table}' not found in agent response"
    
    sleep(sleep_time)

def test_quering(open_genes_db):
    load_dotenv(override = True)
    agent = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form commands to work with the {open_genes_db} sqlite database",
                    task="formulate appropriate commands to operate in the given database and always include the table names in your response.",
                    tools=[sqlite_query],
                    llm_options=LLAMA3_3,
                    key_list_env="GROQ_API_KEY"
                    )
    _test_database_tables(agent, open_genes_db)

def test_quering_gemini(open_genes_db):
    load_dotenv(override = True)
    agent = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form commands to work with the {open_genes_db} sqlite database",
                    task="formulate appropriate commands to operate in the given database using 'sqlite_query' tool and always include the table names in your response.",
                    tools=[sqlite_query],
                    llm_options=GEMINI_2_FLASH
                    )
    _test_database_tables(agent, open_genes_db)


def test_query_structural():
    agent = ChatAgent(role="helpful agent that provides structured information",
                    goal="help users by providing structured information in JSON format",
                    task="analyze user questions and provide comprehensive responses in a structured format",
                    format="""all your answers should be represented as a JSON object with the following structure:
                    {
                      "user_question": "the original question asked by the user",
                      "answer": "your analysis of the question",
                      "delegate_to": "",
                      "question": "",
                      "final_answer": "your complete answer to the user's question"
                    }""",
                    llm_options=GEMINI_2_FLASH_EXP
                    )
    
    # Ask the agent a question that doesn't require SQL or tool use
    response = agent.query_structural("What are the main factors that contribute to aging?", parser=AgentResponse, enforce_validation=True)
    
    print("RESPONSE 1 =======================================")
    pprint(response)

    # Validate response structure
    assert isinstance(response, AgentResponse), "Response should be an AgentResponse instance"
    assert response.user_question == "What are the main factors that contribute to aging?"
    # Check that the response follows the expected structure
    assert hasattr(response, "user_question")
    assert hasattr(response, "answer")
    assert hasattr(response, "delegate_to")
    assert hasattr(response, "question")
    #assert hasattr(response, "final_answer")


class AgentResponse(BaseModel):
    user_question: str = Field(..., description="The original question asked by the user")
    answer: Optional[str] = Field(default="", description="Agent's initial answer or analysis")
    delegate_to: Optional[str] = Field(default="", description="Name of the agent to delegate to, empty if no delegation")
    question: Optional[str] = Field(default="", description="Question to delegate to another agent, empty if no delegation")
    final_answer: Optional[str] = Field(default="", description="Final answer to the user's question, empty if pending delegation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_question": "What genes extend lifespan?",
                "answer": "This requires database access to analyze lifespan data",
                "delegate_to": "agent_db",
                "question": "SELECT * FROM lifespan_change WHERE species='mouse' ORDER BY lifespan_change DESC",
                "final_answer": ""
            }
        }
    )


def test_vision():
    load_dotenv(override = True)
    agent = ChatAgent(role="helpful agent that can see",
                goal="help users by providing a description of the image",
                task="analyze the image and provide a description of the image",
                llm_options=OPENAI_GPT4o
                )
    message = Message(
        role=Role.user,
        content=[
        TextContent(text="What is in this image?"),
        ImageContent(image_url="https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg", use_nested_format=True)
        ]
    )
    pprint(message)
    result = agent.query(message)
    #https://arxiv.org/pdf/2410.05780
    assert "cat" in result




@pytest.mark.skip(reason="needs to be rewritten as it exceeds the context window")
def test_delegation():
    
    agent_db = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form comands to work with the {db_path} sqlite database",
                    task="formulate appropriate comands to operate in the given database.",
                    tools=[sqlite_query],
                    llm_options=GEMINI_2_FLASH
                    )
    agent_db.query("What is in this image?")

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

    for i in range(2):
        result = ponder_agent.query_structural(
            "Interventions on which genes extended mice lifespan most of all? Search all the relevant tables in the open-genes sqlite and only for mouse", 
            parser=AgentResponse, 
            enforce_validation=False
        )
        if result.delegate_to == "agent_db":
            result = agent_db.query(result.question)
        print("\nQuery Result:")  # Add a header for clarity
        sleep(sleep_time)
        pprint(result)
        