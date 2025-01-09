from dotenv import load_dotenv
from just_agents.base_agent import ChatAgent
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
import just_agents.llm_options

from just_agents.tools.db import sqlite_query
import pytest
import requests
import os
from pathlib import Path
from pycomfort.logging import to_nice_stdout
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
                    llm_options=just_agents.llm_options.LLAMA3_3)

    
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