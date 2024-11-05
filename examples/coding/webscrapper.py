from pathlib import Path

from dotenv import load_dotenv
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession

from examples.coding.mounts import make_mounts, input_dir, output_dir, coding_examples_dir
from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from examples.coding.tools import write_thoughts_and_results, amino_match_endswith
from examples.coding.mounts import input_dir, output_dir, coding_examples_dir

load_dotenv(override=True)

"""
This example shows how to use a simple code agent to run python code and bash commands, it does not use volumes and is based on basic LLMSession class.
"""

if __name__ == "__main__":
    assistant: LLMSession= build_agent(coding_examples_dir / "webscrapper.yaml") #probably should make seaparate webscrapping agent
    query = """
    Here is a list of events from Zelar: https://app.sola.day/event/zelarcity 
    Investigate the layout of the page and find all the events.
    Scrape it and save events information in /output/zelar_events.txt 
    If you get zero events, try to find a way to navigate to subpages and scrape them.
    """
    result, thoughts = assistant.query(query)
    write_thoughts_and_results("zelar_events", thoughts, result)