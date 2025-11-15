#test to confirm all the bugfixes are inplace and ok
import json
from dotenv import load_dotenv
import pytest
import os
from pathlib import Path
from typing import Callable, Any, Dict

from just_agents.data_classes import Message, Role
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.base_agent import BaseAgent, BaseAgentWithLogging
from just_agents.web.web_agent import WebAgent
from just_agents.just_tool import JustToolsBus
from just_agents.web.chat_ui_agent import ChatUIAgent
from just_agents.web.streaming import response_from_stream
from just_agents.llm_options import OPENAI_GPT5_NANO
import tests.tools.tool_test_module as tool_test_module

TESTS_DIR = os.path.dirname(__file__)  # Get the directory where this test file is located
MODELS_DIR = os.path.join(TESTS_DIR, "models.d")  # Path to models.d inside tests

@pytest.fixture(scope="module", autouse=True)
def load_env():
    load_dotenv(override=True)

def load_agents(load_env,tmp_path) -> Dict[str,BaseAgent]:
    config_path = Path(TESTS_DIR) / "profiles" / "problematic_configs.yaml"
    os.environ["TMP_DIR"] = str(tmp_path)
    agents : dict[str,BaseAgent] = WebAgent.from_yaml_dict(yaml_path=config_path, required_base_class=WebAgent )
    return agents

def test_agents_loaded(load_env,tmp_path):
    agents = load_agents(load_env,tmp_path)
    assert agents
   # assert len(agents.keys()) == 7
    assert "groq_agent" in agents
    assert "openai_agent" in agents
    assert "deepseek_agent" in agents
    assert "deepseek_distilled_agent" in agents
    assert "custom_agent" in agents
    assert "learnLM_agent" in agents
    assert "gemini_agent" in agents


@pytest.mark.skip(reason="expensive and long, run manually")
def test_deepseek_reasoner_agent(load_env,tmp_path):
    agents = load_agents(load_env,tmp_path)
    assert agents
    assert "deepseek_agent" in agents
    assert agents["deepseek_agent"].llm_options["model"] == "deepseek/deepseek-reasoner"
    response = agents["deepseek_agent"].query("Count the number of letters e in deepseek/deepseek-reasoner", send_system_prompt=False)
    assert response
    assert "ten" in response or "10" in response
    response = agents["deepseek_agent"].query("Count the number of letters r in 'strawberry crumble'", send_system_prompt=False)
    assert "four" in response or "4" in response

@pytest.mark.skip(reason="needs server part in test, todo")
def test_custom_agent_config(load_env,tmp_path):
    agents = load_agents(load_env,tmp_path)
    assert agents
    assert "custom_agent" in agents
    #assert agents["custom_agent"].llm_options["custom_llm_provider"] == "openai"
    assert agents["custom_agent"].llm_options["model"] == "sugar_genie"
    assert agents["custom_agent"].llm_options["api_base"] == "http://127.0.0.1:8089/v1"
    response = agents["custom_agent"].query("Who are the founders of GlucoseDao??", send_system_prompt=False)
    assert response

@pytest.mark.skip(reason="https://github.com/BerriAI/litellm/issues/9296")
def test_empty_resp(load_env, tmp_path):
    config_path = Path(TESTS_DIR) / "profiles" / "tool_problem.yaml"
    agent_empty: WebAgent = ChatUIAgent.from_yaml(file_path=config_path, section_name="sugar_genie_empty_response",
                                              parent_section="agent_profiles")
    messages=[Message(
        role=Role.user,
        content="Who are the founders of GlucoseDao??"
    )]
    agent_empty.litellm_tool_description = True
    agent_empty.debug_enabled = True
    stream_generator = agent_empty.stream(
        messages
    )

    response_content = response_from_stream(stream_generator)
    assert response_content #stop with 0 chinks 70% of the time


def test_bound_method_double_self_regression(load_env):
    """Assembly test replicating the original PDF analysis scenario that caused 'double self' bug.
    
    This replicates the exact problematic pattern from the original code:
    
    Original pattern that failed:
        pdf = PagedPDF(infile)
        bot = ChatAgent(tools=[pdf.get_next_page, pdf.get_prev_page]) 
        res = bot.query("Use the tools to read the PDF and extract the title.")
    
    Would cause: TypeError: method() got multiple values for argument 'self'
    
    Battle scenario: PDF contains spell components (Abra, Shwabra, Kadabra!) that the agent
    must find by navigating through all pages to compose the complete spell.
    """
    from just_agents.just_tool import JustTransientTool, JustToolsBus
    
    # Create a spell PDF that requires navigation through all pages
    pdf_reader = tool_test_module.StatefulPDFReader.create_spell_pdf()
    
    # Create agent exactly like in the original problematic code
    # Role + Task pattern, bound method tools, realistic PDF analysis prompt
    agent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT5_NANO,
        system_prompt="You are a magical PDF analysis assistant. Navigate the PDF using the tools to gather spell components.",
        tools=[pdf_reader.get_next_page, pdf_reader.get_previous_page, pdf_reader.get_current_page, pdf_reader.jump_to_page],
        max_tool_calls=10,  # Allow multiple tool calls for spell composition
        debug=True
    )
    
    # Verify bound methods are handled as transient tools
    assert agent.tools is not None
    assert len(agent.tools) == 4
    for tool_name in ["get_next_page", "get_previous_page", "get_current_page", "jump_to_page"]:
        assert tool_name in agent.tools
        assert isinstance(agent.tools[tool_name], JustTransientTool)
        assert agent.tools[tool_name].is_transient
    
    # Set up event tracking to verify tools are actually called multiple times
    bus = JustToolsBus()
    tool_call_log = []
    
    def track_tool_calls(event_name: str, **kwargs):
        if event_name.endswith("execute") and "kwargs" in kwargs:
            # Extract tool name from event_name pattern: "tool_name.id.execute" 
            tool_name = event_name.split('.')[0]
            tool_call_log.append({
                'tool': tool_name,
                'args': kwargs.get('kwargs', {}),
                'timestamp': len(tool_call_log)
            })
    
    # Subscribe to all tool execution events
    for tool_name in agent.tools.keys():
        bus.subscribe(f"{tool_name}.*", track_tool_calls)
    
    # Store initial state for verification
    initial_page = pdf_reader.current_page
    assert initial_page == 0  # Should start at page 0 (displays as "Page 1")
    
    # THE MAIN ASSEMBLY TEST - BATTLE SCENARIO:
    # Agent must navigate through ALL pages to find spell components and compose the spell
    # This forces multiple bound method calls that would trigger the "double self" bug
    try:
        response = agent.query(
            "This PDF contains an ancient spell with three magic words hidden across its pages. "
            "Use the navigation tools [get_next_page, get_previous_page, get_current_page, jump_to_page] "
            "to read through ALL pages of the document and find the three magic words. "
            "Then compose the complete spell by combining all three words in the correct order. "
            "Start by checking the current page, then navigate through each page systematically.",
            send_system_prompt=True
        )
        
        # If we reach here, the "double self" bug is NOT present
        assert response is not None, "Agent should return a response"
        assert len(response) > 0, "Response should not be empty"
        
        # BATTLE SCENARIO VERIFICATION: Check if the agent found all spell components
        response_lower = response.lower()
        spell_words_found = 0
        if "abra" in response_lower:
            spell_words_found += 1
        if "shwabra" in response_lower:
            spell_words_found += 1
        if "kadabra" in response_lower:
            spell_words_found += 1
            
        # The agent should have found at least 2 of the 3 spell words (allowing for some LLM variation)
        # This proves it actually navigated through the pages using the bound methods
        assert spell_words_found >= 2, (
            f"Agent should have found at least 2 spell words by navigating pages, "
            f"but only found {spell_words_found} in response: {response}"
        )
        
    except TypeError as e:
        error_msg = str(e).lower()
        if "multiple values for argument 'self'" in error_msg or "got multiple values for" in error_msg:
            pytest.fail(
                f"REGRESSION FAILURE: The 'double self' bug is present! "
                f"Bound instance methods are receiving duplicate 'self' parameters. "
                f"Error: {e}"
            )
        else:
            # Different TypeError, re-raise for investigation
            raise e
    
    # Verify tools were actually called multiple times (assembly behavior)
    assert len(tool_call_log) >= 3, f"Expected multiple tool calls, got {len(tool_call_log)}: {tool_call_log}"
    
    # Verify different tools were called (not just one tool repeatedly)
    called_tools = set(call['tool'] for call in tool_call_log)
    assert len(called_tools) >= 2, f"Expected multiple different tools to be called, got: {called_tools}"
    
    # Verify the PDF reader state actually changed due to tool calls
    # This proves the bound methods' 'self' references are working correctly
    final_page = pdf_reader.current_page
    assert final_page != initial_page, f"PDF reader state should have changed from page {initial_page} to {final_page}"
    
    # DIRECT TOOL TESTING: Verify tools still work correctly after agent interaction
    # Test parameterized tool call (jump to a specific page)
    jump_result = agent.tools["jump_to_page"](page_number=1)
    assert "Abra" in jump_result, f"Should find 'Abra' on page 1, got: {jump_result}"
    
    # Navigate to page 2 and verify
    page2_result = agent.tools["get_next_page"]()
    assert "Shwabra" in page2_result, f"Should find 'Shwabra' on page 2, got: {page2_result}"
    
    # Navigate to page 3 and verify
    page3_result = agent.tools["get_next_page"]()
    assert "Kadabra!" in page3_result, f"Should find 'Kadabra!' on page 3, got: {page3_result}"
    
    print(f"✅ Battle scenario assembly test passed! ")
    print(f"   Tools called {len(tool_call_log)} times: {[c['tool'] for c in tool_call_log]}")
    print(f"   Spell words found in response: {spell_words_found}/3")
    print(f"   Agent successfully navigated PDF using bound methods without 'double self' bug!")
