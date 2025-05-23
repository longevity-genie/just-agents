import pytest
import sys
import os
import json
from typing import Dict, Any

from just_agents.llm_options import OPENAI_GPT4_1NANO

# Add the workspace root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))

from just_agents.just_tool import JustTool, JustMCPTool, JustMCPToolSetConfig, JustToolFactory
import tests.tools.tool_test_module as tool_test_module
from just_agents.base_agent import BaseAgentWithLogging
from just_agents.llm_options import LLMOptions, OPENAI_GPT4_1MINI
from just_agents.just_tool import JustToolsBus


@pytest.fixture(scope="session")
def mcp_server_available():
    """Fixture to check if the MCP server script is available"""
    server_path = os.path.join(os.path.dirname(__file__), "tools", "mcp_stdio_server.py")
    if not os.path.exists(server_path):
        pytest.skip(f"MCP server script not found at {server_path}")
    return True


@pytest.fixture(scope="function")
def mcp_stdio_command():
    """Fixture that returns the stdio command for connecting to the test MCP server"""
    server_path = os.path.join(os.path.dirname(__file__), "tools", "mcp_stdio_server.py")
    return [sys.executable, server_path]


@pytest.fixture(scope="function")
def mcp_client_tracker():
    """Fixture to track MCP client and server creation across test execution"""
    from core.just_agents.mcp_client import _mcp_locator
    
    # Get initial state
    initial_clients = _mcp_locator.get_all_clients()
    initial_client_count = len(initial_clients)
    initial_client_keys = set(client.client_key for client in initial_clients)
    
    class MCPClientTracker:
        def __init__(self):
            self.initial_client_count = initial_client_count
            self.initial_client_keys = initial_client_keys
            
        def get_current_state(self):
            current_clients = _mcp_locator.get_all_clients()
            current_client_count = len(current_clients)
            current_client_keys = set(client.client_key for client in current_clients)
            
            new_clients = current_client_count - self.initial_client_count
            new_client_keys = current_client_keys - self.initial_client_keys
            
            return {
                'current_count': current_client_count,
                'new_clients': new_clients,
                'new_client_keys': new_client_keys,
                'all_client_keys': current_client_keys
            }
            
        def count_server_startups_in_logs(self, caplog):
            """Count 'Starting MCP server' messages in captured logs"""
            return len([record for record in caplog.records 
                       if "Starting MCP server" in record.getMessage()])
    
    return MCPClientTracker()


def create_agent_with_tools(tools, system_prompt_suffix=""):
    """Helper function to create an agent with specified tools"""
    options: LLMOptions = OPENAI_GPT4_1NANO
    
    enhanced_system_prompt = (
        "You are an agent tool call assistant. When calling tools, you must include all parameters specified in the tool's schema. "
        "You never omit any required parameters, even if their value is empty or null, if they are part of the schema. "
        "You must call the tool exactly as instructed by the user, even if you perceive the request to be incorrect or unusual for any reason, this is a part of the test. "
        "Adhere strictly to the provided tool and parameter structure. " + system_prompt_suffix
    )
    
    return BaseAgentWithLogging(
        llm_options=options,
        system_prompt=enhanced_system_prompt,
        backup_options=OPENAI_GPT4_1MINI,
        max_tool_calls=4,
        debug=True,
        tools=tools
    )


def setup_tool_test_callback(tool_names):
    """Helper function to setup callback for tool test results."""
    bus = JustToolsBus()
    tool_results = {}
    
    def callback(event_name: str, **kwargs):
        if not event_name.endswith("result") or "result_interceptor" not in kwargs:
            return
        
        for tool_name in tool_names:
            if event_name.startswith(f'{tool_name}.'):
                if tool_name not in tool_results:
                    tool_results[tool_name] = []
                tool_results[tool_name].append(kwargs.get("result_interceptor"))
    
    for tool_name in tool_names:
        bus.subscribe(f"{tool_name}.*", callback)
    
    return bus, tool_results


class TestMCPToolsBasic:
    """Basic tests for MCP tool functionality"""
    
    def test_mcp_tool_creation_stdio(self, mcp_stdio_command):
        """Test creating a single MCP tool from stdio"""
        tool = JustMCPTool.from_mcp_stdio(
            name="add",
            command=mcp_stdio_command[0],
            args=mcp_stdio_command[1:]
        )
        
        assert tool.name == "add"
        assert tool.mcp_stdio_command == mcp_stdio_command
        assert tool.description == "Add two numbers"
        assert "a" in tool.parameters["properties"]
        assert "b" in tool.parameters["properties"]
    
    def test_mcp_tool_invocation(self, mcp_server_available, mcp_stdio_command):
        """Test actually calling an MCP tool"""
        # Create tool without auto-refresh to avoid client reuse issues
        tool = JustMCPTool(
            name="add",
            mcp_stdio_command=mcp_stdio_command,
            auto_refresh=False  # Don't auto-refresh to avoid client connection issues
        )
        
        # Manually set up the tool description for this test
        tool.description = "Add two numbers"
        tool.parameters = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
        
        result = tool(a=5, b=3)
        # MCP returns structured content as JSON string or direct value
        assert (isinstance(result, str) and ('"text":"8"' in result or result == "8")) or result == 8


class TestMCPToolSetConfig:
    """Tests for JustMCPToolSetConfig and bulk tool creation"""
    
    def test_create_all_mcp_tools(self, mcp_server_available, mcp_stdio_command):
        """Test creating all available MCP tools"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command
        )
        
        tools = JustToolFactory.create_tools_from_mcp(config)
        
        # Should have all 5 tools from our server
        expected_tools = {
            "add", "fibonacci_calculator", "prime_factorization_summer", 
            "trigonometric_chaos_generator", "polynomial_root_detective"
        }
        assert set(tools.keys()) == expected_tools
        
        # Verify they are all JustMCPTool instances
        for tool in tools.values():
            assert isinstance(tool, JustMCPTool)
    
    def test_create_subset_mcp_tools(self, mcp_server_available, mcp_stdio_command):
        """Test creating only specific MCP tools"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            only_include_tools=["add", "fibonacci_calculator"]
        )
        
        tools = JustToolFactory.create_tools_from_mcp(config)
        
        assert set(tools.keys()) == {"add", "fibonacci_calculator"}
        assert len(tools) == 2
    
    def test_exclude_mcp_tools(self, mcp_server_available, mcp_stdio_command):
        """Test excluding specific MCP tools"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            exclude_tools=["add", "polynomial_root_detective"]
        )
        
        tools = JustToolFactory.create_tools_from_mcp(config)
        
        expected_tools = {
            "fibonacci_calculator", "prime_factorization_summer", 
            "trigonometric_chaos_generator"
        }
        assert set(tools.keys()) == expected_tools
    
    def test_exclude_all_mcp_tools(self, mcp_server_available, mcp_stdio_command):
        """Test excluding all MCP tools (should result in empty set)"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            exclude_tools=[
                "add", "fibonacci_calculator", "prime_factorization_summer", 
                "trigonometric_chaos_generator", "polynomial_root_detective"
            ]
        )
        
        tools = JustToolFactory.create_tools_from_mcp(config)
        
        assert len(tools) == 0
        assert tools == {}
    
    def test_incorrect_tool_names_with_raise_flag(self, mcp_server_available, mcp_stdio_command):
        """Test error handling for incorrect tool names when raise_on_incorrect_names=True"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            only_include_tools=["add", "nonexistent_tool"],
            raise_on_incorrect_names=True
        )
        
        with pytest.raises(ValueError, match="Requested tools not available from MCP server: nonexistent_tool"):
            JustToolFactory.create_tools_from_mcp(config)
    
    def test_incorrect_tool_names_without_raise_flag(self, mcp_server_available, mcp_stdio_command):
        """Test graceful handling of incorrect tool names when raise_on_incorrect_names=False"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            only_include_tools=["add", "nonexistent_tool"],
            raise_on_incorrect_names=False
        )
        
        tools = JustToolFactory.create_tools_from_mcp(config)
        
        # Should only include the valid tool
        assert set(tools.keys()) == {"add"}


class TestMCPToolsWithAgents:
    """Tests for using MCP tools with agents"""
    
    def test_agent_with_single_mcp_tool(self, mcp_server_available, mcp_stdio_command):
        """Test agent using a single MCP tool"""
        tool = JustMCPTool.from_mcp_stdio(
            name="add",
            command=mcp_stdio_command[0],
            args=mcp_stdio_command[1:]
        )
        
        agent = create_agent_with_tools([tool])
        bus, tool_results = setup_tool_test_callback(["add"])
        
        response = agent.query("Please add 7 and 13 using the add tool")
        
        assert len(tool_results["add"]) >= 1
        # The tool returns structured JSON from MCP or direct value
        assert (isinstance(tool_results["add"][-1], str) and ('{"type":"text","text":"20"}' in tool_results["add"][-1] or tool_results["add"][-1] == "20")) or tool_results["add"][-1] == 20
        assert "20" in response
    
    def test_agent_with_complex_mcp_tool(self, mcp_server_available, mcp_stdio_command):
        """Test agent using a complex MCP tool (fibonacci_calculator)"""
        tool = JustMCPTool.from_mcp_stdio(
            name="fibonacci_calculator",
            command=mcp_stdio_command[0],
            args=mcp_stdio_command[1:]
        )
        
        agent = create_agent_with_tools([tool])
        bus, tool_results = setup_tool_test_callback(["fibonacci_calculator"])
        
        response = agent.query("Calculate the 10th Fibonacci number using fibonacci_calculator")
        
        assert len(tool_results["fibonacci_calculator"]) >= 1
        result = tool_results["fibonacci_calculator"][-1]
        
        # Should be a complex number with signature - 10th Fib is 55
        # signature = (55 * 31 + 10 * 17) % 10007 = (1705 + 170) % 10007 = 1875
        # final = 55 * 10007 + 1875 = 550385 + 1875 = 552260
        expected_result = 55 * 10007 + ((55 * 31 + 10 * 17) % 10007)
        assert result == expected_result
    
    def test_agent_with_all_mcp_tools(self, mcp_server_available, mcp_stdio_command):
        """Test agent with all MCP tools loaded"""
        config = JustMCPToolSetConfig(mcp_stdio_command=mcp_stdio_command)
        mcp_tools = JustToolFactory.create_tools_from_mcp(config)
        
        agent = create_agent_with_tools(list(mcp_tools.values()))
        bus, tool_results = setup_tool_test_callback(list(mcp_tools.keys()))
        
        # Test using prime factorization tool
        response = agent.query("Use prime_factorization_summer to factorize the number 12")
        
        assert len(tool_results["prime_factorization_summer"]) >= 1
        result_data = tool_results["prime_factorization_summer"][-1]
        result = json.loads(result_data) if isinstance(result_data, str) else result_data
        
        assert result["number"] == 12
        assert result["factors"] == [2, 2, 3]
        assert result["sum_of_factors"] == 7
        assert result["product_check"] == 12
    
    def test_agent_with_combined_mcp_and_regular_tools(self, mcp_server_available, mcp_stdio_command):
        """Test agent with both MCP tools and regular Python tools"""
        # Get some MCP tools
        mcp_config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            only_include_tools=["add", "fibonacci_calculator"]
        )
        mcp_tools = JustToolFactory.create_tools_from_mcp(mcp_config)
        
        # Add regular Python tools
        regular_tool = JustTool.from_callable(tool_test_module.regular_function)
        
        # Combine tools
        all_tools = list(mcp_tools.values()) + [regular_tool]
        
        agent = create_agent_with_tools(all_tools)
        tool_names = list(mcp_tools.keys()) + ["regular_function"]
        bus, tool_results = setup_tool_test_callback(tool_names)
        
        # Test using both types of tools
        response1 = agent.query("Use the add tool to add 5 and 8")
        response2 = agent.query("Use regular_function with x=10 and y=15")
        
        # Check MCP tool was called - expect structured JSON format
        assert len(tool_results["add"]) >= 1
        result_val = tool_results["add"][-1]
        assert (isinstance(result_val, str) and ('{"type":"text","text":"13"}' in result_val or result_val == "13")) or result_val == 13
        
        # Check regular tool was called  
        assert len(tool_results["regular_function"]) >= 1
        assert tool_results["regular_function"][-1] == 25
    
    def test_agent_with_no_mcp_tools_excluded_all(self, mcp_server_available, mcp_stdio_command):
        """Test agent behavior when all MCP tools are excluded"""
        # Create config that excludes all tools
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            exclude_tools=[
                "add", "fibonacci_calculator", "prime_factorization_summer", 
                "trigonometric_chaos_generator", "polynomial_root_detective"
            ]
        )
        mcp_tools = JustToolFactory.create_tools_from_mcp(config)
        
        # Add only regular tools
        regular_tool = JustTool.from_callable(tool_test_module.regular_function)
        
        agent = create_agent_with_tools([regular_tool])
        bus, tool_results = setup_tool_test_callback(["regular_function"])
        
        response = agent.query("Use regular_function to add 20 and 30")
        
        # Should work with only regular tools
        assert len(tool_results["regular_function"]) >= 1
        assert tool_results["regular_function"][-1] == 50
        
        # Verify no MCP tools were loaded
        assert len(mcp_tools) == 0


class TestMCPToolsComplexScenarios:
    """Advanced tests for complex MCP tool scenarios"""
    
    def test_trigonometric_chaos_tool(self, mcp_server_available, mcp_stdio_command):
        """Test the complex trigonometric tool to ensure execution vs hallucination"""
        tool = JustMCPTool.from_mcp_stdio(
            name="trigonometric_chaos_generator",
            command=mcp_stdio_command[0],
            args=mcp_stdio_command[1:]
        )
        
        agent = create_agent_with_tools([tool])
        bus, tool_results = setup_tool_test_callback(["trigonometric_chaos_generator"])
        
        response = agent.query("Use trigonometric_chaos_generator with angle_degrees=45 and iterations=5")
        
        assert len(tool_results["trigonometric_chaos_generator"]) >= 1
        result_data = tool_results["trigonometric_chaos_generator"][-1]
        result = json.loads(result_data) if isinstance(result_data, str) else result_data
        
        # Verify the complex structure that would be impossible to hallucinate
        assert result["original_angle_degrees"] == 45
        assert result["iterations_performed"] == 5
        assert "chaos_signature" in result
        assert "sin_sum" in result
        assert "cos_sum" in result
        assert "tan_sum" in result
        assert "final_result" in result
        
        # The chaos_signature should be a specific calculated value
        assert isinstance(result["chaos_signature"], int)
        assert 0 <= result["chaos_signature"] < 99991
    
    def test_polynomial_root_detective_tool(self, mcp_server_available, mcp_stdio_command):
        """Test the polynomial analysis tool with complex calculations"""
        tool = JustMCPTool.from_mcp_stdio(
            name="polynomial_root_detective",
            command=mcp_stdio_command[0],
            args=mcp_stdio_command[1:]
        )
        
        agent = create_agent_with_tools([tool])
        bus, tool_results = setup_tool_test_callback(["polynomial_root_detective"])
        
        # Test with quadratic x^2 - 5x + 6 = 0 (roots at 2 and 3)
        response = agent.query("Use polynomial_root_detective with coefficients=[1, -5, 6]")
        
        assert len(tool_results["polynomial_root_detective"]) >= 1
        result_data = tool_results["polynomial_root_detective"][-1]
        result = json.loads(result_data) if isinstance(result_data, str) else result_data
        
        assert result["coefficients"] == [1, -5, 6]
        assert result["degree"] == 2
        assert result["evaluation_at_1"] == 2  # 1 - 5 + 6 = 2
        assert result["evaluation_at_minus_1"] == 12  # 1 + 5 + 6 = 12
        assert "polynomial_signature" in result
        
        # Should find approximate roots near 2 and 3
        roots = result["approximate_roots"]
        if len(roots) >= 2:
            assert any(abs(root - 2.0) < 0.1 for root in roots)
            assert any(abs(root - 3.0) < 0.1 for root in roots)
    
    def test_multiple_mcp_tools_in_sequence(self, mcp_server_available, mcp_stdio_command):
        """Test using multiple MCP tools in sequence within a single query"""
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            only_include_tools=["add", "fibonacci_calculator", "prime_factorization_summer"]
        )
        mcp_tools = JustToolFactory.create_tools_from_mcp(config)
        
        agent = create_agent_with_tools(list(mcp_tools.values()))
        bus, tool_results = setup_tool_test_callback(list(mcp_tools.keys()))
        
        response = agent.query(
            "First, use add to calculate 7 + 5. "
            "Then use fibonacci_calculator to get the 6th Fibonacci number. "
            "Finally, use prime_factorization_summer to factor the number 15."
        )
        
        # Check that all three tools were called
        assert len(tool_results["add"]) >= 1
        assert len(tool_results["fibonacci_calculator"]) >= 1  
        assert len(tool_results["prime_factorization_summer"]) >= 1
        
        # Verify results
        assert tool_results["add"][-1] == "12" or tool_results["add"][-1] == 12
        
        # 6th Fibonacci is 8, with signature
        fib_result = tool_results["fibonacci_calculator"][-1]
        expected_fib = 8 * 10007 + ((8 * 31 + 6 * 17) % 10007)
        assert fib_result == expected_fib or fib_result == str(expected_fib)
        
        # Prime factorization of 15 should be [3, 5]
        prime_result_data = tool_results["prime_factorization_summer"][-1]
        prime_result = json.loads(prime_result_data) if isinstance(prime_result_data, str) else prime_result_data
        assert prime_result["factors"] == [3, 5]
        assert prime_result["sum_of_factors"] == 8


class TestMCPToolErrors:
    """Tests for error handling in MCP tools"""
    
    def test_mcp_tool_with_invalid_server(self, mcp_server_available):
        """Test error handling when MCP server is not available"""
        with pytest.raises(Exception):  # Should raise some kind of connection error
            tool = JustMCPTool.from_mcp_stdio(
                name="add",
                command="nonexistent_command",
                args=[]
            )
            tool(a=1, b=2)  # This should fail when trying to connect
    
    def test_mcp_tool_with_invalid_tool_name(self, mcp_server_available, mcp_stdio_command):
        """Test error handling for invalid tool names"""
        with pytest.raises(ImportError, match="Tool 'nonexistent_tool' not found in MCP"):
            tool = JustMCPTool.from_mcp_stdio(
                name="nonexistent_tool",
                command=mcp_stdio_command[0],
                args=mcp_stdio_command[1:]
            )
            # The error should occur during refresh/initialization 


class TestMCPServerReuse:
    """Tests specifically for STDIO server reuse functionality"""
    
    def test_stdio_server_reuse(self, mcp_server_available, mcp_stdio_command, mcp_client_tracker, caplog):
        """Test that STDIO servers are reused across multiple tool calls"""
        import logging
        
        # Capture all logs to count server startups
        with caplog.at_level(logging.INFO):
            # Get initial state
            initial_state = mcp_client_tracker.get_current_state()
            
            # Create multiple tools that should use the same server
            config = JustMCPToolSetConfig(
                mcp_stdio_command=mcp_stdio_command,
                only_include_tools=["add", "fibonacci_calculator"]
            )
            tools = JustToolFactory.create_tools_from_mcp(config)
            
            add_tool = tools["add"]
            fib_tool = tools["fibonacci_calculator"]
            
            # Ensure clients are instantiated
            add_tool._ensure_mcp_client_instantiated()
            fib_tool._ensure_mcp_client_instantiated()
            
            # Verify same client is being used
            assert add_tool._mcp_client is fib_tool._mcp_client, "Tools should share the same MCP client"
            
            # Clear logs before making calls
            caplog.clear()
            
            # Make multiple tool calls
            results = []
            for i in range(6):  # Make 6 calls to really test reuse
                if i % 2 == 0:
                    result = add_tool(a=i, b=i+1)
                else:
                    result = fib_tool(n=5)
                results.append(result)
            
            # Count server startups during our test
            server_startups = mcp_client_tracker.count_server_startups_in_logs(caplog)
            
            # Should only see 1 server startup (for the first connection) if reuse is working
            # Allow up to 2 in case of retry scenarios, but anything more indicates poor reuse
            assert server_startups <= 2, (
                f"Expected at most 2 server startups during 6 tool calls, "
                f"but found {server_startups} startups. "
                f"This suggests STDIO servers are not being reused properly."
            )
            
            # Verify all calls succeeded
            assert len(results) == 6, "All tool calls should have succeeded"
            for result in results:
                assert result is not None, "All results should be non-None"
    
    def test_client_locator_prevents_duplicate_clients(self, mcp_server_available, mcp_stdio_command, mcp_client_tracker):
        """Test that the client locator prevents duplicate clients for the same server"""
        # Get initial state
        initial_state = mcp_client_tracker.get_current_state()
        
        # Create multiple tools with same server parameters
        config = JustMCPToolSetConfig(
            mcp_stdio_command=mcp_stdio_command,
            only_include_tools=["add", "fibonacci_calculator", "prime_factorization_summer"]
        )
        tools = JustToolFactory.create_tools_from_mcp(config)
        
        # Ensure all clients are instantiated
        for tool in tools.values():
            tool._ensure_mcp_client_instantiated()
        
        # Get final state
        final_state = mcp_client_tracker.get_current_state()
        
        # Check that at most 1 new client was created for our server
        # (Could be 0 if a client for this server already existed from previous tests)
        new_clients_for_our_server = len(final_state['new_client_keys'])
        assert new_clients_for_our_server <= 1, (
            f"Expected at most 1 new client to be created for our server, "
            f"but {new_clients_for_our_server} new clients were created. "
            f"New client keys: {final_state['new_client_keys']}"
        )
        
        # More importantly: verify all tools use the same client instance
        clients = [tool._mcp_client for tool in tools.values()]
        unique_clients = set(id(client) for client in clients)
        
        assert len(unique_clients) == 1, (
            f"All tools should use the same client instance, but found "
            f"{len(unique_clients)} unique clients: {unique_clients}"
        )
        
        # Also verify that all tools have the same client key (same server parameters)
        client_keys = set(client.client_key for client in clients)
        assert len(client_keys) == 1, (
            f"All tools should use clients with the same client_key, but found "
            f"{len(client_keys)} unique keys: {client_keys}"
        )
        
        # Verify the client key matches our server command
        expected_client_key = f"stdio:{' '.join(mcp_stdio_command)}"
        actual_client_key = next(iter(client_keys))
        assert actual_client_key == expected_client_key, (
            f"Client key should match server command. Expected: {expected_client_key}, "
            f"Got: {actual_client_key}"
        ) 