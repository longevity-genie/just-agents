#!/usr/bin/env python3
"""
Comprehensive Just-Tools Example

This example showcases all available tool types in the just-agents framework:
1. Stateless Tools: Pure functions without dependencies on external state
2. Static Methods: Class-based static methods with full type safety  
3. Stateful Tools: Instance methods supported as transient tools (automatically excluded from serialization)
4. MCP Tools: Model Context Protocol tools with support for both STDIO and SSE modes via FastMCP integration
5. Google Built-in Tools: Native support for Google's built-in search and code execution capabilities
6. Prompt Tools: Pre-configured tools with predefined inputs, ideal for listings and templated operations

The INTENDED USAGE is to pass callables directly to the agent's tools list:
    tools=[function_a, Class.static_method, instance.method]

The framework automatically handles different tool types without needing explicit factory calls.
"""

import json
import sys
import os
import tempfile
import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable, cast
import pprint

from dotenv import load_dotenv
#import keys from .env file
load_dotenv(override=True)

from just_agents.data_classes import JustMCPServerParameters, MCPServersConfig, GoogleBuiltInTools
from just_agents.base_agent import BaseAgentWithLogging
from just_agents import llm_options
import just_agents.examples.toy_mcp_servers.mcp_stdio_server as mcp_server
import just_agents.examples.toy_mcp_servers.bip_bop_server as bip_bop_server

from just_agents.examples.tools import (
    get_current_weather, 
    calculate_fibonacci, 
    text_analyzer,
    letter_count,
    MathUtilities, 
    DocumentProcessor, 
    TaskManager
)

# Import the MCP client locator for debugging
from just_agents.mcp_client import MCPClientLocator

# ========================================
# CUSTOM LOGGING FUNCTIONS
# ========================================

def create_file_logger(log_file_path: str) -> Callable:
    """
    Create a custom logging function that writes to a file instead of stdout.
    
    This demonstrates how to redirect agent logs from stdio to a local file,
    which helps clean up console output from verbose debug messages.
    
    Args:
        log_file_path: Path to the log file where messages will be written
        
    Returns:
        A logging function that matches the LogFunction protocol
    """
    def file_log_function(log_string: str, action: str, source: str, *args, **kwargs) -> None:
        """Custom logging function that writes to file instead of stdout.
        
        Implements LogFunction protocol: Callable[[str, str, str, Any], None]
        """
        shortname = kwargs.pop("agent_shortname", "")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Format the log message similar to the default log_print but with timestamp
        if kwargs:
            log_message = f"[{timestamp}] {action} from {source}: {log_string}, extra args: {str(kwargs)}"
        elif shortname:
            log_message = f"[{timestamp}] {action} from <{shortname}> {source}: {log_string}"
        else:
            log_message = f"[{timestamp}] {action} from {source}: {log_string}"
        
        # Write to file instead of stdout
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
                f.flush()  # Ensure immediate write
        except Exception as e:
            # Fallback to stdout if file writing fails
            print(f"[LOG ERROR] Failed to write to {log_file_path}: {e}")
            print(log_message)
    
    return file_log_function






# Get the server path dynamically from the imported module
_MCP_SERVER_PATH_STR = str(mcp_server.__file__)
_MCP_SERVER_PATH = Path(_MCP_SERVER_PATH_STR).absolute()
_MCP_SERVER_PATH_STR_BIP_BOP = str(bip_bop_server.__file__)
# Get the directory path where this example is located
basic_examples_dir = Path(__file__).parent.absolute()

# ========================================
# TOOL ORGANIZATION FUNCTIONS
# ========================================

def get_stateless_tools() -> List[Callable]:
    """Get stateless tools (pure functions) as callables."""
    return [
        get_current_weather,
        calculate_fibonacci, 
        text_analyzer,
        letter_count
    ]


def get_static_method_tools() -> List[Callable]:
    """Get static method tools as callables."""
    return [
        MathUtilities.calculate_circle_area,
        MathUtilities.convert_temperature,
        MathUtilities.GeometryTools.calculate_triangle_area,
        MathUtilities.GeometryTools.calculate_rectangle_perimeter
    ]


def get_stateful_tool_instances() -> tuple:
    """Create instances for stateful tools and return them."""
    # Create instances that will be used for stateful tools
    doc_processor = DocumentProcessor("ExampleProcessor")
    task_manager = TaskManager("ExampleTaskManager")
    
    return doc_processor, task_manager


def get_stateful_tools() -> List[Callable]:
    """Get stateful (transient) tools as bound methods."""
    doc_processor, task_manager = get_stateful_tool_instances()
    
    return [
        doc_processor.process_document,
        doc_processor.get_processing_history,
        doc_processor.reset_session,
        task_manager.add_task,
        task_manager.complete_task,
        task_manager.list_tasks
    ]


def get_google_builtin_tools() -> List[Dict[str, str]]:
    """Get Google built-in tools as configuration dictionaries."""
    return [
        {"name": GoogleBuiltInTools.search},
        {"name": GoogleBuiltInTools.code}
    ]


def create_mcp_python_config_example() -> MCPServersConfig:
    """
    Create a Python configuration example for MCP tools. 
    Returns:
        Complete MCP servers configuration dictionary.
    """
    
    full_config_json = {
        "mcpServers": {
            "math_wizardry": {
                "transport": "stdio",
                "command": "python3",
                "args": [str(_MCP_SERVER_PATH.name), "--verbose"],
                "env": {"DEBUG": "true"},
                "cwd": str(_MCP_SERVER_PATH.parent)
            }
        }
    }
    return cast(MCPServersConfig, full_config_json)

def create_mcp_json_config_example() -> str:
    """
    Create a JSON configuration file example for MCP tools.
    
    Returns:
        JSON string showing complete configuration format.
    """
    return json.dumps(create_mcp_python_config_example(), indent=2)


def get_prompt_tools() -> List[tuple]:
    """
    Get prompt tools as (callable, call_arguments) tuples.
    
    CRITICAL: Each tuple must provide ALL required arguments for the callable.
    Format: (function, {param1: value1, param2: value2, ...})
    
    The framework will call function(**arguments) so all required parameters must be included.
    """
    return [
        # Pre-configured weather checks for specific cities
        (get_current_weather, {"location": "Tokyo"}),
        (get_current_weather, {"location": "San Francisco"}),
        
        # Pre-configured text analysis - MUST include required 'text' parameter
        (text_analyzer, {"text": "This is a sample text for detailed analysis with multiple sentences and various words.", "analysis_type": "detailed"}),
        
        # Pre-configured temperature conversion - MUST include required 'temperature' parameter  
        (MathUtilities.convert_temperature, {"temperature": 25.0, "from_unit": "celsius", "to_unit": "fahrenheit"}),
        
        # Pre-configured Fibonacci calculation
        (calculate_fibonacci, {"n": 8}),
        
        # Pre-configured letter counting
        (letter_count, {"word": "programming", "letter": "m"})
    ]


def get_mcp_tools_examples() -> List[Any]:
    """
    Get MCP tools using the actual test MCP server.
    Shows ALL MCP configuration approaches:
    1. String configuration (simplest - just the server command/path)
    2. JustMCPServerParameters (more granular control)
    3. Single tool constructor (JustMCPServerParameters.single_tool)
    4. Complete named configuration (Python dict and JSON)
    
    Returns:
        List of MCP tool configurations that can be added directly to tools list.
    """


    mcp_tools = []
    
    # Method 1: Simple string configuration
    # When you pass a string, it becomes JustMCPServerParameters(mcp_client_config=string)
    # All tools from server are included
    #works just as well with server_configs, see below
    
    server_command = str(_MCP_SERVER_PATH_STR_BIP_BOP)
    #  📊 Available Tools:
    # • bip() - Returns "BIP"
    # • bop() - Returns "BOP"
    mcp_tools.append(server_command)  # All tools from server

 # 📊 Available Tools:
    # • add(a, b) - Basic addition of two numbers
    # • fibonacci_calculator(n) - Calculate the nth Fibonacci number with verification signatures
    # • prime_factorization_summer(n) - Decompose numbers into prime factors with detailed analysis
    # • trigonometric_chaos_generator(angle_degrees, iterations) - Complex trigonometric transformations
    # • polynomial_root_detective(coefficients) - Analyze polynomials and find approximate roots
    # • divide(a, b) - Safe division with remainder and verification
    # • modulo(a, b) - Modulo operation with detailed analysis
    # • div_mod_combo(a, b) - Combined division and modulo with extensive verification
    # • gcd_calculator(a, b) - Greatest common divisor using Euclidean algorithm
    # • lcm_calculator(a, b) - Least common multiple with step-by-step verification

    # Method 2: JustMCPServerParameters for granular control  
    # Include only specific tools
    mcp_config_subset = JustMCPServerParameters( 
        mcp_client_config=_MCP_SERVER_PATH_STR,
        only_include_tools=["add", "prime_factorization_summer"]
    )
    mcp_tools.append(mcp_config_subset)
    
    # Method 3: JustMCPServerParameters excluding specific tools
    # adds fibonacci_calculator, trigonometric_chaos_generator
     
    mcp_config_exclude = JustMCPServerParameters(
        mcp_client_config=_MCP_SERVER_PATH_STR,
        exclude_tools=["add", 
                       "prime_factorization_summer", 
                       "polynomial_root_detective",
                       "divide",
                       "modulo",
                       "div_mod_combo",
                       "gcd_calculator",
                       "lcm_calculator",
                       "NONEXISTENT_FUNCTION", 
                       ],
        raise_on_incorrect_names=False # NONEXISTENT_FUNCTION will not raise an error, just be ignored

    )
    mcp_tools.append(mcp_config_exclude)
    
    # Method 4: Single tool constructor (simple way to get just one tool)
    # adds polynomial_root_detective
    single_add_tool = JustMCPServerParameters.single_tool("polynomial_root_detective", _MCP_SERVER_PATH_STR)
    mcp_tools.append(single_add_tool)
    
    # Method 5: Complete named configuration (Python dict format)
    config_python = create_mcp_python_config_example()

    # adds divide, modulo
    mcp_config_full_python = JustMCPServerParameters(
        mcp_client_config=config_python,
        only_include_tools=["divide", "modulo"]  # Limit for demo
    )
    mcp_tools.append(mcp_config_full_python)

    # Method 6: Complete named configuration (json format)
    config_json = create_mcp_json_config_example() 
    mcp_config_full_json = JustMCPServerParameters(
        mcp_client_config=config_json,
        only_include_tools=["gcd_calculator", "lcm_calculator"]  # Limit for demo
    )

    mcp_tools.append(mcp_config_full_json)
    # should have everything except div_mod_combo

    # Note: Even though we have 6 MCP tool configurations here, the MCPClientLocator singleton
    # will only create 3 actual client instances because several configs point to the same servers:
    # 1. BipBop server (1 config)
    # 2. Math server standard (4 configs but same client_key) 
    # 3. Math server with DEBUG=true (1 config with different env = different client_key)
    # The locator deduplicates based on standardized client_key, not number of tool configs.

    return mcp_tools








# ========================================
# AGENT CREATION AND DEMONSTRATION
# ========================================

def create_comprehensive_agent(log_to_file: bool = False, log_file_path: str = None) -> BaseAgentWithLogging:
    """
    Create an agent with all tool types using the INTENDED USAGE pattern.
    
    This demonstrates the correct way to use just-agents tools:
    - Pass callables directly to the tools list
    - Framework automatically handles different tool types
    - No need for explicit factory method calls
    - MCP tools can be added as strings or JustMCPServerParameters
    
    Args:
        log_to_file: If True, redirect logs to file instead of stdout
        log_file_path: Path to log file (auto-generated if None)
    """
    
    # Gather all tools using the intended pattern: direct callables
    all_tools = []
    
    # 1. Add stateless tools (pure functions)
    all_tools.extend(get_stateless_tools())
    
    # 2. Add static method tools (Class.static_method)
    all_tools.extend(get_static_method_tools())
    
    # 3. Add stateful tools (instance.method) - these become transient automatically
    all_tools.extend(get_stateful_tools())
    
    # 4. Add Google built-in tools (as configuration dictionaries)
    all_tools.extend(get_google_builtin_tools())
    
    # 5. Add MCP tools - using actual working server!
    try:
        if os.path.exists(_MCP_SERVER_PATH_STR):
            # Add just a subset to avoid overwhelming the example
            
            all_tools.extend(get_mcp_tools_examples())
        else:
            print(f"⚠️  MCP server not found at {_MCP_SERVER_PATH_STR}, skipping MCP tools")
    except Exception as e:
        print(f"⚠️  Could not load MCP tools: {e}")
    
    # Get prompt tools (callable, arguments) tuples
    prompt_tools = get_prompt_tools()
    
    # Create the agent - this is the INTENDED USAGE!
    agent = BaseAgentWithLogging(
        llm_options=llm_options.GEMINI_2_5_FLASH,
        system_prompt="""You are a comprehensive tool assistant demonstrating just-agents tool types.
                Pay attention to tools names and input arguments, never omit required arguments.
                Use the appropriate tools based on user requests and output results in concise format.
                """,
        tools=all_tools,           # Pass callables directly - framework handles the rest!
        prompt_tools=prompt_tools  # Pass (callable, arguments) tuples
    )

    # DEMONSTRATE CUSTOM LOGGING: Redirect logs from stdio to file
    if log_to_file:
        if log_file_path is None:
            # Create a temporary log file if none specified
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"/tmp/just_agents_log_{timestamp}.log"
        
        print(f"🔧 CUSTOM LOGGING DEMO: Redirecting agent logs to file: {log_file_path}")
        print("   This cleans up stdio from verbose debug messages!")
        
        # Replace the default log_print function with our custom file logger
        # This uses the @log_function.setter to redirect all agent logging
        agent.log_function = create_file_logger(log_file_path)
        
        print(f"✅ Agent logging redirected to: {log_file_path}")
        print("   Console output will now be much cleaner!")

    return agent


def demonstrate_individual_tool_types():
    """Demonstrate each tool type individually using the INTENDED USAGE pattern."""
    print("=" * 70)
    print("JUST-AGENTS COMPREHENSIVE TOOL DEMONSTRATION")
    print("Using the INTENDED USAGE: Pass callables directly to tools list")
    print("=" * 70)
    
    # 1. Stateless Tools (Pure Functions)
    print("\n1. STATELESS TOOLS (Pure Functions)")
    print("-" * 50)
    print("INTENDED USAGE: tools=[function_a, function_b]")
    print()
    
    stateless_tools = get_stateless_tools()
    print("Available stateless tools:")
    for tool in stateless_tools:
        print(f"  - {tool.__name__}: {tool.__doc__.strip() if tool.__doc__ else 'No description'}")
    
    print("\nTesting tools directly:")
    result = get_current_weather("Tokyo")
    print(f"get_current_weather('Tokyo'): {result}")
    
    result = calculate_fibonacci(10)
    print(f"calculate_fibonacci(10): {result}")
    
    result = letter_count("hello", "l")
    print(f"letter_count('hello', 'l'): {result}")
    
    # 2. Static Method Tools
    print("\n2. STATIC METHOD TOOLS")
    print("-" * 50)
    print("INTENDED USAGE: tools=[Class.static_method, NestedClass.static_method]")
    print()
    
    static_tools = get_static_method_tools()
    print("Available static method tools:")
    for tool in static_tools:
        print(f"  - {tool.__qualname__}: {tool.__doc__.strip() if tool.__doc__ else 'No description'}")
    
    print("\nTesting tools directly:")
    result = MathUtilities.calculate_circle_area(5.0)
    print(f"MathUtilities.calculate_circle_area(5.0): {result}")
    
    result = MathUtilities.convert_temperature(25, "celsius", "fahrenheit")
    print(f"MathUtilities.convert_temperature(25, 'celsius', 'fahrenheit'): {result}")
    
    # 3. Stateful Tools (Instance Methods - become Transient automatically)
    print("\n3. STATEFUL TOOLS (Instance Methods)")
    print("-" * 50)
    print("INTENDED USAGE: tools=[instance.method1, instance.method2]")
    print("These automatically become transient tools (excluded from serialization)")
    print()
    
    # Create instances
    doc_processor, task_manager = get_stateful_tool_instances()
    stateful_tools = get_stateful_tools()
    
    print("Available stateful tools (bound methods):")
    for tool in stateful_tools:
        print(f"  - {tool.__self__.__class__.__name__}.{tool.__name__}")
    
    print("\nTesting stateful tools (they maintain state):")
    
    # Process multiple documents to show state maintenance
    result1 = doc_processor.process_document("This is the first document with several words.", "doc1")
    print(f"First document: {result1}")
    
    result2 = doc_processor.process_document("This is a second document that is longer and contains more words than the first.", "doc2")
    print(f"Second document: {result2}")
    
    history = doc_processor.get_processing_history()
    print(f"Processing history shows state: {history['documents_in_session']} documents processed")
    
    # Task management
    task = task_manager.add_task("Learn just-agents", "Study the framework", "high")
    print(f"Added task: {task}")
    
    # 4. Google Built-in Tools
    print("\n4. GOOGLE BUILT-IN TOOLS")
    print("-" * 50)
    print(f"INTENDED USAGE: tools=[{{'name': '{GoogleBuiltInTools.search}'}}, {{'name': '{GoogleBuiltInTools.code}'}}]")
    print()
    
    google_tools = get_google_builtin_tools()
    print("Available Google built-in tools:")
    for tool_config in google_tools:
        print(f"  - {tool_config['name']}: Handled by Google's interface")
    
    # 5. MCP Tools (Using Actual Working Server)
    print("\n5. MCP TOOLS (Model Context Protocol)")
    print("-" * 50)
    print("INTENDED USAGE: Multiple configuration methods available")
    print()
    
    print("\nTesting MCP tools if server is available:")
    try:
        if os.path.exists(_MCP_SERVER_PATH_STR):
            
            # Test Method 3: single_tool constructor
            print("  Method 3 - Single tool constructor:")
            add_tool = JustMCPServerParameters.single_tool("add", _MCP_SERVER_PATH_STR)
            result = add_tool(a=15, b=27)
            print(f"    single_tool('add') → add(15, 27) = {result}")
            
            fib_tool = JustMCPServerParameters.single_tool("fibonacci_calculator", _MCP_SERVER_PATH_STR)
            result = fib_tool(n=7)
            print(f"    single_tool('fibonacci_calculator') → fib(7) = {result}")
            
            # Test Method 4: Complete configuration
            print("  Method 4 - Complete configuration (with DEBUG=true):")
            full_config = {
                "mcpServers": {
                    "math_wizardry": {
                        "transport": "stdio",
                        "command": "python3", 
                        "args": [str(_MCP_SERVER_PATH_STR)],
                        "env": {"DEBUG": "true"},
                        "cwd": str(Path(_MCP_SERVER_PATH_STR).parent)
                    }
                }
            }
            full_mcp_config = JustMCPServerParameters(
                mcp_client_config=full_config,
                only_include_tools=["add"]
            )
            # Note: We don't call it here as it would create a new server instance
            print(f"    Full config created successfully - would start server with DEBUG enabled")
            
        else:
            print(f"  MCP server not found at {_MCP_SERVER_PATH_STR}")
            print("  Example configurations shown above")
    except Exception as e:
        print(f"  Could not test MCP tools: {e}")
        print("  This is normal if MCP dependencies are not installed")
    
    # 6. Prompt Tools (6 tools total)
    print("\n6. PROMPT TOOLS (Pre-configured)")
    print("-" * 50)
    print("INTENDED USAGE: prompt_tools=[(callable, call_arguments), ...]")
    print("NOTE: ALL required arguments must be provided in the configuration!")
    print()
    
    prompt_tools = get_prompt_tools()
    print("Available prompt tools:")
    for i, (tool, args) in enumerate(prompt_tools):
        print(f"  - {tool.__name__} with pre-configured args: {args}")
    
    print("\nTesting prompt tool behavior:")
    # Test the actual callable with pre-configured arguments
    tokyo_weather_func, tokyo_args = prompt_tools[0]  # Tokyo weather
    result = tokyo_weather_func(**tokyo_args)
    print(f"Pre-configured Tokyo weather: {result}")
    
    # Test text analysis with pre-configured detailed mode
    detailed_analysis_func, analysis_args = prompt_tools[2]  # Detailed text analysis
    result = detailed_analysis_func(**analysis_args)  # All arguments already provided
    print(f"Pre-configured detailed analysis: {result}")
    
    # Test temperature conversion with pre-configured values
    temp_conversion_func, temp_args = prompt_tools[3]  # Temperature conversion
    result = temp_conversion_func(**temp_args)
    print(f"Pre-configured temperature conversion (25°C to °F): {result}")
    
    # Test Fibonacci calculation
    fibonacci_func, fib_args = prompt_tools[4]  # Fibonacci
    result = fibonacci_func(**fib_args)
    print(f"Pre-configured Fibonacci (n=8): {result}")


def demonstrate_serialization():
    """Demonstrate YAML serialization of different tool types."""
    print("\n" + "=" * 60)
    print("YAML SERIALIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create agent
    agent = create_comprehensive_agent()
    
    # Save to YAML
 
    config_path = basic_examples_dir / "agent_profiles.yaml"
    agent.save_to_yaml("comprehensive_agent", file_path=config_path)
    
    
    print("Agent saved successfully!")
    print("\nNOTE: Transient tools (stateful tools) are automatically excluded from serialization")
    print("This is by design - stateful tools cannot be reliably recreated from configuration")
    
    # Load agent back
    try:
        loaded_agent = BaseAgentWithLogging.from_yaml("ComprehensiveToolsDemo", file_path=config_path)
        print(f"Agent loaded successfully with {len(loaded_agent.tools)} tools")
        print(f"Prompt tools loaded: {len(loaded_agent.prompt_tools) if loaded_agent.prompt_tools else 0}")
        
        # Show which tools were preserved
        if loaded_agent.tools:
            print("Preserved tools:")
            for tool_name in loaded_agent.tools.keys():
                print(f"  - {tool_name}")
        
        if loaded_agent.prompt_tools:
            print("Preserved prompt tools:")
            for tool_name in loaded_agent.prompt_tools.keys():
                print(f"  - {tool_name}")
                
    except Exception as e:
        print(f"Error loading agent: {e}")


def demonstrate_logging_redirection():
    """Demonstrate redirecting agent logs from stdio to a file."""
    print("\n" + "=" * 70)
    print("LOGGING REDIRECTION DEMONSTRATION")
    print("Showing how to redirect verbose debug messages from stdio to file")
    print("=" * 70)
    
    # Create a temporary log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"/tmp/just_agents_demo_{timestamp}.log"
    
    print(f"📝 Creating agent with file logging: {log_file_path}")
    
    # Create agent with file logging enabled
    agent = create_comprehensive_agent(log_to_file=True, log_file_path=log_file_path)
    
    print("🔇 Notice: Console is now clean of verbose agent logs!")
    print("🔍 All agent logs are being written to the file instead.")
    
    # Test a simple query to generate logs
    print("\n🧪 Testing agent query (logs go to file)...")
    try:
        response = agent.query("What's the weather like in Tokyo?")
        print(f"✅ Agent response: {response}")
    except Exception as e:
        print(f"⚠️ Query failed (likely missing API key): {e}")
    
    # Show log file contents
    print(f"\n📖 Log file contents from {log_file_path}:")
    print("-" * 50)
    try:
        with open(log_file_path, 'r') as f:
            log_contents = f.read()
            if log_contents:
                print(log_contents)
            else:
                print("(Log file is empty)")
    except FileNotFoundError:
        print("(Log file not found)")
    
    print("-" * 50)
    print(f"💡 Key insight: All verbose debug messages are in {log_file_path}")
    print("   instead of cluttering the console output!")
    
    return log_file_path


def demonstrate_agent_usage():
    """Demonstrate using the comprehensive agent with the INTENDED USAGE pattern."""
    print("\n" + "=" * 70)
    print("AGENT USAGE DEMONSTRATION")
    print("Showing the INTENDED USAGE: Pass callables directly to tools list")
    print("=" * 70)
    
    # Create agent with available tools using the intended pattern
    # Enable file logging to keep console clean during testing
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"/tmp/just_agents_usage_demo_{timestamp}.log"
    agent = create_comprehensive_agent(log_to_file=True, log_file_path=log_file_path)
    
    print(f"📝 Agent logs redirected to: {log_file_path}")
    print("🔇 Console output will be clean during agent testing!")
    
    print("✓ Created comprehensive agent using INTENDED USAGE pattern:")
    print("  tools=[function_a, Class.static_method, instance.method, config_dict]")
    print()
    print(f"Total tools available: {len(agent.tools)}")
    print(f"Prompt tools available: {len(agent.prompt_tools) if agent.prompt_tools else 0}")
    
    # Print detailed tools list using .list_tools() method
    print("\n🔧 Detailed tools list from agent.list_tools():")
    try:
        tools_list = agent.list_tools()
        if tools_list:
            for tool_name, tool_class in tools_list.items():
                tool_type = tool_class.__name__ if hasattr(tool_class, '__name__') else str(tool_class)
                print(f"  • {tool_name}: {tool_type}")
        else:
            print("  No tools found")
    except Exception as e:
        print(f"Error calling list_tools(): {e}")
        import traceback
        traceback.print_exc()
    
    # Show what types of tools were automatically detected
    if hasattr(agent, 'tools') and agent.tools:
        print("\nFramework automatically detected and created:")
        tool_type_counts = {}
        for tool_name, tool in agent.tools.items():
            tool_type = type(tool).__name__
            tool_type_counts[tool_type] = tool_type_counts.get(tool_type, 0) + 1
            
        for tool_type, count in tool_type_counts.items():
            print(f"  - {count} {tool_type} tools")
    
    print("\nThis demonstrates the power of the INTENDED USAGE pattern:")
    print("  ✓ No manual factory method calls needed")
    print("  ✓ Framework automatically detects tool types")
    print("  ✓ Stateful tools become transient automatically")
    print("  ✓ Clean, intuitive API")
    
    # Test queries (commented out as they require actual LLM calls)
    print("\nExample queries you could ask this agent:")
    print("1. 'What's the weather like in Tokyo?'")
    print("2. 'Calculate the area of a circle with radius 10'")
    print("3. 'Process this document: [some text]'")
    print("4. 'Add a task to organize my schedule'") 
    print("5. 'Convert 25 degrees Celsius to Fahrenheit'")
    print("6. 'Count the letter 'l' in the word 'hello''")
    print("7. 'Calculate the 8th Fibonacci number'")
    print("8. 'Use the MCP add tool to add 15 and 27'")
    print("9. 'Calculate the 6th Fibonacci number using MCP fibonacci_calculator'")
    print("10. 'Show me how to use single_tool constructor for specific MCP tools'")
    
    # Add actual agent calls following the pattern from function_calling_example.py
    print("\n" + "=" * 70)
    print("🤖 TESTING EACH TOOL TYPE WITH ACTUAL AGENT CALLS")
    print("=" * 70)
    
    # Note: Memory callback for pprint is disabled to keep console clean
    # All detailed logging goes to the log file instead via agent.log_function
    # agent.memory.add_on_message(lambda m: pprint.pprint(m))  # Commented out to avoid console spam
    responses = []
    
    def test_tool(test_num: int, emoji: str, description: str, query: str, condition: bool = True) -> Any:
        """Helper function to reduce test code duplication."""
        print(f"\n{test_num}. {emoji} Testing {description}:")
        print("-" * 50)
        if condition:
            response = agent.query(query)
            print("")
            pprint.pprint(response)
            return response
        else:
            print("MCP server not available, skipping MCP tool test")
            return None
    
    try:
        responses.append(test_tool(1, "🌤️", "STATELESS TOOL (get_current_weather)", 
                                  "What's the weather like in Tokyo?"))
        
        responses.append(test_tool(2, "📐", "STATIC METHOD TOOL (calculate_circle_area)", 
                 "Calculate the area of a circle with radius 5"))
        
        responses.append(test_tool(3, "📄", "STATEFUL TOOL (document processor)", 
                 "Process this document: 'The quick brown fox jumps over the lazy dog. This is a sample text for processing.' with document ID 'demo-doc'"))
        
        responses.append(test_tool(4, "🔢", "MCP TOOL (if available)", 
                 "Use the MCP add tool to calculate 15 + 27", 
                 condition=os.path.exists(_MCP_SERVER_PATH_STR)))
        
        responses.append(test_tool(5, "📊", "MULTIPLE TOOL TYPES in one query", 
                 "Convert 25 degrees Celsius to Fahrenheit, then calculate the 7th Fibonacci number, and tell me the weather in San Francisco"))
        
        responses.append(test_tool(6, "📝", "TASK MANAGER (stateful tool with state persistence)", 
                 "Add a task called 'Learn just-agents framework' with description 'Study all tool types' and high priority, then list all current tasks"))
        
        # pprint.pprint(responses)  # Commented out to keep console clean

        print("\n" + "=" * 70)
        print("✅ ALL TOOL TYPES TESTED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📖 Detailed log file is available in: {log_file_path}")

        agent.memory.pretty_print_all_messages()  

    except Exception as e:
        print(f"❌ Error during agent testing: {e}")
        print("This might be due to missing API keys or MCP server issues")
        import traceback
        traceback.print_exc()


def print_active_mcp_servers():
    """Print the list of active MCP servers from the locator.
    
    Expected behavior: Should show 3 active MCP clients:
    1. BipBop server - Simple server with 'bip' and 'bop' tools
    2. Math server - Standard math server instance  
    3. Math server with DEBUG=true - Separate instance due to different env config
    
    Note: Each unique configuration creates a separate client instance in the singleton locator.
    Different environment variables (like DEBUG=true) create distinct client keys.
    """
    print("\n" + "=" * 70)
    print("🔍 ACTIVE MCP SERVERS IN LOCATOR")
    print("=" * 70)
    
    try:
        # Get all active clients from the global locator
        all_clients = MCPClientLocator().get_all_clients()
        
        print(f"Total active MCP clients: {len(all_clients)}")
        print("(Expected: 3 - BipBop server, Math server, Math server with DEBUG=true)")
        print()
        
        if all_clients:
            for i, client in enumerate(all_clients, 1):
                print(f"{i}. Client Key: {client.client_key}")
                
                # Show a truncated config for readability
                config = client.get_standardized_client_config()
                if len(config) > 100:
                    config_display = config[:100] + "... (truncated)"
                else:
                    config_display = config
                print(f"   Config: {config_display}")
                
                # Check if client is connected
                try:
                    if hasattr(client, '_client') and client._client:
                        is_connected = client._client.is_connected() if hasattr(client._client, 'is_connected') else "Unknown"
                        print(f"   Connected: {is_connected}")
                    else:
                        print(f"   Connected: Not initialized")
                except Exception as e:
                    print(f"   Connected: Error checking - {e}")
                print()
        else:
            print("No active MCP clients found in locator.")
            
        # Document why we have 3 servers
        print("📋 Why 3 servers?")
        print("   • BipBop server: One instance for simple bip/bop tools")  
        print("   • Math server: Standard instance without DEBUG")
        print("   • Math server + DEBUG: Separate instance due to different env config")
        print("   • Different configs = different client_key = separate instances")
            
    except Exception as e:
        print(f"Error accessing MCP client locator: {e}")
        import traceback
        traceback.print_exc()


def get_active_mcp_servers_info() -> dict:
    """Get active MCP servers info as a dictionary for programmatic access.
    
    Returns info about the 3 expected MCP clients:
    1. BipBop server, 2. Math server, 3. Math server with DEBUG=true
    """
    try:
        all_clients = MCPClientLocator().get_all_clients()
        return {
            "total_count": len(all_clients),
            "clients": [
                {
                    "client_key": client.client_key,
                    "config": client.get_standardized_client_config(),
                    "connected": getattr(client._client, 'is_connected', lambda: False)() if hasattr(client, '_client') and client._client else False
                }
                for client in all_clients
            ]
        }
    except Exception as e:
        return {"error": str(e), "total_count": 0, "clients": []}


def main():
    """Main demonstration function showing the INTENDED USAGE pattern."""
    print("🚀 Starting Just-Agents Comprehensive Tool Demonstration")
    print("📋 Demonstrating INTENDED USAGE: Pass callables directly to tools list")
    print()
    
    try:
        # Demonstrate individual tool types
        demonstrate_individual_tool_types()
        
        # Demonstrate serialization
        demonstrate_serialization()
        
        # Demonstrate logging redirection
        demonstrate_logging_redirection()
        
        # Demonstrate agent usage
        demonstrate_agent_usage()
        
        # Print active MCP servers before finishing
        print_active_mcp_servers()
        
        print("\n" + "=" * 70)
        print("🎉 DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\n🎯 This example showcased the INTENDED USAGE pattern:")
        print("   tools=[function, Class.static_method, instance.method, config_dict]")
        print()
        print("✅ What was demonstrated:")
        print("   ✓ Stateless tools (4) - pure functions like weather, fibonacci")
        print("   ✓ Static method tools (4) - Class.method, nested classes")
        print("   ✓ Stateful tools (6) - instance methods that maintain state")
        print(f"   ✓ Google built-in tools (2) - {GoogleBuiltInTools.search}, {GoogleBuiltInTools.code}")
        print("   ✓ MCP tools (11) - distributed across 3 server instances")
        print("   ✓ Prompt tools (6) - pre-configured callable + arguments")
        print("   ✓ YAML serialization - transient tools excluded by design")
        print("   ✓ LOGGING REDIRECTION - Custom log_function to redirect stdio to file")
        print("   ✓ MCP server debugging - locator inspection functions")
        print("   ✓ Automatic tool type detection (29 total tools created)")
        print("   ✓ ACTUAL AGENT TESTING - Real LLM calls with pprint output")
        print("   ✓ Function calling demonstration - Each tool type tested live")
        print("   ✓ Memory callback integration - Shows internal conversation flow")
        print()
        print("🔧 Key Benefits of INTENDED USAGE:")
        print("   • No need for manual JustTool.from_callable() calls")
        print("   • Framework automatically detects and handles tool types")
        print("   • Stateful tools become transient automatically")
        print("   • MCP tools support both simple strings and granular config")
        print("   • Clean, intuitive API with less boilerplate code")
        print("   • Custom logging redirection via agent.log_function setter")
        print("   • Built-in debugging for MCP server management")
        print("   • Live testing with memory callbacks shows internal tool calls")
        print("   • Easy integration with pprint for detailed conversation tracing")
        print()
        print("📚 Quick Reference - INTENDED USAGE:")
        print("   agent = BaseAgentWithLogging(")
        print("       tools=[")
        print("           function_a,                    # Pure function")
        print("           Class.static_method,           # Static method")
        print("           instance.method,               # Instance method (becomes transient)")
        print(f"           {{'name': '{GoogleBuiltInTools.search}'}},      # Google built-in")
        print("           'path/to/mcp_server.py',       # MCP tools (string config)")
        print("           JustMCPServerParameters(...),  # MCP tools (granular config)")
        print("       ],")
        print("       prompt_tools=[")
        print("           (function_a, {'arg': 'value'}) # Pre-configured tool")
        print("       ]")
        print("   )")
        print()
        print("📝 Logging Redirection:")
        print("   # Redirect all agent logs from stdio to file")
        print("   agent.log_function = create_file_logger('/path/to/logfile.log')")
        print("   # Uses @log_function.setter to replace default log_print")
        print()
        print("🔧 MCP Tools Configuration (6 Methods Demonstrated):")
        print("   • Method 1 - String: 'server_command' → gets all tools")
        print("   • Method 2 - Include: JustMCPServerParameters(only_include_tools=[...])")
        print("   • Method 3 - Exclude: JustMCPServerParameters(exclude_tools=[...])")
        print("   • Method 4 - Single Tool: JustMCPServerParameters.single_tool('name', 'path')")
        print("   • Method 5 - Python Dict: full_config_dict with transport, env, cwd, args")
        print("   • Method 6 - JSON String: JSON representation of complete config")
        print("   • All methods support error handling and filter options")
        print()
        print("🔧 MCP Server Debugging (NEW):")
        print("   • print_active_mcp_servers() - displays 3 active server instances")
        print("   • get_active_mcp_servers_info() - programmatic access to server data")
        print("   • MCPClientLocator is a singleton - same instance via MCPClientLocator()")
        print("   • Expected: 3 clients (BipBop, Math std, Math w/DEBUG) due to config deduplication")
        print("   • 6 tool configurations → 3 actual client instances (smart caching)")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
