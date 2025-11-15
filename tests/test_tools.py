import pytest
from dotenv import load_dotenv
import sys
import os
import json
from pathlib import Path

# Add the workspace root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))

from just_agents.just_tool import JustTool, JustGoogleBuiltIn, JustToolFactory, JustImportedTool, JustPromptTool, JustTransientTool
import tests.tools.tool_test_module as tool_test_module
from just_agents.base_agent import BaseAgentWithLogging
from just_agents.llm_options import LLMOptions, OPENAI_GPT5_NANO
from just_agents.just_tool import JustToolsBus
from just_agents.protocols.litellm_protocol import LiteLLMAdapter
from pydantic import ValidationError

@pytest.fixture(scope="module", autouse=True)
def load_env():
    load_dotenv(override=True)

@pytest.fixture(scope="module", autouse=True)
def ensure_assets_in_path():
    """Ensure assets directory is in sys.path for the test module."""
    # The path manipulation is done at the module level above.
    # This fixture is mainly to acknowledge and organize this setup.
    pass

def test_from_callable_regular_function():
    """Test JustTool.from_callable with a regular function."""
    tool = JustTool.from_callable(tool_test_module.regular_function)
    assert tool.package == "tool_test_module" or tool.package == "tests.tools.tool_test_module"
    assert tool.name == "regular_function"
    assert tool.static_class is None
    assert tool.description == "A regular function."
    assert tool.get_callable()(2, 3) == 5
    assert tool(x=5, y=5) == 10

def test_from_callable_static_method_top_level_class():
    """Test JustTool.from_callable with a static method in a top-level class."""
    tool = JustTool.from_callable(tool_test_module.TopLevelClass.static_method_top)
    assert tool.package == "tool_test_module" or tool.package == "tests.tools.tool_test_module"
    assert tool.name == "static_method_top"
    assert tool.static_class == "TopLevelClass"
    assert tool.description == "A static method in a top-level class."
    assert tool.get_callable()("test") == "test-default"
    assert tool(a="hello", b="world") == "hello-world"

def test_from_callable_static_method_nested_class():
    """Test JustTool.from_callable with a static method in a nested class."""
    tool = JustTool.from_callable(tool_test_module.TopLevelClass.NestedClass.static_method_nested)
    assert tool.package == "tool_test_module" or tool.package == "tests.tools.tool_test_module"
    assert tool.name == "static_method_nested"
    assert tool.static_class == "TopLevelClass.NestedClass"
    assert tool.description == "A static method in a nested class."
    assert tool.get_callable()(True) == "NestedTrue"
    assert tool(flag=False) == "NestedFalse"

def test_from_callable_static_method_deeper_nested_class():
    """Test JustTool.from_callable with a static method in a deeper nested class."""
    func = tool_test_module.TopLevelClass.NestedClass.DeeperNestedClass.static_method_deeper
    tool = JustTool.from_callable(func)
    assert tool.package == "tool_test_module" or tool.package == "tests.tools.tool_test_module"
    assert tool.name == "static_method_deeper"
    assert tool.static_class == "TopLevelClass.NestedClass.DeeperNestedClass"
    assert tool.description == "A static method in a deeper nested class."
    assert tool.get_callable()(2.5) == 5.0
    assert tool(val=3.0) == 6.0

def test_manual_creation_static_method():
    """Test manual creation of JustTool for a static method."""
    # Callable is resolved at init.
    tool = JustTool(
        package="tests.tools.tool_test_module",
        name="static_method_nested",
        static_class="TopLevelClass.NestedClass",
        description="Manual Description" # This remains as set since we no longer update description
    )
    assert tool(flag=True) == "NestedTrue"
    # Description remains as originally set
    assert tool.description == "Manual Description"
    assert tool(flag=False) == "NestedFalse"

def test_error_non_existent_module():
    """Test error handling for a non-existent module."""
    # The name 'some_function' and static_class 'None' will be part of the error message from initialization
    with pytest.raises(ImportError, match=r"Error initializing some_function \(class: None\) from non_existent_module: Could not import module 'non_existent_module'"):
        JustTool(
            package="non_existent_module",
            name="some_function"
        )

def test_error_non_existent_class_in_module():
    """Test error handling for a non-existent class in an existing module."""
    # name='some_method', static_class='NonExistentClass', package='tests.tools.tool_test_module'
    expected_match = (
        r"Error initializing some_method \(class: NonExistentClass\) from tests\.tools\.tool_test_module: "
        r"Could not resolve inner class segment 'NonExistentClass' in 'tests\.tools\.tool_test_module' "
        r"while trying to find 'NonExistentClass' in module 'tests\.tools\.tool_test_module'\. "
        r"Error: module 'tests\.tools\.tool_test_module' has no attribute 'NonExistentClass'"
    )
    with pytest.raises(ImportError, match=expected_match):
        JustTool(
            package="tests.tools.tool_test_module",
            name="some_method",
            static_class="NonExistentClass"
        )

def test_error_non_existent_nested_class():
    """Test error handling for a non-existent part of a nested class path."""
    # name='some_method', static_class='TopLevelClass.NonExistentInner.SomethingElse', package='tests.tools.tool_test_module'
    expected_match = (
        r"Error initializing some_method \(class: TopLevelClass\.NonExistentInner\.SomethingElse\) "
        r"from tests\.tools\.tool_test_module: Could not resolve inner class segment 'NonExistentInner' in 'tests\.tools\.tool_test_module\.TopLevelClass' "
        r"while trying to find 'TopLevelClass\.NonExistentInner\.SomethingElse' in module 'tests\.tools\.tool_test_module'\. "
        r"Error: type object 'TopLevelClass' has no attribute 'NonExistentInner'"
    )
    with pytest.raises(ImportError, match=expected_match):
        JustTool(
            package="tests.tools.tool_test_module",
            name="some_method",
            static_class="TopLevelClass.NonExistentInner.SomethingElse"
        )

def test_error_non_existent_method_in_static_class():
    """Test error handling for a non-existent method in an existing static class."""
    # name='non_existent_method', static_class='TopLevelClass.NestedClass', package='tests.tools.tool_test_module'
    expected_match = (
        r"Error initializing non_existent_method \(class: TopLevelClass\.NestedClass\) "
        r"from tests\.tools\.tool_test_module: Could not resolve method 'non_existent_method' on class 'TopLevelClass\.NestedClass' "
        r"in module 'tests\.tools\.tool_test_module'\. Error: type object 'NestedClass' has no attribute 'non_existent_method'"
    )
    with pytest.raises(ImportError, match=expected_match):
        JustTool(
            package="tests.tools.tool_test_module",
            name="non_existent_method",
            static_class="TopLevelClass.NestedClass"
        )

def test_error_regular_function_not_found_in_module():
    """Test error handling for a non-existent regular function in an existing module."""
    # name='non_existent_function', static_class=None, package='tests.tools.tool_test_module'
    with pytest.raises(ImportError, match=r"Error initializing non_existent_function \(class: None\) from tests\.tools\.tool_test_module: Could not resolve function 'non_existent_function' from module 'tests\.tools\.tool_test_module'"):
        JustTool(
            package="tests.tools.tool_test_module",
            name="non_existent_function"
        )

def test_tool_is_callable_itself():
    """Test that the JustTool instance itself is callable."""
    tool = JustTool.from_callable(tool_test_module.regular_function)
    assert callable(tool)
    assert tool(10, 20) == 30

def test_get_callable_wrap_false():
    """Test get_callable with wrap=False to retrieve the raw function."""
    original_func = tool_test_module.TopLevelClass.NestedClass.static_method_nested
    tool = JustTool.from_callable(original_func)
    raw_callable = tool.get_callable(wrap=False)
    assert raw_callable is original_func # Check for identity
    assert raw_callable(True) == "NestedTrue"

    # Ensure the wrapped version is still different and works
    wrapped_callable = tool.get_callable(wrap=True)
    assert wrapped_callable is not original_func
    assert wrapped_callable(False) == "NestedFalse"
    # Also check __call__
    assert tool(True) == "NestedTrue"


# To make sure the changes in from_callable (using **tool_params) are fine
# and that name is correctly picked if tool_params['name'] is None (though function_to_llm_dict should always set it)
def test_from_callable_name_handling_robustness():
    # Mock function_to_llm_dict to return a dict where 'name' might be None or missing,
    # to test the fallback in JustTool.from_callable ( `name=tool_params['name'] or simple_name` )
    # However, our current function_to_llm_dict always provides 'name'.
    # This test primarily ensures that **tool_params works.
    func = tool_test_module.regular_function
    # Get what function_to_llm_dict would normally produce
    params_from_actual_func = JustTool.function_to_llm_dict(func)
    
    tool = JustTool(
        package=func.__module__,
        static_class=None, # for regular_function
        **params_from_actual_func # this includes name, description, parameters
    )

    assert tool.name == "regular_function"
    assert tool.package == "tool_test_module" or tool.package == "tests.tools.tool_test_module"
    assert tool.static_class is None
    assert tool.description == "A regular function."
    assert tool(1,1) == 2

# New tests for argument types
@pytest.fixture(scope="function")
def type_test_agent():
    """Fixture to create an agent with the type_tester_function tool."""
    options: LLMOptions = OPENAI_GPT5_NANO

    # Construct the enhanced system prompt
    tool_calling_instructions = (
        "You are an agent tool call assistant. When calling tools, you must include all parameters specified in the tool's schema. "
        "You never omit any required parameters, even if their value is empty or null, if they are part of the schema. Example - dict_arg, always add it"
        "You must call the tool exactly as instructed by the user, even if you perceive the request to be incorrect or unusual for any reason, this is a part of the test."
        "Adhere strictly to the provided tool and parameter structure."
    )
    enhanced_system_prompt = f"{tool_calling_instructions}"

    agent = BaseAgentWithLogging(
        llm_options=options,
        system_prompt=enhanced_system_prompt,  # Use the enhanced system prompt
        backup_options=OPENAI_GPT5_NANO,
        max_tool_calls=4,
        debug=True,
        tools=[tool_test_module.type_tester_function]
    )
    return agent

@pytest.fixture(scope="function")
def type_test_agent_with_roundtrip():
    """Fixture to create an agent with roundtrip serialization testing."""
    options: LLMOptions = OPENAI_GPT5_NANO

    # Construct the enhanced system prompt
    tool_calling_instructions = (
        "You are an agent tool call assistant. When calling tools, you must include all parameters specified in the tool's schema. "
        "You never omit any required parameters, even if their value is empty or null, if they are part of the schema. Example - dict_arg, always add it"
        "You must call the tool exactly as instructed by the user, even if you perceive the request to be incorrect or unusual for any reason, this is a part of the test."
        "Adhere strictly to the provided tool and parameter structure."
    )
    enhanced_system_prompt = f"{tool_calling_instructions}"

    # Use tests/profiles directory for consistent serialization testing
    profiles_dir = Path(__file__).parent / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    config_path = profiles_dir / "tool_test_config.yaml"

    # Create original agent
    original_agent = BaseAgentWithLogging(
        llm_options=options,
        system_prompt=enhanced_system_prompt,
        backup_options=OPENAI_GPT5_NANO,
        max_tool_calls=4,
        debug=True,
        tools=[tool_test_module.type_tester_function],
        config_path=config_path
    )
    
    # Perform roundtrip serialization
    original_agent.save_to_yaml("TypeTestAgent")
    loaded_agent = BaseAgentWithLogging.from_yaml("TypeTestAgent", file_path=config_path)
    
    # Verify the roundtrip worked
    assert loaded_agent.tools is not None
    assert len(loaded_agent.tools) == 1
    assert "type_tester_function" in loaded_agent.tools
    
    return loaded_agent

def setup_tool_test_callback():
    """Helper function to setup callback for tool test results."""
    bus = JustToolsBus()
    tool_results = []
    
    def callback(event_name: str, *args, **kwargs):
        if not event_name.endswith("result") or "result_interceptor" not in kwargs:
            return
        if event_name.startswith('type_tester_function'):
            tool_results.append(kwargs.get("result_interceptor"))
    
    bus.subscribe("type_tester_function.*", callback)
    return bus, tool_results

def create_tool_test_prompt(s_arg='any', i_arg=1, f_arg=1.0, b_arg=True, 
                            l_arg=None, composite_list_arg=None, dict_arg=None, model_arg=None):
    """Helper function to create consistent tool test prompts."""
    if l_arg is None:
        l_arg = []
    if composite_list_arg is None:
        composite_list_arg = []
    if dict_arg is None:
        dict_arg = {}
    if model_arg is None:
        model_arg = {'name': 'test', 'age': 25, 'score': 85.5, 'is_active': True}
        
    return (
        f"Call type_tester_function with s_arg='{s_arg}', i_arg={i_arg}, f_arg={f_arg}, b_arg={b_arg}, "
        f"l_arg={json.dumps(l_arg)}, composite_list_arg={json.dumps(composite_list_arg)}, "
        f"dict_arg={json.dumps(dict_arg)}, model_arg={json.dumps(model_arg)}"
    )

def test_tool_with_string_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with a string argument (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    prompt = create_tool_test_prompt(s_arg='test string')
    type_test_agent_with_roundtrip.query(prompt)

    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    assert tool_output["s_arg_type"] == "str"
    assert tool_output["s_arg_value"] == "test string"

def test_tool_with_integer_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with an integer argument (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    prompt = create_tool_test_prompt(i_arg=123)
    type_test_agent_with_roundtrip.query(prompt)
    
    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    assert tool_output["i_arg_type"] == "int"
    assert tool_output["i_arg_value"] == 123

def test_tool_with_float_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with a float argument (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    prompt = create_tool_test_prompt(f_arg=3.14)
    type_test_agent_with_roundtrip.query(prompt)
    
    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    assert tool_output["f_arg_type"] == "float"
    assert tool_output["f_arg_value"] == 3.14

def test_tool_with_boolean_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with a boolean argument (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    prompt_true = create_tool_test_prompt(b_arg=True)
    type_test_agent_with_roundtrip.query(prompt_true)
    assert len(tool_results) >= 1 #allow retries
    tool_output_true = tool_results[-1]
    assert tool_output_true["b_arg_type"] == "bool"
    assert tool_output_true["b_arg_value"] is True
    
    tool_results.clear() # Clear for the next call within the same test
    prompt_false = create_tool_test_prompt(b_arg=False)
    type_test_agent_with_roundtrip.query(prompt_false)
    assert len(tool_results) >= 1 #allow retries
    tool_output_false = tool_results[-1]
    assert tool_output_false["b_arg_type"] == "bool"
    assert tool_output_false["b_arg_value"] is False

def test_tool_with_list_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with a list argument (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    
    list_val = ["item1", 2, True, 3.14]
    dict_arg_val = {"text": "value", "number": 42, "flag": True, "decimal": 3.14}
    
    prompt = create_tool_test_prompt(l_arg=list_val, dict_arg=dict_arg_val)
    type_test_agent_with_roundtrip.query(prompt)
    
    assert len(tool_results) >= 1
    tool_output = tool_results[-1]
    assert tool_output["l_arg_type"] == "list"
    # Expected: Pydantic preserves original types in the list
    expected_l_arg_value = ["item1", 2, True, 3.14]
    assert tool_output["l_arg_value"] == expected_l_arg_value
    assert tool_output["l_items_type"] == ["str", "int", "bool", "float"]

def test_tool_with_dict_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with a dictionary argument (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    
    dict_val = {"key_str": "value", "key_int": 100, "key_bool": False, "key_float": 0.5}
    model_val = {"name": "test", "age": 25, "score": 85.5, "is_active": True}
    
    prompt = create_tool_test_prompt(dict_arg=dict_val, model_arg=model_val)
    type_test_agent_with_roundtrip.query(prompt)
    
    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    
    # Check dictionary argument
    assert tool_output["dict_arg_type"] == "dict"
    assert tool_output["dict_arg_value"] == dict_val
    assert tool_output["d_items_type"] == {
        "key_str": "str",
        "key_int": "int",
        "key_bool": "bool",
        "key_float": "float"
    }
    
    # Check Pydantic model argument
    assert tool_output["model_arg_type"] == "SimpleModel"
    assert tool_output["model_arg_value"] == model_val
    assert tool_output["model_arg_fields"] == {
        "name": "str",
        "age": "int",
        "score": "float",
        "is_active": "bool"
    }

def test_tool_with_combined_args(type_test_agent_with_roundtrip):
    """Test agent calling a tool with combined argument types (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()

    s_val = "hello world"
    i_val = 42
    f_val = 3.14159
    b_val = True
    l_val = ["apple", 123, False, 2.71]
    d_val = {"text": "value", "number": 42, "flag": True, "decimal": 3.14}
    model_val = {"name": "test", "age": 25, "score": 85.5, "is_active": True}
    
    prompt = create_tool_test_prompt(
        s_arg=s_val, 
        i_arg=i_val, 
        f_arg=f_val, 
        b_arg=b_val, 
        l_arg=l_val, 
        dict_arg=d_val, 
        model_arg=model_val
    )

    llm_response = type_test_agent_with_roundtrip.query(prompt)
    print(f"LLM Response: {llm_response}") # For debugging

    assert len(tool_results) >= 1 #allow retries, f"Tool was not called or event not captured. LLM response: {llm_response}"
    tool_output = tool_results[-1]

    assert tool_output["s_arg_type"] == "str"
    assert tool_output["s_arg_value"] == s_val
    assert tool_output["i_arg_type"] == "int"
    assert tool_output["i_arg_value"] == i_val
    assert tool_output["f_arg_type"] == "float"
    assert tool_output["f_arg_value"] == f_val
    assert tool_output["b_arg_type"] == "bool"
    assert tool_output["b_arg_value"] == b_val
    
    assert tool_output["l_arg_type"] == "list"
    assert tool_output["l_arg_value"] == l_val
    assert tool_output["l_items_type"] == ['str', 'int', 'bool', 'float']
    
    assert tool_output["dict_arg_type"] == "dict"
    assert tool_output["dict_arg_value"] == d_val
    assert tool_output["d_items_type"] == {
        "text": "str",
        "number": "int",
        "flag": "bool",
        "decimal": "float"
    }
    
    assert tool_output["model_arg_type"] == "SimpleModel"
    assert tool_output["model_arg_value"] == model_val
    assert tool_output["model_arg_fields"] == {
        "name": "str",
        "age": "int",
        "score": "float",
        "is_active": "bool"
    }

def test_tool_with_composite_list_arg(type_test_agent_with_roundtrip):
    """Test agent calling a tool with a composite list argument (List[Union[str, int, bool, float]]) (with roundtrip serialization)."""
    bus, tool_results = setup_tool_test_callback()
    
    composite_val = ["text", 10, True, 2.5]
    d_val = {"text": "value", "number": 42, "flag": True, "decimal": 3.14}
    
    prompt = create_tool_test_prompt(
        composite_list_arg=composite_val, 
        dict_arg=d_val
    )

    llm_response = type_test_agent_with_roundtrip.query(prompt)
    print(f"LLM Response for composite_list_arg test: {llm_response}") # For debugging

    assert len(tool_results) >= 1 #allow retries, f"Tool was not called or event not captured. LLM response: {llm_response}"
    tool_output = tool_results[-1]

    assert tool_output["composite_list_arg_type"] == "list"
    assert tool_output["composite_list_arg_value"] == composite_val
    assert tool_output["composite_list_items_type"] == ["str", "int", "bool", "float"]

# Note: Tool cleanup is handled by pytest automatically via function-scoped fixture


# Google Built-in Tools Tests
def test_google_builtin_tool_creation():
    """Test creating JustGoogleBuiltIn tools for both valid names."""
    # Test googleSearch tool
    google_search = JustGoogleBuiltIn(name="googleSearch")
    assert google_search.name == "googleSearch"
    assert google_search.description == "Built-in tool to search the web"
    assert google_search.parameters == {}
    assert google_search.strict is None
    
    # Test codeExecution tool
    code_execution = JustGoogleBuiltIn(name="codeExecution")
    assert code_execution.name == "codeExecution"
    assert code_execution.description == "Built-in tool to execute code"
    assert code_execution.parameters == {}
    assert code_execution.strict is None

def test_google_builtin_tool_invalid_name():
    """Test that JustGoogleBuiltIn rejects invalid tool names."""
    with pytest.raises(ValueError, match=r"Input should be 'googleSearch' or 'codeExecution'"):
        JustGoogleBuiltIn(name="invalidTool")

def test_google_builtin_tool_litellm_format():
    """Test that JustGoogleBuiltIn tools produce the correct LiteLLM format."""
    google_search = JustGoogleBuiltIn(name="googleSearch")
    google_search_llm = google_search.get_litellm_description()
    
    expected = {
        'name': 'googleSearch',
        'description': 'Built-in tool to search the web',
        'parameters': {}
    }
    assert google_search_llm == expected
    
    code_execution = JustGoogleBuiltIn(name="codeExecution")
    code_execution_llm = code_execution.get_litellm_description()
    
    expected = {
        'name': 'codeExecution',
        'description': 'Built-in tool to execute code',
        'parameters': {}
    }
    assert code_execution_llm == expected

def test_google_builtin_tool_callable_raises_error():
    """Test that calling Google built-in tools raises appropriate errors."""
    google_search = JustGoogleBuiltIn(name="googleSearch")
    
    with pytest.raises(RuntimeError, match=r"Google built-in tool 'googleSearch' should not be called directly"):
        google_search()
    
    code_execution = JustGoogleBuiltIn(name="codeExecution")
    
    with pytest.raises(RuntimeError, match=r"Google built-in tool 'codeExecution' should not be called directly"):
        code_execution()

def test_google_builtin_tool_factory_creation():
    """Test creating JustGoogleBuiltIn tools via JustToolFactory."""
    # Test creation from dictionary
    google_dict = {"name": "googleSearch"}
    google_tool = JustToolFactory.create_tool(google_dict)
    
    assert isinstance(google_tool, JustGoogleBuiltIn)
    assert google_tool.name == "googleSearch"
    assert google_tool.description == "Built-in tool to search the web"
    assert google_tool.parameters == {}
    
    # Test creation from dictionary for codeExecution
    code_dict = {"name": "codeExecution"}
    code_tool = JustToolFactory.create_tool(code_dict)
    
    assert isinstance(code_tool, JustGoogleBuiltIn)
    assert code_tool.name == "codeExecution"
    assert code_tool.description == "Built-in tool to execute code"
    assert code_tool.parameters == {}

def test_google_builtin_tools_in_tools_dict():
    """Test that Google built-in tools work correctly when processed by create_tools_dict."""
    tools_dict_input = {
        "search": {"name": "googleSearch"},
        "code": {"name": "codeExecution"},
        "regular": tool_test_module.regular_function
    }
    
    tools_dict = JustToolFactory.create_tools_dict(tools_dict_input)
    
    # Verify Google built-in tools
    assert isinstance(tools_dict["googleSearch"], JustGoogleBuiltIn)
    assert tools_dict["googleSearch"].name == "googleSearch"
    
    assert isinstance(tools_dict["codeExecution"], JustGoogleBuiltIn)
    assert tools_dict["codeExecution"].name == "codeExecution"
    
    # Verify regular tool still works
    assert isinstance(tools_dict["regular_function"], JustTool)
    assert tools_dict["regular_function"].name == "regular_function"

def test_google_builtin_litellm_adapter_integration():
    """Test that LiteLLMAdapter correctly handles Google built-in tools by returning empty dicts."""
    adapter = LiteLLMAdapter()
    
    # Create Google built-in tools
    google_search = JustGoogleBuiltIn(name="googleSearch")
    code_execution = JustGoogleBuiltIn(name="codeExecution")
    
    # Test that the adapter returns empty dict for Google built-in tools
    google_search_callable = google_search.get_callable(wrap=False)
    google_search_tool = adapter.tool_from_function(google_search_callable)
    assert google_search_tool == {}
    
    code_execution_callable = code_execution.get_callable(wrap=False)
    code_execution_tool = adapter.tool_from_function(code_execution_callable)
    assert code_execution_tool == {}
    
    # Verify that regular functions still work normally
    regular_tool = adapter.tool_from_function(tool_test_module.regular_function)
    assert regular_tool != {}
    assert regular_tool['function']['name'] == 'regular_function'

def test_google_builtin_tool_raw_callable_properties():
    """Test that the raw callable for Google built-in tools has the correct properties."""
    google_search = JustGoogleBuiltIn(name="googleSearch")
    raw_callable = google_search.get_callable(wrap=False)
    
    # Verify the callable has the correct name
    assert raw_callable.__name__ == "googleSearch"
    assert callable(raw_callable)
    
    # Verify it has a docstring
    assert raw_callable.__doc__ is not None
    assert "Google built-in tool stub" in raw_callable.__doc__
    
    code_execution = JustGoogleBuiltIn(name="codeExecution")
    raw_callable = code_execution.get_callable(wrap=False)
    
    assert raw_callable.__name__ == "codeExecution"
    assert callable(raw_callable)
    assert raw_callable.__doc__ is not None
    assert "Google built-in tool stub" in raw_callable.__doc__

def test_tool_factory_conflict_detection_mcp_vs_import():
    """Test that JustToolFactory properly detects and rejects conflicting MCP vs import parameters."""
    # Test MCP endpoint (URL) vs package conflict
    conflict_dict = {
        "name": "conflicted_tool",
        "package": "some.module",
        "mcp_client_config": "http://test.example.com"
    }
    
    with pytest.raises(ValueError, match=r"Tool 'conflicted_tool' has conflicting parameters:.+"):
        JustToolFactory.create_tool(conflict_dict)
    
    # Test MCP endpoint (command) vs static_class conflict
    conflict_dict2 = {
        "name": "conflicted_tool2", 
        "static_class": "SomeClass",
        "mcp_client_config": "python server.py"
    }
    
    with pytest.raises(ValueError, match=r"Tool 'conflicted_tool2' has conflicting parameters:.+"):
        JustToolFactory.create_tool(conflict_dict2)


def test_tool_factory_conflict_detection_multiple_mcp_client_configs():
    """Test that JustToolFactory properly detects and rejects multiple MCP endpoint parameters."""
    # This test is no longer relevant since we have only one mcp_client_config parameter
    # Instead, test that a tool with only mcp_client_config works correctly
    valid_dict = {
        "name": "valid_mcp_tool",
        "mcp_client_config": "http://test.example.com"
    }
    
    # This should not raise an error during type detection (though it will fail during instantiation)
    with pytest.raises(ValueError, match=r"Failed to create JustMCPTool tool from dictionary:.+"):
        JustToolFactory.create_tool(valid_dict)


def test_tool_factory_missing_name():
    """Test that JustToolFactory requires a name parameter."""
    nameless_dict = {
        "package": "some.module"
    }
    
    with pytest.raises(ValueError, match=r"Failed to create JustImportedTool tool from dictionary.+"):
        JustToolFactory.create_tool(nameless_dict)


def test_tool_factory_insufficient_parameters():
    """Test that JustToolFactory requires sufficient parameters to determine tool type."""
    insufficient_dict = {
        "name": "mystery_tool"
        # No other identifying parameters
    }
    
    with pytest.raises(ValueError, match=r"Tool 'mystery_tool' has insufficient parameters to determine tool type"):
        JustToolFactory.create_tool(insufficient_dict)


def test_tool_deserialization_from_dict():
    """Test that JustToolFactory can handle serialized tool dictionaries (as from JSON/YAML deserialization)."""
    
    # Test Google built-in tool from dict
    google_dict = {"name": "googleSearch"}
    google_tool = JustToolFactory.create_tool(google_dict)
    assert isinstance(google_tool, JustGoogleBuiltIn)
    assert google_tool.name == "googleSearch"
    
    # Test imported tool from dict  
    import_dict = {
        "name": "regular_function", 
        "package": "tests.tools.tool_test_module",
        "description": "A test function"
    }
    import_tool = JustToolFactory.create_tool(import_dict)
    assert isinstance(import_tool, JustImportedTool)
    assert import_tool.name == "regular_function"
    assert import_tool.package == "tests.tools.tool_test_module"
    
    # Test prompt tool from dict
    prompt_dict = {
        "name": "regular_function",
        "package": "tests.tools.tool_test_module", 
        "call_arguments": {"x": 10, "y": 20}
    }
    prompt_tool = JustToolFactory.create_tool(prompt_dict)
    assert isinstance(prompt_tool, JustPromptTool)
    assert prompt_tool.name == "regular_function"
    assert prompt_tool.call_arguments == {"x": 10, "y": 20}


def test_tools_dict_creation_with_serialized_dicts():
    """Test that create_tools_dict can handle dictionaries containing serialized tool representations."""
    
    # Simulate what we'd get from JSON deserialization
    tools_input = {
        "google_search": {"name": "googleSearch"},
        "my_function": {
            "name": "regular_function",
            "package": "tests.tools.tool_test_module"
        }
    }
    
    tools_dict = JustToolFactory.create_tools_dict(tools_input)
    
    # Verify the tools were created correctly
    assert len(tools_dict) == 2
    assert "googleSearch" in tools_dict
    assert "regular_function" in tools_dict
    
    assert isinstance(tools_dict["googleSearch"], JustGoogleBuiltIn)
    assert isinstance(tools_dict["regular_function"], JustImportedTool)
    
    # Test with sequence of dicts
    tools_sequence = [
        {"name": "googleSearch"},
        {
            "name": "regular_function", 
            "package": "tests.tools.tool_test_module"
        }
    ]
    
    tools_dict_from_sequence = JustToolFactory.create_tools_dict(tools_sequence)
    
    assert len(tools_dict_from_sequence) == 2
    assert "googleSearch" in tools_dict_from_sequence
    assert "regular_function" in tools_dict_from_sequence


def test_prompt_tools_dict_creation_with_serialized_dicts():
    """Test that create_prompt_tools_dict can handle dictionaries containing serialized tool representations."""
    
    # Simulate what we'd get from JSON deserialization for prompt tools
    prompt_tools_input = {
        "my_prompt_tool": {
            "name": "regular_function",
            "package": "tests.tools.tool_test_module",
            "call_arguments": {"x": 5, "y": 7}
        }
    }
    
    prompt_tools_dict = JustToolFactory.create_prompt_tools_dict(prompt_tools_input)
    
    assert len(prompt_tools_dict) == 1
    assert "regular_function" in prompt_tools_dict
    
    tool = prompt_tools_dict["regular_function"]
    assert isinstance(tool, JustPromptTool)
    assert tool.call_arguments == {"x": 5, "y": 7}
    
    # Test with sequence of dicts
    prompt_tools_sequence = [
        {
            "name": "regular_function",
            "package": "tests.tools.tool_test_module", 
            "call_arguments": {"x": 15, "y": 25}
        }
    ]
    
    prompt_tools_dict_from_sequence = JustToolFactory.create_prompt_tools_dict(prompt_tools_sequence)
    
    assert len(prompt_tools_dict_from_sequence) == 1
    assert "regular_function" in prompt_tools_dict_from_sequence
    
    tool = prompt_tools_dict_from_sequence["regular_function"]
    assert tool.call_arguments == {"x": 15, "y": 25}


def test_integration_json_deserialization_simulation():
    """Integration test simulating JSON deserialization of a profile with tools."""
    import json
    
    # Simulate a serialized profile with tools (as it would come from JSON)
    profile_data = {
        "name": "TestProfile",
        "tools": [
            {"name": "googleSearch"},
            {
                "name": "regular_function",
                "package": "tests.tools.tool_test_module",
                "description": "A test function"
            }
        ],
        "prompt_tools": {
            "my_prompt_tool": {
                "name": "regular_function",
                "package": "tests.tools.tool_test_module",
                "call_arguments": {"x": 1, "y": 2}
            }
        }
    }
    
    # Simulate the JSON roundtrip (serialize then deserialize)
    json_str = json.dumps(profile_data)
    deserialized_data = json.loads(json_str)
    
    # Now process the tools as if they were deserialized from JSON
    tools_dict = JustToolFactory.create_tools_dict(deserialized_data["tools"])
    prompt_tools_dict = JustToolFactory.create_prompt_tools_dict(deserialized_data["prompt_tools"])
    
    # Verify the tools were processed correctly
    assert len(tools_dict) == 2
    assert "googleSearch" in tools_dict
    assert "regular_function" in tools_dict
    
    assert isinstance(tools_dict["googleSearch"], JustGoogleBuiltIn)
    assert isinstance(tools_dict["regular_function"], JustImportedTool)
    
    # Verify prompt tools
    assert len(prompt_tools_dict) == 1
    assert "regular_function" in prompt_tools_dict
    assert isinstance(prompt_tools_dict["regular_function"], JustPromptTool)
    assert prompt_tools_dict["regular_function"].call_arguments == {"x": 1, "y": 2}
    
    # Test that the tools actually work
    google_tool = tools_dict["googleSearch"]
    regular_tool = tools_dict["regular_function"]
    prompt_tool = prompt_tools_dict["regular_function"]
    
    # Google tool should raise an error when called
    with pytest.raises(RuntimeError, match="should not be called directly"):
        google_tool()
    
    # Regular tool should work
    result = regular_tool(x=10, y=20)
    assert result == 30
    
    # Prompt tool should work  
    prompt_result = prompt_tool(x=5, y=7)
    assert prompt_result == 12


def test_roundtrip_serialization_mixed_tool_types():
    """Test roundtrip serialization for agents with mixed tool types."""
    profiles_dir = Path(__file__).parent / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    config_path = profiles_dir / "mixed_tools_test.yaml"
    
    # Create agent with mixed tool types
    original_agent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT5_NANO,
        system_prompt="Test agent with mixed tools",
        tools=[
            {"name": "googleSearch"},  # Google built-in
            tool_test_module.regular_function,  # Regular callable  
            {  # Imported tool dict
                "name": "static_method_top",
                "package": "tests.tools.tool_test_module",
                "static_class": "TopLevelClass"
            }
        ],
        config_path=config_path
    )
    
    # Perform roundtrip serialization
    original_agent.save_to_yaml("MixedToolsAgent")
    loaded_agent = BaseAgentWithLogging.from_yaml("MixedToolsAgent", file_path=config_path)
    
    # Verify tools were preserved
    assert loaded_agent.tools is not None
    assert len(loaded_agent.tools) == 3
    assert "googleSearch" in loaded_agent.tools
    assert "regular_function" in loaded_agent.tools
    assert "static_method_top" in loaded_agent.tools
    
    # Verify tool types
    assert isinstance(loaded_agent.tools["googleSearch"], JustGoogleBuiltIn)
    assert isinstance(loaded_agent.tools["regular_function"], JustImportedTool)
    assert isinstance(loaded_agent.tools["static_method_top"], JustImportedTool)
    
    # Test that tools work correctly
    with pytest.raises(RuntimeError, match="should not be called directly"):
        loaded_agent.tools["googleSearch"]()
    
    # Test regular function
    result = loaded_agent.tools["regular_function"](x=10, y=20)
    assert result == 30
    
    # Test static method
    result = loaded_agent.tools["static_method_top"](a="test")
    assert result == "test-default"


def test_roundtrip_serialization_prompt_tools():
    """Test roundtrip serialization for agents with prompt tools."""
    profiles_dir = Path(__file__).parent / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    config_path = profiles_dir / "prompt_tools_test.yaml"
    
    # Create agent with prompt tools
    original_agent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT5_NANO,
        system_prompt="Test agent with prompt tools",
        prompt_tools=[
            {
                "name": "regular_function",
                "package": "tests.tools.tool_test_module",
                "call_arguments": {"x": 5, "y": 10}
            }
        ],
        config_path=config_path
    )
    
    # Perform roundtrip serialization
    original_agent.save_to_yaml("PromptToolsAgent")
    loaded_agent = BaseAgentWithLogging.from_yaml("PromptToolsAgent", file_path=config_path)
    
    # Verify prompt tools were preserved
    assert loaded_agent.prompt_tools is not None
    assert len(loaded_agent.prompt_tools) == 1
    assert "regular_function" in loaded_agent.prompt_tools
    
    # Verify prompt tool type and configuration
    prompt_tool = loaded_agent.prompt_tools["regular_function"]
    assert isinstance(prompt_tool, JustPromptTool)
    assert prompt_tool.call_arguments == {"x": 5, "y": 10}
    
    # Test that prompt tool works correctly
    result = prompt_tool(x=1, y=2)  # Should use actual args, not call_arguments
    assert result == 3


def test_roundtrip_serialization_comprehensive_agent():
    """Test roundtrip serialization for an agent with both tools and prompt_tools."""
    profiles_dir = Path(__file__).parent / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    config_path = profiles_dir / "comprehensive_agent_test.yaml"
    
    # Create comprehensive agent
    original_agent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT5_NANO,
        system_prompt="Comprehensive test agent",
        backup_options=OPENAI_GPT5_NANO,
        max_tool_calls=5,
        debug=True,
        tools=[
            {"name": "googleSearch"},
            tool_test_module.regular_function,
            {
                "name": "static_method_nested",
                "package": "tests.tools.tool_test_module",
                "static_class": "TopLevelClass.NestedClass"
            }
        ],
        prompt_tools=[
            {
                "name": "regular_function",
                "package": "tests.tools.tool_test_module",
                "call_arguments": {"x": 1, "y": 1}
            }
        ],
        config_path=config_path
    )
    
    # Perform roundtrip serialization
    original_agent.save_to_yaml("ComprehensiveAgent")
    loaded_agent = BaseAgentWithLogging.from_yaml("ComprehensiveAgent", file_path=config_path)
    
    # Verify all agent properties were preserved
    assert loaded_agent.llm_options == original_agent.llm_options
    assert loaded_agent.system_prompt == original_agent.system_prompt
    assert loaded_agent.backup_options == original_agent.backup_options
    assert loaded_agent.max_tool_calls == original_agent.max_tool_calls
    assert loaded_agent.shortname == original_agent.shortname
    
    # Verify tools
    assert loaded_agent.tools is not None
    assert len(loaded_agent.tools) == 3
    assert "googleSearch" in loaded_agent.tools
    assert "regular_function" in loaded_agent.tools
    assert "static_method_nested" in loaded_agent.tools
    
    # Verify prompt tools
    assert loaded_agent.prompt_tools is not None
    assert len(loaded_agent.prompt_tools) == 1
    assert "regular_function" in loaded_agent.prompt_tools
    
    # Test tools functionality
    with pytest.raises(RuntimeError, match="should not be called directly"):
        loaded_agent.tools["googleSearch"]()
    
    assert loaded_agent.tools["regular_function"](x=5, y=5) == 10
    assert loaded_agent.tools["static_method_nested"](flag=True) == "NestedTrue"
    assert loaded_agent.prompt_tools["regular_function"](x=3, y=4) == 7


# Transient Tools Tests

def test_bound_method_detection():
    """Test that bound methods are properly detected and routed to JustTransientTool."""
    # Create an instance with bound methods
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    counter = tool_test_module.SimpleCounterTool(initial_value=5)
    
    # Test bound instance method
    bound_method = pdf_reader.get_next_page
    tool = JustToolFactory.create_tool(bound_method)
    
    assert isinstance(tool, JustTransientTool)
    assert tool.is_transient is True
    assert tool.name == "get_next_page"
    
    # Test another bound method
    counter_method = counter.increment
    counter_tool = JustToolFactory.create_tool(counter_method)
    
    assert isinstance(counter_tool, JustTransientTool)
    assert counter_tool.is_transient is True
    assert counter_tool.name == "increment"


def test_static_method_vs_bound_method():
    """Test that static methods are not treated as transient while bound methods are."""
    # Static method should create regular JustTool
    static_tool = JustToolFactory.create_tool(tool_test_module.TopLevelClass.static_method_top)
    assert isinstance(static_tool, JustImportedTool)
    assert static_tool.is_transient is False
    
    # Bound method should create JustTransientTool
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    bound_tool = JustToolFactory.create_tool(pdf_reader.get_current_page)
    assert isinstance(bound_tool, JustTransientTool)
    assert bound_tool.is_transient is True


def test_transient_tool_from_bound_method():
    """Test creating transient tools directly from bound methods."""
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    
    # Test from_callable class method
    tool = JustTransientTool.from_callable(pdf_reader.get_next_page)
    
    assert tool.name == "get_next_page"
    assert tool.is_transient is True
    assert tool.description == "Move to next page and return its content."
    
    # Test with regular callable too
    regular_tool = JustTransientTool.from_callable(tool_test_module.regular_function)
    assert regular_tool.name == "regular_function"
    assert regular_tool.is_transient is True


def test_transient_tool_functionality():
    """Test that transient tools work correctly and maintain state."""
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    counter = tool_test_module.SimpleCounterTool(initial_value=10)
    
    # Create tools
    next_page_tool = JustToolFactory.create_tool(pdf_reader.get_next_page)
    increment_tool = JustToolFactory.create_tool(counter.increment)
    get_count_tool = JustToolFactory.create_tool(counter.get_count)
    
    # Test PDF reader functionality
    result1 = next_page_tool()
    assert "Page 2/3: Second page content" in result1
    
    result2 = next_page_tool()
    assert "Page 3/3: Third page content" in result2
    
    # Test counter functionality
    current_count = get_count_tool()
    assert current_count == 10
    
    new_count = increment_tool(amount=5)
    assert new_count == 15
    
    # Verify state was maintained
    current_count_after = get_count_tool()
    assert current_count_after == 15


def test_transient_tool_no_double_self():
    """Test that transient tools don't have the 'double self' problem."""
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    
    # Create tool from bound method
    jump_tool = JustToolFactory.create_tool(pdf_reader.jump_to_page)
    
    # This should work without 'multiple values for argument self' error
    result = jump_tool(page_number=2)
    assert "Page 2/3: Second page content" in result
    
    # Test with different page
    result2 = jump_tool(page_number=1)
    assert "Page 1/3: First page content" in result2


def test_transient_tool_serialization_exclusion():
    """Test that transient tools are excluded from serialization."""
    from just_agents.just_tool import JustTools
    
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    
    # Create a mix of regular and transient tools
    tools_dict = {
        "regular": tool_test_module.regular_function,
        "static": tool_test_module.TopLevelClass.static_method_top,
        "google": {"name": "googleSearch"},
        "transient_pdf": pdf_reader.get_next_page,
        "transient_current": pdf_reader.get_current_page
    }
    
    # Create JustTools instance
    tools = JustTools.from_tools(tools_dict)
    
    # Verify all tools are present in the dict
    assert len(tools) == 5
    assert "regular_function" in tools
    assert "static_method_top" in tools
    assert "googleSearch" in tools
    assert "get_next_page" in tools
    assert "get_current_page" in tools
    
    # Check tool types
    assert isinstance(tools["regular_function"], JustImportedTool)
    assert isinstance(tools["static_method_top"], JustImportedTool)
    assert isinstance(tools["googleSearch"], JustGoogleBuiltIn)
    assert isinstance(tools["get_next_page"], JustTransientTool)
    assert isinstance(tools["get_current_page"], JustTransientTool)
    
    # Serialize the tools
    serialized = tools.model_dump()
    
    # Transient tools should be excluded from serialization
    serialized_names = [tool.get('function', tool.get('name')) for tool in serialized]
    assert "regular_function" in serialized_names
    assert "static_method_top" in serialized_names  
    assert "googleSearch" in serialized_names
    assert "get_next_page" not in serialized_names  # Excluded
    assert "get_current_page" not in serialized_names  # Excluded
    
    # Should have 3 tools in serialized form (excluding 2 transient ones)
    assert len(serialized) == 3


def test_transient_tool_event_bus():
    """Test that transient tools work correctly with the event bus."""
    counter = tool_test_module.SimpleCounterTool(initial_value=0)
    increment_tool = JustToolFactory.create_tool(counter.increment)
    
    # Set up event bus callback
    bus = JustToolsBus()
    results = []
    
    def callback(event_name: str, **kwargs):
        if event_name.endswith("result") and "result_interceptor" in kwargs:
            results.append(kwargs["result_interceptor"])
    
    bus.subscribe(f"increment.{id(increment_tool)}.*", callback)
    
    # Call the tool
    result = increment_tool(amount=3)
    assert result == 3
    
    # Check that event was published
    assert len(results) == 1
    assert results[0] == 3


def test_transient_tool_error_handling():
    """Test error handling for transient tools."""
    # Test that any callable works with JustTransientTool
    regular_tool = JustTransientTool.from_callable(tool_test_module.regular_function)
    assert regular_tool.is_transient is True
    assert regular_tool.name == "regular_function"
    
    # Test with static method 
    static_tool = JustTransientTool.from_callable(tool_test_module.TopLevelClass.static_method_top)
    assert static_tool.is_transient is True
    assert static_tool.name == "static_method_top"
    
    # Test error when trying to create a tool without raw_callable
    with pytest.raises(ValidationError, match="Field required"):
        JustTransientTool(name="broken", is_transient=True)
    
    # Test that the tool works correctly after creation
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    working_tool = JustTransientTool.from_callable(pdf_reader.get_current_page)
    result = working_tool()
    assert "Page 1/3: First page content" in result


def test_agent_with_transient_tools():
    """Integration test with agents using transient tools."""
    profiles_dir = Path(__file__).parent / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    config_path = profiles_dir / "transient_tools_test.yaml"
    
    # Create stateful objects
    pdf_reader = tool_test_module.StatefulPDFReader.create_sample_pdf()
    counter = tool_test_module.SimpleCounterTool(initial_value=5)
    
    # Create agent with mix of regular and transient tools
    agent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT5_NANO,
        system_prompt="Test agent with transient tools",
        tools=[
            tool_test_module.regular_function,  # Regular tool
            pdf_reader.get_next_page,  # Transient tool
            counter.increment,  # Transient tool
            counter.get_count  # Transient tool
        ],
        config_path=config_path
    )
    
    # Verify tools are present
    assert agent.tools is not None
    assert len(agent.tools) == 4
    assert "regular_function" in agent.tools
    assert "get_next_page" in agent.tools
    assert "increment" in agent.tools
    assert "get_count" in agent.tools
    
    # Verify tool types
    assert isinstance(agent.tools["regular_function"], JustImportedTool)
    assert isinstance(agent.tools["get_next_page"], JustTransientTool)
    assert isinstance(agent.tools["increment"], JustTransientTool)
    assert isinstance(agent.tools["get_count"], JustTransientTool)
    
    # Test serialization - transient tools should be excluded
    agent.save_to_yaml("TransientToolsAgent")
    
    # Load the agent back
    loaded_agent = BaseAgentWithLogging.from_yaml("TransientToolsAgent", file_path=config_path)
    
    # Only non-transient tools should be loaded
    assert loaded_agent.tools is not None
    assert len(loaded_agent.tools) == 1  # Only regular_function
    assert "regular_function" in loaded_agent.tools
    assert "get_next_page" not in loaded_agent.tools
    assert "increment" not in loaded_agent.tools
    assert "get_count" not in loaded_agent.tools


def test_roundtrip_serialization_google_builtin_tools():
    """Test roundtrip serialization for agents with Google built-in tools."""
    profiles_dir = Path(__file__).parent / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    config_path = profiles_dir / "google_tools_test.yaml"
    
    # Create agent with Google built-in tools
    original_agent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT5_NANO,
        system_prompt="Test agent with Google tools",
        tools=[
            {"name": "googleSearch", "description": "Built-in tool to search the web"},
            {"name": "codeExecution", "description": "Built-in tool to execute code"}
        ],
        config_path=config_path
    )
    
    # Perform roundtrip serialization
    original_agent.save_to_yaml("GoogleToolsAgent")
    loaded_agent = BaseAgentWithLogging.from_yaml("GoogleToolsAgent", file_path=config_path)
    
    # Verify tools were preserved
    assert loaded_agent.tools is not None
    assert len(loaded_agent.tools) == 2
    assert "googleSearch" in loaded_agent.tools
    assert "codeExecution" in loaded_agent.tools
    
    # Verify tool types
    assert isinstance(loaded_agent.tools["googleSearch"], JustGoogleBuiltIn)
    assert isinstance(loaded_agent.tools["codeExecution"], JustGoogleBuiltIn)
    
    # Test that tools raise errors when called (as expected for Google built-ins)
    with pytest.raises(RuntimeError, match="should not be called directly"):
        loaded_agent.tools["googleSearch"]()
        
    with pytest.raises(RuntimeError, match="should not be called directly"):
        loaded_agent.tools["codeExecution"]()


def test_add_task_schema_bug_minimal_reproduction():
    """
    MINIMAL FAILING TEST: Just check the schema generation for add_task.
    This isolates the exact schema generation issue without LLM calls.
    """
    import sys
    from pathlib import Path
    
    # Add examples directory to path
    examples_path = Path(__file__).parent.parent / "examples" 
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    
    from just_agents.examples.tools import TaskManager
    
    # Create instance and tool
    task_manager = TaskManager("SchemaTestManager")
    add_task_tool = JustToolFactory.create_tool(task_manager.add_task)
    
    # Get the raw function to examine its signature 
    raw_func = task_manager.add_task
    print(f"\n🔍 Function signature: {raw_func}")
    print(f"🔍 Function name: {raw_func.__name__}")
    
    # Use JustTool's function_to_llm_dict to see what it generates
    llm_dict = JustTool.function_to_llm_dict(raw_func)
    
    print(f"\n📊 LLM dict generated by function_to_llm_dict:")
    import pprint
    pprint.pprint(llm_dict)
    
    # Check the parameters
    parameters = llm_dict.get('parameters', {})
    properties = parameters.get('properties', {})
    required = parameters.get('required', [])
    
    print(f"\n🐛 SCHEMA BUG ANALYSIS:")
    print(f"   Properties keys: {list(properties.keys())}")
    print(f"   Required array: {required}")
    print(f"   'title' in properties: {'title' in properties}")
    print(f"   'title' in required: {'title' in required}")
    
    # This will fail and show us exactly what's wrong
    if 'title' in required and 'title' not in properties:
        raise AssertionError(
            f"SCHEMA BUG CONFIRMED: 'title' is in required array {required} "
            f"but missing from properties {list(properties.keys())}. "
            f"This causes LiteLLM to fail with 'property is not defined' error."
        )

