import pytest
import sys
import os
import json

from just_agents.llm_options import OPENAI_GPT4_1NANO

# Add the workspace root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))

from just_agents.just_tool import JustTool
import tests.tools.tool_test_module as tool_test_module
from just_agents.base_agent import BaseAgentWithLogging
from just_agents.llm_options import LLMOptions, OPENAI_GPT4_1MINI
from just_agents.just_tool import JustToolsBus


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

def test_manual_creation_and_refresh_static_method():
    """Test manual creation of JustTool for a static method and then calling refresh."""
    # auto_refresh is True by default, so callable is resolved at init.
    # We can test refresh explicitly if needed by setting auto_refresh=False initially,
    # but the current setup tests that _resolve_callable works during init.
    tool = JustTool(
        package="tests.tools.tool_test_module",
        name="static_method_nested",
        static_class="TopLevelClass.NestedClass",
        description="Manual Description" # This will be overwritten by refresh if it fetches docstring
    )
    assert tool(flag=True) == "NestedTrue"
    # Test refresh explicitly (it should re-fetch metadata)
    tool.auto_refresh = True # Ensure refresh is enabled for the call
    refreshed_tool = tool.refresh()
    assert refreshed_tool.description == "A static method in a nested class."
    assert refreshed_tool(flag=False) == "NestedFalse"

def test_error_non_existent_module():
    """Test error handling for a non-existent module."""
    # The name 'some_function' and static_class 'None' will be part of the error message from refresh()
    with pytest.raises(ImportError, match=r"Error refreshing some_function \(class: None\) from non_existent_module: Could not import module 'non_existent_module'"):
        JustTool(
            package="non_existent_module",
            name="some_function",
            auto_refresh=True # Triggers get_callable -> _resolve_callable -> refresh
        )

def test_error_non_existent_class_in_module():
    """Test error handling for a non-existent class in an existing module."""
    # name='some_method', static_class='NonExistentClass', package='tests.tools.tool_test_module'
    expected_match = (
        r"Error refreshing some_method \(class: NonExistentClass\) from tests\.tools\.tool_test_module: "
        r"Could not resolve inner class segment 'NonExistentClass' in 'tests\.tools\.tool_test_module' "
        r"while trying to find 'NonExistentClass' in module 'tests\.tools\.tool_test_module'\. "
        r"Error: module 'tests\.tools\.tool_test_module' has no attribute 'NonExistentClass'"
    )
    with pytest.raises(ImportError, match=expected_match):
        JustTool(
            package="tests.tools.tool_test_module",
            name="some_method",
            static_class="NonExistentClass",
            auto_refresh=True
        )

def test_error_non_existent_nested_class():
    """Test error handling for a non-existent part of a nested class path."""
    # name='some_method', static_class='TopLevelClass.NonExistentInner.SomethingElse', package='tests.tools.tool_test_module'
    expected_match = (
        r"Error refreshing some_method \(class: TopLevelClass\.NonExistentInner\.SomethingElse\) "
        r"from tests\.tools\.tool_test_module: Could not resolve inner class segment 'NonExistentInner' in 'tests\.tools\.tool_test_module\.TopLevelClass' "
        r"while trying to find 'TopLevelClass\.NonExistentInner\.SomethingElse' in module 'tests\.tools\.tool_test_module'\. "
        r"Error: type object 'TopLevelClass' has no attribute 'NonExistentInner'"
    )
    with pytest.raises(ImportError, match=expected_match):
        JustTool(
            package="tests.tools.tool_test_module",
            name="some_method",
            static_class="TopLevelClass.NonExistentInner.SomethingElse",
            auto_refresh=True
        )

def test_error_non_existent_method_in_static_class():
    """Test error handling for a non-existent method in an existing static class."""
    # name='non_existent_method', static_class='TopLevelClass.NestedClass', package='tests.tools.tool_test_module'
    expected_match = (
        r"Error refreshing non_existent_method \(class: TopLevelClass\.NestedClass\) "
        r"from tests\.tools\.tool_test_module: Could not resolve method 'non_existent_method' on class 'TopLevelClass\.NestedClass' "
        r"in module 'tests\.tools\.tool_test_module'\. Error: type object 'NestedClass' has no attribute 'non_existent_method'"
    )
    with pytest.raises(ImportError, match=expected_match):
        JustTool(
            package="tests.tools.tool_test_module",
            name="non_existent_method",
            static_class="TopLevelClass.NestedClass",
            auto_refresh=True
        )

def test_error_regular_function_not_found_in_module():
    """Test error handling for a non-existent regular function in an existing module."""
    # name='non_existent_function', static_class=None, package='tests.tools.tool_test_module'
    with pytest.raises(ImportError, match=r"Error refreshing non_existent_function \(class: None\) from tests\.tools\.tool_test_module: Could not resolve function 'non_existent_function' from module 'tests\.tools\.tool_test_module'"):
        JustTool(
            package="tests.tools.tool_test_module",
            name="non_existent_function",
            auto_refresh=True
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
    options: LLMOptions = OPENAI_GPT4_1NANO

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
        backup_options=OPENAI_GPT4_1MINI,
        max_tool_calls=4,
        debug=True,
        tools=[tool_test_module.type_tester_function]
    )
    return agent

def setup_tool_test_callback():
    """Helper function to setup callback for tool test results."""
    bus = JustToolsBus()
    tool_results = []
    
    def callback(event_name: str, **kwargs):
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

def test_tool_with_string_arg(type_test_agent):
    """Test agent calling a tool with a string argument."""
    bus, tool_results = setup_tool_test_callback()
    prompt = create_tool_test_prompt(s_arg='test string')
    type_test_agent.query(prompt)

    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    assert tool_output["s_arg_type"] == "str"
    assert tool_output["s_arg_value"] == "test string"

def test_tool_with_integer_arg(type_test_agent):
    """Test agent calling a tool with an integer argument."""
    bus, tool_results = setup_tool_test_callback()
    prompt = create_tool_test_prompt(i_arg=123)
    type_test_agent.query(prompt)
    
    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    assert tool_output["i_arg_type"] == "int"
    assert tool_output["i_arg_value"] == 123

def test_tool_with_float_arg(type_test_agent):
    """Test agent calling a tool with a float argument."""
    bus, tool_results = setup_tool_test_callback()
    prompt = create_tool_test_prompt(f_arg=3.14)
    type_test_agent.query(prompt)
    
    assert len(tool_results) >= 1 #allow retries
    tool_output = tool_results[-1]
    assert tool_output["f_arg_type"] == "float"
    assert tool_output["f_arg_value"] == 3.14

def test_tool_with_boolean_arg(type_test_agent):
    """Test agent calling a tool with a boolean argument."""
    bus, tool_results = setup_tool_test_callback()
    prompt_true = create_tool_test_prompt(b_arg=True)
    type_test_agent.query(prompt_true)
    assert len(tool_results) >= 1 #allow retries
    tool_output_true = tool_results[-1]
    assert tool_output_true["b_arg_type"] == "bool"
    assert tool_output_true["b_arg_value"] is True
    
    tool_results.clear() # Clear for the next call within the same test
    prompt_false = create_tool_test_prompt(b_arg=False)
    type_test_agent.query(prompt_false)
    assert len(tool_results) >= 1 #allow retries
    tool_output_false = tool_results[-1]
    assert tool_output_false["b_arg_type"] == "bool"
    assert tool_output_false["b_arg_value"] is False

def test_tool_with_list_arg(type_test_agent):
    """Test agent calling a tool with a list argument."""
    bus, tool_results = setup_tool_test_callback()
    
    list_val = ["item1", 2, True, 3.14]
    dict_arg_val = {"text": "value", "number": 42, "flag": True, "decimal": 3.14}
    
    prompt = create_tool_test_prompt(l_arg=list_val, dict_arg=dict_arg_val)
    type_test_agent.query(prompt)
    
    assert len(tool_results) >= 1
    tool_output = tool_results[-1]
    assert tool_output["l_arg_type"] == "list"
    # Expected: Pydantic preserves original types in the list
    expected_l_arg_value = ["item1", 2, True, 3.14]
    assert tool_output["l_arg_value"] == expected_l_arg_value
    assert tool_output["l_items_type"] == ["str", "int", "bool", "float"]

def test_tool_with_dict_arg(type_test_agent):
    """Test agent calling a tool with a dictionary argument."""
    bus, tool_results = setup_tool_test_callback()
    
    dict_val = {"key_str": "value", "key_int": 100, "key_bool": False, "key_float": 0.5}
    model_val = {"name": "test", "age": 25, "score": 85.5, "is_active": True}
    
    prompt = create_tool_test_prompt(dict_arg=dict_val, model_arg=model_val)
    type_test_agent.query(prompt)
    
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

def test_tool_with_combined_args(type_test_agent):
    """Test agent calling a tool with combined argument types."""
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

    llm_response = type_test_agent.query(prompt)
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

def test_tool_with_composite_list_arg(type_test_agent):
    """Test agent calling a tool with a composite list argument (List[Union[str, int, bool, float]])."""
    bus, tool_results = setup_tool_test_callback()
    
    composite_val = ["text", 10, True, 2.5]
    d_val = {"text": "value", "number": 42, "flag": True, "decimal": 3.14}
    
    prompt = create_tool_test_prompt(
        composite_list_arg=composite_val, 
        dict_arg=d_val
    )

    llm_response = type_test_agent.query(prompt)
    print(f"LLM Response for composite_list_arg test: {llm_response}") # For debugging

    assert len(tool_results) >= 1 #allow retries, f"Tool was not called or event not captured. LLM response: {llm_response}"
    tool_output = tool_results[-1]

    assert tool_output["composite_list_arg_type"] == "list"
    assert tool_output["composite_list_arg_value"] == composite_val
    assert tool_output["composite_list_items_type"] == ["str", "int", "bool", "float"]

# Note: Tool cleanup is handled by pytest automatically via function-scoped fixture
