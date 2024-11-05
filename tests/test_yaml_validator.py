from just_agents.interfaces.IAgent import build_agent
from just_agents.utils import SchemaValidationError

def test_yaml_validator():
    try:
        agent = build_agent("tests/wrong_yaml.yaml")
        # If we reach here, the validation didn't fail as expected
        assert False, "Expected SchemaValidationError was not raised"
    except SchemaValidationError as e:
        error_message = str(e)
        # Assert that both expected error strings are present in the error message
        assert "jus_streming_method" in error_message, "Expected 'jus_streming_method' error not found"
        assert "backup_optionss" in error_message, "Expected 'backup_optionss' error not found"