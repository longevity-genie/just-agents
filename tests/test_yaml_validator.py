from just_agents.interfaces.IAgent import build_agent
from just_agents.utils import SchemaValidationError

def test_yaml_validator():
    try:
        agent = build_agent("wrong_yaml.yaml")
    except SchemaValidationError as e:
        assert "jus_streming_method" in str(e)
        assert "backup_optionss" in str(e)