from dotenv import load_dotenv
from pathlib import Path

from just_agents.just_profile import JustAgentProfile

if __name__ == "__main__":
    load_dotenv(override=True)
    basic_examples_dir = Path(__file__).parent.absolute()
    config_path = basic_examples_dir / "yaml_initialization_example_new.yaml"
    legacy_config_path = basic_examples_dir / "agent_from_yaml.yaml"
    agent_from_legacy_schema = JustAgentProfile.load_legacy_schema(legacy_config_path)
    res = agent_from_legacy_schema.query("Count the number of occurrences of the letter ’L’ in the word - ’LOLLAPALOOZA’.")
    print(res)
    agent_from_legacy_schema.save_to_yaml("ConvertedLegacySchemaExample", file_path=config_path)

