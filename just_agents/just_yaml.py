import yaml
import os
from typing import Optional, List, Dict, Any, ClassVar
from pathlib import Path

from pydantic import BaseModel

class JustYaml(BaseModel):
    """
    A utility static class for reading and saving data to YAML files.

    Constants:
        DEFAULT_CONFIG_PATH (Path): Default path to the configuration YAML file.
        DEFAULT_SECTION_NAME (str): Default section name to use when none is provided.
        DEFAULT_AGENT_PROFILES_SECTION (str): Default parent section name.

    Methods:
        read_yaml_data(file_path: Path, section_name: str, parent_section: str = DEFAULT_AGENT_PROFILES_SECTION) -> Dict:
            Reads data from a given section within a YAML file.

        save_to_yaml(file_path: Path, section_data: Dict, section_name: str, parent_section: str = DEFAULT_AGENT_PROFILES_SECTION) -> None:
            Updates a section within a YAML file with new data.
    """
    MODULE_DIR : ClassVar[Path] = Path(os.path.abspath(os.path.dirname(__file__)))
    DEFAULT_CONFIG_PATH : ClassVar[Path] = Path('config/agent_profiles.yaml')
    DEFAULT_SECTION_NAME : ClassVar[str] = 'agent'
    DEFAULT_AGENT_PROFILES_SECTION : ClassVar[str] = 'agent_profiles'

    @staticmethod
    def read_yaml_data(
            file_path: Path,
            section_name: str,
            parent_section: str = DEFAULT_AGENT_PROFILES_SECTION
    ) -> Dict:
        """
        Reads data from a given section within a YAML file.

        Args:
            file_path (Path): The path to the YAML file.
            section_name (str): The name of the section to read.
            parent_section (str): The parent section name. Defaults to 'agent_profiles'.

        Returns:
            Dict: The data from the specified section.

        Raises:
            ValueError: If the section or parent section is not found in the YAML file.
        """
        if file_path.exists():
            with file_path.open('r') as f:
                data = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(
                f"File '{file_path}' not found."
            )
        try:
            # Retrieve the data for the specified section
            if parent_section:
                return data[parent_section][section_name]
            else:
                return data[section_name]
        except KeyError:
            raise ValueError(
                f"Section '{section_name}' under parent section '{parent_section}' not found in '{file_path}'"
            )

    @staticmethod
    def read_yaml_data_safe(
            file_path: Path,
            section_name: str,
            parent_section: str = DEFAULT_AGENT_PROFILES_SECTION
    ) -> Optional[Dict]:
        """
        Reads data from a given section within a YAML file.

        Args:
            file_path (Path): The path to the YAML file.
            section_name (str): The name of the section to read.
            parent_section (str): The parent section name. Defaults to 'agent_profiles'.

        Returns:
            Optional[Dict]: The data from the specified section, or None if not found or error occurred while reading.
        """
        if file_path.exists():
            with file_path.open('r') as f:
                data = yaml.safe_load(f) or {}
        else:
            return None

        try:
            # Retrieve the data for the specified section
            if parent_section:
                if parent_section in data:
                    if section_name and section_name in data[parent_section]:
                        return data[parent_section][section_name]
            else:
                if section_name and section_name in data:
                    return data[section_name]
            return None
        except KeyError:
            return None

    @staticmethod
    def save_to_yaml(
            file_path: Path,
            section_data: Dict,
            section_name: str,
            parent_section: str = DEFAULT_AGENT_PROFILES_SECTION
    ) -> None:
        """
        Updates a section within a YAML file with new data.

        Args:
            file_path (Path): The path to the YAML file.
            section_data (Dict): The data to be saved in the specified section.
            section_name (str): The name of the section to update.
            parent_section (str): The parent section name. Defaults to 'agent_profiles'.

        Returns:
            None
        """
        data = {}
        # Check if the YAML file exists and load existing data
        if file_path.exists():
            with file_path.open('r') as f:
                existing_data = yaml.safe_load(f) or {}
                data.update(existing_data)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the parent section exists
        if parent_section not in data:
            data[parent_section] = {}

        # Update the data with the current instance's data
        data[parent_section][section_name] = section_data

        # Write the updated data back to the YAML file
        with file_path.open('w') as f:
            yaml.safe_dump(data, f)
