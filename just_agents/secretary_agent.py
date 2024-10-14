import json
from typing import Optional, Any, Dict, Sequence
from pydantic import BaseModel, Field, PrivateAttr
from just_agents.just_profile import JustAgentProfile
from just_agents.just_yaml import JustYaml
from just_agents.just_agent import JustAgent
from just_agents.llm_options import OPENAI_GPT4oMINI

DEFAULT_SECRETARY_PROMPT = """
You are a skilled AI assistant specializing in analysis and description of AI agents. 
You are tasked with generation of a detailed profile for an AI agent, based on the provided information, 
including 'system_prompt', 'tools', 'llm_options', 'existing_profile' or any other available attributes.
Your task is to fill in values of a JSON-formatted profile that matches the PROFILE_UPDATE_TEMPLATE provided below.
Values of the template describe what output is expected for each field.
Double-check that the output is a valid JSON and contains all the fields specified in PROFILE_UPDATE_TEMPLATE.
Never include any additional text or explanations.
"""

class SecretaryAgent(JustAgent):
    PROFILE_TEMPLATE : str = 'PROFILE_UPDATE_TEMPLATE'
    llm_options : Dict[str, Any] = Field(OPENAI_GPT4oMINI)
    system_prompt: str = Field(default_factory=lambda: DEFAULT_SECRETARY_PROMPT )
    autoload_from_yaml: bool = True

    exclude_list: Optional[Sequence[str]] = None
    refresh_list: Optional[Sequence[str]] = None
    extra_list: Dict[str,str] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the LLMSession for the SecretaryAgent.
        """
        if not self.llm_options:
            self.llm_options = OPENAI_GPT4oMINI

        super().model_post_init(__context)

    @staticmethod
    def get_profile_from_agent(agent: JustAgent, renew: bool) -> Optional[JustAgentProfile]:
        if not agent:
            return None
        if isinstance(agent.description, str):
            return JustAgentProfile(
                description=agent.description,
                system_prompt=agent.system_prompt,
            )
        elif isinstance(agent.description, JustAgentProfile):
            if renew:
                return JustAgentProfile(
                    description=agent.description.description,
                    system_prompt=agent.system_prompt,
                )
            else:
                return agent.description
        else:
            return None

    @staticmethod
    def get_agent_info(agent: JustAgent, renew: bool = False) -> Optional[str]:
        """
        Extracts and returns detailed information about a specific JustAgent instance in JSON format.

        Args:
            agent (JustAgent): The JustAgent instance to extract information from.
            renew (bool): Whether to include 'existing_profile' in the returned information.

        Returns:
            Optional[str]: A JSON string containing the extracted agent information, or None if the agent is not provided.
        """
        if not agent:
            return None
        system_prompt : str = ""
        if isinstance(agent.description, JustAgentProfile):
            data = agent.description.model_dump_with_extras()
            system_prompt = data.pop('system_prompt', "")  # deduplicate
            description = json.dumps(data)
        elif isinstance(agent.description, str):
            description = agent.description
        else:
            raise NotImplementedError(
                f"Unsupported description type: {type(agent.description)}"
            )
        if agent.system_prompt:
            system_prompt = agent.system_prompt

        # Extract information from the JustAgent instance
        agent_info = {
            'system_prompt': system_prompt,
            'tools': [tool.name for tool in agent.tools] if agent.tools else [],
            'llm_options': json.dumps(agent.llm_options, indent=2)
        }
        if not renew:
            agent_info['existing_profile'] = description
        if agent.agent_name:
            agent_info['agent_name'] = agent.agent_name
        return json.dumps(agent_info)

    def generate_profile(
            self,
            agent: JustAgent,
            renew: bool,
            autosave: bool,
            parent_section: str = None,
            exclude_list: Optional[Sequence[str]] = None,
            refresh_list : Optional[Sequence[str]] = None,
            extra_list: Dict[str,str] = None,
    ) -> Optional[JustAgentProfile]:

        """
        Generates a JustAgentProfile for the given JustAgent object.

        Args:
            agent (JustAgent): The agent for which to generate the profile.
            renew (bool): Whether to renew the profile.
            autosave (bool): Whether to autosave the profile.
            parent_section (str): YAML parent section name, defaults to 'agent_profiles'.
            exclude_list (Optional[Sequence[str]]): Fields to exclude from the process, defaults to SERVICE_FIELDS.
            refresh_list (Optional[Sequence[str]]): Fields to refresh if the 'refresh' flag is set, defaults to TO_REFRESH.
            extra_list (Dict[str,str]): Extra fields to populate, defaults to empty list. Must be supplied with descriptions

        Returns:
            Optional[JustAgentProfile]: The generated agent profile or None if the profile cannot be created.
        """

        agent_profile = self.get_profile_from_agent(agent, renew)
        if agent_profile and isinstance(agent_profile, JustAgentProfile):
            to_populate = agent_profile.to_populate(
                renew,
                self.exclude_list or exclude_list,
                self.refresh_list or refresh_list,
                self.extra_list or extra_list,
            )
        else:
            return None

        if not to_populate and not renew:  #This agent already has complete description, no need to autosave
            return agent_profile

        info = self.get_agent_info(agent, renew) # extract existing info

        prompt = (
                self.system_prompt + "\n\n"
                + info + "\n\n"
                + self.PROFILE_TEMPLATE + "\n"
                + json.dumps(to_populate) + "\n\n"
        )

        # Use self.llm_session.query() to get the LLM to generate the profile
        # print (str(prompt))
        response = self.query(prompt)

        # Parse the response as JSON to create a JustAgentProfile instance
        try:
            profile_data = json.loads(response)
            updated_profile = JustAgentProfile(**profile_data)
            agent_profile.update(updated_profile, renew)
            if autosave:
                agent_profile.save_to_yaml(
                    parent_section=parent_section
                )
        except json.JSONDecodeError as e:
            print("Failed to parse LLM response as JSON:", str(e))
            agent_profile = None
        except Exception as e:
            print("An error occurred while creating the profile:", str(e))
            agent_profile = None

        return agent_profile


def selftest():
    import just_agents
    import dotenv

    dotenv.load_dotenv(override=True)
    opt = just_agents.llm_options.OPENAI_GPT4oMINI.copy()

    secretary = SecretaryAgent(
        extra_list={
            "personality_traits": "Agents personality traits go here",
        }
    )
    agent_profile = secretary.generate_profile(
        secretary,
        True,
        True,
        'test'
    )

    print("Results: ", json.dumps(agent_profile.model_dump_with_extras(), indent=2))
    print("Results for LLM: ", json.dumps(agent_profile.model_dump_for_llm(), indent=2))

if __name__ == "__main__":
    selftest()
