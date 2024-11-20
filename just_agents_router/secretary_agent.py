import json
from typing import Optional, Any, Dict, Sequence, ClassVar, Tuple
from pydantic import Field
from just_agents.base_agent import BaseAgent
from just_agents.just_profile import JustAgentFullProfile

class SecretaryAgent(BaseAgent, JustAgentFullProfile):
    AVAILABLE_INFO: ClassVar[str] = 'AVAILABLE_ATTRIBUTES'
    PROFILE_TEMPLATE: ClassVar[str] = 'PROFILE_UPDATE_TEMPLATE'
    DEFAULT_SECRETARY_PROMPT : ClassVar[str] = """
    You are a skilled AI assistant specializing in analysis and description of AI agents. 
    You are tasked with generation of a minimalistic and concise yet detail-rich profile for an AI agent, based on the AVAILABLE_ATTRIBUTES, 
    including 'system_prompt', 'llm_options' and any other. Your task is to fill in values of a JSON-formatted profile 
    that matches the PROFILE_UPDATE_TEMPLATE provided below. Values of the template describe what output is expected for each field. 
    Only populate fields based on the well-established information, don't make up anything. 
    Double-check that the output contains only a valid JSON with all the fields specified in PROFILE_UPDATE_TEMPLATE. 
    Never include any additional text or explanations in your reply.
    """
    KEEP_IN_INFO : ClassVar[Tuple[str]] = (
        'shortname',
        'description',
        'system_prompt',
        'tools',
        'llm_options',
        'class_hierarchy',
        'class_qualname',
    )
    EXCLUDE_FROM_INFO : ClassVar[Tuple[str]] = (
        'include_list',
        'exclude_list',
        'refresh_list',
        'extra_dict',
        'filter_list'
    )
    EXCLUDE_FROM_RENEWAL: ClassVar[Tuple[str]] = (
        'system_prompt',
        'tools',
        'llm_options',
        'exclude_list',
        'refresh_list',
        'extra_dict',
    )
    REFRESH_LIST : ClassVar[Sequence[str]] = (
        'shortname',
        'description',
    )

    include_list: Optional[Sequence[str]] = Field(KEEP_IN_INFO, description="Fields to explicitly display to LLM")
    exclude_list: Optional[Sequence[str]] = Field(EXCLUDE_FROM_INFO, description="Fields to exclude from the attributes shown to LLM")
    filter_list: Optional[Sequence[str]] = Field(EXCLUDE_FROM_RENEWAL, description="Fields to exclude from the population process")
    refresh_list: Optional[Sequence[str]] = Field(None, description="Fields to refresh if the 'refresh' flag is set")
    extra_dict: Optional[Dict[str,str]] = Field(None, description="Extra arbitrary fields to populate")


    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the LLMSession for the SecretaryAgent.
        """
        if not self.system_prompt or self.system_prompt == self.DEFAULT_GENERIC_PROMPT:
            self.system_prompt = self.DEFAULT_SECRETARY_PROMPT
        super().model_post_init(__context)

    def get_info(
            self,
            agent: BaseAgent,
            include_list: Optional[Sequence[str]] = None,
            exclude_list: Optional[Sequence[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generates the 'info' dictionary for the given agent.

        Args:
            agent (JustAgent): The agent for which to generate the info.
            include_list (Optional[Sequence[str]]): Fields to include, defaults to self.include_list.
            exclude_list (Optional[Sequence[str]]): Fields to exclude, defaults to self.exclude_list.

        Returns:
            Optional[Dict[str, Any]]: The info dictionary or None if the agent is None.
        """
        if not agent:
            return None

        exclude_list = list(exclude_list or self.exclude_list)

        if agent.shortname == agent.DEFAULT_SECTION_NAME:
            exclude_list.append('shortname')
        if agent.description == agent.DEFAULT_DESCRIPTION:
            exclude_list.append('description')

        info = agent.to_json_inclusive(
            include_list=include_list or self.include_list,
            exclude_list=exclude_list
        )

        return info

    def get_to_populate(
            self,
            agent: BaseAgent,
            filter_list: Optional[Sequence[str]] = None,
            refresh_list: Optional[Sequence[str]] = None,
            extra_dict: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generates the 'to_populate' dictionary for the given agent.

        Args:
            agent (JustAgent): The agent for which to generate the to_populate.
            filter_list (Optional[Sequence[str]]): Fields to exclude, defaults to self.filter_list.
            refresh_list (Optional[Sequence[str]]): Fields to refresh if 'renew' is True, defaults to self.REFRESH_LIST.
            extra_dict (Optional[Dict[str, str]]): Extra fields to populate.

        Returns:
            Optional[Dict[str, Any]]: The to_populate dictionary or None if the agent is None.
        """
        if not agent:
            return None

        refresh_list = list(refresh_list or self.refresh_list or ())

        if agent.shortname == agent.DEFAULT_SECTION_NAME:
            refresh_list.append('shortname')
        if agent.description == agent.DEFAULT_DESCRIPTION:
            refresh_list.append('description')

        to_populate = agent.fields_to_populate(
            False,
            include_list=refresh_list or self.refresh_list,
            exclude_list=filter_list or self.filter_list,
            extra_list=extra_dict or self.extra_dict,
        )

        return to_populate

    def update_profile(
            self,
            agent: BaseAgent,
            info: Dict[str, Any],
            to_populate: Dict[str, Any],
            verbose: bool = False,
    ) -> bool:
        """
        Updates a JustAgentProfile using the LLM call with provided 'info' and 'to_populate' dictionaries.

        Args:
            agent (JustAgent): The agent for which to generate the profile.
            info (Dict[str, Any]): The info dictionary containing current details of the agent.
            to_populate (Dict[str, Any]): The dictionary specifying fields to populate or update.
            verbose (bool): If True, prints the prompt and response for debugging.

        Returns:
            bool: True if the profile was updated successfully, False otherwise.
        """

        if not agent:
            return False

        if not to_populate:  # This agent already has a complete description, no need to autosave
            if verbose:
                print("Profile already complete!")
            return False

        prompt = (
                self.AVAILABLE_INFO + "\n"
                + json.dumps(info, indent=2) + "\n\n"
                + self.PROFILE_TEMPLATE + "\n"
                + json.dumps(to_populate, indent=2) + "\n\n"
        )

        # Use self.llm_session.query() to get the LLM to generate the profile
        if verbose:
            print('Prompt:', str(prompt))
        response = self.query(prompt)
        if verbose:
            print('Response:', str(response))
        # Parse the response as JSON to create a JustAgent instance
        try:
            profile_data = json.loads(response)
            agent.update(profile_data,overwrite=True)
        except json.JSONDecodeError as e:
            print("Failed to parse LLM response as JSON:", str(e))
            return False
        except Exception as e:
            print("An error occurred while creating the profile:", str(e))
            return False

        return True
