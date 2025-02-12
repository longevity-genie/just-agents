
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Optional, Dict, Any, Callable, Literal, Type
from just_agents.base_agent import BaseAgent, BaseAgentWithLogging, VariArgs, LogFunction
from just_agents.just_serialization import JustSerializable
from pydantic import Field,BaseModel,PrivateAttr
import yaml
from eliot import start_action,start_task, Action, to_file, add_destinations, remove_destination
from just_agents.just_bus import SingletonMeta
from pycomfort.logging import to_nice_file, to_nice_stdout

LogDestinations = Literal["stdout","file","both","print_fallback"]
class EliotLogger(metaclass=SingletonMeta):
    """
    A simple singleton class to provide Eliot logging functionality to a WebAgent.
    """

    _logger_output: LogDestinations
    _logdir : Path
    log_path: Path
    
    def stdout_logger(self,message: str, **kwargs: Any) -> None:
        print(message)
        if kwargs:
            print(kwargs)

    def __init__(self, log_dir: Path, temp_dir: Path, logger_output: LogDestinations):
        if not log_dir or not temp_dir:
            raise ValueError("logdir is not set")
        self._logdir = log_dir
        self._logdir.mkdir(exist_ok=True)

        if not logger_output:
            logger_output: LogDestinations = "both"
        self._logger_output = logger_output
        # Generate unique log filename if not provided
       
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:4]
        uniq_name = f"{timestamp}_{unique_id}"
        self.log_path = self._logdir / f"{uniq_name}.log"
        json_path = self._logdir / f"{uniq_name}.json.log"
        if self._logger_output == "stdout" or self._logger_output == "both":
            to_nice_stdout(
                output_file=temp_dir / f"{uniq_name}.json"
            )
            # add_destinations(self.stdout_logger)
        if self._logger_output == "file" or self._logger_output == "both":
            to_nice_file(
                output_file=json_path,
                rendered_file=self.log_path
            )
            # to_file(open(self.log_path, "ab"))



class WebAgentEliotLoggerMixin(BaseModel):
    """
    A mixin class that provides Eliot logging functionality to a WebAgent.
    """

    logger_output: LogDestinations = Field(default="both",description="The output destination for Eliot logs")
    _logger: EliotLogger = PrivateAttr(default=None)
    _logdir: Path = PrivateAttr(default_factory=lambda: Path(os.getenv('LOG_DIR', 'logs')))
    _tempdir: Path = PrivateAttr(default_factory=lambda: Path(os.getenv('TMP_DIR', 'tmp')))
    _log_function: LogFunction = PrivateAttr(default=None)
    _task : Action = PrivateAttr(default_factory=lambda: start_task(action_type="WebAgent"))
    

    def log_action(self,log_string: str, action: str, source: str, *args: VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        """
        Log an action using Eliot's task logging mechanism.
        This method logs a message with additional context using Eliot's structured logging.

        Args:
            log_string (str): The primary message to log
            action (str): The type of action being logged
            source (str): The source of the log message
            *args (VariArgs.args): Variable positional arguments
            **kwargs (VariArgs.kwargs): Variable keyword arguments for additional logging details
        """

        # Transform kwargs into a string-to-string dictionary for logging
        str_kwargs = {str(k): str(v) for k, v in kwargs.items()}
        
        with self._task as log_action:
            log_action.log(
                message_type="WebAgent.log",
                message=log_string,
                action=action,
                source=source,
            #    extra_args=args,
                **str_kwargs
            )

    
    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._logdir.mkdir(exist_ok=True)
        self._tempdir.mkdir(exist_ok=True)

        self._logger = EliotLogger(self._logdir, self._tempdir, self.logger_output)

        if self.logger_output == "stdout":
            self._log_function=self.log_action
        elif self.logger_output == "file":
            self._log_function=self.log_action
        elif self.logger_output == "both":
            self._log_function=self.log_action
        elif self.logger_output == "print_fallback":
            self._log_function=self.default_logging_function
        else:
            raise ValueError(f"Invalid logger_output value: {self.logger_output}")

        self._log_function("Logging initialization complete", action="startup", source="EliotLogger",
                           log_destination = self.logger_output, path=self._logger.log_path)

class WebAgent(BaseAgentWithLogging,WebAgentEliotLoggerMixin):
    """
    A WebAgent is a REST API agent that can be used within an OpenAI compatible API endpoint.
    Any instance of BaseAgent can be used as a WebAgent
    """
    REQUIRED_CLASS: ClassVar[Type[BaseAgent]] = BaseAgent
    DEFAULT_DESCRIPTION: ClassVar[str] = "Generic all-purpose Web AI agent"
    description: str = Field(
        DEFAULT_DESCRIPTION,
        description="Short description of what the agent does")
    """Short description of what the agent does."""

    enforce_agent_prompt: bool = Field(
        default=False,
        description="Queries containing 'system' messages fall back to completion")

    continue_conversation: bool = Field(
        default=False,
        description="Concatenate memory messages and query messages ")

    remember_query: bool = Field(
        default=False,
        description="Add new query messages to memory")

    def __str__(self):
        name = self.description or self.shortname
        return f"{name}"

    @classmethod
    def from_yaml_dict(
        cls,
        yaml_path: Path | str,
        parent_section: Optional[str] = "agent_profiles",
        section: Optional[str] = None,
        required_base_class: Optional[Type[BaseAgent]] = None,
        fail_on_any_error: bool = True
    ) -> Dict[str, 'BaseAgent']:
        """
        Creates a dictionary of WebAgent (or subclass) instances from a YAML file.
        """
        with start_action(action_type="agent.load") as action:
            if isinstance(yaml_path, str):
                yaml_path = Path(yaml_path)

            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")

            if required_base_class is None:
                required_base_class = getattr(cls, 'REQUIRED_CLASS', None) or cls

            with yaml_path.open('r') as f:
                config_data = yaml.safe_load(f) or {}

            agents : Dict[str,BaseAgent] = {}

            # Get the correct section data
            if parent_section:
                sections = config_data.get(parent_section, {})
            else:
                if "agent_profiles" in config_data:
                    sections = config_data["agent_profiles"]
                    parent_section = "agent_profiles"
                elif "agents" in config_data:
                    sections = config_data["agents"]
                    parent_section = "agents"
                else:
                    sections = config_data

            # Process each section
            selected_sections = {}
            if section:
                if section in sections:
                    sections = {section: sections[section]}
                else:
                    action.log(
                        message_type="agent.config_error",
                        error=f"Section {section} not found in {parent_section} of {yaml_path}",
                        action="agent_config_error"
                    )
            else:
                selected_sections = sections

            for section_name, section_data in selected_sections.items():
                try:
                    auto_instance : JustSerializable = WebAgent.from_yaml_auto(
                        section_name,
                        parent_section,
                        yaml_path
                    )
                except Exception as e:
                    action.log(
                        message_type="agent.config_error",
                        error=f"str(e)"
                    )
                    if fail_on_any_error:
                        raise e
                    else:
                        continue
 
                if isinstance(auto_instance,BaseAgent) and isinstance(auto_instance, required_base_class):
                    agent = auto_instance
                    action.log(
                        message_type="agent.type_check",
                        instance=auto_instance.__class__.__qualname__,
                        required_class=required_base_class.__qualname__,
                        message=f"Agent is an instance of {auto_instance.__class__.__name__} matching or inheriting from {required_base_class.__name__}"
                    )
                else:
                    action.log(
                        message_type="agent.config_error",
                        instance=auto_instance,
                        error=f"Agent is not an instance or a descendant of {str(required_base_class.__name__)}, bound=BaseAgent! It will be discarded"
                    )
                    continue
                agents[section_name] = agent
                if agent.llm_options.get("tools", None):
                    action.log(
                        message_type="agent.config_error",
                        llm_options=agent.llm_options,
                        tools=agent.tools,
                        error="LLM options section contains tools information! It will be discarded"
                    )
                action.log(
                    message_type="agent.loaded",
                    section_name=section_name,
                    parent_section=parent_section,
                    name=agent.shortname
                )

            return agents
