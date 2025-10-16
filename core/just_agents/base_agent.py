import copy
from copy import deepcopy

from pydantic import Field, PrivateAttr, computed_field, BaseModel, field_serializer, field_validator, model_validator
from typing import Optional, List, Union, Any, Generator, Dict, ClassVar, Protocol, Type, Callable, AsyncGenerator, get_args
from functools import partial
from pydantic_core import PydanticSerializationUnexpectedValue
from just_agents.data_classes import FinishReason, ToolCall, Message, Role, ReasoningEffort, GoogleBuiltInTools
from just_agents.types import MessageDict, SupportedMessages

from just_agents.llm_options import LLMOptions
from just_agents.interfaces.function_call import IFunctionCall
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, BaseModelResponse
from just_agents.interfaces.agent import IAgentWithInterceptors, QueryListener, ResponseListener, VariArgs

from just_agents.base_memory import IBaseMemory, BaseMemory, OnToolCallable, OnMessageCallable
from just_agents.just_profile import JustAgentProfile, JustAgentProfileChatMixin, JustAgentProfileToolsetMixin
from just_agents.rotate_keys import RotateKeys
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.protocols.protocol_factory import StreamingMode, ProtocolAdapterFactory
from just_agents.just_tool import SubscriberCallback, GOOGLE_BUILTIN_SEARCH, GOOGLE_BUILTIN_CODE
from just_agents.just_bus import JustLogBus
from just_agents.just_locator import JustAgentsLocator
from just_agents.just_schema import ModelHelper


class BaseAgent(
    JustAgentProfile,
    IAgentWithInterceptors[
        SupportedMessages, #Input
        SupportedMessages, #Output
        SupportedMessages #StreamingOutput
    ]
):
    """
    A base agent that can query and stream LLM inputs and outputs.

    Note: it is based on pydantic and the only required field is llm_options.
    However, it is also recommended to set system_prompt.
    """

    # Core configuration for the LLM
    llm_options: LLMOptions = Field(
        ...,  # ... means this field is required
        validation_alias="options",
        description="options that will be passed to the LLM, see https://platform.openai.com/docs/api-reference/completions/create for more details")
    
    # Fallback options if primary LLM call fails
    backup_options: Optional[LLMOptions] = Field(
        default=None,
        description="options that will be used after we give up with main options, one more completion call will be done with backup options")

    parser: Optional[
        Union[Type[BaseModel],Dict[str,Any]]
    ] = Field(
        default=None,
        description="Pydantic model class to validate against, or, in serialized form - a dict describing such class fields explicitly, e.g: {'name': 'str', 'value': 'int', 'score': 'float'}")

    @field_serializer('parser', mode="plain", when_used="always")
    def serialize_parser(self, parser: Optional[Type[BaseModel]]) -> Optional[Dict[str, Any]]:
        """
        Serializes the parser field (Pydantic model class) to a dictionary representation.
        
        Args:
            parser: The Pydantic model class to serialize
            
        Returns:
            A dictionary with the model name and schema, or None if parser is None
        """
        if parser is None:
            return None

        # Use ModelHelper to convert the model class to a serialized schema
        return ModelHelper.serialize_model_schema(parser)

    @model_validator(mode='after')
    def validate_base_agent(self) -> 'BaseAgent': #don't use model_validate name or inheritance breaks #https://github.com/pydantic/pydantic/discussions/9202
        if isinstance(self.parser, dict):
            try:
                # Create a new model class from the schema
                self.parser = ModelHelper.create_model_from_flat_yaml(
                    self.codename + 'OutputParser',
                    self.parser,
                    optional_fields=False  # Make fields required for validation
                )
            except Exception as e:
                raise ValueError(f"Invalid YAML format for parser: {e}")
        return self

    # API key management settings
    completion_remove_key_on_error: bool = Field(
        default=True,
        description="In case of using list of keys removing key from the list after error call with this key")
    completion_max_tries: Optional[int]  = Field(
        2, ge=0,
        description="Maximum retry attempts before failing or falling back to backup_options")
    
    # Memory system to store conversation history
    memory: IBaseMemory = Field(
        default_factory=BaseMemory,
        exclude=True,
        description="Stores conversation history and maintains context between messages")

    streaming_method: StreamingMode = Field(
        StreamingMode.openai,
        description="protocol to handle llm format for function calling")

    key_list_path: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Path to text file with list of api keys, one key per line")
    
    key_list_env: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Environment variable name containing comma-separated API keys")

    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="The effort level for reasoning. One of: 'low', 'medium', 'high'")

    drop_unsupported_params: bool = Field(
        default=True,
        description="Drop parameters unsupported by the LLM provider/model from the query")

    max_tool_calls: int = Field(
        ge=1,
        default=50,
        description="Controls the number of LLM interaction cycles for tool usage.\n- Without `backup_options`: The agent attempts tool resolution up to `max_tool_calls` times using the primary LLM. The final attempt is tool-less if prior attempts continued to request tools.\n- With `backup_options`:\n    - If `max_tool_calls` is 1: Two attempts are made, both using the backup LLM (first with tools, second tool-less if needed).\n    - If `max_tool_calls` > 1: The agent makes `max_tool_calls - 1` attempts with the primary LLM. If tools are still needed, two further attempts are made with the backup LLM (first with tools, second tool-less if needed).\nThis results in `max_tool_calls` iterations if no backup, or `max_tool_calls + 1` iterations if backup is active (for `max_tool_calls >= 1`). Prevents tool call loops.")

    raise_on_completion_status_errors: bool = Field(
        default=True,
        description="Raise an exception on completion status 4xx and 5xx errors")

    send_system_prompt: bool = Field(
        default=True,
        description="When set, system prompt is used in query. ")
    enforce_agent_prompt: bool = Field(
        default=True,
        description="When set, replaces 'system' messages within queries containing them with agent's prompt setup")
    """
    Behavior of send_system_prompt and enforce_agent_prompt flags combinations:
    - send_system_prompt=True, enforce_agent_prompt=True:
      "Agent mode" -- This is the default behavior
        - Any 'system' messages in the query are replaced with the agent's prompt
        - Agent's prompt is updated and embedded in the query as 'system' message
    - send_system_prompt=True, enforce_agent_prompt=False:
      "Agent with override mode" -- Use when you need to override the agents behavior by custom system messages
        If any 'system' messages are present in the query:
        - they are preserved and sent to the LLM 'as is' 
        - Agent's prompt or prompt_tools are not used. 
        If the user query doesn't contain 'system' messages:
        - Behavior is identical to the 'Agent mode' (True,True)
    - send_system_prompt=False, enforce_agent_prompt=True:
      "Forced user mode" -- Use when you need to force the user query to be sent to the LLM without any additional 'system' messages
        - All 'system' messages are force-removed from the query, including the ones contained in the user query
        - Query is sent to LLM as is, without any additional 'system' messages
    - send_system_prompt=False, enforce_agent_prompt=False:
      "Passive mode" -- Use when you only need to pass the user query, tools, and LLM options to the LLM
        - If any 'system' messages are present in the query, they are preserved and sent to the LLM 'as is'.
        - Agent's prompt or prompt_tools are not used
        - If the user query doesn't contain 'system' messages, no system messages are sent to the LLM
    """

    continue_conversation: bool = Field(
        default=True,
        description="Concatenate memory messages and query messages ")
    remember_query: bool = Field(
        default=True,
        description="Add new query messages to memory")
    """
    Behavior of continue_conversation and remember_query flags combinations:
    - continue_conversation=True, remember_query=True:
     "Full Memory Mode" -- This is the default behavior
        - All previous memory messages are included in the context for this query
        - New query and response messages are added to memory for future interactions
        - Creates continuous, persistent conversation with full history

    - continue_conversation=True, remember_query=False:
      "Reference-Only Mode" -- Use when you need history for context but don't want to save the current interaction
        - All previous memory messages are included in the context for this query
        - New query and response messages are NOT saved to memory
        - Allows one-off questions that reference history without changing it

    - continue_conversation=False, remember_query=True:
      "Fresh Start Mode" -- Use when you want to start a new conversation thread but save it for later
        - Previous memory messages are NOT included in the context (starts from scratch)
        - New query and response messages ARE saved to memory for future interactions
        - Useful for beginning new topics that should be remembered

    - continue_conversation=False, remember_query=False:
      "Stateless Mode" -- Use for completely independent, one-shot interactions
        - Previous memory messages are NOT included in the context (starts from scratch)
        - New query and response messages are NOT saved to memory
        - Creates completely isolated interactions with no persistence
    """

    debug: bool = Field(
        default=False,
        exclude=True,
        description="Whether to enable debug mode on instantiation")

    observability: bool = Field(
        default=True,
        description="Whether to enable observability on instantiation")

    use_proxy: Optional[bool] = Field(
        default=None,
        description="Whether to use a proxy to connect to the LLM provider")

    proxy_address: Optional[str] = Field(
        default=None,
        description="The address of the proxy to use")

    # Protected handlers implementation
    _on_query : List[QueryListener] = PrivateAttr(default_factory=list)
    _on_response : List[ResponseListener] = PrivateAttr(default_factory=list)

    # Private attributes for internal state management
    _protocol: Optional[IProtocolAdapter] = PrivateAttr(None)  # Handles LLM-specific message formatting
    _partial_streaming_chunks: List[BaseModelResponse] = PrivateAttr(
        default_factory=list)  # Buffers streaming responses
    _key_getter: Optional[RotateKeys] = PrivateAttr(None)  # Manages API key rotation
    _tool_fuse_broken: bool = PrivateAttr(False) #Fuse to prevent tool loops

    _locator: JustAgentsLocator = PrivateAttr(default_factory=lambda: JustAgentsLocator())

    #@computed_field
    @property
    def codename(self) -> str:
        """
        Get the agent's unique codename from the locator.
        If the agent is not yet registered, it will be registered automatically.
        
        Returns:
            str: The unique codename for this agent
        """
        # Try to get existing codename
        existing_codename = self._locator.get_codename(self)
        if existing_codename:
            return existing_codename
        
        # Register and get a new codename if not already registered
        return self._locator.publish_agent(self)

    def dynamic_prompt(self, prompt: str) -> str:
        extended_prompt = prompt
        if self.prompt_tools:
            for tool_name, tool in self.prompt_tools.items():
                tool_input = tool.call_arguments
                tool_description = tool.description
                tool_output = tool.get_callable()(**tool_input)
                call_string = f'Result of {str(tool_name)}({str(tool_input)}) tool execution:\n'
                call_string += f"{str(tool_output)}\n"
                if tool_description:
                    call_string += f" # {str(tool_name)} short description: {tool_description}\n"

                extended_prompt += f"\n{call_string}\n"
        return extended_prompt


    def instruct(self, prompt: str, memory: IBaseMemory = None, dynamic_prompt: bool = True) -> None: 
        if not memory:
            memory = self.memory

        if dynamic_prompt:
            prompt = self.dynamic_prompt(prompt)
        if prompt:
            memory.add_system_message(prompt)

    def add_to_memory(self, messages: SupportedMessages, memory: IBaseMemory = None) -> None:
        if not memory:
            memory = self.memory
        memory.add_message(messages)

    def get_last_message(self, memory: IBaseMemory = None) -> Optional[MessageDict]:
        if not memory:
            memory = self.memory
        msg = memory.last_message
        # if msg is None:
        #    raise ValueError("No messages in memory")
        return msg

    @property
    def debug_enabled(self) -> bool:
        return self._protocol.is_debug_enabled()

    @debug_enabled.setter
    def debug_enabled(self, value: bool) -> None:
        if value:
            self._protocol.enable_debug()


    def model_post_init(self, __context: Any) -> None:
        # Call parent class's post_init first (from JustAgentProfile)
        super().model_post_init(__context)

        # Initialize protocol adapter if not already set
        # Protocol adapter handles formatting messages for specific LLM providers
        if not self._protocol:
            self._protocol = ProtocolAdapterFactory.get_protocol_adapter(
                self.streaming_method,  # currently, only LiteLLM universal adapter is available
                log_name=self.codename
            )

        # causes bugs, let LiteLLM handle it. https://github.com/BerriAI/litellm/issues/14271
        # If tools (functions) are defined, configure LLM to use them
        # if self.tools is not None:
        #     # Enable automatic tool selection if not explicitly set
        #     if not self.llm_options.get("tool_choice", None):
        #         self.llm_options["tool_choice"] = "auto"

        # Set up API key rotation based on file or environment variable
        if self.key_list_path is not None:
            self._key_getter = RotateKeys.from_path(self.key_list_path)
        elif self.key_list_env is not None:
            self._key_getter = RotateKeys.from_env(self.key_list_env)
            
        # Warn if both direct API key and key rotation are configured
        if (self._key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
            print("Warning api_key will be rewritten by key_getter. Both are present in llm_options.")

        self._protocol.set_logging(enable=self.observability)
        
        # Auto-register with the locator
        self._locator.publish_agent(self)

        if self.debug:
            self._protocol.enable_debug()

    def _fork_memory(self, copy_values: bool) -> IBaseMemory:
        new_memory : IBaseMemory
        if copy_values:
            new_memory = self.memory.deepcopy()
        else:
            #new_memory = type(self.memory)()  # Call the default constructor of same class
            new_memory = self.memory.deepcopy()
            new_memory.messages.clear()

        return new_memory

    @classmethod
    def from_json(cls, json_data: Dict[str, Any], qualname_check: bool = True) -> 'BaseAgent':
        instance : BaseAgent = super().from_json(json_data, qualname_check)
        if not instance.tools and instance.llm_options.get('tools', None):
            raise ValueError(f"Tools mismatch: agent tools empty, but llm_options has tools section:'{instance.llm_options.get('tools')}'")
        return instance

    def _prepare_options(self, options: LLMOptions, **kwargs) -> Dict[str, Any]:
        import warnings
        use_litellm = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            use_litellm = getattr(self, "litellm_tool_description", False)
        opt: LLMOptions = copy.deepcopy(options)
        if self.tools is not None and not self._tool_fuse_broken:  # populate llm_options based on available tools
            opt["tools"] = [
                self._protocol.tool_from_function(
                    self.tools[tool].get_callable(wrap=False),
                    function_dict = self.tools[tool].get_litellm_description(),
                    use_litellm=use_litellm
                ) for tool in self.tools if (
                    not self.tools[tool].max_calls_per_query
                        or self.tools[tool].remaining_calls > 0
                )
            ]
        tool_count = len(opt.get("tools", [])) #active tools

        if tool_count == 0:
            opt.pop("tools", None) #Ensure no tools are passed to adapter
        else:
            for tool in opt["tools"]:
                for built_in in get_args([GoogleBuiltInTools.search, GoogleBuiltInTools.code]):
                    if built_in in tool:
                        tool_count -= 1 #remove built-in tools

        if tool_count == 0: #if no tools, or built-in tools only, remove tool_choice
            opt.pop("tool_choice", None) 
 

        if self.use_proxy and self.proxy_address:
            opt["api_base"] = self.proxy_address
        
        if self.reasoning_effort:
            opt["reasoning_effort"] = self.reasoning_effort

        opt.update(kwargs)
        return opt
    
    def _execute_completion(
            self,
            messages: SupportedMessages,
            stream: bool,
            active_options_for_this_call: LLMOptions, # New parameter
            **kwargs
    ) -> BaseModelResponse:
        
        # Prepare option sets to try for this specific call
        option_sets_to_try = [active_options_for_this_call]
        # If the active options are the primary ones, and a global backup exists,
        # add the global backup as a potential fallback for this specific call's retries.
        if active_options_for_this_call is self.llm_options and self.backup_options:
            option_sets_to_try.append(self.backup_options)

        last_exception: Optional[Exception] = None

        for opt_set_idx, current_options_config_template in enumerate(option_sets_to_try):
            opt = self._prepare_options(current_options_config_template, **kwargs)
            local_key_getter: Optional[RotateKeys] = None
            if self._key_getter is not None and "api_key" not in opt:
                local_key_getter = copy.deepcopy(self._key_getter)

            max_tries = self.completion_max_tries if self.completion_max_tries is not None else 1
            if local_key_getter:
                if self.completion_remove_key_on_error:
                    max_tries = min(max_tries, local_key_getter.len())
                max_tries = max(1, max_tries)
            
            # If this is the second set of options being tried (i.e., internal fallback to self.backup_options),
            # then only attempt it once.
            if opt_set_idx == 1: 
                max_tries = 1 

            for attempt in range(max_tries):
                current_key = opt.get('api_key', None)
                if not current_key and local_key_getter:
                    try:
                        current_key = local_key_getter()
                        opt["api_key"] = current_key
                    except IndexError:
                        last_exception = last_exception or IndexError("Ran out of API keys during rotation.")
                        break

                try:
                    # Only one call site for completion, for both primary and backup options
                    return self._protocol.completion(
                        drop_params=self.drop_unsupported_params,
                        raise_on_completion_status_errors=self.raise_on_completion_status_errors,
                        **opt, 
                        messages=messages,
                        stream=stream,
                    )
                except Exception as e:
                    last_exception = e
                    if hasattr(self, '_log_bus'):
                        self._log_bus.warn(
                            source=f"{self.codename}.completion.error",
                            message=f"Attempt {attempt + 1}/{max_tries} failed with options_set {opt_set_idx}",
                            api_key=JustLogBus.mask_api_key(current_key),
                            action="completion.error",
                            exception=e,
                            options=current_options_config_template,
                            attempt=attempt + 1,
                            max_tries=max_tries
                        )
                    if local_key_getter and self.completion_remove_key_on_error and current_key is not None:
                        local_key_getter.remove(current_key)

            # If we are about to try backup options (because active_options_for_this_call failed), log the fallback
            if opt_set_idx == 0 and len(option_sets_to_try) > 1 and hasattr(self, '_log_bus'): # opt_set_idx == 0 means primary active_options failed
                self._log_bus.info(
                    source=f"{self.codename}.completion.fallback",
                    message=f"Attempts with initial options failed. Falling back to agent's backup_options for this call.",
                    action="completion.fallback"
                )

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Completion failed after all attempts, but no specific exception was caught.")

    def _process_function_calls(
            self,
            function_calls: List[IFunctionCall[SupportedMessages]],
            memory: IBaseMemory = None
    ) -> SupportedMessages:
        if not memory:
            memory = self.memory
        messages: SupportedMessages = []
        for call in function_calls:
            msg = call.execute_function(lambda function_name: self.tools[function_name].get_callable())
            self.handle_on_response(msg, action='response', source='tool')
            self.add_to_memory(msg, memory)
            messages.append(msg)
        return messages

    def _preprocess_input(
            self,
            query_input: SupportedMessages,
            enforce_agent_prompt: Optional[bool] = None,
            continue_conversation: Optional[bool] = None,
            send_system_prompt: Optional[bool] = None,
            **kwargs
    ) -> IBaseMemory:
        # Set default values if not provided
        if enforce_agent_prompt is None:
            enforce_agent_prompt = self.enforce_agent_prompt
        if continue_conversation is None:
            continue_conversation = self.continue_conversation
        if send_system_prompt is None:
            send_system_prompt = self.send_system_prompt

        self.handle_on_query(query_input, action='query', source='input')  # handle the input query
        memory_instance = self._fork_memory(copy_values=continue_conversation)  # Handlers from main memory need to fire even if messages are discarded

        memory_instance.clear_system_messages(clear_non_empty=True) 
        # No old system messages in history in any case, either user's or agent's will be used
        self.add_to_memory(query_input, memory_instance)  # Now add query to ephemeral memory
        memory_instance.clear_system_messages(clear_non_empty=enforce_agent_prompt) #clear again, no system in query for 1 and 3 cases
        
        if send_system_prompt:
            if enforce_agent_prompt or not memory_instance.has_system_messages(): #Cases 1 and 2, 'Agent mode' and 'Agent with override mode'
                self.instruct(self.system_prompt, memory_instance)

        self.handle_on_query(memory_instance.messages, action='query', source='preprocessor')  # handle the modified query
        return memory_instance

    def _postprocess_query(
            self,
            memory_instance: IBaseMemory,
            remember_query: Optional[bool] = None,
            **kwargs
    ) -> IBaseMemory:
        self._tool_fuse_broken = False  # defuse call limiter
        if self.tools:
            for tool in self.tools.values():
                tool.reset() #reset tool calls, if set
        if remember_query is None:
            remember_query = self.remember_query
        if remember_query: #replace with shallow copy of the fork if remember is set
            self.memory.messages = memory_instance.messages.copy()
        return memory_instance

    def query(
            self,
            query_input: SupportedMessages,
            send_system_prompt: Optional[bool] = None,
            enforce_agent_prompt: Optional[bool] = None,
            continue_conversation: Optional[bool] = None,
            remember_query: Optional[bool] = None,
            response_format: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        Query the agent and return the last message.
        The method iteratively calls the LLM. If `backup_options` are configured and tool calls persist,
        the final two attempts (a tool-enabled call, then a potentially tool-less graceful response call)
        will utilize the backup model. The `_tool_fuse_broken` attribute controls whether tools are
        stripped in the final attempt to ensure a graceful exit from tool loops.
        """
        memory = self._preprocess_input(
            query_input,
            send_system_prompt=send_system_prompt,
            enforce_agent_prompt=enforce_agent_prompt,
            continue_conversation=continue_conversation,
        )

        self._tool_fuse_broken = False # Ensure a clean slate for the fuse for this specific call
        effective_max_tool_calls = self.max_tool_calls
        if self.backup_options:
            effective_max_tool_calls += 1

        for step in range(effective_max_tool_calls):
            options_for_this_step = self.llm_options
            if self.backup_options:
                # If backup options are available, use them for the last two attempts
                if step >= effective_max_tool_calls - 2:
                    options_for_this_step = self.backup_options
            
            response = self._execute_completion(
                memory.messages, 
                stream=False, 
                active_options_for_this_call=options_for_this_step,
                response_format=response_format, # Pass response_format here explicitly
                **kwargs
            )
            msg: SupportedMessages = self._protocol.message_from_response(response) # type: ignore
            self.handle_on_response(msg, action='response', source='llm')
            self.add_to_memory(msg, memory)

            if not self.tools or self._tool_fuse_broken:
               break # If  the fuse is broken or there are no tools available, exit the loop

            tool_calls = self._protocol.tool_calls_from_message(msg)
            
            # Fuse logic: if this is the second-to-last attempt overall, and it has tool calls,
            # set the fuse to True so the *final* attempt will be tool-less.
            if step == effective_max_tool_calls - 2: 
                if tool_calls:
                    self._tool_fuse_broken = True # Arm fuse for the *final* iteration (next one)

            if not tool_calls:
                break # Exit loop if no tool calls are made
            
            # Process each tool call if they exist
            self._process_function_calls(
                tool_calls,
                memory
            ) 
            
    

        return self._postprocess_query(
            memory,
            remember_query=remember_query,
        ).last_message_str

    def stream(
            self,
            query_input: SupportedMessages,
            send_system_prompt: Optional[bool] = None,
            enforce_agent_prompt: Optional[bool] = None,
            continue_conversation: Optional[bool] = None,
            remember_query: Optional[bool] = None,
            reconstruct_chunks : bool = False,
            restream_tools: Optional[bool] = None,
            response_format: Optional[str] = None,
            **kwargs
    ) -> Generator[Union[BaseModelResponse, SupportedMessages],None,None]:
        """
        Streams responses from the agent.
        The method iteratively calls the LLM. If `backup_options` are configured and tool calls persist,
        the final two attempts (a tool-enabled call, then a potentially tool-less graceful response call)
        will utilize the backup model. The `_tool_fuse_broken` attribute controls whether tools are
        stripped in the final attempt to ensure a graceful exit from tool loops.
        """
        memory = self._preprocess_input(
            query_input,
            send_system_prompt=send_system_prompt,
            enforce_agent_prompt=enforce_agent_prompt,
            continue_conversation=continue_conversation,
        )

        self._tool_fuse_broken = False # Ensure a clean slate for the fuse for this specific call
        effective_max_tool_calls = self.max_tool_calls
        if self.backup_options:
            effective_max_tool_calls += 1

        for step in range(effective_max_tool_calls):
            self._partial_streaming_chunks.clear()
            
            options_for_this_step = self.llm_options
            if self.backup_options:
                # If backup options are available, use them for the last two attempts
                if step >= effective_max_tool_calls - 2:
                    options_for_this_step = self.backup_options

            response = self._execute_completion(
                memory.messages, 
                stream=True, 
                active_options_for_this_call=options_for_this_step,
                response_format=response_format, 
                **kwargs
            )
            yielded = False
            tool_calls = [] # Initialize tool_calls for this step
            for i, part in enumerate(response):
                self._partial_streaming_chunks.append(part)
                msg: SupportedMessages = self._protocol.message_from_response(part)
                delta = self._protocol.content_from_delta(msg)
                finish_reason: FinishReason = self._protocol.finish_reason_from_response(part)
                if delta or restream_tools:  # stream content as is
                    yielded = True
                    if reconstruct_chunks:
                        yield SSE.sse_wrap(
                            self._protocol.create_chunk_from_content(
                                delta, part["model"], role=msg.get("role", None)
                            ).model_dump(mode='json')
                        )
                    else:
                        yield SSE.sse_wrap(part.model_dump(mode='json'))
                elif finish_reason == FinishReason.function_call:
                    raise NotImplementedError("Function calls are deprecated, use Tool calls instead")
                elif finish_reason == FinishReason.tool_calls:
                    pass #processed separately
                else:
                    yielded = True
                    yield SSE.sse_wrap(part.model_dump(mode='json'))

            if len(self._partial_streaming_chunks) > 0:
                assembly = self._protocol.response_from_deltas(self._partial_streaming_chunks)
                self._partial_streaming_chunks.clear()
                msg: SupportedMessages = self._protocol.message_from_response(assembly)  # type: ignore
                self.handle_on_response(msg, action='response', source='llm')
                self.add_to_memory(msg, memory)

                tool_calls = self._protocol.tool_calls_from_message(msg)
                if not tool_calls and not yielded:
                    yield SSE.sse_wrap(
                        assembly.model_dump(mode='json')
                    )  # not delta and not tool, pass as is

            if not self.tools: # or self._tool_fuse_broken or not tool_calls: (old logic)
                # self._tool_fuse_broken = False # (old logic, fuse reset is handled by _postprocess_query)
                break  # If there are no tools available, exit the loop
            
            # Fuse logic: if this is the second-to-last attempt overall, and it has tool calls,
            # set the fuse to True so the *final* attempt will be tool-less.
            if step == effective_max_tool_calls - 2: 
                if tool_calls: # tool_calls should be populated from the assembled response
                    self._tool_fuse_broken = True # Arm fuse for the *final* iteration (next one)

            if not tool_calls:
                break # Exit loop if no tool calls are made
            else:
                tool_messages = self._process_function_calls(tool_calls,memory)
                if restream_tools:
                    for i, tool_message in enumerate(tool_messages):
                        yield SSE.sse_wrap(
                            self._protocol.create_chunk_from_content(
                                tool_message, kwargs.get("model",self.shortname), role=Role.tool.value
                            ).model_dump(mode='json')
                        )
    

        self._postprocess_query(
            memory,
            remember_query=remember_query,
        )
        yield SSE.sse_wrap(self._protocol.stop)


    def query_structural(
        self, 
        query_input: SupportedMessages, 
        parser: Type[BaseModel] = BaseModel,
        response_format: Optional[str] = None,
        enforce_validation: bool = False,
        **kwargs
    ) -> Union[dict, BaseModel]:
        """
        Query the agent and parse the response according to the provided parser.
        
        If no response_format is provided and parser is a Pydantic model,
        automatically generates and sends a JSON schema to guide the response format.
        
        Args:
            query_input: The input to send to the model
            parser: The Pydantic model class to validate against or dict
            response_format: Optional JSON schema to guide response format
            enforce_validation: Whether to enforce validation on the model side
            **kwargs: Additional arguments to pass to the query method
            
        Returns:
            Either a dictionary or validated Pydantic model
        """
        
        if response_format is None and parser is not dict and issubclass(parser, BaseModel) and enforce_validation:
            schema = ModelHelper.get_response_schema(parser)
            response_format = ModelHelper.make_all_fields_required(parser)
            
            schema_wrapper = {
                "name": "query_structural", 
                "schema": schema,
                "strict": enforce_validation
            }

            response_format_obj = {"type": "json_schema", "json_schema": schema_wrapper}
        
        # Get raw response from the model
        raw_response = self.query(query_input, response_format=response_format, **kwargs)
        
        # Process and validate the response
        return ModelHelper.get_structured_output(raw_response, parser)


    @property
    def model_supported_parameters(self) -> list[str]:
        """Returns the list of parameters supported by the current model"""
        model = self.llm_options.get("model")
        if not model:
            return []
        return self._protocol.get_supported_params(model)
    
    @property
    def supports_vision(self) -> bool:
        """Checks if the current model supports vision"""
        return "vision" in self.model_supported_parameters
    
    
    @property
    def supports_response_format(self) -> bool:
        """Checks if the current model supports the response_format parameter"""
        #TODO: implement provider specific check
        return "response_format" in self.model_supported_parameters
    


class LogFunction(Protocol):
    def __call__(self, log_string: str, action: str, source: str, *args: VariArgs.args,
                 **kwargs: VariArgs.kwargs) -> None:
        ...

# Simple default logging implementation that matches LogFunction type
def log_print(log_string: str, action: str, source: str, *args: VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
    """Basic print-based logging stub intended to be replaced with actual logger.
    Implements LogFunction protocol: Callable[[str, str, str, Any], None]

    Args:
        log_string: The message to log
        action: The type of action being logged
        source: The source of the log message
        **kwargs: Additional logging parameters
    """
    shortname = kwargs.pop("agent_shortname", "")

    if kwargs:
        print(f"{action} from {source}: {log_string}, extra args: {str(kwargs)}")
    elif shortname:
        print(f"{action} from <{shortname}> {source}: {log_string}")
    else:
        print(f"{action} from {source}: {log_string}")




class BaseAgentWithLogging(BaseAgent):
    default_logging_function: ClassVar[LogFunction] = partial(log_print) #default logging function
    _log_function: LogFunction = PrivateAttr(default=None) #a hook to attach a custom logging function dynamically
    
    #Handlers protocols/signatures typehints aggregated for reference:
    _log_query: QueryListener[SupportedMessages] = PrivateAttr(default=None)
    _log_response: ResponseListener[SupportedMessages]  = PrivateAttr(default=None)
    _log_tool_request: OnToolCallable = PrivateAttr(default=None)
    _log_message: OnMessageCallable = PrivateAttr(default=None)
    _log_tool_call: SubscriberCallback = PrivateAttr(default=None)
    _log_tool_result: SubscriberCallback = PrivateAttr(default=None)
    _log_tool_error: SubscriberCallback = PrivateAttr(default=None)

    _log_bus : JustLogBus = PrivateAttr(default_factory= lambda: JustLogBus())
    
    @property
    def log_function(self) -> LogFunction:
        return lambda message, action, source, *args, **kwargs: (
                self._logger_instance_info_wrapper(message, action, source, *args, **kwargs)
        )
    
    @log_function.setter
    def log_function(self, value: LogFunction) -> None:
        if not callable(value):
            raise ValueError("log_function must be a callable")
        else:
            self._log_function = value

    def _logger_instance_info_wrapper(self, message, action, source, *args, **kwargs):
        kwargs_cpy = kwargs.copy() #defensive copy
        agent_codename=self.codename
        kwargs_cpy["agent_shortname"] = self.shortname
        # kwargs_cpy["agent_id"] = agent_codename
        self._log_function(
                message,
                action,
                f"{agent_codename}.{source}",
                *args,
                **kwargs_cpy
        )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self._log_function is None:
            self.log_function=self.default_logging_function

        #self._log_query = self.query_handler
        # Wrap the bound query_handler in a lambda to match the expected QueryListener signature.
        self._log_query = (
            lambda input_query, action, source, *args, **kwargs: self.query_handler(input_query, action, source, *args, **kwargs)
        )

        self.add_on_query_listener(self._log_query) #using listener to log query
        self._log_response = self.response_handler
        self.add_on_response_listener(self._log_response) #using listener to log response
        self._log_tool_request = self.tool_handler   
        self.memory.add_on_tool_call(self._log_tool_request) #using memory handler to log tool call
        self._log_message = self.memory_handler
        self.memory.add_on_message(self._log_message) #using handler to log message
        self._log_tool_call = self.tool_call_callback
        self.subscribe_to_tool_call(self._log_tool_call) #using listener to log tool call
        self.subscribe_to_prompt_tool_call(self._log_tool_call)
        self._log_tool_result = self.tool_result_callback
        self.subscribe_to_tool_result(self._log_tool_result) #using listener to log tool results
        self.subscribe_to_prompt_tool_result(self._log_tool_result)
        self._log_tool_error = self.tool_error_callback
        self.subscribe_to_tool_error(self._log_tool_error) #using listener to log tool errors
        self.subscribe_to_prompt_tool_error(self._log_tool_error)
        
        self._log_bus.subscribe(f"{self.codename}.*", self.log_event_handler)
        self.log_function(
            f"Loaded {self.shortname} instance with codename {self.codename}",
            "instantiation.success",
            f"model_post_init",
            class_name=f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        )


    def _message_to_strings(self, message: SupportedMessages) -> List[str]:
        if isinstance(message, list):
            return [self._message_to_strings(item)[0] for item in message]
        elif isinstance(message, Message):
            return [str(message.model_dump_json(
                exclude_none=True,
                exclude_unset=False,
                exclude_defaults=False,
                by_alias=True,
            ))]
        else:
            return [str(message or '')]


    def query_handler(self, input_query: SupportedMessages, action: str, source: str, *args:VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        _ = args
        for item in self._message_to_strings(input_query):
            self.log_function(item, action, source, **kwargs)


    def response_handler(self, response: SupportedMessages, action: str, source: str) -> None:
        for item in self._message_to_strings(response):
            self.log_function(item, action, source)

    def tool_call_callback(self, event_name: str, *args, **kwargs) -> None:
        """
        Callback function for tool call events.
        
        Args:
            event_name (str): Name of the event (format: "{tool_name}.call")
            *args: Positional arguments passed to the tool
            **kwargs: Keyword arguments passed to the tool
        """
        # Log or process the tool call
        #self.log_function(event_name, "tool.execute", "tools_bus") #auxilary event for tracing if parsing fails
        if args:
          self.log_function(f"{str(args)}", "tool.args", event_name)
        self.log_function(f"{str(kwargs)}", "tool.kwargs", event_name)


    def tool_result_callback(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Callback function for tool result events.
        
        Args:
            event_name (str): Name of the event (format: "{tool_name}.result")
            *args: Additional positional arguments (if any).
            **kwargs: Keyword arguments that should include a "result_interceptor" key.
        """
        # Extract the result_interceptor from kwargs (if present)
        result_interceptor: Optional[Any] = kwargs.get("result_interceptor")
        
        # Log the event
        #self.log_function(event_name, "tool.result", "tools_bus") #auxilary event for tracing if parsing fails
       
        if result_interceptor is not None:
            self.log_function(f"{str(result_interceptor)}", "tool.result", event_name)
        else:
            self.log_function("None", "tool.result", event_name)

    def tool_error_callback(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Callback function for tool error events.
        
        Args:
            event_name (str): Name of the event (format: "{tool_name}.error")
            *args: Additional positional arguments (if any).
            **kwargs: Keyword arguments that should include an "error" key.
        """
        # Extract the error value from kwargs (if present)
        error: Optional[Exception] = kwargs.get("error")
        
        # Log the event and error
        #self.log_function(event_name, "tool.error", "tools_bus")
        self.log_function(f"{str(error)}", "tool.error", event_name)
    
    def tool_handler(self, tool_call: ToolCall) -> None:
        """Handler for tool calls. Implements OnToolCallable protocol.
        
        Args:
            tool_call: The tool call to handle
        """
        self.log_function(tool_call.name, action="memory.add", source="tool_call")
        self.log_function(f"{str(tool_call.arguments)}", action="memory.add", source=tool_call.name)

    def memory_handler(self, message: MessageDict) -> None:
        """Handler for messages. Implements OnMessageCallable protocol.
        
        Args:
            message: The message dictionary to handle
        """
        if message:
            self.log_function(str(message), action="memory.add", source=message.get("role","Error"))

    def log_event_handler(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Handler for auxiliary log events. 
        
        Args:
            event_name: The name of the event
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        message: Optional[str] = kwargs.pop("message","") or kwargs.pop("log_message","Missing log message!!")
        action: Optional[str] = kwargs.pop("action", "log_bus.receive")
        self.log_function(message, action=action, source=event_name, **kwargs)


class ChatAgent(BaseAgent, JustAgentProfileChatMixin):
    """
    An agent that has role/goal/task attributes and can call other agents
    """

    def model_post_init(self, __context: Any) -> None:
        # Call parent's post_init to maintain core functionality
        super().model_post_init(__context)
        if self.system_prompt == self.DEFAULT_GENERIC_PROMPT:
            self.system_prompt = ""

        if self.role is not None:
            self.system_prompt = self.system_prompt + "\n" + self.role
        if self.goal is not None:
            self.system_prompt = self.system_prompt + "\n" + self.goal
        if self.task is not None:
            self.system_prompt = self.system_prompt + "\n" + self.task
        if self.format is not None:
            self.system_prompt = self.system_prompt + "\n" + self.format

class ChatAgentWithLogging(ChatAgent, BaseAgentWithLogging):
    pass

