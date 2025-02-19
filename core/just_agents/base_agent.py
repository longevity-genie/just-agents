from pydantic import Field, PrivateAttr
from typing import Optional, List, Union, Any, Generator, Dict, ClassVar, Protocol
from functools import partial
from just_agents.data_classes import FinishReason, ToolCall, Message, Role
from just_agents.types import MessageDict, SupportedMessages

from just_agents.llm_options import LLMOptions
from just_agents.interfaces.function_call import IFunctionCall
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, BaseModelResponse
from just_agents.interfaces.agent import IAgentWithInterceptors, QueryListener, ResponseListener, VariArgs

from just_agents.base_memory import IBaseMemory, BaseMemory, OnToolCallable, OnMessageCallable
from just_agents.just_profile import JustAgentProfile, JustAgentProfileChatMixin
from just_agents.rotate_keys import RotateKeys
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.protocols.protocol_factory import StreamingMode, ProtocolAdapterFactory
from just_agents.just_tool import SubscriberCallback


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
        exclude=True,
        description="options that will be used after we give up with main options, one more completion call will be done with backup options")

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

    max_tool_calls: int = Field(
        ge=1,
        default=50,
        description="A safeguard to prevent tool calls stuck in a loop")

    drop_params: bool = Field(
        default=True,
        description="Drop params from the request, useful for some models that do not support them")

    enforce_agent_prompt: bool = Field(
        default=True,
        description="When set, replaces query containing 'system' messages with agent prompt")

    continue_conversation: bool = Field(
        default=True,
        description="Concatenate memory messages and query messages ")

    remember_query: bool = Field(
        default=True,
        description="Add new query messages to memory")

    # Protected handlers implementation
    _on_query : List[QueryListener] = PrivateAttr(default_factory=list)
    _on_response : List[ResponseListener] = PrivateAttr(default_factory=list)

    # Private attributes for internal state management
    _protocol: Optional[IProtocolAdapter] = PrivateAttr(None)  # Handles LLM-specific message formatting
    _partial_streaming_chunks: List[BaseModelResponse] = PrivateAttr(
        default_factory=list)  # Buffers streaming responses
    _key_getter: Optional[RotateKeys] = PrivateAttr(None)  # Manages API key rotation
    _tool_fuse_broken: bool = PrivateAttr(False) #Fuse to prevent tool loops




    def instruct(self, prompt: str, memory: IBaseMemory = None): #backward compatibility
        if not memory:
            memory = self.memory
        memory.add_system_message(prompt)

    def deepcopy_memory(self, memory: IBaseMemory = None) -> IBaseMemory:
        if not memory:
            memory = self.memory
        return memory.deepcopy()

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
        return self._protocol.debug_enabled

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
                self.streaming_method,  # e.g., OpenAI, Azure, etc.
            )

        # If tools (functions) are defined, configure LLM to use them
        if self.tools is not None:
            # Enable automatic tool selection if not explicitly set
            if not self.llm_options.get("tool_choice", None):
                self.llm_options["tool_choice"] = "auto"

        # Set up API key rotation based on file or environment variable
        if self.key_list_path is not None:
            self._key_getter = RotateKeys.from_path(self.key_list_path)
        elif self.key_list_env is not None:
            self._key_getter = RotateKeys.from_env(self.key_list_env)
            
        # Warn if both direct API key and key rotation are configured
        if (self._key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
            print("Warning api_key will be rewritten by key_getter. Both are present in llm_options.")

        self._protocol.enable_logging()

    def _fork_memory(self, copy_values: bool) -> IBaseMemory:
        new_memory : IBaseMemory
        if copy_values:
            new_memory = self.memory.model_copy() #Shallow copy
        else:
            new_memory = type(self.memory)()  # Call the default constructor of same class
        return new_memory

    @classmethod
    def from_json(cls, json_data: Dict[str, Any], qualname_check: bool = True) -> 'BaseAgent':
        instance : BaseAgent = super().from_json(json_data, qualname_check)
        if not instance.tools and instance.llm_options.get('tools', None):
            raise ValueError(f"Tools mismatch: agent tools empty, but llm_options has tools section:'{instance.llm_options.get('tools')}'")
        return instance

    def _prepare_options(self, options: LLMOptions):
        opt = options.copy()
        if self.tools is not None and not self._tool_fuse_broken:  # populate llm_options based on available tools
            opt["tools"] = [{"type": "function",
                             "function": self.tools[tool].get_litellm_description()} for tool in self.tools]
        else:
            opt.pop("tools", None) #Ensure no tools are passed to adapter
        return opt
    
    def _execute_completion(
            self,
            messages: SupportedMessages,
            stream: bool,
            **kwargs
    ) -> BaseModelResponse:
        
        opt = self._prepare_options(self.llm_options)
        opt.update(kwargs)
        
        max_tries = self.completion_max_tries or 1  # provide default if None
        if self._key_getter is not None:
            if max_tries < 1:
                max_tries = self._key_getter.len()
            else:
                if self.completion_remove_key_on_error:
                    max_tries = min(max_tries, self._key_getter.len())
            last_exception = None
            for _ in range(max_tries):
                opt["api_key"] = self._key_getter()
                try:
                    return self._protocol.completion(messages=messages, stream=stream, **opt)
                except Exception as e:
                    last_exception = e
                    if self.completion_remove_key_on_error:
                        self._key_getter.remove(opt["api_key"])

            if self.backup_options:
                opt = self._prepare_options(self.backup_options)
                return self._protocol.completion(messages=self.memory.messages, stream=stream, **opt)
            if last_exception:
                raise last_exception
            else:
                raise Exception(
                    f"Run out of tries to execute completion. Check your keys! Keys {self._key_getter.len()} left.")
        else:
            return self._protocol.completion(messages=self.memory.messages, stream=stream, **opt)


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
            **kwargs
    ) -> IBaseMemory:

        if enforce_agent_prompt is None:
            enforce_agent_prompt = self.enforce_agent_prompt
        if continue_conversation is None:
            continue_conversation = self.continue_conversation

        self.handle_on_query(query_input, action='query', source='input') # handle the input query
        memory_instance = self._fork_memory(copy_values=True) #Handlers from main memory need to fire even if messages are discarded
        if not continue_conversation:
            memory_instance.clear_messages() #Clear copied messages list instead
        self.add_to_memory(query_input, memory_instance) #Now add query to ephemeral memory
        memory_instance.clear_system_messages(clear_non_empty=enforce_agent_prompt) #Clear prompt messages
        if not memory_instance.prompt_messages:
            self.instruct(self.system_prompt, memory_instance) #Ensure system prompt

        self.handle_on_query(memory_instance.messages, action='query', source='preprocessor')  # handle the modified query
        return memory_instance

    def _postprocess_query(
            self,
            memory_instance: IBaseMemory,
            remember_query: Optional[bool] = None,
            **kwargs
    ) -> IBaseMemory:
        self._tool_fuse_broken = False  # defuse
        if remember_query is None:
            remember_query = self.remember_query
        if remember_query: #replace with shallow copy of the fork if remember is set
            self.memory.messages = memory_instance.messages.copy()
        return memory_instance

    def query(
            self,
            query_input: SupportedMessages,
            enforce_agent_prompt: Optional[bool] = None,
            continue_conversation: Optional[bool] = None,
            remember_query: Optional[bool] = None,
            **kwargs
    ) -> str:
        """
        Query the agent and return the last message
        """
        memory = self._preprocess_input(
            query_input,
            enforce_agent_prompt=enforce_agent_prompt,
            continue_conversation=continue_conversation,
        )

        for step in range(self.max_tool_calls):
            # individual llm call, unpacking the message, processing handlers
            response = self._execute_completion(memory.messages ,stream=False, **kwargs)
            msg: SupportedMessage = self._protocol.message_from_response(response) # type: ignore
            self.handle_on_response(msg, action='response', source='llm')
            self.add_to_memory(msg, memory)

            if not self.tools or self._tool_fuse_broken:
               break         # If there are no tool calls or tools available, exit the loop

            tool_calls = self._protocol.tool_calls_from_message(msg)
            # Process each tool call if they exist and re-execute query
            self._process_function_calls(
                tool_calls,
                memory
            )  # NOTE: no kwargs here as tool calls might need different parameters

            if not tool_calls:
                break
            elif step == self.max_tool_calls - 2: #special case where we ran out of tool calls or stuck in a loop
                self._tool_fuse_broken = True #one last attempt at graceful response

        return self._postprocess_query(
            memory,
            remember_query=remember_query,
        ).last_message_str

    def stream(
            self,
            query_input: SupportedMessages,
            enforce_agent_prompt: Optional[bool] = None,
            continue_conversation: Optional[bool] = None,
            remember_query: Optional[bool] = None,
            reconstruct_chunks : bool = False,
            restream_tools: Optional[bool] = None,
            **kwargs
    ) -> Generator[Union[BaseModelResponse, SupportedMessages],None,None]:
        memory = self._preprocess_input(
            query_input,
            enforce_agent_prompt=enforce_agent_prompt,
            continue_conversation=continue_conversation,
        )

        for step in range(self.max_tool_calls):
            self._partial_streaming_chunks.clear()
            response = self._execute_completion(memory.messages, stream=True, **kwargs)
            yielded = False
            tool_calls = []
            for i, part in enumerate(response):
                self._partial_streaming_chunks.append(part)
                msg: SupportedMessages = self._protocol.message_from_response(part)
                delta = self._protocol.content_from_delta(msg)
                finish_reason: FinishReason = self._protocol.finish_reason_from_response(part)
                if delta or restream_tools:  # stream content as is
                    yielded = True
                    if reconstruct_chunks:
                        yield SSE.sse_wrap(self._protocol.message_as_chunk(i, delta, part["model"], msg.get("role", None)))
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

            if not self.tools or self._tool_fuse_broken or not tool_calls:
                self._tool_fuse_broken = False
                break  # If there are no tool calls or tools available, exit the loop
            else:
                tool_messages = self._process_function_calls(tool_calls,memory)
                if restream_tools:
                    for i, tool_message in enumerate(tool_messages):
                        yield SSE.sse_wrap(self._protocol.message_as_chunk(i, tool_message, part["model"]))
            if step == self.max_tool_calls - 2:  # special case where we ran out of tool calls or stuck in a loop
                self._tool_fuse_broken = True  # one last attempt at graceful response without tools

        self._postprocess_query(
            memory,
            remember_query=remember_query,
        )
        yield SSE.sse_wrap(self._protocol.stop)

    @property
    def model_supported_parameters(self) -> list[str]:
        """Returns the list of parameters supported by the current model"""
        model = self.llm_options.get("model")
        if not model:
            return []
        return self._protocol.get_supported_params(model)
    
    
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
    print(f"{action} from {source}: {log_string}")
    if kwargs:
        print(f"{action} from {source}, extra args: {str(kwargs)}")


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


    @property
    def log_function(self) -> LogFunction:
        return self._log_function
    
    @log_function.setter
    def log_function(self, value: LogFunction) -> None:
        if not callable(value):
            raise ValueError("log_function must be a callable")
        else:
            self._log_function = value

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self._log_function is None:
            self._log_function=self.default_logging_function

        #self._log_query = self.query_handler
        # Wrap the bound query_handler in a lambda to match the expected QueryListener signature.
        # Probably a linter bug due to generics + variadic arguments, works both ways
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
        self._log_tool_result = self.tool_result_callback
        self.subscribe_to_tool_result(self._log_tool_result) #using listener to log tool results
        self._log_tool_error = self.tool_error_callback
        self.subscribe_to_tool_error(self._log_tool_error) #using listener to log tool errors

        self._log_function(f"Loaded {self.shortname}", "instantiation.success", "BaseAgentWithLogging", class_name=self.__class__)


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
            self._log_function(item, action, source, **kwargs)


    def response_handler(self, response: SupportedMessages, action: str, source: str) -> None:
        for item in self._message_to_strings(response):
            self._log_function(item, action, source)

    def tool_call_callback(self, event_name: str, *args, **kwargs) -> None:
        """
        Callback function for tool call events.
        
        Args:
            event_name (str): Name of the event (format: "{tool_name}.call")
            *args: Positional arguments passed to the tool
            **kwargs: Keyword arguments passed to the tool
        """
        # Log or process the tool call
        #self._log_function(event_name, "tool.execute", "tools_bus") #auxilary event for tracing if parsing fails
        self._log_function(f"{str(args)}", "tool.args", event_name)
        self._log_function(f"{str(kwargs)}", "tool.kwargs", event_name)


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
        #self._log_function(event_name, "tool.result", "tools_bus") #auxilary event for tracing if parsing fails
       
        if result_interceptor is not None:
            self._log_function(f"{str(result_interceptor)}", "tool.result", event_name)
        else:
            self._log_function("None", "tool.result", event_name)

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
        #self._log_function(event_name, "tool.error", "tools_bus")
        self._log_function(f"{str(error)}", "tool.error", event_name)


    
    def tool_handler(self, tool_call: ToolCall) -> None:
        """Handler for tool calls. Implements OnToolCallable protocol.
        
        Args:
            tool_call: The tool call to handle
        """
        self._log_function(tool_call.name, action="memory.add", source="tool_call")
        self._log_function(f"{str(tool_call.arguments)}", action="memory.add", source=tool_call.name)

    def memory_handler(self, message: MessageDict) -> None:
        """Handler for messages. Implements OnMessageCallable protocol.
        
        Args:
            message: The message dictionary to handle
        """
        if message:
            self._log_function(str(message), action="memory.add", source=message.get("role","Error"))


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

