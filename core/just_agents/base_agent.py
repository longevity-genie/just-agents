from pydantic import Field, PrivateAttr
from typing import Optional, List, Union, Any, Generator

from just_agents.core.interfaces.IMemory import IMemory
from just_agents.core.types import Role, AbstractMessage, SupportedMessages, SupportedMessage

from just_agents.llm_options import LLMOptions
from just_agents.streaming.protocols.interfaces.IFunctionCall import IFunctionCall
from just_agents.streaming.protocols.interfaces.IProtocolAdapter import IProtocolAdapter, BaseModelResponse
from just_agents.core.interfaces.IAgent import IAgentWithInterceptors, QueryListener, ResponseListener

from just_agents.base_memory import IBaseMemory, BaseMemory
from just_agents.just_profile import JustAgentProfile
from just_agents.core.rotate_keys import RotateKeys
from just_agents.streaming.protocol_factory import StreamingMode, ProtocolAdapterFactory
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params


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

    max_tool_calls: int = Field(
        ge=1,
        default=50,
        description="A safeguard to prevent tool calls stuck in a loop")

    drop_params: bool = Field(
        default=True,
        description="Drop params from the request, useful for some models that do not support them")

    # Protected handlers implementation
    _on_query : List[QueryListener] = PrivateAttr(default_factory=list)
    _on_response : List[ResponseListener] = PrivateAttr(default_factory=list)

    # Private attributes for internal state management
    _protocol: Optional[IProtocolAdapter] = PrivateAttr(None)  # Handles LLM-specific message formatting
    _partial_streaming_chunks: List[BaseModelResponse] = PrivateAttr(
        default_factory=list)  # Buffers streaming responses
    _key_getter: Optional[RotateKeys] = PrivateAttr(None)  # Manages API key rotation
    _tool_fuse_broken: bool = PrivateAttr(False) #Fuse to prevent tool loops

    def instruct(self, prompt: str): #backward compatibility
        self.memory.add_message({"role": Role.system, "content": prompt})

    def clear_memory(self) -> None:
        self.memory.clear_messages()
        self.instruct(self.system_prompt)

    def deepcopy_memory(self) -> IMemory:
        return self.memory.deepcopy()

    def add_to_memory(self, messages: SupportedMessages) -> None:
        self.memory.add_message(messages)

    def get_last_message(self) -> SupportedMessage:  # type: ignore
        msg = self.memory.last_message
        if msg is None:
            raise ValueError("No messages in memory")
        return msg

    def model_post_init(self, __context: Any) -> None:
        # Call parent class's post_init first (from JustAgentProfile)
        super().model_post_init(__context)

        # Initialize protocol adapter if not already set
        # Protocol adapter handles formatting messages for specific LLM providers
        if not self._protocol:
            self._protocol = ProtocolAdapterFactory.get_protocol_adapter(
                self.streaming_method,  # e.g., OpenAI, Azure, etc.
                execute_functions=lambda calls: self._process_function_calls(calls),
            )

        # If tools (functions) are defined, configure LLM to use them
        if self.tools is not None:
            # Enable automatic tool selection if not explicitly set
            if not self.llm_options.get("tool_choice", None):
                self.llm_options["tool_choice"] = "auto"

        # Set up API key rotation if a key list file is provided
        if self.key_list_path is not None:
            self._key_getter = RotateKeys(self.key_list_path)
        # Warn if both direct API key and key rotation are configured
        if (self._key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
            print("Warning api_key will be rewritten by key_getter. Both are present in llm_options.")

        # Initialize the agent with its system prompt
        self.instruct(self.system_prompt)

    def _prepare_options(self, options: LLMOptions):
        opt = options.copy()
        if self.tools is not None and not self._tool_fuse_broken:  # populate llm_options based on available tools
            opt["tools"] = [{"type": "function",
                             "function": self.tools[tool].get_litellm_description()} for tool in self.tools]
        return opt

    def _execute_completion(
            self,
            stream: bool,
            **kwargs
    ) -> Union[AbstractMessage, BaseModelResponse]:
        
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
                    return self._protocol.completion(messages=self.memory.messages, stream=stream, **opt)
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


    def _process_function_calls(self, function_calls: List[IFunctionCall[AbstractMessage]]) -> SupportedMessages:
        messages: SupportedMessages = []
        for call in function_calls:
            msg = call.execute_function(lambda function_name: self.tools[function_name].get_callable())
            self.handle_on_response(msg)
            self.add_to_memory(msg)
            messages.append(msg)
        return messages
    
    def query_with_current_memory(self, **kwargs): #former proceed() aka llm_think()
        for step in range(self.max_tool_calls):
            # individual llm call, unpacking the message, processing handlers
            response = self._execute_completion(stream=False, **kwargs)
            msg: AbstractMessage = self._protocol.message_from_response(response) # type: ignore
            self.handle_on_response(msg)
            self.add_to_memory(msg)

            if not self.tools or self._tool_fuse_broken:
               self._tool_fuse_broken = False
               break
            # If there are no tool calls or tools available, exit the loop
            tool_calls = self._protocol.tool_calls_from_message(msg)
            # Process each tool call if they exist and re-execute query
            self._process_function_calls(
                tool_calls)  # NOTE: no kwargs here as tool calls might need different parameters

            if not tool_calls:
                break
            elif step == self.max_tool_calls - 2: #special case where we ran out of tool calls or stuck in a loop
                self._tool_fuse_broken = True #one last attempt at graceful response


    def streaming_query_with_current_memory(self, reconstruct_chunks = False, **kwargs):
        try:
            self._partial_streaming_chunks.clear()
            for step in range(self.max_tool_calls):
                response = self._execute_completion(stream=True, **kwargs)
                tool_messages: list[AbstractMessage] = []
                for i, part in enumerate(response):
                    self._partial_streaming_chunks.append(part)
                    msg: AbstractMessage = self._protocol.message_from_delta(response) # type: ignore
                    delta = self._protocol.content_from_delta(msg)
                    if delta:
                        if reconstruct_chunks:
                            yield self._protocol.get_chunk(i, delta, options={'model': part["model"]})
                        else:
                            yield response
                    if self.tools and not self._tool_fuse_broken:
                        tool_calls = self._protocol.tool_calls_from_message(msg)
                        if tool_calls:
                            self.add_to_memory(
                                self._protocol.function_convention.reconstruct_tool_call_message(tool_calls)
                            )
                            self._process_function_calls(tool_calls)
                            tool_messages.append(self._process_function_calls(tool_calls))

                if not tool_messages:
                    break
                elif step == self.max_tool_calls - 2:  # special case where we ran out of tool calls or stuck in a loop
                    self._tool_fuse_broken = True  # one last attempt at graceful response

        finally:
            self._tool_fuse_broken = False #defuse
            yield self._protocol.done()
            if len(self._partial_streaming_chunks) > 0:
                response = self._protocol.response_from_deltas(self._partial_streaming_chunks)
                msg: AbstractMessage = self._protocol.message_from_response(response) # type: ignore
                self.handle_on_response(msg)
                self.add_to_memory(msg)
            self._partial_streaming_chunks.clear()


    def query(self, query_input: SupportedMessages, **kwargs) -> str:
        """
        Query the agent and return the last message
        """
        self.handle_on_query(query_input)
        self.add_to_memory(query_input)
        self.query_with_current_memory(**kwargs)
        result = self.memory.last_message_str
        if result is None:
            raise ValueError("No response generated")
        return result
    

    def stream(self, query_input: SupportedMessages, reconstruct_chunks = False, **kwargs) -> Generator[Union[BaseModelResponse, AbstractMessage],None,None]:
        self.handle_on_query(query_input)
        self.add_to_memory(query_input)
        return self.streaming_query_with_current_memory(reconstruct_chunks=reconstruct_chunks, **kwargs)


    @property
    def model_supported_parameters(self) -> list[str]:
        """Returns the list of parameters supported by the current model"""
        model = self.llm_options.get("model")
        if not model:
            return []
        return get_supported_openai_params(model)  # type: ignore
    
    
    @property
    def supports_response_format(self) -> bool:
        """Checks if the current model supports the response_format parameter"""
        #TODO: implement provider specific check
        return "response_format" in self.model_supported_parameters