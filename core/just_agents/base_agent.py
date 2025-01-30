from pydantic import Field, PrivateAttr
from typing import Optional, List, Union, Any, Generator

from just_agents.data_classes import FinishReason, ToolCall
from just_agents.types import MessageDict, SupportedMessages

from just_agents.llm_options import LLMOptions
from just_agents.interfaces.function_call import IFunctionCall
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, BaseModelResponse
from just_agents.interfaces.agent import IAgentWithInterceptors, QueryListener, ResponseListener

from just_agents.base_memory import IBaseMemory, BaseMemory
from just_agents.just_profile import JustAgentProfile
from just_agents.rotate_keys import RotateKeys
from just_agents.protocols.protocol_factory import StreamingMode, ProtocolAdapterFactory



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

    def _fork_memory(self, copy_values: bool) -> IBaseMemory:
        new_memory : IBaseMemory
        if copy_values:
            new_memory = self.memory.model_copy() #Shallow copy
        else:
            new_memory = type(self.memory)()  # Call the default constructor of same class
        return new_memory

    def _prepare_options(self, options: LLMOptions):
        opt = options.copy()
        if self.tools is not None and not self._tool_fuse_broken:  # populate llm_options based on available tools
            opt["tools"] = [{"type": "function",
                             "function": self.tools[tool].get_litellm_description()} for tool in self.tools]
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
            self.handle_on_response(msg)
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

        self.handle_on_query(query_input) # handle the input query
        memory_instance = self._fork_memory(copy_values=True) #Handlers from main memory need to fire even if messages are discarded
        if not continue_conversation:
            memory_instance.clear_messages() #Clear copied messages list instead
        self.add_to_memory(query_input, memory_instance) #Now add query to ephemeral memory
        memory_instance.clear_system_messages(clear_non_empty=enforce_agent_prompt) #Clear prompt messages
        if not memory_instance.prompt_messages:
            self.instruct(self.system_prompt, memory_instance) #Ensure system prompt

        self.handle_on_query(memory_instance.messages)  # handle the modified query
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
            self.handle_on_response(msg)
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
                if delta:  # stream content as is
                    yielded = True
                    if reconstruct_chunks:
                        yield self._protocol.get_chunk(i, delta, options={'model': part["model"]})
                    else:
                        yield self._protocol.sse_wrap(part.model_dump(mode='json'))
                elif finish_reason == FinishReason.function_call:
                    raise NotImplementedError("Function calls are deprecated, use Tool calls instead")
                elif finish_reason == FinishReason.tool_calls:
                    pass #processed separately
                else:
                    yielded = True
                    yield self._protocol.sse_wrap(part.model_dump(mode='json'))

            if len(self._partial_streaming_chunks) > 0:
                assembly = self._protocol.response_from_deltas(self._partial_streaming_chunks)
                self._partial_streaming_chunks.clear()
                msg: SupportedMessages = self._protocol.message_from_response(assembly)  # type: ignore
                self.handle_on_response(msg)
                self.add_to_memory(msg, memory)

                tool_calls = self._protocol.tool_calls_from_message(msg)
                if not tool_calls and not yielded:
                    yield self._protocol.sse_wrap(
                        assembly.model_dump(mode='json'))  # not delta and not tool, pass as is

            if not self.tools or self._tool_fuse_broken or not tool_calls:
                self._tool_fuse_broken = False
                break  # If there are no tool calls or tools available, exit the loop
            else:
                self._process_function_calls(
                    tool_calls,
                    memory
                )
            if step == self.max_tool_calls - 2:  # special case where we ran out of tool calls or stuck in a loop
                self._tool_fuse_broken = True  # one last attempt at graceful response without tools

        self._postprocess_query(
            memory,
            remember_query=remember_query,
        )
        yield self._protocol.done()

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
    


class ChatAgent(BaseAgent):
    """
    An agent that has role/goal/task attributes and can call other agents
    """

    role: Optional[str] = Field(default=None, description="Defines the agent's persona or identity")
    goal: Optional[str] = Field (default=None, description="Specifies the agent's broader objective.")
    task: Optional[str] = Field (default=None, description="Describes the specific task the agent is responsible for.")
    format: Optional[str] = Field (default=None, description="Describes the specific format the agent is responsible for.")
    


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

