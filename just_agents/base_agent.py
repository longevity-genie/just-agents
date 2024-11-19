
from pydantic import Field, PrivateAttr
from typing import Optional, List, Union, Any, Generator

from just_agents.interfaces.IMemory import IMemory
from just_agents.types import Role, AbstractMessage, SupportedMessages, SupportedMessage

from just_agents.llm_options import LLMOptions
from just_agents.interfaces.IFunctionCall import IFunctionCall
from just_agents.interfaces.IProtocolAdapter import IProtocolAdapter, BaseModelResponse
from just_agents.interfaces.IAgent import IAgentWithInterceptors, QueryListener, ResponseListener

from just_agents.base_memory import BaseMemory
from just_agents.just_profile import JustAgentProfile
from just_agents.rotate_keys import RotateKeys
from just_agents.streaming.protocol_factory import StreamingMode, ProtocolAdapterFactory

class BaseAgent(
    JustAgentProfile,
    IAgentWithInterceptors[
        SupportedMessages, #Input
        SupportedMessages, #Output
        SupportedMessages #StreamingOutput
    ]
):

    llm_options: LLMOptions = Field(
        ...,
        validation_alias="options",
        description="options that will be passed to the LLM, see https://platform.openai.com/docs/api-reference/completions/create for more details")
    backup_options: Optional[LLMOptions] = Field(
        None,
        exclude=True,
        description="options that will be used after we give up with main options, one more completion call will be done with backup options")
    completion_remove_key_on_error: bool = Field(
        True,
        description="In case of using list of keys removing key from the list after error call with this key")
    completion_max_tries: Optional[int]  = Field(
        2, ge=0,
        description="maximum number of completion retries before giving up")
    streaming_method: StreamingMode = Field(
        StreamingMode.openai,
        description="protocol to handle llm format for function calling")

    key_list_path: Optional[str] = Field(
        None,
        exclude=True,
        description="path to text file with list of api keys, one key per line")
    drop_params: bool = Field(
        True,
        description=" drop params from the request, useful for some models that do not support them")

    _on_query : List[QueryListener] = PrivateAttr(default_factory=list)
    _on_response : List[ResponseListener] = PrivateAttr(default_factory=list)

    memory: BaseMemory = Field(default_factory=BaseMemory,
        exclude=True,
        description="memory for the agent that stores previous messages") #is supposed to be IMemory[Role,SupportedMessage]
    
    _protocol: Optional[IProtocolAdapter] = PrivateAttr(None)
    _partial_streaming_chunks: List[BaseModelResponse] = PrivateAttr(default_factory=list)
    _key_getter: Optional[RotateKeys] = PrivateAttr(None)


    def instruct(self, prompt: str): #backward compatibility
        self.memory.add_message({"role": Role.system, "content": prompt})

    def clear_memory(self) -> None:
        self.memory.clear_messages()
        self.instruct(self.system_prompt)

    def deepcopy_memory(self) -> IMemory:
        return self.memory.deepcopy()

    def add_to_memory(self, messages: SupportedMessages) -> None:
        self.memory.add_message(messages)

    def get_last_message(self) -> SupportedMessage:
        return self.memory.last_message

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        if not self._protocol:
            self._protocol = ProtocolAdapterFactory.get_protocol_adapter(
                self.streaming_method,
                execute_functions= lambda calls: self._process_function_calls(calls),
            )

        if self.tools is not None:
            if not self.llm_options.get("tool_choice", None):
                self.llm_options["tool_choice"] = "auto"

        if self.key_list_path is not None:
            self._key_getter = RotateKeys(self.key_list_path)
        if (self._key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
            print("Warning api_key will be rewritten by key_getter. Both are present in llm_options.")

        self.instruct(self.system_prompt)

    def _prepare_options(self, options: LLMOptions):
        opt = options.copy()
        if self.tools is not None:  # populate llm_options based on available tools
            opt["tools"] = [{"type": "function",
                             "function": self.tools[tool].get_litellm_description()} for tool in self.tools]
        return opt

    def _execute_completion(
            self,
            stream: bool
    ) -> Union[AbstractMessage, BaseModelResponse]:
        opt = self._prepare_options(self.llm_options)
        max_tries = self.completion_max_tries
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

    def query_with_currentmemory(self): #former proceed() aka llm_think()
        while True:
            # individual llm call, unpacking the message, processing handlers
            response = self._execute_completion(stream=False)
            msg: AbstractMessage = self._protocol.message_from_response(response)
            self.handle_on_response(msg)
            self.add_to_memory(msg)

            if not self.tools:
               break
            # If there are no tool calls or tools available, exit the loop
            tool_calls = self._protocol.tool_calls_from_message(msg)
            if not tool_calls:
                break
            # Process each tool call if they exist and re-execute query
            self._process_function_calls(tool_calls)

    def streaming_query_with_currentmemory(self, reconstruct = False):
        try:
            self._partial_streaming_chunks.clear()
            while True:
                response = self._execute_completion(stream=True)
                tool_messages: list[AbstractMessage] = []
                for i, part in enumerate(response):
                    self._partial_streaming_chunks.append(part)
                    msg: AbstractMessage = self._protocol.message_from_delta(response)
                    delta = self._protocol.content_from_delta(msg)
                    if delta:
                        if reconstruct:
                            yield self._protocol.get_chunk(i, delta, options={'model': part["model"]})
                        else:
                            yield response
                    if self.tools:
                        tool_calls = self._protocol.tool_calls_from_message(msg)
                        if tool_calls:
                            self.add_to_memory(
                                self._protocol.function_convention.reconstruct_tool_call_message(tool_calls)
                            )
                            self._process_function_calls(tool_calls)
                            tool_messages.append(self._process_function_calls(tool_calls))
                if not tool_messages:
                    break

        finally:
            yield self._protocol.done()
            if len(self._partial_streaming_chunks) > 0:
                response = self._protocol.response_from_deltas(self._partial_streaming_chunks)
                msg: AbstractMessage = self._protocol.message_from_response(response)
                self.handle_on_response(msg)
                self.add_to_memory(msg)
            self._partial_streaming_chunks.clear()


    def query(self, query_input: SupportedMessages) -> str: #remembers query in handler, executes query and returns str
        self.handle_on_query(query_input)
        self.add_to_memory(query_input)
        self.query_with_currentmemory()
        return self.memory.last_message_str()

    def stream(self, query_input: SupportedMessages, reconstruct = False ) \
            -> Generator[Union[BaseModelResponse, AbstractMessage],None,None]:
        self.handle_on_query(query_input)
        self.add_to_memory(query_input)
        return self.streaming_query_with_currentmemory( reconstruct = reconstruct )






