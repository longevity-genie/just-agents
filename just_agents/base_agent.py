
import json
from litellm import ModelResponse, completion

from pydantic import Field, PrivateAttr
from typing import Optional, Dict, List, Any, AsyncGenerator

from just_agents.types import StreamingMode, SupportedMessages, message_from_response
from just_agents.interfaces.IAgent import IAgentWithInterceptors, QueryListener, ResponseListener, \
    StreamingResponseListener, AbstractStreamingGeneratorResponseType

from just_agents.streaming.abstract_streaming import AbstractStreaming
from just_agents.streaming.openai_streaming import AsyncSession

from just_agents.base_memory import BaseMemory
from just_agents.just_profile import JustAgentProfile
from just_agents.rotate_keys import RotateKeys

class BaseAgent(
    JustAgentProfile,
    IAgentWithInterceptors[
        SupportedMessages, #Input
        SupportedMessages, #Output
        AsyncGenerator[Any,None] #StreamingOutput
    ]
):

    llm_options: Dict[str, Any] = Field(
        ...,
        validation_alias="options",
        description="options that will be passed to the LLM, see https://platform.openai.com/docs/api-reference/completions/create for more details")
    completion_remove_key_on_error: bool = Field(
        True,
        description="In case of using list of keys removing key from the list after error call with this key")
    completion_max_tries: Optional[int]  = Field(
        2, ge=0,
        description="maximum number of completion retries before giving up")
    streaming_method: StreamingMode = Field(
        StreamingMode.openai,
        description="protocol to handle llm format for function calling")
    backup_options: Optional[Dict] = Field(
        None,
        exclude=True,
        description="options that will be used after we give up with main options, one more completion call will be done with backup options")
    key_list_path: Optional[str] = Field(
        None,
        exclude=True,
        description="path to text file with list of api keys, one key per line")
    drop_params: bool = Field(
        True,
        description=" drop params from the request, useful for some models that do not support them")

    _on_query : List[QueryListener] = PrivateAttr(default_factory=list)
    _on_response : List[ResponseListener] = PrivateAttr(default_factory=list)
    _on_streaming_response : List[StreamingResponseListener] = PrivateAttr(default_factory=list)

    _conversation: BaseMemory = PrivateAttr(default_factory=BaseMemory)
    _streaming: Optional[AbstractStreaming] = PrivateAttr()
    _key_getter: Optional[RotateKeys] = PrivateAttr(None)

    @property
    def on_query(self) -> List[QueryListener]:
        return self._on_query

    @property
    def on_response(self) -> List[ResponseListener]:
        return self._on_response

    @property
    def on_streaming_response(self) -> List[StreamingResponseListener]:
        return self._on_streaming_response

    def set_memory(self, new_conversation: BaseMemory) -> None:
        """Replace the current memory with a new one."""
        self._conversation = new_conversation

    def instruct(self, prompt: str): #backward compatibility
        self._conversation.add_system_message(prompt)

    def clear_memory(self) -> None:
        self._conversation.clear_messages()
        self.instruct(self.system_prompt)

    def deepcopy_memory(self) -> BaseMemory:
        return self._conversation.deepcopy()

    def add_to_memory(self, messages: SupportedMessages) -> None:
        self._conversation.add_message(messages)

    def _memorize_on_query(self, messages: SupportedMessages, *_args, **_kwargs ) -> None:
        self.add_to_memory(messages)

    def _memorize_on_response(self, response: SupportedMessages, *_args, **_kwargs ) -> None:
        self.add_to_memory(response)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        if self.tools is not None:
            if not self.llm_options.get("tool_choice", None):
                self.llm_options["tool_choice"] = "auto"

        if self.key_list_path is not None:
            self._key_getter = RotateKeys(self.key_list_path)
        if (self._key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
            print("Warning api_key will be rewritten by key_getter. Both are present in llm_options.")

        if self.streaming_method == StreamingMode.openai: #todo: this block is best encapsulated in StreamingMethod class internal logic
            self._streaming = AsyncSession(self)
        elif self.streaming_method == StreamingMode.qwen2:
            from just_agents.streaming.qwen2_streaming import Qwen2AsyncSession
            self._streaming = Qwen2AsyncSession(self)
        elif self.streaming_method == StreamingMode.chain_of_thought:
            from just_agents.streaming.chain_of_thought import ChainOfThought
            self._streaming = ChainOfThought(self)
        else:
            raise ValueError("just_streaming_method is incorrect. "
                             "It should be one of this ['openai', 'qwen2', 'chain_of_thought']")

        #Add listeners hooks to remember queries and replies:
        self.add_on_query_listener(self._memorize_on_query)
        self.add_on_response_listener(self._memorize_on_response)
        #todo: streaming responses memorization

        self.instruct(self.system_prompt)

    def _execute_completion(self, stream: bool) -> ModelResponse: #wrapper of llm call that supplies keys and options
        opt = self.llm_options.copy()
        if self.tools is not None:  # populate llm_options based on available tools
            opt["tools"] = [{"type": "function",
                             "function": self.tools[tool].get_litellm_description()} for tool in self.tools]
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
                    response = completion(messages=self._conversation.messages, stream=stream, **opt)
                    return response
                except Exception as e:
                    last_exception = e
                    if self.completion_remove_key_on_error:
                        self._key_getter.remove(opt["api_key"])

            if self.backup_options:
                return completion(messages=self._conversation.messages, stream=stream, **self.backup_options)
            if last_exception:
                raise last_exception
            else:
                raise Exception(
                    f"Run out of tries to execute completion. Check your keys! Keys {self._key_getter.len()} left.")
        else:
            return completion(messages=self._conversation.messages, stream=stream, **opt)

    @IAgentWithInterceptors.response_handler
    def _atomic_query(self) -> SupportedMessages: # individual llm call, unpacking the message, processing handlers
        response: ModelResponse = self._execute_completion(stream=False)
        return message_from_response(response)

    @IAgentWithInterceptors.response_handler
    def _execute_function_call(self, function_name: str, function_args_jsons: str, fn_id: str):
        try:
            function_to_call = self.tools[function_name].get_callable()
            function_args = json.loads(function_args_jsons)
            function_response = str(function_to_call(**function_args))
        except Exception as e:
            function_response = str(e)
        return {"role": "tool", "content": function_response, "name": function_name, "tool_call_id": fn_id}

    def query_with_current_conversation(self): #former proceed() aka llm_think()
        while True:
            message = self._atomic_query()
            tool_calls = message.get("tool_calls")

            # If there are no tool calls or tools available, exit the loop
            if not tool_calls or self.tools is None:
                break

            # Process each tool call if they exist
            for tool_call in tool_calls:
                _fn_message = self._execute_function_call(
                    tool_call["function"]["name"],
                    tool_call["function"]["arguments"],
                    tool_call["id"]
                )

    @IAgentWithInterceptors.query_handler
    def query(self, query_input: SupportedMessages) -> str: #remembers query in handler, executes query and returns str
        self.query_with_current_conversation()
        return self._conversation.last_message_str()

    @IAgentWithInterceptors.streaming_response_handler
    @IAgentWithInterceptors.query_handler
    def stream(self, query_input: SupportedMessages) -> AbstractStreamingGeneratorResponseType:
        return self.streaming.resp_async_generator()
        ##todo: remove recurrent llm_session, handle tools, handle async, handle response memory





