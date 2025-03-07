from abc import ABC, abstractmethod
import ast
import json
import re
from typing import Type, Union, Generator, AsyncGenerator, Any, TypeVar, Generic, List, Optional, Callable, Coroutine, \
    Protocol, ParamSpec, ParamSpecArgs, ParamSpecKwargs
from pydantic import BaseModel, ConfigDict
import sys

# Define generic types for inputs and outputs
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar("Self")

AbstractQueryInputType = TypeVar("AbstractQueryInputType")
AbstractQueryResponseType = TypeVar("AbstractQueryResponseType")
AbstractStreamingChunkType = TypeVar("AbstractStreamingChunkType")

AbstractAgentInputType = TypeVar('AbstractAgentInputType', bound=BaseModel)
AbstractAgentOutputType = TypeVar('AbstractAgentOutputType', bound=BaseModel)

# Define the type that represents streaming responses
AbstractStreamingGeneratorResponseType = Union[
    Coroutine[Any, Any, AbstractQueryResponseType],
    Coroutine[Any, Any, AsyncGenerator[AbstractStreamingChunkType, None]],
    Generator[AbstractStreamingChunkType, None ,None],
    AsyncGenerator[AbstractStreamingChunkType, None]
]

# Signature for a query function
ResponseFunction = Callable[...,AbstractQueryResponseType]
StreamingResponseFunction = Callable[...,AbstractStreamingGeneratorResponseType]

class IAgent(ABC, Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType]):

    @abstractmethod
    def query(self, query_input: AbstractQueryInputType, **kwargs) -> Optional[AbstractQueryResponseType]:
        raise NotImplementedError("You need to implement query() abstract method first!")
    
    @abstractmethod
    def stream(self, query_input: AbstractQueryInputType) -> Optional[AbstractStreamingGeneratorResponseType]:
        raise NotImplementedError("You need to implement stream() abstract method first!")

    def _clean_fallback_result(self, raw: str) -> str:
        """
        Remove any markdown code fences from the fallback JSON response.
        This regex checks if the entire string is enclosed within triple-backticks with an
        optional "json" language tag, and if so, returns just the content.
        """
        raw = raw.strip()
        pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
        match = pattern.match(raw)
        if match:
            return match.group(1)
        return raw

    
    def query_structural(
        self, 
        query_input: AbstractQueryInputType, 
        parser: Type[BaseModel] = BaseModel,
        response_format: Optional[str] = None,
        enforce_validation: bool = False,
        **kwargs
    ) -> Union[dict, BaseModel]:
        """
        Query the agent and parse the response according to the provided parser.
        Attempts multiple parsing strategies in the following order:
        1. Direct dict usage if already a dict
        2. Clean markdown code fences if present
        3. Standard json parsing
        4. AST literal eval as final fallback
        
        If no response_format is provided and parser is a Pydantic model,
        automatically generates and sends a JSON schema to guide the response format.
        """
        
        # Check if we should auto-generate response format from parser
        if response_format is None and parser is not dict and issubclass(parser, BaseModel) and enforce_validation:
            schema = self._get_response_schema(parser)
            
            # Check if this is a Gemini model to add enforce_validation
            #provider = getattr(self.llm_options, "provider", None) if hasattr(self, "llm_options") else None
            #model_name = getattr(self.llm_options, "model", "") if hasattr(self, "llm_options") else ""
            
            # response_format_obj = {"type": "json_object", "response_schema": schema}
            #response_format_obj["enforce_validation"] = enforce_validation

            schema_wrapper = {
                "name": "query_structural", 
                "schema": schema,
                "strict": enforce_validation
            }

            response_format_obj = {"type": "json_schema", "json_schema": schema_wrapper}
            
            #TODO: 400 response for strict, investigate, maybe litellm bug
            #if enforce_validation:
            #    response_format_obj["strict"] = True
            
           #  response_format = json.dumps(response_format_obj)
        
        # raw_response = self.query(query_input, response_format=response_format_obj, **kwargs)
        raw_response = self.query(query_input, response_format=parser, **kwargs)

        # If already a dict, no parsing needed
        if isinstance(raw_response, dict):
            return raw_response if parser == dict else parser.model_validate(raw_response)

        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        parsing_errors = []
        
        # Try standard json first on the raw response
        try:
            response_dict = json.loads(raw_response)
            return response_dict if parser == dict else parser.model_validate(response_dict)
        except json.JSONDecodeError as e:
            parsing_errors.append(f"Standard JSON parsing failed: {str(e)}")
            
            # Only clean markdown code blocks if initial parsing failed
            cleaned_response = self._clean_fallback_result(raw_response)

            # Check for and fix common double-escaping issues
            if '\\\\' in cleaned_response:
                cleaned_response = cleaned_response.replace('\\\\', '\\')
                
            # Try parsing the cleaned response
            try:
                response_dict = json.loads(cleaned_response)
                return response_dict if parser == dict else parser.model_validate(response_dict)
            except json.JSONDecodeError as e:
                parsing_errors.append(f"Cleaned JSON parsing failed: {str(e)}")

            # Final fallback: AST literal eval
            try:
                response_dict = ast.literal_eval(cleaned_response)
                if isinstance(response_dict, dict):
                    return response_dict if parser == dict else parser.model_validate(response_dict)
                parsing_errors.append("AST parsing succeeded but result was not a dict")
            except (ValueError, SyntaxError) as e:
                parsing_errors.append(f"AST literal_eval parsing failed: {str(e)}")

        # If all parsing attempts failed, raise detailed error
        raise ValueError(
            f"Failed to parse response using multiple methods:\n"
            f"{chr(10).join(parsing_errors)}\n"
            f"Original response:\n{raw_response}\n"
            f"Cleaned response:\n{cleaned_response}"
        )

    def _get_response_schema(self, parser: Type[BaseModel]) -> dict:
        """
        Extract JSON schema from a Pydantic model for use with LLM response formatting.
        This is an internal helper method used by query_structural.
        
        Args:
            parser: A Pydantic model class to extract schema from
            
        Returns:
            A dictionary representing the JSON schema
        """
        # Get the JSON schema from the Pydantic model
        schema = parser.model_json_schema()
        
        def clean_schema_recursively(schema_obj):
            """Recursively remove Pydantic-specific fields that cause issues with LLM APIs"""
            if isinstance(schema_obj, dict):
                # Remove problematic fields at current level
                for field in ["default", "title", "$defs"]:
                    if field in schema_obj:
                        del schema_obj[field]
                
                # Handle nested properties
                if "properties" in schema_obj:
                    for prop_name, prop_value in schema_obj["properties"].items():
                        clean_schema_recursively(prop_value)
                
                # Handle items in arrays
                if "items" in schema_obj:
                    clean_schema_recursively(schema_obj["items"])
                
                # Handle additional properties
                if "additionalProperties" in schema_obj and isinstance(schema_obj["additionalProperties"], dict):
                    clean_schema_recursively(schema_obj["additionalProperties"])
        
        # Clean the schema recursively
        clean_schema_recursively(schema)
        
        # Make sure required fields are specified for OpenAI models
        if "properties" in schema:
            # If required field doesn't exist, create it
            if "required" not in schema:
                schema["required"] = []
                # schema["required"] = [
                #     field_name for field_name, field in schema["properties"].items() 
                #     if not field.get("nullable", False) and not "None" in str(field.get("type", ""))
                # ]
            # Get all property names
            all_properties = list(schema["properties"].keys())
            
            # For OpenAI models, we need to include all properties in the required array
            # even if they're optional in the Pydantic model
            for field_name in all_properties:
                if field_name not in schema["required"]:
                    schema["required"].append(field_name)
        
        return schema


VariArgs = ParamSpec('VariArgs')

# Signatures for listener templates
class QueryListener(Protocol[AbstractQueryInputType]):
    def __call__(self, input_query: AbstractQueryInputType, action: str, source: str, *args:VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        ...

class ResponseListener(Protocol[AbstractQueryResponseType]):
    def __call__(self, response: AbstractQueryResponseType, action: str, source: str) -> None:
        ...

# Define IAgentWithInterceptors with methods to manage interceptors
class IAgentWithInterceptors(
        IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
        ABC,
        Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType]
    ):

    _on_query: List[QueryListener[AbstractQueryInputType]]
    _on_response: List[ResponseListener[AbstractQueryResponseType]]

    # Methods to manage on_query listeners
    def handle_on_query(self, input_query: AbstractQueryResponseType, *args:VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        for handler in self._on_query:
            handler(input_query, *args, **kwargs)

    def add_on_query_listener(self, listener: QueryListener[AbstractQueryInputType]) -> None:
        self._on_query.append(listener)

    def remove_on_query_listener(self, listener: QueryListener[AbstractQueryInputType]) -> None:
        self._on_query.remove(listener)

    # Methods to manage on_response listeners
    def handle_on_response(self, query_result: AbstractQueryResponseType, *args, **kwargs) -> None:
        for handler in self._on_response:
            handler(query_result, *args, **kwargs)

    def add_on_response_listener(
            self,
            listener: ResponseListener[AbstractQueryResponseType]
    ) -> None:
        self._on_response.append(listener)

    def remove_on_response_listener(
            self,
            listener: ResponseListener[AbstractQueryResponseType]
    ) -> None:
        self._on_response.remove(listener)
