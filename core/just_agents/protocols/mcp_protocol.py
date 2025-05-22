from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Union

import litellm

from mcp import ClientSession
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import Tool as MCPTool

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from mcp.types import TextPart

import os
import json
from weakref import WeakValueDictionary

from just_agents.data_classes import ToolDefinition

# Global client registry - uses weak references to allow garbage collection when not in use
_mcp_clients: WeakValueDictionary = WeakValueDictionary()

def get_mcp_client_by_inputs(sse_endpoint: Optional[str] = None, stdio_command: Optional[List[str]] = None) -> 'MCPClient':
    """
    Factory function to get (or create) an MCPClient instance based on connection parameters.
    Uses a global registry to reuse existing clients when possible.
    
    Args:
        sse_endpoint: The MCP endpoint URL for SSE mode
        stdio_command: The command with arguments to run for stdio mode
        
    Returns:
        An MCPClient instance (either existing or newly created)
        
    Raises:
        ValueError: If neither sse_endpoint nor stdio_command is provided
    """
    if sse_endpoint:
        # Create a unique key for the client based on endpoint
        client_key = f"sse:{sse_endpoint}"
        if client_key not in _mcp_clients:
            _mcp_clients[client_key] = MCPClient(mode="sse", endpoint=sse_endpoint)
        return _mcp_clients[client_key]
    
    elif stdio_command:
        if not stdio_command or len(stdio_command) < 1:
            raise ValueError("stdio_command must contain at least the command to execute")
        
        # Create a unique key for the client based on command + args
        command_str = " ".join(stdio_command)
        client_key = f"stdio:{command_str}"
        
        if client_key not in _mcp_clients:
            command = stdio_command[0]
            args = stdio_command[1:] if len(stdio_command) > 1 else []
            server_params = StdioServerParameters(command=command, args=args)
            _mcp_clients[client_key] = MCPClient(mode="stdio", server_params=server_params)
        
        return _mcp_clients[client_key]
    
    else:
        raise ValueError("Either sse_endpoint or stdio_command must be provided")


class ToolParameter(BaseModel):
    name: str = Field(..., description="Parameter name")
    parameter_type: str = Field(..., description='Parameter type (e.g., "string", "number")')
    description: str = Field(..., description="Parameter description")
    required: bool = Field(False, description="Whether the parameter is required")
    default: Any = Field(None, description="Default value for the parameter")


class MCPToolDef(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: List[ToolParameter] = Field(..., description="List of ToolParameter objects")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary of additional metadata")
    identifier: str = Field("", description="Tool identifier (defaults to name)")


class ToolInvocationResult(BaseModel):
    content: str = Field(..., description="Result content as a string")
    error_code: int = Field(..., description="Error code (0 for success, 1 for error)")


class MCPClient:
    """Client for interacting with Model Context Protocol (MCP) endpoints using either SSE or STDIO."""

    def __init__(self, mode: Literal["sse", "stdio"],
                 endpoint: Optional[str] = None,
                 server_params: Optional[StdioServerParameters] = None):
        """Initialize MCP client.

        Args:
            mode: Connection mode, either "sse" or "stdio".
            endpoint: The MCP endpoint URL (required for "sse" mode).
            server_params: StdioServerParameters (required for "stdio" mode).
        """
        self.mode = mode
        self.endpoint = endpoint
        self.server_params = server_params

        if self.mode == "sse":
            if not self.endpoint:
                raise ValueError("Endpoint must be provided for SSE mode")
            if urlparse(self.endpoint).scheme not in ("http", "https"):
                raise ValueError(f"Endpoint {self.endpoint} is not a valid HTTP(S) URL for SSE mode")
        elif self.mode == "stdio":
            if not self.server_params:
                raise ValueError("Server parameters must be provided for STDIO mode")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sse' or 'stdio'.")

        self.session: Optional[ClientSession] = None
        self._connection_context_manager: Optional[Any] = None
        self._client_session_context_manager: Optional[Any] = None

    async def _connect(self) -> None:
        """Establishes the connection and initializes the MCP session."""
        if self.session and self.session.connected:
            return # Already connected

        if self.mode == "sse":
            if not self.endpoint:
                raise ValueError("Endpoint not set for SSE mode")
            self._connection_context_manager = sse_client(self.endpoint)
        elif self.mode == "stdio":
            if not self.server_params:
                raise ValueError("Server parameters not set for STDIO mode")
            self._connection_context_manager = stdio_client(self.server_params)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        streams_tuple = await self._connection_context_manager.__aenter__()
        
        self._client_session_context_manager = ClientSession(*streams_tuple)
        self.session = await self._client_session_context_manager.__aenter__()
        
        await self.session.initialize()

    async def _close(self) -> None:
        """Closes the MCP session and tears down the connection."""
        if self._client_session_context_manager:
            try:
                await self._client_session_context_manager.__aexit__(None, None, None)
            except Exception:
                pass # Ignore errors during close, best effort
        if self._connection_context_manager:
            try:
                await self._connection_context_manager.__aexit__(None, None, None)
            except Exception:
                pass # Ignore errors during close, best effort
        
        self.session = None
        self._client_session_context_manager = None
        self._connection_context_manager = None

    async def __aenter__(self) -> "MCPClient":
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close()

    async def list_tools(self) -> List[MCPToolDef]:
        """List available tools from the MCP endpoint, formatted as ToolDef objects."""
        if not self.session or not self.session.connected:
            await self._connect()
        if not self.session: # Should be set by _connect
            raise RuntimeError("MCPClient session not established after connect attempt.")
        
        tools_data = []
        mcp_tools = await self._fetch_tool_info()
        
        for tool in mcp_tools: # tool is MCPTool
            parameters = []
            required_params = tool.inputSchema.get("required", [])
            for param_name, param_schema in tool.inputSchema.get("properties", {}).items():
                parameters.append(
                    ToolParameter(
                        name=param_name,
                        parameter_type=param_schema.get("type", "string"),
                        description=param_schema.get("description", ""),
                        required=param_name in required_params,
                        default=param_schema.get("default"),
                    )
                )
            
            metadata_info: Dict[str, Any] = {}
            if self.mode == "sse" and self.endpoint:
                metadata_info["endpoint"] = self.endpoint
            elif self.mode == "stdio":
                metadata_info["connection_type"] = "stdio"

            tools_data.append(
                MCPToolDef(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=parameters,
                    metadata=metadata_info if metadata_info else None,
                    identifier=tool.name
                )
            )
        return tools_data

    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult:
        """Invoke a specific tool with parameters, returning a ToolInvocationResult."""
        if not self.session or not self.session.connected:
            await self._connect()
        if not self.session: # Should be set by _connect
            raise RuntimeError("MCPClient session not established after connect attempt.")
        
        mcp_call_result = await self.session.call_tool(tool_name, kwargs)

        content_str = ""
        if mcp_call_result.content:
            # Ensure each part is serialized to JSON string as per original intent for ToolInvocationResult
            content_items = [part.model_dump_json(exclude_none=True) for part in mcp_call_result.content]
            content_str = "\\n".join(content_items)
        
        return ToolInvocationResult(
            content=content_str,
            error_code=1 if mcp_call_result.isError else 0,
        )

    async def load_tools_formatted(self, format: Literal["mcp", "openai"] = "mcp") -> Union[List[MCPTool], List[ChatCompletionToolParam]]:
        """Load all available MCP tools, with option to format for OpenAI."""
        if not self.session:
            raise RuntimeError("MCPClient session not active. Use in 'async with' block.")
        
        mcp_tools_list = await self._fetch_tool_info()

        if format == "openai":
            return [
                MCPClient._transform_mcp_tool_to_openai_tool(mcp_tool=tool) for tool in mcp_tools_list
            ]
        return mcp_tools_list

    async def call_tool_direct(self, params: MCPCallToolRequestParams) -> MCPCallToolResult:
        """Call an MCP tool directly using MCPCallToolRequestParams."""
        if not self.session:
            raise RuntimeError("MCPClient session not active. Use in 'async with' block.")
        
        tool_result = await self.session.call_tool(
            name=params.name,
            arguments=params.arguments,
        )
        return tool_result

    async def call_openai_tool_formatted(self, openai_tool_call_dict: Dict) -> MCPCallToolResult:
        """Call an OpenAI-formatted tool call (as a dict) via MCP."""
        if not self.session:
            raise RuntimeError("MCPClient session not active. Use in 'async with' block.")
        
        mcp_params = MCPClient._transform_openai_tool_call_request_to_mcp_tool_call_request(
            openai_tool_dict=openai_tool_call_dict,
        )
        return await self.call_tool_direct(params=mcp_params)

    @staticmethod
    def _transform_mcp_tool_to_openai_tool(mcp_tool: MCPTool) -> ChatCompletionToolParam:
        """Convert an MCP tool to an OpenAI tool."""
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                parameters=mcp_tool.inputSchema,
                strict=False,
            ),
        )

    @staticmethod
    def _get_function_arguments(function_dict: Dict) -> dict:
        """Helper to safely get and parse function arguments from a dictionary."""
        arguments = function_dict.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return arguments if isinstance(arguments, dict) else {}

    @staticmethod
    def _transform_openai_tool_call_request_to_mcp_tool_call_request(
        openai_tool_dict: Dict,
    ) -> MCPCallToolRequestParams:
        """Convert an OpenAI ChatCompletionMessageToolCall (as dict) to an MCP CallToolRequestParams."""
        function_dict = openai_tool_dict["function"]
        return MCPCallToolRequestParams(
            name=function_dict["name"],
            arguments=MCPClient._get_function_arguments(function_dict),
        )

    async def _fetch_tool_info(self) -> List[MCPTool]:
        """
        Connects to MCP and retrieves raw information about all tools.
        
        Returns:
            List[MCPTool]: Raw tool information from MCP
        """
        if not self.session or not self.session.connected:
            await self._connect()
        if not self.session: # Should be set by _connect
            raise RuntimeError("MCPClient session not established after connect attempt.")
        
        tools_result = await self.session.list_tools() # tools_result is mcp.types.ListToolsResult
        return tools_result.tools

    async def list_tools_openai(self) -> List[ToolDefinition]:
        """
        Lists available tools from the MCP endpoint as ToolDefinition objects.
        
        Returns:
            List[ToolDefinition]: List of tools formatted as ToolDefinition objects
        """
        tools_raw = await self._fetch_tool_info()
        tool_definitions = []
        
        for tool in tools_raw:
            # Convert MCP tool schema to ToolDefinition format
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param_schema in tool.inputSchema.get("properties", {}).items():
                parameters["properties"][param_name] = param_schema
                if param_name in tool.inputSchema.get("required", []):
                    parameters["required"].append(param_name)
            
            tool_definition = ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                parameters=parameters
            )
            tool_definitions.append(tool_definition)
            
        return tool_definitions

    async def get_tool_openai_by_name(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Gets a specific tool from the MCP endpoint as a ToolDefinition object.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            Optional[ToolDefinition]: The tool as a ToolDefinition object, or None if not found
        """
        tool_definitions = await self.list_tools_openai()
        
        for tool in tool_definitions:
            if tool.name == tool_name:
                return tool
                
        return None

# Helper functions (kept at module level, prefixed with _ or as originally named if conventional)

async def test_mcp():
    # This StdioServerParameters is for the test case.
    server_params = StdioServerParameters(
        command="python3",
        # Make sure to update to the full absolute path to your mcp_server.py file
        args=["./mcp_server.py"], 
    )
    async with MCPClient(mode="stdio", server_params=server_params) as client:
        # Initialize the connection is handled by __aenter__

        # Get tools
        # Assuming tools are openai.types.chat.ChatCompletionToolParam objects
        tools: List[ChatCompletionToolParam] = await client.load_tools_formatted(format="openai") # type: ignore
        print("MCP TOOLS: ", [tool.model_dump_json(indent=2) for tool in tools])


        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        # Ensure litellm.acompletion is awaited
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=[tool.model_dump() for tool in tools], # Pass tools as list of dicts if required by litellm
        )
        print("LLM RESPONSE: ", llm_response.model_dump_json(indent=4))

        # Ensure tool_calls is present and not empty before accessing
        if not llm_response.choices[0].message.tool_calls:
            print("No tool calls in LLM response.")
            return

        openai_tool_call_obj = llm_response.choices[0].message.tool_calls[0]
        openai_tool_call_dict = openai_tool_call_obj.model_dump()

        # Call the tool using MCP client
        call_result = await client.call_openai_tool_formatted(
            openai_tool_call_dict=openai_tool_call_dict,
        )
        print("MCP TOOL CALL RESULT: ", call_result.model_dump_json(indent=2))

        # send the tool result to the LLM
        # Assistant's message that included the tool call
        messages.append(llm_response.choices[0].message.model_dump(exclude_none=True))
        
        tool_content_for_llm = ""
        if call_result.content:
            first_part = call_result.content[0]
            # Check if the first part is TextPart and has text
            if isinstance(first_part, TextPart) and first_part.text is not None: # Requires TextPart import
                tool_content_for_llm = first_part.text
            elif hasattr(first_part, "text") and first_part.text is not None: # Fallback if TextPart not imported/checked
                 tool_content_for_llm = first_part.text
            else:
                # Fallback if not a TextPart or text is None, serialize the first part
                tool_content_for_llm = first_part.model_dump_json(exclude_none=True)
        
        messages.append(
            {
                "role": "tool",
                "content": tool_content_for_llm,
                "tool_call_id": openai_tool_call_obj.id,
            }
        )
        print("final messages with tool result: ", messages)
        
        llm_response_final = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=[tool.model_dump() for tool in tools], # Pass tools again
        )
        print(
            "FINAL LLM RESPONSE: ", llm_response_final.model_dump_json(indent=4)
        )