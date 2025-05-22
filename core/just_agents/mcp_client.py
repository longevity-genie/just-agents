from typing import Any, Dict, List, Optional, Literal
from urllib.parse import urlparse
from weakref import WeakValueDictionary

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool as MCPTool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from pydantic import BaseModel, Field, PrivateAttr
from just_agents.data_classes import ToolDefinition

# Global client registry - uses weak references to allow garbage collection when not in use
_mcp_clients: WeakValueDictionary = WeakValueDictionary()

class MCPServerParameters(BaseModel):
    """
    Parameters for an MCP server.
    """
    mcp_sse_endpoint: Optional[str] = Field(None, description="The MCP endpoint URL for SSE mode")
    mcp_stdio_command: Optional[List[str]] = Field(None, description="The command with arguments to run for stdio mode")

class MCPToolInvocationResult(BaseModel):
    content: str = Field(..., description="Result content as a string")
    error_code: int = Field(..., description="Error code (0 for success, 1 for error)")


def get_mcp_client_by_inputs(server_params: MCPServerParameters) -> 'MCPClient':
    """
    Factory function to get (or create) an MCPClient instance based on connection parameters.
    Uses a global registry to reuse existing clients when possible.
    
    Args:
        server_params: MCPServerParameters object containing connection parameters
        
    Returns:
        An MCPClient instance (either existing or newly created)
        
    Raises:
        ValueError: If no valid connection parameters are provided in server_params
    """
    # Create a unique key based on the connection parameters
    if server_params.mcp_sse_endpoint:
        client_key = f"sse:{server_params.mcp_sse_endpoint}"
    elif server_params.mcp_stdio_command:
        command_str = " ".join(server_params.mcp_stdio_command)
        client_key = f"stdio:{command_str}"
    else:
        raise ValueError("Either mcp_sse_endpoint or mcp_stdio_command must be provided in server_params")
    
    # Return existing client if available
    if client_key in _mcp_clients:
        return _mcp_clients[client_key]
    
    # Create new client
    client = MCPClient(**server_params.model_dump())
    _mcp_clients[client_key] = client
    return client


class MCPClient(MCPServerParameters):
    """Client for interacting with Model Context Protocol (MCP) endpoints using either SSE or STDIO."""

    # Private attributes using Pydantic 2 PrivateAttr
    _mode: str = PrivateAttr(default="")
    _endpoint: Optional[str] = PrivateAttr(default=None)
    _stdio_params: Optional[StdioServerParameters] = PrivateAttr(default=None)
    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _connection_context_manager: Optional[Any] = PrivateAttr(default=None)
    _client_session_context_manager: Optional[Any] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        # Determine connection mode based on parameters
        if self.mcp_sse_endpoint:
            self._mode = "sse"
            self._endpoint = self.mcp_sse_endpoint
            if urlparse(self._endpoint).scheme not in ("http", "https"):
                raise ValueError(f"Endpoint {self._endpoint} is not a valid HTTP(S) URL for SSE mode")
            self._stdio_params = None
        elif self.mcp_stdio_command:
            self._mode = "stdio"
            self._endpoint = None
            if not self.mcp_stdio_command or len(self.mcp_stdio_command) < 1:
                raise ValueError("stdio_command must contain at least the command to execute")
            command = self.mcp_stdio_command[0]
            args = self.mcp_stdio_command[1:] if len(self.mcp_stdio_command) > 1 else []
            self._stdio_params = StdioServerParameters(command=command, args=args)
        else:
            raise ValueError("Either mcp_sse_endpoint or mcp_stdio_command must be provided")

    async def _connect(self) -> None:
        """Establishes the connection and initializes the MCP session."""
        if self._session and self._session.connected:
            return # Already connected

        if self._mode == "sse":
            self._connection_context_manager = sse_client(self._endpoint)
        elif self._mode == "stdio":
            self._connection_context_manager = stdio_client(self._stdio_params)
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")

        streams_tuple = await self._connection_context_manager.__aenter__()
        
        self._client_session_context_manager = ClientSession(*streams_tuple)
        self._session = await self._client_session_context_manager.__aenter__()
        
        await self._session.initialize()

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
        
        self._session = None
        self._client_session_context_manager = None
        self._connection_context_manager = None

    async def __aenter__(self) -> "MCPClient":
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close()

    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> MCPToolInvocationResult:
        """Invoke a specific tool with parameters, returning a ToolInvocationResult."""
        if not self._session or not self._session.connected:
            await self._connect()
        if not self._session: # Should be set by _connect
            raise RuntimeError("MCPClient session not established after connect attempt.")
        
        mcp_call_result = await self._session.call_tool(tool_name, kwargs)

        content_str = ""
        if mcp_call_result.content:
            # Ensure each part is serialized to JSON string as per original intent for ToolInvocationResult
            content_items = [part.model_dump_json(exclude_none=True) for part in mcp_call_result.content]
            content_str = "\\n".join(content_items)
        
        return MCPToolInvocationResult(
            content=content_str,
            error_code=1 if mcp_call_result.isError else 0,
        )

    async def _fetch_tool_info(self) -> List[MCPTool]:
        """
        Connects to MCP and retrieves raw information about all tools.
        
        Returns:
            List[MCPTool]: Raw tool information from MCP
        """
        if not self._session or not self._session.connected:
            await self._connect()
        if not self._session: # Should be set by _connect
            raise RuntimeError("MCPClient session not established after connect attempt.")
        
        tools_result = await self._session.list_tools() # tools_result is mcp.types.ListToolsResult
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
