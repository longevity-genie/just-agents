from typing import Any, Dict, List, Optional, Literal
from urllib.parse import urlparse
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool as MCPTool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from pydantic import BaseModel, Field, PrivateAttr
from just_agents.data_classes import ToolDefinition
from just_agents.just_bus import SingletonMeta
from just_agents.just_locator import JustLocator


class MCPClientLocator(JustLocator['MCPClient'], metaclass=SingletonMeta):
    """
    A singleton registry for MCP clients.
    Manages the registration and lookup of MCP clients by their connection parameters.
    """
    def __init__(self) -> None:
        """Initialize the MCP client locator with empty registries."""
        # Initialize the parent JustLocator with 'client_key' as the config identifier attribute
        super().__init__(entity_config_identifier_attr="client_key")
    
    def publish_client(self, client: 'MCPClient') -> str:
        """
        Register an MCP client with the locator. 
        
        Args:
            client: The MCP client instance to register
            
        Returns:
            str: The unique codename assigned to the client
            
        Raises:
            ValueError: If the client doesn't have a client_key attribute
        """
        # Check if client has client_key attribute
        if not hasattr(client, 'client_key'):
            raise ValueError("MCP client must have a 'client_key' attribute to be published")
        
        client_key = getattr(client, 'client_key')
        if not client_key:
            raise ValueError("MCP client's 'client_key' cannot be empty")
            
        # Delegate to parent implementation
        return self.publish_entity(client)
    
    def get_client_by_key(self, client_key: str) -> Optional['MCPClient']:
        """
        Get the first MCP client with the given client key.

        Args:
            client_key: The client key to match
            
        Returns:
            Optional[MCPClient]: The first matching client instance, or None if not found
        """
        clients = self.get_entities_by_config_identifier(client_key)
        return clients[0] if clients else None
    
    def get_all_clients(self) -> List['MCPClient']:
        """
        Get all MCP clients in the locator.
        This is primarily a debug/inspection method.
        
        Returns:
            List[MCPClient]: A list of all MCP client instances
        """
        return self.get_all_entities()
    
    async def cleanup_dead_clients(self) -> None:
        """
        Clean up clients with dead sessions to prevent memory leaks.
        Should be called periodically in long-running applications.
        """
        dead_clients = []
        
        for client in self.get_all_entities():
            if client._session:
                try:
                    # Check if session is bound to a closed loop
                    session_loop = getattr(client._session, '_receive_loop', None)
                    if session_loop and session_loop.is_closed():
                        dead_clients.append(client)
                        continue
                    
                    # For STDIO connections, check if process is dead
                    if hasattr(client._session, '_write_stream') and hasattr(client._session._write_stream, '_transport'):
                        transport = client._session._write_stream._transport
                        if hasattr(transport, 'get_returncode') and transport.get_returncode() is not None:
                            dead_clients.append(client)
                except (AttributeError, OSError):
                    # Any error accessing session suggests it's dead
                    dead_clients.append(client)
        
        # Remove dead clients and close their sessions
        for client in dead_clients:
            try:
                await client._close()
            except Exception:
                pass  # Ignore errors during cleanup
            self.unregister_entity(client)


# Module-level singleton locator instance
_mcp_locator = MCPClientLocator()


class MCPServerParameters(BaseModel):
    """
    Parameters for an MCP server.
    """
    mcp_sse_endpoint: Optional[str] = Field(None, description="The MCP endpoint URL for SSE mode")
    mcp_stdio_command: Optional[List[str]] = Field(None, description="The command with arguments to run for stdio mode")

class MCPToolInvocationResult(BaseModel):
    content: str = Field(..., description="Result content as a string")
    error_code: int = Field(..., description="Error code (0 for success, 1 for error)")





class MCPClient(MCPServerParameters):
    """Client for interacting with Model Context Protocol (MCP) endpoints using either SSE or STDIO."""

    # Client key for locator identification
    client_key: str = Field(..., description="Unique key identifying this client's connection parameters")

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

    @classmethod
    def get_client_by_inputs(cls, server_params: MCPServerParameters) -> 'MCPClient':
        """
        Factory classmethod to get (or create) an MCPClient instance based on connection parameters.
        Uses the module-level MCPClientLocator to reuse existing clients when possible.
        
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
            # SSE connections are lightweight HTTP - no need for reuse, create fresh each time
            client = cls(client_key=client_key, **server_params.model_dump())
            return client
        elif server_params.mcp_stdio_command:
            command_str = " ".join(server_params.mcp_stdio_command)
            client_key = f"stdio:{command_str}"
        else:
            raise ValueError("Either mcp_sse_endpoint or mcp_stdio_command must be provided in server_params")
        
        # Re-enable client reuse for STDIO connections with improved session lifecycle management
        existing_client = _mcp_locator.get_client_by_key(client_key)
        if existing_client:
            return existing_client
        
        # Create new client with the client_key
        client = cls(client_key=client_key, **server_params.model_dump())
        
        # Register the client with the locator
        _mcp_locator.publish_client(client)
        
        return client

    async def _connect(self) -> None:
        """Establishes the connection and initializes the MCP session."""
        if self._session:
            # Check if the session is still usable by trying to access its event loop
            try:
                current_loop = asyncio.get_running_loop()
                
                # Get the loop that the session is bound to (ClientSession uses _receive_loop)
                session_loop = getattr(self._session, '_receive_loop', None)
                
                # If session has no loop attribute or loop is closed, recreate session
                if session_loop is None:
                    await self._close()
                elif hasattr(session_loop, 'is_closed') and session_loop.is_closed():
                    await self._close()
                elif hasattr(session_loop, '__call__'):
                    # _receive_loop is a method, but calling it creates a coroutine
                    # We should not call it as it's an async method that manages the receive loop
                    # Instead, let's check if the session is still valid in other ways
                    try:
                        # For STDIO connections, check if the underlying process is still alive
                        if hasattr(self._session, '_write_stream') and hasattr(self._session._write_stream, '_transport'):
                            transport = self._session._write_stream._transport
                            if hasattr(transport, 'get_returncode'):
                                returncode = transport.get_returncode()
                                if returncode is not None:
                                    # Process has exited, need to recreate
                                    await self._close()
                                else:
                                    return # Session exists and process is alive
                            else:
                                return # Session exists, assume it's valid
                        else:
                            return # Session exists, assume it's valid
                    except (AttributeError, OSError):
                        # Any error accessing session internals suggests issues, recreate
                        await self._close()
                else:
                    await self._close()
            except (AttributeError, RuntimeError):
                # Any error suggests session issues, recreate to be safe
                await self._close()

        # If we get here and still have a session, we can reuse it
        if self._session:
            return

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
            except Exception as e:
                pass # Ignore errors during close, best effort
        if self._connection_context_manager:
            try:
                await self._connection_context_manager.__aexit__(None, None, None)
            except Exception as e:
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
        # Ensure we have a valid connection, reusing if possible
        await self._connect()
        
        try:
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
        except Exception as e:
            # If we get an error, it might be due to a stale session
            # Try to reconnect once and retry
            await self._close()
            await self._connect()
            
            mcp_call_result = await self._session.call_tool(tool_name, kwargs)

            content_str = ""
            if mcp_call_result.content:
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
        # Ensure we have a valid connection, reusing if possible
        await self._connect()
        
        try:
            tools_result = await self._session.list_tools() # tools_result is mcp.types.ListToolsResult
            return tools_result.tools
        except Exception as e:
            # If we get an error, it might be due to a stale session
            # Try to reconnect once and retry
            await self._close()
            await self._connect()
            
            tools_result = await self._session.list_tools()
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
