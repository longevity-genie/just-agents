from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import asyncio
import shlex

from mcp.types import Tool as MCPTool, TextContent, ImageContent, EmbeddedResource
from fastmcp import Client
from fastmcp.exceptions import ToolError

from pydantic import BaseModel, Field, PrivateAttr, AnyUrl, ValidationError
from just_agents.data_classes import ToolDefinition
from just_agents.just_bus import SingletonMeta
from just_agents.just_locator import JustLocator, generate_unique_name


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
            if client._client:
                try:
                    # Check if client is connected
                    if not client._client.is_connected():
                        dead_clients.append(client)
                except (AttributeError, OSError):
                    # Any error accessing client suggests it's dead
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
    mcp_endpoint: Optional[str] = Field(None, description="The MCP endpoint (URL for SSE, command for stdio, or file path)")

class MCPToolInvocationResult(BaseModel):
    content: str = Field(..., description="Result content as a string")
    error_code: int = Field(..., description="Error code (0 for success, 1 for error)")





class MCPClient(MCPServerParameters):
    """Client for interacting with Model Context Protocol (MCP) endpoints using either SSE or STDIO."""

    # Client key for locator identification
    client_key: str = Field(..., description="Unique key identifying this client's connection parameters")

    # Private attributes using Pydantic 2 PrivateAttr
    _transport_spec: Union[AnyUrl, Dict[str, Any], Path, str] = PrivateAttr(default=None)
    _client: Optional[Client] = PrivateAttr(default=None)
    _client_context_manager: Optional[Any] = PrivateAttr(default=None)
    _client_event_loop: Optional[Any] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        # Determine transport specification based on parameters
        if self.mcp_endpoint:
            self._transport_spec = self._parse_mcp_endpoint(self.mcp_endpoint)
        else:
            raise ValueError("mcp_endpoint must be provided")
    
    @staticmethod
    def _parse_mcp_endpoint(endpoint: str, server_name: Optional[str] = None) -> Union[AnyUrl, Dict[str, Any], Path]:
        """
        Parse the MCP endpoint and return the appropriate transport specification for FastMCP.
        
        Args:
            endpoint: The MCP endpoint string
            
        Returns:
            Union[AnyUrl, Dict[str, Any], Path]: Parsed transport specification:
                - AnyUrl: for HTTP/HTTPS URLs
                - Dict[str, Any]: for command specifications (like {"command": "python", "args": ["script.py"]})  
                - Path: for single script file paths
        """
        # Check if it's a URL (HTTP/HTTPS) using Pydantic's AnyUrl
        try:
            url = AnyUrl(endpoint)
            return url
        except ValidationError:
            # Not a valid URL, continue to other checks
            pass
        
        # Check if it's a single file path that exists
        endpoint_path = Path(endpoint)
        if endpoint_path.exists() and endpoint_path.is_file():
            return endpoint_path
        
        # Check if it's a command string (like "python script.py" or "/path/to/python /path/to/script.py")
        # This is common for STDIO MCP servers
        try:
            # Use shlex to properly split the command while handling quoted arguments
            parts = shlex.split(endpoint)
            server_name = server_name or generate_unique_name()
            if len(parts) >= 2:
                # Check if the last part looks like a supported script path
                potential_script = Path(parts[-1])
                supported_extensions = {'.py', '.js'}
                if potential_script.exists() and potential_script.suffix in supported_extensions:
                    # This looks like a command to run a script
                    # Return it as a command dict for FastMCP
                    return {
                        "mcpServers": {
                            server_name: {
                                "command": parts[0],
                                "args": [str(part) for part in parts[1:]]
                            }
                        }
                    }
        except ValueError:
            # shlex.split failed, probably not a valid command string
            pass
        
        # Default to returning as-is (let FastMCP handle it)
        return endpoint


    @classmethod
    def get_client_by_inputs(cls, mcp_endpoint: Optional[str] = None, **kwargs) -> 'MCPClient':
        """
        Factory classmethod to get (or create) an MCPClient instance based on connection parameters.
        Uses the module-level MCPClientLocator to reuse existing clients when possible.
        
        Args:
            mcp_endpoint: The MCP endpoint (URL for SSE, command for stdio, or file path)
            **kwargs: Additional parameters (ignored, for compatibility with unpacked objects)
            
        Returns:
            An MCPClient instance (either existing or newly created)
            
        Raises:
            ValueError: If no valid connection parameters are provided
        """
        if not mcp_endpoint:
            raise ValueError("mcp_endpoint must be provided")
            
        # Parse the endpoint to determine transport type
        transport_spec = cls._parse_mcp_endpoint(mcp_endpoint)
        
        # Check if it's HTTP transport (URLs)
        if isinstance(transport_spec, AnyUrl) or (isinstance(transport_spec, str) and transport_spec.startswith(("http://", "https://"))):
            # HTTP/HTTPS endpoints are lightweight - create fresh each time
            client_key = f"sse:{mcp_endpoint}"
            client = cls(client_key=client_key, mcp_endpoint=mcp_endpoint)
            return client
        else:
            # Non-HTTP endpoints (stdio commands, file paths, command dicts) use multiton pattern
            client_key = f"stdio:{mcp_endpoint}"
            
            # Re-enable client reuse for non-HTTP connections
            existing_client = _mcp_locator.get_client_by_key(client_key)
            if existing_client:
                return existing_client
            
            # Create new client with the client_key
            client = cls(client_key=client_key, mcp_endpoint=mcp_endpoint)
            
            # Register the client with the locator
            _mcp_locator.publish_client(client)
            
            return client

    async def _connect(self) -> None:
        """Establishes the connection and initializes the FastMCP client."""
        # Get current event loop at the start to ensure it's available throughout
        current_loop = asyncio.get_running_loop()
        
        if self._client:
            # Check if the client is still usable by trying to access its event loop
            try:
                # If we have a stored event loop, check if it matches current loop
                if hasattr(self, '_client_event_loop') and self._client_event_loop is not None:
                    if self._client_event_loop != current_loop:
                        # Event loop has changed, close old connection
                        await self._close()
                    elif hasattr(self._client_event_loop, 'is_closed') and self._client_event_loop.is_closed():
                        # Event loop is closed, recreate connection
                        await self._close()
                    else:
                        # Event loop is valid, check if client is still connected
                        try:
                            if self._client.is_connected():
                                return # Client is already connected and working
                            else:
                                # Client exists but not connected, clean up
                                await self._close()
                        except (AttributeError, RuntimeError, OSError):
                            # Client has issues, recreate
                            await self._close()
                else:
                    # No stored event loop, check if client is connected
                    try:
                        if self._client.is_connected():
                            # Store current loop for future reference
                            self._client_event_loop = current_loop
                            return
                        else:
                            await self._close()
                    except (AttributeError, RuntimeError, OSError):
                        await self._close()
            except (AttributeError, RuntimeError):
                # Any error suggests client issues, recreate to be safe
                await self._close()
        
        # If we get here and still have a client, we can reuse it
        if self._client:
            return

        # Create new FastMCP client
        self._client = Client(self._transport_spec)
        self._client_context_manager = self._client.__aenter__()
        await self._client_context_manager
        
        # Store the current event loop for future reference
        self._client_event_loop = current_loop

    async def _close(self) -> None:
        """Closes the FastMCP client and tears down the connection."""
        if self._client_context_manager and self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass # Ignore errors during close, best effort
        
        self._client = None
        self._client_context_manager = None
        self._client_event_loop = None

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
            # FastMCP call_tool returns List[TextContent | ImageContent | EmbeddedResource]
            content_list = await self._client.call_tool(tool_name, kwargs)
            
            # Convert content list to string format as expected by MCPToolInvocationResult
            content_str = ""
            if content_list:
                content_items = []
                for content in content_list:
                    if isinstance(content, (TextContent, ImageContent, EmbeddedResource)):
                        content_items.append(content.model_dump_json(exclude_none=True))
                    else:
                        # Fallback for any other content types
                        content_items.append(content.model_dump_json(exclude_none=True))
                content_str = "\\n".join(content_items)
            
            return MCPToolInvocationResult(
                content=content_str,
                error_code=0,  # Success
            )
        except ToolError as e:
            # FastMCP raises ToolError for tool execution errors (wrong inputs, etc.)
            return MCPToolInvocationResult(
                content=str(e),
                error_code=1,  # Error
            )
        except Exception as e:
            # Connection or other unexpected errors
            return MCPToolInvocationResult(
                content=f"Connection error: {str(e)}",
                error_code=1,
            )

    async def _fetch_tool_info(self) -> List[MCPTool]:
        """
        Connects to MCP and retrieves raw information about all tools.
        
        Returns:
            List[MCPTool]: Raw tool information from MCP
        """
        # Ensure we have a valid connection, reusing if possible
        await self._connect()
        
        # FastMCP list_tools returns List[mcp.types.Tool] directly
        tools = await self._client.list_tools()
        return tools

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
