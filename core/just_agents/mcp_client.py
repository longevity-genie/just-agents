from typing import Any, Dict, List, Optional, Union, cast

from pathlib import Path
import asyncio
import shlex
import threading
import json

from mcp.types import Tool as MCPTool, TextContent, ImageContent, EmbeddedResource
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.exceptions import ToolError

from pydantic import BaseModel, Field, PrivateAttr, AnyUrl, ValidationError
from just_agents.data_classes import ToolDefinition
from just_agents.just_bus import SingletonMeta
from just_agents.just_locator import JustLocator, generate_unique_name
from just_agents.data_classes import JustMCPServerParameters, MCPServersConfig

AcceptedMCPClientConfig = Union[AnyUrl, str, MCPServersConfig, Path]

class MCPToolInvocationResult(BaseModel):
    content: str = Field(..., description="Result content as a string")
    error_code: int = Field(..., description="Error code (0 for success, 1 for error)")

class MCPClientLocator(JustLocator['MCPClient'], metaclass=SingletonMeta):
    """
    A singleton registry for MCP clients.
    Manages the registration and lookup of MCP clients by their connection parameters.
    """
    def __init__(self) -> None:
        """Initialize the MCP client locator with empty registries."""
        # Initialize the parent JustLocator with 'client_key' as the config identifier attribute
        super().__init__(entity_config_identifier_attr="client_key")
        # Dedicated background loop for all MCP work
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready: threading.Event = threading.Event()
        self._start_loop_thread()

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return

        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass
                loop.close()

        self._loop_thread = threading.Thread(target=_worker, daemon=True, name="MCPBackgroundLoop")
        self._loop_thread.start()
        self._loop_ready.wait()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the dedicated MCP background asyncio loop."""
        # Ensure loop is started
        if self._loop is None:
            self._start_loop_thread()
        assert self._loop is not None
        return self._loop
    
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



class MCPClient(JustMCPServerParameters):
    """Client for interacting with Model Context Protocol (MCP) endpoints using either SSE or STDIO."""

    # Client key for locator identification
    client_key: str = Field(..., description="Unique key identifying this client's connection parameters")

    # Private attributes using Pydantic 2 PrivateAttr
    _transport_spec: AcceptedMCPClientConfig = PrivateAttr(default=None)
    _client: Optional[Client] = PrivateAttr(default=None)
    _client_context_manager: Optional[Any] = PrivateAttr(default=None)
    _client_event_loop: Optional[Any] = PrivateAttr(default=None)
    # Per-client dedicated loop/thread
    _loop: Optional[asyncio.AbstractEventLoop] = PrivateAttr(default=None)
    _loop_thread: Optional[threading.Thread] = PrivateAttr(default=None)
    _loop_ready: Optional[threading.Event] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        # Determine transport specification based on parameters
        if self.mcp_client_config:
            self._transport_spec = self._parse_mcp_client_config(self.mcp_client_config)
        else:
            raise ValueError("mcp_client_config must be provided")
        # Initialize per-client loop signaling
        if self._loop_ready is None:
            self._loop_ready = threading.Event()
        # Update the config to unified representation
        self.mcp_client_config = self._serialize_transport_spec_for_key(self._transport_spec)

    def get_standardized_client_config(self, serialize_dict: bool = False) -> str:
        """Get the client key for the client.""" 
        reparsed_config = self._serialize_transport_spec_for_key(self._transport_spec)
        if isinstance(self._transport_spec, dict) and serialize_dict:
            return json.loads(reparsed_config)
        return reparsed_config
    
    
    # ---------------- Per-client loop management ----------------
    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return

        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            if self._loop_ready:
                self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass
                loop.close()

        self._loop_thread = threading.Thread(target=_worker, daemon=True, name=f"MCPClientLoop:{self.client_key}")
        self._loop_thread.start()
        if self._loop_ready:
            self._loop_ready.wait()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._start_loop_thread()
        assert self._loop is not None
        return self._loop

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Public accessor for the client's loop."""
        return self._ensure_loop()

    async def _run_on_client_loop(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        # If already on client loop, await directly
        try:
            current = asyncio.get_running_loop()
            if current is loop:
                return await coro
        except RuntimeError:
            pass
        # Schedule on client loop and await from current loop
        cfut = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wrap_future(cfut)
    
    @staticmethod
    def _parse_mcp_client_config(endpoint: Union[str, MCPServersConfig], server_name: Optional[str] = None) -> AcceptedMCPClientConfig:
        """
        Parse the MCP endpoint and return the appropriate transport specification for FastMCP.
        
        Args:
            endpoint: The MCP endpoint string or a multi-server config dictionary
            
        Returns:
            Union[AnyUrl, Dict[str, Any], Path]: Parsed transport specification:
                - AnyUrl: for HTTP/HTTPS URLs
                - Dict[str, Any]: for command specifications (like {"command": "python", "args": ["script.py"]})  
                - Path: for single script file paths
        """
        # Direct dict input is treated as config dict (e.g., {"mcpServers": {...}})
        if isinstance(endpoint, dict):
            if "mcpServers" in endpoint:
                return cast(MCPServersConfig, endpoint)
            else:
                raise ValueError("Invalid MCP client config: must contain 'mcpServers' key")

        endpoint_str: str = endpoint

        # Check if it's a JSON string representing a config dict with mcpServers
        try:
            parsed = json.loads(endpoint_str)
            if isinstance(parsed, dict) and "mcpServers" in parsed:
                return cast(MCPServersConfig, parsed)
        except (json.JSONDecodeError, TypeError):
            # Not JSON or not a dict; continue with other checks
            pass
        # Check if it's a URL (HTTP/HTTPS) using Pydantic's AnyUrl
        try:
            url = AnyUrl(endpoint_str)
            return url
        except ValidationError:
            # Not a valid URL, continue to other checks
            pass
        
        # Check if it's a single file path that exists
        endpoint_path = Path(endpoint_str)
        if endpoint_path.exists() and endpoint_path.is_file():
            return endpoint_path
        
        # Check if it's a command string (like "python script.py" or "/path/to/python /path/to/script.py")
        # This is common for STDIO MCP servers
        try:
            # Use shlex to properly split the command while handling quoted arguments
            parts = shlex.split(endpoint_str)
            server_name = server_name or generate_unique_name()
            if len(parts) >= 2:
                # Check if the last part looks like a supported script path
                potential_script = Path(parts[-1])
                supported_extensions = {'.py', '.js'}
                if potential_script.exists() and potential_script.suffix in supported_extensions:
                    # This looks like a command to run a script
                    # Return it as a command dict for FastMCP
                    return cast(MCPServersConfig, {
                        "mcpServers": {
                            server_name: {
                                "command": parts[0],
                                "args": [str(part) for part in parts[1:]]
                            }
                        }
                    })
        except ValueError:
            # shlex.split failed, probably not a valid command string
            pass
        
        # Default to returning as-is (let FastMCP handle it)
        return endpoint_str

    @staticmethod
    def _serialize_transport_spec_for_key(transport_spec: AcceptedMCPClientConfig) -> str:
        """
        Produce a deterministic string key from the parsed transport specification.
        - Paths: absolute resolved path string
        - URLs: canonical string representation
        - Dict configs: fully compact, sorted representation (stable order), including server names
        - Strings: returned as-is
        """
        # Dict config: produce a stable, compact, sorted JSON
        if isinstance(transport_spec, dict):
            return json.dumps(transport_spec, separators=(",", ":"), sort_keys=True)

        # Path: resolve to absolute
        if isinstance(transport_spec, Path):
            try:
                return str(transport_spec.resolve())
            except Exception:
                return str(transport_spec)

        # URL: use its string form
        if isinstance(transport_spec, AnyUrl):
            return str(transport_spec)

        # Default string form
        return str(transport_spec)


    @classmethod
    def get_client_by_inputs(cls, mcp_client_config: Optional[Union[str, MCPServersConfig]] = None, **kwargs) -> 'MCPClient':
        """
        Factory classmethod to get (or create) an MCPClient instance based on connection parameters.
        Uses the module-level MCPClientLocator to reuse existing clients when possible.
        
        Args:
            mcp_client_config: The MCP endpoint (URL for SSE, command for stdio, file path, or config dict)
            **kwargs: Additional parameters (ignored, for compatibility with unpacked objects)
            
        Returns:
            An MCPClient instance (either existing or newly created)
            
        Raises:
            ValueError: If no valid connection parameters are provided
        """
        if not mcp_client_config:
            raise ValueError("mcp_client_config must be provided")
            
        # Parse the endpoint to determine transport type
        transport_spec = cls._parse_mcp_client_config(mcp_client_config)

        # Build a deterministic client key from the parsed transport specification
        client_key = cls._serialize_transport_spec_for_key(transport_spec)

        # Reuse existing client if present (applies to all transports)
        existing_client = _mcp_locator.get_client_by_key(client_key)
        if existing_client:
            return existing_client

        # Create and register new client
        client = cls(client_key=client_key, mcp_client_config=mcp_client_config)
        _mcp_locator.publish_client(client)
        return client

    async def _connect(self) -> None:
        """Establishes the connection and initializes the FastMCP client on its own loop."""
        async def _inner() -> None:
            # Reuse if connected
            if self._client:
                try:
                    if self._client.is_connected():
                        self._client_event_loop = self._loop
                        return
                except Exception:
                    pass
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._client = None
                self._client_context_manager = None
            # Create client
            self._client = Client(self._transport_spec)
            self._client_context_manager = self._client.__aenter__()
            await self._client_context_manager
            self._client_event_loop = self._loop

        await self._run_on_client_loop(_inner())

    async def _close(self) -> None:
        """Closes the FastMCP client and tears down the connection."""
        if self._client:
            async def _inner_close():
                try:
                    await self._client.close()
                except Exception:
                    pass
            try:
                await self._run_on_client_loop(_inner_close())
            except Exception:
                pass

        # Always clean up references
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
        async def _inner() -> MCPToolInvocationResult:
            await self._connect()
            try:
                call_result: CallToolResult = await self._client.call_tool(tool_name, kwargs)
                if call_result.is_error:
                    error_message = "Tool execution failed"
                    if call_result.content:
                        content_items = []
                        for content in call_result.content:
                            if isinstance(content, (TextContent, ImageContent, EmbeddedResource)):
                                content_items.append(content.model_dump_json(exclude_none=True))
                            else:
                                content_items.append(str(content))
                        error_message = "\\n".join(content_items)
                    return MCPToolInvocationResult(content=error_message, error_code=1)
                content_str = ""
                if call_result.content:
                    content_items = []
                    for content in call_result.content:
                        if isinstance(content, (TextContent, ImageContent, EmbeddedResource)):
                            content_items.append(content.model_dump_json(exclude_none=True))
                        else:
                            content_items.append(str(content))
                    content_str = "\\n".join(content_items)
                return MCPToolInvocationResult(content=content_str, error_code=0)
            except ToolError as e:
                return MCPToolInvocationResult(content=str(e), error_code=1)
            except Exception as e:
                return MCPToolInvocationResult(content=f"Connection error: {str(e)}", error_code=1)

        return await self._run_on_client_loop(_inner())

    async def _fetch_tool_info(self) -> List[MCPTool]:
        """
        Connects to MCP and retrieves raw information about all tools.
        
        Returns:
            List[MCPTool]: Raw tool information from MCP
        """
        async def _inner() -> List[MCPTool]:
            await self._connect()
            tools = await self._client.list_tools()
            return tools
        return await self._run_on_client_loop(_inner())

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
