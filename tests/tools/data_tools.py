from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Literal
import os
from pydantic import BaseModel, Field, RootModel

from just_agents.just_bus import JustLogBus


# Initialize the singleton logger
log_bus = JustLogBus()

class FileInfo(BaseModel):
    """Information about a file in the filesystem.
    
    Attributes:
        type: The type of the item ("file")
        path: Relative path to the file from base directory
        extension: File extension including the dot
        size: Size of the file in bytes
    """
    type: Literal["file"] = "file"
    path: str = Field(..., description="Relative path to the file from base directory")
    extension: str = Field(..., description="File extension including the dot")
    size: int = Field(..., description="Size of the file in bytes")

class DirectoryTree(RootModel):
    """A model representing a directory tree structure.
    
    This is a recursive structure where each key is a filename/directory name
    and the value is either a FileInfo object or another DirectoryTree.
    """
    root: Dict[str, Union["DirectoryTree", FileInfo]] = Field(
        default_factory=dict, 
        description="Directory tree structure"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }

def validate_path_security(path: str, base_dir: str = "tests/data") -> str:
    """Validates that a path is secure and within the allowed base directory.
    
    Args:
        path: The path to validate as a string
        base_dir: The base directory that all paths must be contained within
            
    Returns:
        str: The resolved path as a string if valid
            
    Raises:
        ValueError: If the path attempts to access files outside the base directory
    """
    # Convert strings to Path objects
    base_dir_path = Path(base_dir)
    
    # Handle both absolute and relative paths
    if os.path.isabs(path):
        path_obj = Path(path)
    else:
        path_obj = (base_dir_path / path)
    
    # Resolve to absolute path
    resolved_path = path_obj.resolve()
    
    # Security check - ensure we're not escaping the base directory
    if not str(resolved_path).startswith(str(base_dir_path)):
        raise ValueError(f"Security error: Cannot access paths outside {base_dir}")
    
    # Return string representation instead of Path object
    return str(resolved_path)


def read_file(file_path: str) -> str:
    """Read content from a single file.
    
    Args:
        file_path: Path to the file to read as a string
        
    Returns:
        str: Content of the file
        
    Raises:
        ValueError: If the file path is outside the allowed directory
        FileNotFoundError: If the file doesn't exist
    """
    base_dir = "tests/data"
    
    try:
        # Validate path security
        secure_path_str = validate_path_security(file_path, base_dir)
        secure_path = Path(secure_path_str)  # Convert back to Path for internal use
        
        # Check if file exists
        if not secure_path.exists() or not secure_path.is_file():
            log_bus.log_message(
                f"File not found: {secure_path_str}",
                source="data_tools.read_file",
                action="file_check",
                path=file_path,
                resolved_path=secure_path_str
            )
            
            # Raise FileNotFoundError with clear message
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read and return content
        content = secure_path.read_text(encoding='utf-8')
        return content
        
    except ValueError as e:
        # Re-raise security errors
        log_bus.log_message(
            f"Security validation error: {str(e)}",
            source="data_tools.read_file",
            action="security_validation",
            error=str(e),
            path=file_path
        )
        raise e


def list_files(
    show_all: bool = False,
    subdir: Optional[str] = None,
    as_json: bool = True
) -> Union[List[str], Dict[str, Any]]:
    """Lists files in the data directory with various filtering options.
    
    Args:
        show_all: If True, returns all files regardless of extension.
            If False, returns only text files (*.txt, *.md, and *.csv).
        subdir: Optional subdirectory path to filter results as a string. 
            If provided, only files in this subdirectory will be returned.
        as_json: If True, returns a dictionary with the full directory tree.
            If False, returns a flat list of file paths as strings.
    
    Returns:
        Union[List[str], Dict[str, Any]]: Either a list of file paths as strings
            or a dictionary representing the directory structure.
            
    Raises:
        ValueError: If the provided subdir attempts to access files outside /app/data.
    """
    # Base directory
    base_dir = "tests/data"
    base_dir_path = Path(base_dir)
    
    # Handle subdir if provided
    if subdir:
        try:
            # Validate path security and get resolved path
            subdir_path_str = validate_path_security(subdir, base_dir)
            subdir_path = Path(subdir_path_str)  # Convert back to Path for internal use
            
            # Check if the subdirectory exists
            if not subdir_path.exists() or not subdir_path.is_dir():
                log_bus.log_message(
                    f"Subdirectory not found: {subdir_path_str}",
                    source="data_tools.list_files",
                    action="directory_check",
                    subdir=subdir,
                    resolved_path=subdir_path_str
                )
                return [] if not as_json else {"error": f"Subdirectory not found: {subdir}"}
            
            root_dir = subdir_path
        except ValueError as e:
            # Return empty result for security errors
            log_bus.log_message(
                f"Security error: {str(e)}",
                source="data_tools.list_files",
                action="security_validation",
                error=str(e),
                subdir=subdir
            )
            return [] if not as_json else {"error": str(e)}
    else:
        root_dir = base_dir_path
    
    # If we just want a flat list of files
    if not as_json:
        if show_all:
            # Return strings instead of Path objects
            return [str(p) for p in root_dir.glob("**/*") if p.is_file()]
        else:
            # Include txt, md, and csv files
            file_list = []
            file_list.extend([str(p) for p in root_dir.glob("**/*.txt") if p.is_file()])
            file_list.extend([str(p) for p in root_dir.glob("**/*.md") if p.is_file()])
            file_list.extend([str(p) for p in root_dir.glob("**/*.csv") if p.is_file()])
            return file_list
    
    # Build a tree structure for JSON output
    def build_tree(directory: Path) -> Dict[str, Any]:
        """Helper function to recursively build directory tree."""
        result = {}
        
        for path in directory.iterdir():
            rel_path = str(path.relative_to(base_dir_path))
            
            if path.is_dir():
                result[path.name] = build_tree(path)
            elif show_all or path.suffix.lower() in ['.txt', '.md', '.csv']:
                result[path.name] = {
                    "type": "file",
                    "path": rel_path,
                    "extension": path.suffix,
                    "size": path.stat().st_size
                }
        
        return result
    
    # Build and return the tree
    tree = build_tree(root_dir)
    return tree
