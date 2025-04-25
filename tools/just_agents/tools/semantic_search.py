from typing import Optional
import requests
import os

from typing import List

import requests


def list_search_indexes(non_empty: bool = True) -> List[str]:
    """
    Get list of available search indexes.
    
    :param bool non_empty: If True, only return non-empty indexes
    :return: List of index names
    """
    # SEARCH_DB_URL: The database URL to query (default: "http://localhost:9200")
    db = os.getenv('SEARCH_DB_URL') or 'http://localhost:8090'
    
    response = requests.post(
        f"{db}/list_indexes",
        json={"non_empty": non_empty}
    )
    response.raise_for_status()
    
    return response.json()

def semantic_search(
    query: str, 
    index: str, 
    limit: int = 10, 
    semantic_ratio: float = 0.5
) -> list[str]:
    """
    Search for documents using semantic search.

    Args:
        query: The search query
        index: The index to search in
        limit: The maximum number of results to return (default: 10)
        semantic_ratio: The ratio of semantic search to use (0.0 to 1.0, default: 0.5)

    Returns:
        List of matching documents with their metadata

    Raises:
        requests.exceptions.HTTPError: If the server returns an error response
        requests.exceptions.RequestException: If there's a connection error
        ValueError: If semantic_ratio is not between 0 and 1
    """
    db = os.getenv('SEARCH_DB_URL', 'http://localhost:8090')  # Updated default port
    
    # Validate semantic ratio
    if not 0 <= semantic_ratio <= 1:
        raise ValueError("semantic_ratio must be between 0 and 1")

    # Build payload, excluding None values
    payload = {
        "query": query,
        "index": index,
        "limit": limit,
        "semantic_ratio": semantic_ratio
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        response = requests.post(
            f"{db}/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to perform search: {str(e)}") from e
    


def agentic_semantic_search(query: str, 
                            index: Optional[str] = None, 
                            additional_instructions: Optional[str] = None) -> str:
    """
    Perform an advanced search using the RAG agent that can provide contextual answers.

    Args:
        query: The search query.
        index: Optional - The index to search in (takes all indexes if not specified)
        additional_instructions: Optional - Additional instructions to the agent

    Returns:
        A detailed response from the RAG agent incorporating retrieved documents

    Raises:
        requests.exceptions.HTTPError: If the server returns an error response
        requests.exceptions.RequestException: If there's a connection error or other request-related issue
    """
    db = os.getenv("SEARCH_DB_URL", "http://localhost:8090")  # Updated default port to match server
    
    payload = {
        "query": query,
        "index": index,
        "additional_instructions": additional_instructions
    }
    # Remove None values from payload
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.post(
            f"{db}/search_agent",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.text  # Changed from json() since we expect a string response
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to perform search: {str(e)}") from e