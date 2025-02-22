from typing import Optional, Union
import requests
import os

import json
from typing import List

import requests
from semanticscholar import SemanticScholar
from semanticscholar.PaginatedResults import PaginatedResults


def get_semantic_paper(query: str):
    """
    Search for academia papers by keyword.

    :param str query: plain-text search query string.
    """
    sch = SemanticScholar()
    
    # Environment variables for search parameters:
    # SEMANTIC_YEAR: Publication year filter
    year = os.getenv('SEMANTIC_YEAR') or None
    
    # SEMANTIC_PUB_TYPES: Publication types list (comma-separated)
    publication_types = os.getenv('SEMANTIC_PUB_TYPES', '').strip().split(',') if os.getenv('SEMANTIC_PUB_TYPES') and os.getenv('SEMANTIC_PUB_TYPES').strip() else None
    
    # SEMANTIC_OPEN_ACCESS: Restrict to papers with public PDFs (true/false)
    open_access_pdf = True if os.getenv('SEMANTIC_OPEN_ACCESS', '').lower() == 'true' else None
    
    # SEMANTIC_VENUE: Venue list (comma-separated)
    venue = os.getenv('SEMANTIC_VENUE', '').strip().split(',') if os.getenv('SEMANTIC_VENUE') and os.getenv('SEMANTIC_VENUE').strip() else None
    
    # SEMANTIC_FIELDS_OF_STUDY: Fields of study list (comma-separated)
    fields_of_study = os.getenv('SEMANTIC_FIELDS_OF_STUDY', '').strip().split(',') if os.getenv('SEMANTIC_FIELDS_OF_STUDY') and os.getenv('SEMANTIC_FIELDS_OF_STUDY').strip() else None
    
    # SEMANTIC_FIELDS: Fields to return (comma-separated)
    fields = os.getenv('SEMANTIC_FIELDS', '').strip().split(',') if os.getenv('SEMANTIC_FIELDS') and os.getenv('SEMANTIC_FIELDS').strip() else None
    
    # SEMANTIC_PUB_DATE: Publication date range (format: YYYY-MM-DD:YYYY-MM-DD)
    publication_date_or_year = os.getenv('SEMANTIC_PUB_DATE') or None
    
    # SEMANTIC_MIN_CITATIONS: Minimum citation count
    try:
        min_citation_count = int(os.getenv('SEMANTIC_MIN_CITATIONS')) if os.getenv('SEMANTIC_MIN_CITATIONS') else None
    except ValueError:
        min_citation_count = None
    
    # SEMANTIC_LIMIT: Maximum results to return (default: 100)
    try:
        limit = int(os.getenv('SEMANTIC_LIMIT', '100'))
    except ValueError:
        limit = 100
    
    # SEMANTIC_BULK: Bulk retrieval mode (true/false)
    bulk = True if os.getenv('SEMANTIC_BULK', '').lower() == 'true' else None
    
    # SEMANTIC_SORT: Sort order (format: field:order)
    sort = os.getenv('SEMANTIC_SORT') or None

    results: PaginatedResults = sch.search_paper(
        query,
        year=year,
        publication_types=publication_types,
        open_access_pdf=open_access_pdf,
        venue=venue,
        fields_of_study=fields_of_study,
        fields=fields,
        publication_date_or_year=publication_date_or_year,
        min_citation_count=min_citation_count,
        limit=limit,
        bulk=bulk,
        sort=sort
    )
    return results



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


def hybrid_opensearch(text: str,
                  collections = ["aging_papers_paragraphs_bge_base_en_v1.5", "aging_papers_paragraphs_specter2"],
                  limit: int = 10,
                  db: str = "https://localhost:9200",
                  verbose: bool = False,
                  host: str = "https://api.longevity-genie.info", string: bool = True) -> Union[str, list[str]]:
    """
    Searching in academic literature. Note: do not change default parameters for db and collections unless there are explicit users or systems instructions

    Parameters:
    - text (str): The text to search for.
    - collections (List[str]): A list of collection names to search within
    - limit (int): The maximum number of results to return.
    - db (str): The database URL to query.
    - verbose (bool): Whether to include verbose output. Default is False.
    - host (str): The host URL for the search endpoint. Default is "https://agingkills.eu".
    - string (str): if result should be concatenated to str and not list

    Returns:
    - List[str]: response from the hybrid search API.
    """
    # Ensure all parameters have valid values even if None is passed
    collections = collections or ["aging_papers_paragraphs_bge_base_en_v1.5", "aging_papers_paragraphs_specter2"]
    limit = limit if isinstance(limit, int) and limit > 0 else 10
    db = db or "https://localhost:9200"
    host = host or "https://api.longevity-genie.info"

    # Endpoint for the hybrid search
    endpoint = f"{host}/hybrid_search"

    # Payload to send in the POST request
    payload = {
        "text": text,
        "collections": collections,
        "limit": limit,
        "db": db,
        "verbose": verbose
    }

    # Perform the POST request
    response = requests.post(endpoint, json=payload)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Return the JSON response
    results = response.json()
    return ".".join(results).replace("\\n", "\n") if string else results


def literature_opensearch(query: str, limit: int = 10):
    """
    Search in the academic literature
    :param query:
    :param limit: limit number of results
    :return: extracts from scientific papers where SOURCE field is the id of the paper
    """
    return hybrid_opensearch(text=query, limit=limit)




def brave_search(query: str, search_kwargs: dict = None) -> str:
    """
    Query the Brave search engine and return the results as a JSON string.
    
    Args:
        query (str): The search query to look up
        search_kwargs (dict, optional): Additional search parameters to pass to the Brave API
    
    Returns:
        str: A JSON string containing a list of search results, where each result has:
            - title: The title of the webpage
            - link: The URL of the webpage
            - snippet: Combined description and extra snippets from the page
    """
    # BRAVE_API_KEY: API key for Brave Search authentication
    api_key = os.getenv('BRAVE_API_KEY')
    if not api_key:
        raise ValueError("Brave API key must be provided in BRAVE_API_KEY environment variable")

    search_kwargs = search_kwargs or {}
    
    base_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    
    # Prepare search parameters
    params = {"q": query, "extra_snippets": True}
    params.update(search_kwargs)
    
    # Prepare and validate URL
    req = requests.PreparedRequest()
    req.prepare_url(base_url, params)
    if req.url is None:
        raise ValueError("prepared url is None, this should not happen")

    # Make the request
    response = requests.get(req.url, headers=headers)
    if not response.ok:
        raise Exception(f"HTTP error {response.status_code}")

    # Process results
    web_search_results = response.json().get("web", {}).get("results", [])
    final_results = [
        {
            "title": item.get("title"),
            "link": item.get("url"),
            "snippet": " ".join(
                filter(
                    None, [item.get("description"), *item.get("extra_snippets", [])]
                )
            ),
        }
        for item in web_search_results
    ]
    return json.dumps(final_results)
