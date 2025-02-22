from typing import Union
import requests


import requests


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