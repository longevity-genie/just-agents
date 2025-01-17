from typing import Union
import requests

from semanticscholar import SemanticScholar
from semanticscholar.PaginatedResults import PaginatedResults

def get_semantic_paper(query: str,
                       year: str = None,
                       publication_types: list = None,
                       open_access_pdf: bool = None,
                       venue: list = None,
                       fields_of_study: list = None,
                       fields: list = None,
                       publication_date_or_year: str = None,
                       min_citation_count: int = None,
                       limit: int = 100,
                       bulk: bool = False,
                       sort: str = None):
    """
    Search for academia papers by keyword.

        :param str query: plain-text search query string.
        :param str year: (optional) restrict results to the given range of \
               publication year.
        :param list publication_types: (optional) restrict results to the given \
               publication type list.
        :param bool open_access_pdf: (optional) restrict results to papers \
               with public PDFs.
        :param list venue: (optional) restrict results to the given venue list.
        :param list fields_of_study: (optional) restrict results to given \
               field-of-study list, using the s2FieldsOfStudy paper field.
        :param list fields: (optional) list of the fields to be returned.
        :param str publication_date_or_year: (optional) restrict results to \
               the given range of publication date in the format \
               <start_date>:<end_date>, where dates are in the format \
               YYYY-MM-DD, YYYY-MM, or YYYY.
        :param int min_citation_count: (optional) restrict results to papers \
               with at least the given number of citations.
        :param int limit: (optional) maximum number of results to return \
               (must be <= 100).
        :param bool bulk: (optional) bulk retrieval of basic paper data \
               without search relevance (ignores the limit parameter if True \
               and returns up to 1,000 results in each page).
        :param str sort: (optional) sorts results (only if bulk=True) using \
               <field>:<order> format, where "field" is either paperId, \
               publicationDate, or citationCount, and "order" is asc \
               (ascending) or desc (descending).
        :returns: query results
    """
    sch = SemanticScholar()
    results: PaginatedResults = sch.search_paper(query,
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


def hybrid_search(text: str,
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


def literature_search(query: str, limit: int = 10):
    """
    Search in the academic literature
    :param query:
    :param limit: limit number of results
    :return: extracts from scientific papers where SOURCE field is the id of the paper
    """
    return hybrid_search(text=query, limit=limit)
