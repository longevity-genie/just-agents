from typing import Optional

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

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
        :param list publication_type: (optional) restrict results to the given \
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
    results = sch.search_paper(query)
    return results
