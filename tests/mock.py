from typing import List

def hybrid_search(query: str) -> str:
    """Mock function for hybrid search."""
    return f"Mocked hybrid search result for query: {query}"

def rsid_lookup(rsid: str) -> str:
    """Retrieves information about rsid. It could be the effects of longevity or its influence on diseases.
     rsid is identifier for single nucleotide polymorphism, and looks like rs123 where instead of 123 could be any integer."""
    return f"{rsid} has pro-longevity influence"

def gene_lookup(gene: str) -> str:
    """Mock function for gene lookup."""
    return f"Mocked information for gene: {gene}"

def pathway_lookup(pathway: str) -> str:
    """Mock function for pathway lookup."""
    return f"Mocked information for pathway: {pathway}"

def disease_lookup(disease: str) -> str:
    """Mock function for disease lookup."""
    return f"Mocked information for disease: {disease}"

def sequencing_info() -> str:
    """Mock function for sequencing info."""
    return "Mocked DNA sequencing information"

def _process_sql(sql_query: str) -> List[dict]:
    """Mock function for processing SQL query."""
    return [{"mocked_key": "mocked_value"}]

def clinical_trails_full_trial(trial_id: str) -> dict:
    """Mock function for full clinical trial details."""
    return {"trial_id": trial_id, "mocked_detail": "This is a mocked clinical trial detail"}
