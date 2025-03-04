#import numpy as np
#import pandas as pd
from pathlib import Path
#import markdown
import re
from typing import Optional, List, Dict, Any
import os
import random


def about_glucosedao(question: str) -> str:
    """
    Answer questions about GlucoseDAO
    """
    print("question ", question)
    return """
    Leadership Structure
    Core Team
    -Founder: Zaharia Livia (Type 1 diabetic)
    -Founding Members: Anton Kulaga, Brandon Houten
    Scientific Advisory Board
    -Irina Gaianova (University of Michigan, Biostatistics)
    -Prof. Georg Fullen (Rostock University)
    -Renat Sergazinov (GlucoBench author)
    """


def search_documents_raw(query: str, index: str, limit: Optional[int] = 4, semantic_ratio: Optional[float] = 0.5) -> \
list[dict]:
    """
    Search documents in MeiliSearch database. Giving search results in raw format.

    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within.
                     It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 8 by default.

    Example results:
        [ {'_rankingScore': 0.718,  # Relevance score of the document
          '_rankingScoreDetails': {'vectorSort': {'order': 0,  # Ranking order 'similarity': 0.718}},  # Similarity score
          'hash': 'e22c1616...',  # Unique document identifier
          'source': '/path/to/document.txt',  # Source document path
          'text': 'Ageing as a risk factor...',  # Document content
          'token_count': None,  # Number of tokens (if applicable)
          'total_fragments': None},  # Total fragments (if applicable)
          ]
    """
    # Simulate search results based on the example in the docstring
    
    # Simulated document content based on query and index
    documents = [
        {
            '_rankingScore': round(random.uniform(0.6, 0.95), 3),
            '_rankingScoreDetails': {
                'vectorSort': {
                    'order': i,
                    'similarity': round(random.uniform(0.6, 0.95), 3)
                }
            },
            'hash': f'e22c1616{i}{"".join(random.choices("0123456789abcdef", k=8))}',
            'source': f'/path/to/{index}/document_{i}.txt',
            'text': f'Sample content related to {query} in {index} database. Document #{i}',
            'token_count': random.randint(50, 200),
            'total_fragments': random.randint(1, 5)
        }
        for i in range(min(limit, 10))  # Generate up to 'limit' documents, max 10
    ]
    
    # Sort by ranking score in descending order
    documents.sort(key=lambda x: x['_rankingScore'], reverse=True)
    
    print(f"Simulated search for '{query}' in index '{index}' with {len(documents)} results")
    return documents


def search_documents(query: str, index: str, limit: Optional[int] = 30, semantic_ratio: Optional[float] = 0.5) -> list[str]:
    """
    Search documents in MeiliSearch database.

    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within. It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 30 by default.
        semantic_ratio (float): Semantic ratio

    Returns:
        list[str]: A list of strings containing the search results.

    Example result:
        [ {'_rankingScore': 0.718,  # Relevance score of the document
          '_rankingScoreDetails': {'vectorSort': {'order': 0,  # Ranking order
                                                  'similarity': 0.718}},  # Similarity score
          'hash': 'e22c1616...',  # Unique document identifier
          'source': '/path/to/document.txt',  # Source document path
          'text': 'Ageing as a risk factor...',  # Document content
          'token_count': None,  # Number of tokens (if applicable)
          'total_fragments': None},  # Total fragments (if applicable)
          ]

    """
    # Get semantic ratio from environment variable or use default
    semantic_ratio = float(os.getenv("MEILISEARCH_SEMANTIC_RATIO", 0.5))
    
    # Get raw search results
    hits = search_documents_raw(query, index, limit, semantic_ratio=semantic_ratio)
    
    # Print debug info
    print(f"Found {len(hits)} results for query '{query}' in index '{index}'")
    
    # Format results as strings with source information
    result: list[str] = [h["text"] + "\n SOURCE: " + h["source"] for h in hits]
    return result


def all_indexes(non_empty: bool = True) -> List[str]:
    """
    Get all indexes that you can use for search.

    Args:
        non_empty (bool): If True, returns only indexes that contain documents, otherwise returns all available indexes. True by default.

    Returns:
        List[str]: A list of index names that can be used for document searches.
    """
    # Simulate available indexes
    all_available_indexes = [
        "research_papers", 
        "clinical_trials",
        "medical_guidelines", 
        "patient_data",
        "diabetes_research",
        "glucose_monitoring"
    ]
    
    # Simulate empty indexes
    empty_indexes = ["patient_data", "glucose_monitoring"]
    
    # Filter out empty indexes if requested
    result_indexes = [idx for idx in all_available_indexes if non_empty is False or idx not in empty_indexes]
    
    print(f"Retrieved {'non-empty' if non_empty else 'all'} indexes: {', '.join(result_indexes)}")
    return result_indexes

def semantic_search(
    query: str, 
    index: str, 
    limit: int = 10, 
    semantic_ratio: float = 0.5
) -> list[str]:
    """
    Search for documents using semantic search (mock implementation).

    Args:
        query: The search query
        index: The index to search in
        limit: The maximum number of results to return (default: 10)
        semantic_ratio: The ratio of semantic search to use (0.0 to 1.0, default: 0.5)

    Returns:
        List[str]: A list of strings containing simulated search results

    Raises:
        ValueError: If semantic_ratio is not between 0 and 1
    """
    # Validate semantic ratio
    if not 0 <= semantic_ratio <= 1:
        raise ValueError("semantic_ratio must be between 0 and 1")
    
    # Print debug info
    print(f"Simulated semantic search for '{query}' in index '{index}' with semantic_ratio={semantic_ratio}")
    
    # Dictionary of rich content with rare words and made-up terms
    rich_content: Dict[str, List[str]] = {
        "research_papers": [
            "The quiescent glycoregulatory mechanisms exhibit profound circadian oscillations, particularly in the context of postprandial hyperglycemia. The pancreatic β-cells' exocytotic machinery demonstrates remarkable plasticity in response to glucolipotoxicity.",
            "Recent investigations into the xenobiotic-induced dysglycemia reveal a complex interplay between hepatic gluconeogenesis and peripheral insulin resistance. The term 'glycofractionation' describes this novel metabolic phenomenon.",
            "Islet amyloid polypeptide aggregation, colloquially termed 'amyloidogenic diabetopathy', represents a pathognomonic feature in type 2 diabetes mellitus. The islet microvasculature undergoes significant remodeling during diabetogenesis."
        ],
        "clinical_trials": [
            "The double-blind, placebo-controlled trial of Zorbinol-X demonstrated significant reductions in HbA1c through its novel mechanism of 'glucomodulatory transduction inhibition' at the cellular level.",
            "Participants receiving the experimental Endofibrase therapy exhibited marked improvements in postprandial glycemic excursions, attributed to enhanced 'pancreatic quiescification' - a term coined specifically for this trial.",
            "The GLIMMER-7 study protocol incorporates rigorous assessment of 'dysglycemic rebounding' following administration of the investigational compound XR-27891, a selective glycokinase activator."
        ],
        "diabetes_research": [
            "The pathophysiology of diabetic nephropathy involves glomerular hyperfiltration and subsequent podocytopathy. The term 'nephrodiabetogenic cascade' has been proposed to describe this sequential deterioration.",
            "Continuous glucose monitoring systems now incorporate advanced algorithms for detecting 'glycemic perturbation patterns' - a neologism describing subtle variations in interstitial glucose concentrations.",
            "The concept of 'insulinomimetic xenohormesis' represents a paradigm shift in understanding how certain phytochemicals may activate insulin signaling pathways through molecular mimicry."
        ],
        "medical_guidelines": [
            "Current recommendations emphasize the importance of 'glycemic equipoise' - maintaining balanced glucose homeostasis through integrated pharmacological and lifestyle interventions.",
            "The guidelines now recognize 'diabetogenic iatrogenesis' as a distinct clinical entity requiring specific management strategies, particularly in hospitalized patients.",
            "Clinicians should be vigilant for signs of 'glucoregulatory decompensation' in patients with comorbid endocrinopathies, especially during periods of physiological stress."
        ]
    }
    
    # Generate mock search results with rich content
    results: list[str] = []
    
    # Add some basic results
    for i in range(1, min(limit + 1, 6)):  # Generate up to 5 basic results
        results.append(f"Semantic result #{i} for query '{query}' in index '{index}'. Relevance: {round(random.uniform(0.6, 0.95), 2)}.")
    
    # Add rich content based on the index
    if index in rich_content:
        # Select 1-3 random rich content items for this index
        selected_content = random.sample(rich_content[index], min(3, len(rich_content[index])))
        for content in selected_content:
            relevance = round(random.uniform(0.75, 0.98), 2)
            results.append(f"{content} Semantic score: {relevance}. SOURCE: /data/{index}/document_{random.randint(1000, 9999)}.txt")
    
    # Add special content for diabetes queries
    if "diabetes" in query.lower():
        results.insert(0, "The novel concept of 'pancreatic beta-cell dedifferentiation' or 'diabetogenic transdifferentiation' represents a fundamental shift in our understanding of type 2 diabetes pathophysiology. The term 'glucolipoapoptosis' describes the specific cellular death pathway. Semantic score: 0.97. SOURCE: /data/diabetes_research/beta_cell_pathophysiology.txt")
    
    # Add clinical trial specific content
    if index == "clinical_trials":
        results.append(f"The GLUCOMET-9 trial investigating 'selective hepatic insulin sensitization' through the novel compound Metaboreglin-D demonstrated statistically significant improvements in fasting plasma glucose levels (p<0.001). SOURCE: /mock/data/trials/trial_{random.randint(1000, 9999)}.txt")
    
    # Shuffle results to make them appear more realistic
    random.shuffle(results)
    
    return results[:limit]  # Ensure we don't exceed the requested limit
