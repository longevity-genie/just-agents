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

    Example of result:
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
        index (str): The name of the index to search within.
                    It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 30 by default.

    Example of result:
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
        non_empty (bool): If True, returns only indexes that contain documents.
                        If False, returns all available indexes. True by default.
    
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


# def generate_random_matrix(rows: int, cols: int) -> np.ndarray:
#     """
#     Generate a random matrix of given dimensions.
#
#     Args:
#         rows (int): Number of rows.
#         cols (int): Number of columns.
#
#     Returns:
#         np.ndarray: A matrix filled with random values.
#     """
#     matrix = np.random.rand(rows, cols)
#     matrix[0][0]=0.2323232323232 #to discern between tool output and hallucinations
#     print("Random Matrix:\n", matrix)
#
#     return matrix
#
# def summarize_dataframe(data: dict) -> pd.DataFrame:
#     """
#     Convert a dictionary into a DataFrame and return basic statistics.
#
#     Args:
#         data (dict): A dictionary where keys are column names and values are lists.
#
#     Returns:
#         pd.DataFrame: A DataFrame summary with mean and standard deviation.
#     """
#     df = pd.DataFrame(data)
#     summary = df.describe()
#     print("\nData Summary:\n", summary)
#     return summary
