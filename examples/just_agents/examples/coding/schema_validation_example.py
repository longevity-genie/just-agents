#!/usr/bin/env python
"""
Schema Validation Example

This example demonstrates how to use Pydantic models for response validation
with BaseAgent's query_structural method to get type-safe, structured responses
from an LLM.
"""

from typing import List, Dict, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import sys
import os
import json
from datetime import date

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from just_agents.base_agent import BaseAgent


# Define an enum for product categories
class ProductCategory(str, Enum):
    """Product categories enumeration"""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"
    FOOD = "food"


# Define a model for product information
class ProductInfo(BaseModel):
    """Information about a product"""
    product_id: str = Field(..., description="Unique identifier for the product")
    name: str = Field(..., description="Name of the product")
    price: float = Field(..., description="Price of the product in USD")
    category: ProductCategory = Field(..., description="Category of the product")
    in_stock: bool = Field(..., description="Whether the product is in stock")
    
    # Complex nested fields
    features: List[str] = Field(default_factory=list, description="List of product features")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Product dimensions (width, height, depth)")
    
    # Field with a union type
    promotion: Optional[Union[str, Dict[str, Union[str, float]]]] = Field(
        None, 
        description="Current promotion for the product, either a description or a structured promotion"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "product_id": "P12345",
                    "name": "Wireless Headphones",
                    "price": 99.99,
                    "category": "electronics",
                    "in_stock": True,
                    "features": ["Noise cancellation", "Bluetooth 5.0", "40-hour battery life"],
                    "dimensions": {"width": 18.5, "height": 20.2, "depth": 8.7},
                    "promotion": "Summer Sale: 15% off"
                }
            ]
        }
    )


# Define an advanced search response
class SearchResult(BaseModel):
    """Product search results with filtering and metadata"""
    
    # Search metadata
    query: str = Field(..., description="The search query that was processed")
    total_results: int = Field(..., description="Total number of matching products")
    page: int = Field(1, description="Current page number")
    
    # Filtering information
    applied_filters: Dict[str, Union[str, List[str], float, List[float]]] = Field(
        default_factory=dict,
        description="Filters that were applied to the search"
    )
    
    # Sorting information
    sort_by: Literal["relevance", "price_low", "price_high", "newest"] = Field(
        "relevance", 
        description="How results are sorted"
    )
    
    # Results
    products: List[ProductInfo] = Field(
        default_factory=list,
        description="List of products matching the search criteria"
    )
    
    # Suggested searches
    related_searches: List[str] = Field(
        default_factory=list,
        description="Suggested related search queries"
    )
    
    # Search timestamp
    search_date: date = Field(..., description="Date when the search was performed")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "wireless headphones",
                    "total_results": 42,
                    "page": 1,
                    "applied_filters": {
                        "price_range": [50.0, 200.0],
                        "brand": ["Sony", "Bose"],
                        "rating": 4.0
                    },
                    "sort_by": "relevance",
                    "products": [
                        {
                            "product_id": "P12345",
                            "name": "Sony WH-1000XM4 Wireless Headphones",
                            "price": 349.99,
                            "category": "electronics",
                            "in_stock": True,
                            "features": ["Industry-leading noise cancellation", "Speak-to-chat technology", "30-hour battery life"],
                            "dimensions": {"width": 18.5, "height": 20.2, "depth": 8.7},
                            "promotion": {"type": "discount", "value": 15.0, "description": "Summer Sale"}
                        }
                    ],
                    "related_searches": ["noise cancelling headphones", "bluetooth earbuds", "sony headphones"],
                    "search_date": "2023-07-15"
                }
            ]
        }
    )


def create_agent_with_schema_validation():
    """Create an agent that uses schema validation for responses"""
    
    # Create options for the LLM as a dictionary
    options = {
        "model": "gpt-4-turbo-preview",  # Use an appropriate model that supports JSON mode
        "temperature": 0.5,
        "max_tokens": 1000
    }
    
    # System prompt that instructs the agent about its role
    system_prompt = """
    You are a product search expert that helps users find products based on their queries.
    When responding to user queries, you should provide structured information about products
    that match their search criteria.
    
    Include relevant details such as:
    - Product names, IDs, and prices
    - Categories and features
    - Stock availability
    - Any current promotions or discounts
    
    For ambiguous queries, suggest related search terms to help narrow down results.
    """
    
    # Create the agent with the SearchResult schema as the parser
    agent = BaseAgent(
        llm_options=options,
        system_prompt=system_prompt,
        parser=SearchResult  # Set the parser to our Pydantic model
    )
    
    return agent


def demonstrate_structured_query():
    """Demonstrate how to use query_structural to get a validated response"""
    
    # Create the agent
    agent = create_agent_with_schema_validation()
    
    # Example user query
    user_query = "I'm looking for wireless headphones with good noise cancellation under $200"
    
    print(f"User query: {user_query}\n")
    print("Sending query to LLM with SearchResult schema validation...\n")
    
    # In a real application, you would uncomment this code to get an actual response
    # result = agent.query_structural(user_query, parser=SearchResult, enforce_validation=True)
    # print("Structured Response (validated against SearchResult schema):")
    # print(json.dumps(result.model_dump(), indent=2))
    
    # For demonstration, show what the structure would look like
    print("For demo purposes, the response would be validated against this schema:")
    print(json.dumps(SearchResult.model_json_schema(), indent=2))
    
    print("\nFields would be properly typed in the result object:")
    print("- result.query: str")
    print("- result.total_results: int")
    print("- result.products: List[ProductInfo]")
    print("- result.products[0].price: float")
    print("- result.products[0].category: ProductCategory (Enum)")
    print("- result.search_date: date")


def show_schema_usage_patterns():
    """Show different patterns for using schemas with agents"""
    
    # Create basic options
    options = {
        "model": "gpt-4-turbo-preview", 
        "temperature": 0.7
    }
    
    print("\n--- Different Schema Usage Patterns ---\n")
    
    # 1. Setting parser at agent creation time
    print("1. Setting parser at agent creation time:")
    agent1 = BaseAgent(
        llm_options=options,
        system_prompt="You are a product expert.",
        parser=SearchResult
    )
    print(f"   Agent created with parser: {agent1.parser.__name__}")
    
    # 2. Using a different parser at query time
    print("\n2. Using a different parser at query time:")
    agent2 = BaseAgent(
        llm_options=options,
        system_prompt="You are a product expert."
    )
    print(f"   Agent created with parser: {agent2.parser}")
    print("   Query would use: agent2.query_structural(query, parser=ProductInfo)")
    
    # 3. Serializing and deserializing an agent with a parser
    print("\n3. Serializing and deserializing an agent with a parser:")
    agent3 = BaseAgent(
        llm_options=options,
        system_prompt="You are a product expert.",
        parser=ProductInfo
    )
    
    # Serialize to a dict
    agent_dict = agent3.model_dump()
    
    # Create a new agent from the dict
    agent3b = BaseAgent.model_validate(agent_dict)
    
    print(f"   Original agent parser: {agent3.parser.__name__}")
    print(f"   Deserialized agent parser: {agent3b.parser.__name__}")
    
    # 4. Creating a custom parser programmatically
    print("\n4. Creating a custom parser programmatically:")
    schema_dict = {
        "name": "str",
        "age": "int",
        "email": "Optional[str]",
        "preferences": "Dict[str, Any]"
    }
    
    from just_agents.just_schema import ModelHelper
    CustomParser = ModelHelper.create_model_from_flat_yaml(
        "CustomParser",
        schema_dict,
        optional_fields=False
    )
    
    agent4 = BaseAgent(
        llm_options=options,
        system_prompt="You are a user profile assistant.",
        parser=CustomParser
    )
    
    print(f"   Agent created with custom parser: {agent4.parser.__name__}")
    print(f"   Parser fields: {list(agent4.parser.model_fields.keys())}")


def main():
    """Main function to run the examples"""
    print("=== Schema Validation Example ===\n")
    
    # Demonstrate structured query
    demonstrate_structured_query()
    
    # Show different usage patterns
    show_schema_usage_patterns()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 