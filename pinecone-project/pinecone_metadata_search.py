#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


def metadata_filter_search(query, metadata_filter, top_k=10):
    """
    Search Pinecone using vector similarity with metadata filtering.
    This combines semantic search with metadata constraints.
    """
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("multimodal-search-v2")
    model = SentenceTransformer(os.getenv("TRANSFORMER_MODEL"))
    namespace = "__default__"

    # Encode the query into a vector
    query_embedding = model.encode(query)

    # Query with metadata filter
    results = index.query(
        namespace=namespace,
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )

    return results


if __name__ == "__main__":
    # Example: Search for content about networking that was created recently
    query = "network protocols and layers"

    # Example metadata filters - adjust based on your actual metadata structure
    metadata_filter = {
        # Example: Filter by category
        # "category": {"$in": ["networking", "protocols", "OSI"]},
        # Example: Filter by date range (if you have date metadata)
        # "created_date": {"$gte": "2024-01-01", "$lte": "2025-01-31"},
        # Example: Filter by source
        # "source": {"$eq": "technical_docs"},
        # Example: Filter by tags
        # "tags": {"$in": ["network", "tcp", "udp"]},
        # Simple text field filter (adjust field name to match your metadata)
        # "text": {"$contains": "transport"}
    }

    # For this example, let's do a simple search without filters first
    # Uncomment and modify the filter above based on your metadata structure
    results = metadata_filter_search(query, metadata_filter={}, top_k=5)

    print(f"Metadata Filter Search Results for: '{query}'")
    print("-" * 50)

    if results.matches:
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Score: {match.score:.4f}")
            print(f"   ID: {match.id}")
            if match.metadata:
                print("   Metadata:")
                for key, value in match.metadata.items():
                    # Truncate long text values for display
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"     - {key}: {value}")
    else:
        print("No results found")
