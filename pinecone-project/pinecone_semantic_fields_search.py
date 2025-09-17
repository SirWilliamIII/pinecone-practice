#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone


def semantic_search_with_fields(query_text, fields=None, top_k=10):
    """
    Perform semantic search using Pinecone's built-in embedding capability.
    This uses the index.search() method with text input, where Pinecone
    handles the embedding automatically (requires configured embedding model).

    Note: This requires your Pinecone index to be configured with an
    embedding model for automatic text-to-vector conversion.
    """
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("multimodal-search-v2")
    namespace = "__default__"

    # Use Pinecone's search method with text input
    # This requires the index to have an associated embedding model
    try:
        results = index.search(
            namespace=namespace,
            query={"inputs": {"text": query_text}, "top_k": top_k},
            fields=fields,  # Specify which metadata fields to return
        )
        return results
    except Exception as e:
        print(
            f"Note: index.search() with text input requires configured embedding model."
        )
        print(f"Error: {e}")
        print("\nFalling back to manual embedding approach...")

        # Fallback: Use manual embedding with sentence transformers
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(os.getenv("TRANSFORMER_MODEL"))

        # Encode the query
        query_embedding = model.encode(query_text)

        # Query with vector
        results = index.query(
            namespace=namespace,
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False,  # Don't need vector values in response
        )

        # If specific fields were requested, filter the metadata
        if fields and hasattr(results, "matches"):
            for match in results.matches:
                if match.metadata:
                    # Filter metadata to only requested fields
                    filtered_metadata = {
                        k: v for k, v in match.metadata.items() if k in fields
                    }
                    match.metadata = filtered_metadata

        return results


def display_results(results, query, fields=None):
    """Helper function to display search results"""
    print(f"Semantic Search Results for: '{query}'")
    if fields:
        print(f"Returning fields: {', '.join(fields)}")
    print("-" * 50)

    if hasattr(results, "matches") and results.matches:
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Score: {match.score:.4f}")
            print(f"   ID: {match.id}")
            if match.metadata:
                print("   Metadata:")
                for key, value in match.metadata.items():
                    # Truncate long text values for display
                    if isinstance(value, str) and len(value) > 150:
                        value = value[:150] + "..."
                    print(f"     - {key}: {value}")
    else:
        print("No results found")


if __name__ == "__main__":
    # Example 1: Basic semantic search with all metadata
    print("\n=== Example 1: Basic Semantic Search ===\n")
    query1 = "What are the different layers in network architecture?"
    results1 = semantic_search_with_fields(query1, fields=None, top_k=3)
    display_results(results1, query1)

    # Example 2: Semantic search returning specific fields only
    print("\n\n=== Example 2: Search with Specific Fields ===\n")
    query2 = "How does TCP handle congestion control?"
    # Adjust these field names based on your actual metadata structure
    specific_fields = ["text", "source", "category", "date"]
    results2 = semantic_search_with_fields(query2, fields=specific_fields, top_k=3)
    display_results(results2, query2, specific_fields)

    # Example 3: Business query example (like your AAPL example)
    print("\n\n=== Example 3: Business Context Query ===\n")
    query3 = "What is the outlook for network security, considering both emerging threats and defense mechanisms?"
    # Fields you might have for business/technical documents
    business_fields = ["chunk_text", "quarter", "section", "document_type"]
    results3 = semantic_search_with_fields(query3, fields=business_fields, top_k=3)
    display_results(results3, query3, business_fields)
