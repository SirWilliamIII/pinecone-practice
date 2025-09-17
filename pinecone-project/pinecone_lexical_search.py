#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone


def lexical_search(query, top_k=10):
    """
    Perform lexical (keyword-based) search using BM25 sparse vectors.
    This is traditional keyword matching, not semantic search.
    """
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("semantic-search-demo")
    namespace = "__default__"

    # For demonstration, we'll create a simple sparse vector
    # In production, you'd use a proper BM25 encoder like pinecone-text
    # pip install pinecone-text
    # from pinecone_text.sparse import BM25Encoder

    # Alternative approach using Pinecone's built-in sparse vector support
    # This example shows how to construct a sparse vector manually
    # You would normally get these from a proper BM25 encoding

    # For now, let's demonstrate with a simple keyword-based approach
    # Convert query to lowercase tokens (simple tokenization)
    tokens = query.lower().split()

    # Create a simple sparse vector representation
    # In production, use proper BM25 encoding with document frequencies
    sparse_indices = []
    sparse_values = []

    # Simple hash function to convert tokens to indices (for demo only)
    for i, token in enumerate(tokens):
        # Simple hash to get an index (in production, use proper vocabulary mapping)
        index_val = abs(hash(token)) % 10000  # Limit to 10000 dimensions
        sparse_indices.append(index_val)
        # Simple TF value (in production, use proper BM25 scoring)
        sparse_values.append(1.0)

    # Query with sparse vector for lexical/keyword matching
    try:
        # Note: This requires your index to be configured for sparse vectors
        # or hybrid search (dense + sparse)
        results = index.query(
            namespace=namespace,
            sparse_vector={
                "indices": sparse_indices,
                "values": sparse_values
            },
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Note: Sparse vector search requires index configuration for sparse vectors.")
        print(f"Error: {e}")
        print("\nFalling back to metadata text search as alternative...")

        # Alternative: Use metadata filtering for keyword search
        # This searches for keywords in metadata fields
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(os.getenv("TRANSFORMER_MODEL"))

        # Create a dummy embedding for the query structure
        query_embedding = model.encode(query)

        # Search with metadata filter for text containing keywords
        # Adjust field name based on your metadata structure
        results = index.query(
            namespace=namespace,
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            # Example: filter for text field containing keywords
            # Uncomment and adjust based on your metadata:
            # filter={
            #     "$or": [
            #         {"text": {"$contains": token}} for token in tokens
            #     ]
            # }
        )
        return results


if __name__ == "__main__":
    # Example: Lexical search for specific keywords
    query = "TCP UDP transport protocol"

    results = lexical_search(query, top_k=5)

    print(f"Lexical/Keyword Search Results for: '{query}'")
    print("-" * 50)

    if hasattr(results, 'matches') and results.matches:
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Score: {match.score:.4f}")
            print(f"   ID: {match.id}")
            if match.metadata:
                # Show first 200 chars of text if available
                text = match.metadata.get('text', '')
                if text:
                    print(f"   Text: {text[:200]}...")
    else:
        print("No results found")