import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


def search_obsidian_knowledge_base(query, top_k=5):
    """Search the Obsidian knowledge base using semantic similarity"""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    MODEL = os.getenv("TRANSFORMER")
    # Initialize Pinecone and SentenceTransformer
    index_name = "semantic-search-demo"
    model = SentenceTransformer(MODEL)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Encode query and perform search
    query_embedding = model.encode(query)
    results = index.query(
        vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
    )

    # Display results
    print("TESTING YOUR OBSIDIAN KNOWLEDGE BASE")
    print("=" * 50)
    print(f"Query: '{query}'")

    if not results["matches"]:
        print("No results found for the query.")
        return None

    for i, match in enumerate(results["matches"], 1):
        print(f"{i}. ID: {match['id']} (Score: {match['score']:.4f})")
        print(f"   File: {match['metadata'].get('filename', 'Unknown')}")
        print(f"   Text: {match['metadata'].get('text', 'No text')[:100]}")
        print()

    return results


# Example query
search_obsidian_knowledge_base("hacking")
