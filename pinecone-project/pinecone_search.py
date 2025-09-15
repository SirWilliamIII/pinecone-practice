

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


def search_obsidian_knowledge_base(query, top_k=5):
    """Search the Obsidian knowledge base using semantic similarity"""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    index_name = "semantic-search-demo"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    query_embedding = model.encode(query)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    # Test search on your knowledge base
    print("\n" + "="*50)
    print("TESTING YOUR OBSIDIAN KNOWLEDGE BASE")
    print("="*50)


    print(f"Query: '{query}'")
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. ID: {match['id']} (Score: {match['score']:.4f})")
        print(f"   File: {match['metadata']['filename']}")
        print(f"   Text: \"{match['metadata']['text'][:100]}...\"")
        print()

    return results


search_obsidian_knowledge_base("GPU performance machine learning")
