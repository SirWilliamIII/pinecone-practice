#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


def search_pinecone(query, top_k=10):
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("multimodal-search-v2")
    model = SentenceTransformer(os.getenv("TRANSFORMER_MODEL"))
    namespace = "__default__"
    # Encode and search
    query_embedding = model.encode(query)

    results = index.query(
        namespace=namespace,
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )

    return results


query = "what is the transport layer"

if __name__ == "__main__":
    # Example usage
    results = search_pinecone(query, top_k=3)
    for match in results.matches:
        print(
            f"Score: {match['score']:.4f} - {match['metadata'].get('text', '')[:100]}..."
        )
