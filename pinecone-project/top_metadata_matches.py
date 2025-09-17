import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL = os.getenv("TRANSFORMER_MODEL")
api_key = os.getenv("PINECONE_API_KEY")
QUERY = "amazon"
top_k = 10


def search_docs():
    index_name = "semantic-search-demo"
    model = SentenceTransformer(MODEL)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    query_embedding = model.encode(query)
    results = index.query(
        vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
    )

    print(f"Query: '{query}'")
    for i, match in enumerate(results["matches"], 1):
        breakpoint()
        print(f"{i}. ID: {match['id']} (Score: {match['score']:.4f})")
        print(f"   File: {match['metadata'].get('filename', 'Unknown')}")
        print(f"   Text: {match['metadata'].get('text', 'No text')[:100]}")
        print()

    ids = [match["id"] for match in results["matches"]]
    # fetched = index.fetch(ids=ids)

    # for id, vector in fetched["vectors"].items():
    # 	print(f"ID: {id}, Metadata: {vector['metadata']}")

    breakpoint()
    return results


# After running your search:
