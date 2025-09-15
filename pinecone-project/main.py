#!/usr/bin/env python3
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from obsidian_loader import load_obsidian_vault
from pinecone_uploader import upload_documents_to_pinecone
from pinecone_search import search_obsidian_knowledge_base

# Configuration
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = "semantic-search-demo"

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# if pc.has_index(name=index_name):
#     print(f"Using Index: ${index_name}")
# else:
# # Create fresh index
#     print("Creating new index...")
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-west-2")
#     )
#     print("Waiting for index to be ready...")
#     time.sleep(30)
    
# print(f"Index '{index_name}' is ready!")

documents = load_obsidian_vault()
documents = documents[:10]



model = SentenceTransformer('all-MiniLM-L6-v2')
print("Creating embeddings for documents...")
for doc in documents:
    embedding = model.encode(doc["text"])
    doc["embedding"] = embedding.tolist()
    print(f"Document '{doc['id']}' embedded: {len(doc['embedding'])} dimensions")


index = upload_documents_to_pinecone(documents, index_name, pc)

# Test search on your knowledge base
print("\n" + "="*50)
print("TESTING YOUR OBSIDIAN KNOWLEDGE BASE")
print("="*50)


