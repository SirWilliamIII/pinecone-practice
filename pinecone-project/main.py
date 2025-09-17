#!/usr/bin/env python3
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from obsidian_loader import load_obsidian_vault
from pinecone_uploader import upload_documents_to_pinecone


# Configuration
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = "multimodal-search-v2"

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

if pc.has_index(name=index_name):
    print(f"Using existing index: {index_name}")
else:
    # Create fresh index
    print("Creating new index...")
    pc.create_index(
        name=index_name,
        dimension=512,  # CLIP model dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
    print("Waiting for index to be ready...")
    time.sleep(30)

print(f"Index '{index_name}' is ready!")

documents = load_obsidian_vault()
MODEL = "clip-ViT-B-32"  # Multimodal model that handles text and images
model = SentenceTransformer(MODEL)
print("Creating embeddings for documents...")
for doc in documents:
    if doc.get("file_type") == "image":
        # For images, encode the actual image file
        try:
            from PIL import Image
            image = Image.open(doc["full_path"])
            embedding = model.encode(image)
        except Exception as e:
            print(f"Error encoding image {doc['filename']}: {e}")
            # Fallback to text embedding
            embedding = model.encode(doc["text"])
    else:
        # For text/PDF content
        embedding = model.encode(doc["text"])

    doc["embedding"] = embedding.tolist()
    print(f"Document '{doc['id']}' ({doc.get('file_type', 'text')}) embedded: {len(doc['embedding'])} dimensions")


index = upload_documents_to_pinecone(documents, index_name, pc)

# Test search on your knowledge base
print("\n" + "=" * 50)
print("TESTING YOUR OBSIDIAN KNOWLEDGE BASE")
print("=" * 50)
