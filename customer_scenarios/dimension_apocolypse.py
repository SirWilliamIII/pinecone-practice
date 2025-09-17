#!/usr/bin/env python3
"""
CSE Training: The Dimension Apocalypse
This demonstrates the #1 customer integration failure.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv

load_dotenv()
MODEL=os.getenv("TRANSFORMER")
# Simulate customer's initial setup - everything works great!
print("=== CUSTOMER'S INITIAL SETUP (Works Fine) ===")


# Customer starts with a small model
model_small = SentenceTransformer(MODEL)# 384 dimensions
sample_text = "This is a test document"

# Generate embeddings
embedding_small = model_small.encode(sample_text)
print(f"Small model dimensions: {embedding_small.shape[0]}")
print(f"Sample embedding (first 5 values): {embedding_small[:5]}")

# Customer creates Pinecone index for 384 dimensions
print(f"\n✅ Created Pinecone index with {embedding_small.shape[0]} dimensions")
print("✅ Uploaded 1000 documents successfully")
print("✅ Queries working perfectly")

print("\n" + "="*60)
print("📅 THREE WEEKS LATER...")
print("Customer decides to 'upgrade' to better model")
print("="*60)

# Customer finds a "better" model (this is where disaster strikes)
model_large = SentenceTransformer('all-mpnet-base-v2')  # 768 dimensions
embedding_large = model_large.encode(sample_text)

print(f"\n🔥 NEW MODEL DIMENSIONS: {embedding_large.shape[0]}")
print(f"🔥 EXISTING INDEX DIMENSIONS: {embedding_small.shape[0]}")

# Show the mismatch
dimension_mismatch = embedding_large.shape[0] != embedding_small.shape[0]
print(f"\n💥 DIMENSION MISMATCH DETECTED: {dimension_mismatch}")

if dimension_mismatch:
    print("\n🚨 CUSTOMER IMPACT:")
    print("   - Cannot add new vectors to existing index")
    print("   - Cannot query with new model embeddings") 
    print("   - Must delete entire index and start over")
    print("   - Loses all existing data and configurations")
    print("   - Downtime during re-indexing")

print("\n" + "="*60)
print("CSE DIAGNOSTIC QUESTION TIME")
print("="*60)