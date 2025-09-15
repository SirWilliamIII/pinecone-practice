#!/usr/bin/env python3
"""
CSE Training: Format Hell
The numpy → JSON → Pinecone data shuttling nightmare
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import json

print("=== FORMAT HELL: The Data Conversion Nightmare ===")

# Customer generates embeddings (this part works)
model = SentenceTransformer('all-MiniLM-L6-v2')
text = "Customer support document"
embedding = model.encode(text)

print(f"✅ Generated embedding: {type(embedding)}")
print(f"✅ Shape: {embedding.shape}")
print(f"✅ First 3 values: {embedding[:3]}")

print(f"\n🔥 PROBLEM: Pinecone needs Python lists, not numpy arrays")
print(f"Customer tries to upsert numpy array directly...")

# This is what breaks in customer code
try:
    # Simulate Pinecone upsert with numpy array (this fails)
    fake_upsert_data = {
        "vectors": [
            {
                "id": "doc1", 
                "values": embedding,  # 🚨 numpy array, not list!
                "metadata": {"source": "support_doc"}
            }
        ]
    }
    
    # Try to serialize (this is where it explodes)
    json_data = json.dumps(fake_upsert_data)
    print("✅ JSON serialization worked somehow...")
    
except TypeError as e:
    print(f"\n💥 JSON SERIALIZATION ERROR: {e}")
    print("💥 numpy arrays are not JSON serializable!")

print(f"\n🔧 CUSTOMER ATTEMPTS FIX #1: Convert to list")
embedding_list = embedding.tolist()
print(f"✅ Converted to list: {type(embedding_list)}")

# But wait, there's more problems...
print(f"\n🔥 HIDDEN PROBLEM: Special float values")

# Create an embedding with problematic values
problematic_embedding = np.array([0.1, 0.2, np.nan, 0.4, np.inf])
problematic_list = problematic_embedding.tolist()

print(f"Problematic values: {problematic_list}")

try:
    json_test = json.dumps(problematic_list)
    print("JSON serialization of special values...")
except Exception as e:
    print(f"💥 SPECIAL VALUES ERROR: {e}")

# Show what actually gets serialized
json_with_special = json.dumps(problematic_list, allow_nan=True)
print(f"JSON with special values: {json_with_special}")
print(f"🚨 Pinecone will reject null/infinity values!")

print(f"\n🔥 PROBLEM #3: Metadata bloat")
huge_metadata = {
    "title": "Customer Support Document" * 100,  # Way too long
    "content": "Full document text here..." * 1000,  # Massive
    "nested": {"deep": {"very": {"deep": {"structure": True}}}},  # Complex nesting
    "list_data": list(range(1000)),  # Large arrays
}

metadata_size = len(json.dumps(huge_metadata))
print(f"Metadata size: {metadata_size} bytes")
print(f"🚨 Pinecone metadata limit: ~40KB per vector")
print(f"🚨 Customer exceeded limit by: {metadata_size - 40000} bytes")

print(f"\n" + "="*60)
print("CSE DIAGNOSTIC SCENARIOS")
print("="*60)