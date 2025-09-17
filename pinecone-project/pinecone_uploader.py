#!/usr/bin/env python3
import time

def upload_documents_to_pinecone(documents, index_name, pc):
    """Upload documents to Pinecone index in batches"""
    # Connect to index and upload in batches
    index = pc.Index(index_name)

    # Prepare vectors
    vectors_to_upsert = []
    for doc in documents:
        vector_data = (
            doc["id"],
            doc["embedding"],
            {
                "text": doc["text"],
                "filename": doc["filename"],
                "category": doc["category"]
            }
        )
        vectors_to_upsert.append(vector_data)

    # Upload in batches
    print(f"Uploading {len(vectors_to_upsert)} vectors in batches...")
    batch_size = 10
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        print(f"Uploading batch {i//batch_size + 1}: {len(batch)} vectors")
        index.upsert(vectors=batch)
        time.sleep(2)

    print("Upload complete! Waiting for indexing...")
    time.sleep(10)

    # Verify upload
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")

    return index
