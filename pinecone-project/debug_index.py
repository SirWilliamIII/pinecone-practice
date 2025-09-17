#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone


def debug_index():
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("multimodal-search-v2")

    # Get detailed index statistics
    stats = index.describe_index_stats()
    print("=== DETAILED INDEX STATISTICS ===")
    print(f"Total vectors: {stats.total_vector_count:,}")
    print(f"Index fullness: {stats.index_fullness:.2%}")

    if stats.namespaces:
        print(f"\nFound {len(stats.namespaces)} namespaces:")
        for ns_name, ns_stats in stats.namespaces.items():
            print(f"  - Namespace: '{ns_name}' -> {ns_stats.vector_count:,} vectors")

            # Try to list a few vectors from each namespace
            try:
                result = index.list_paginated(namespace=ns_name, limit=3)
                if result.vectors:
                    print(
                        f"    Sample vector IDs: {[v.id for v in result.vectors[:3]]}"
                    )

                    # Try to fetch one vector to see if it has metadata
                    sample_id = result.vectors[0].id
                    fetch_result = index.fetch(ids=[sample_id], namespace=ns_name)
                    vector_data = fetch_result.vectors.get(sample_id)

                    if vector_data and vector_data.metadata:
                        print(
                            f"    ✅ Sample metadata keys: {list(vector_data.metadata.keys())}"
                        )
                    else:
                        print(f"    ❌ No metadata found in sample vector")
                else:
                    print(f"    ❌ Could not list vectors from this namespace")
            except Exception as e:
                print(f"    ❌ Error accessing namespace: {e}")
    else:
        print("\n❌ No namespaces found or all namespaces are empty")

    # Try the default namespace specifically
    print(f"\n=== TESTING DEFAULT NAMESPACE ===")
    try:
        result = index.list_paginated(limit=3)  # No namespace = default
        if result.vectors:
            print(
                f"✅ Default namespace has vectors: {[v.id for v in result.vectors[:3]]}"
            )
        else:
            print("❌ Default namespace appears empty")
    except Exception as e:
        print(f"❌ Error accessing default namespace: {e}")


if __name__ == "__main__":
    debug_index()
