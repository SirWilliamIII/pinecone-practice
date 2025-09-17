#!/usr/bin/env python3
"""Debug script to check Pinecone connection and index status."""

import os
from dotenv import load_dotenv
from pinecone import Pinecone


def debug_pinecone():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("INDEX_NAME", "multimodal-search-v2")

    print(f"🔑 API Key: {api_key[:20]}..." if api_key else "❌ No API key found")
    print(f"📋 Index name: {index_name}")

    try:
        # Initialize Pinecone
        print("\n🔄 Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client connected")

        # List existing indexes
        print("\n📊 Checking existing indexes...")
        indexes = pc.list_indexes()
        print(f"Found {len(indexes)} total indexes:")

        for idx in indexes:
            print(f"  - {idx.name} (status: {idx.status.state})")
            if idx.name == index_name:
                print(f"    ✅ Found our target index: {index_name}")

        # Check if our specific index exists
        if pc.has_index(name=index_name):
            print(f"\n✅ Index '{index_name}' exists!")

            # Try to connect to it
            index = pc.Index(index_name)
            stats = index.describe_index_stats()

            print(f"📈 Index stats:")
            print(f"  - Total vectors: {stats.total_vector_count}")
            print(f"  - Dimension: {getattr(stats, 'dimension', 'unknown')}")

            if stats.total_vector_count > 0:
                print("🎉 Index has data! Search should work.")
            else:
                print("⚠️  Index exists but has no data. Need to run indexing.")

        else:
            print(f"\n❌ Index '{index_name}' does not exist")
            print("Need to run indexing to create it.")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    debug_pinecone()
