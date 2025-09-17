#!/usr/bin/env python3
import os
from collections import defaultdict, Counter
from dotenv import load_dotenv
from pinecone import Pinecone


def explore_metadata_values(namespace="__default__", sample_size=1000):
    """
    Explore and list all available metadata keys and their values from a Pinecone index.
    This helps understand the structure and content of your vector database.
    """
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("semantic-search-demo")

    # Get index statistics
    try:
        stats = index.describe_index_stats()
        print("=== INDEX STATISTICS ===")
        print(f"Total vectors: {stats.total_vector_count:,}")
        print(f"Index fullness: {stats.index_fullness:.2%}")
        if stats.namespaces:
            print("\nNamespaces:")
            for ns_name, ns_stats in stats.namespaces.items():
                print(f"  - {ns_name}: {ns_stats.vector_count:,} vectors")
        print()
    except Exception as e:
        print(f"Could not get index stats: {e}\n")

    # Sample vectors to analyze metadata
    print("=== METADATA EXPLORATION ===")
    print(f"Sampling up to {sample_size} vectors to analyze metadata...\n")

    # Collect metadata from sample vectors
    metadata_keys = set()
    metadata_values = defaultdict(set)
    metadata_types = defaultdict(set)
    metadata_samples = defaultdict(list)
    vector_count = 0

    try:
        # List vectors with pagination to get a sample
        pagination_token = None
        collected_samples = 0

        while collected_samples < sample_size:
            try:
                # Get a batch of vector IDs
                if pagination_token:
                    result = index.list_paginated(
                        namespace=namespace,
                        limit=min(100, sample_size - collected_samples),
                        pagination_token=pagination_token
                    )
                else:
                    result = index.list_paginated(
                        namespace=namespace,
                        limit=min(100, sample_size - collected_samples)
                    )

                if not result.vectors:
                    break

                # Get the actual vector data with metadata
                vector_ids = [v.id for v in result.vectors]
                fetch_result = index.fetch(ids=vector_ids, namespace=namespace)

                # Analyze metadata from fetched vectors
                for vector_id, vector_data in fetch_result.vectors.items():
                    if vector_data.metadata:
                        vector_count += 1
                        for key, value in vector_data.metadata.items():
                            metadata_keys.add(key)
                            metadata_types[key].add(type(value).__name__)

                            # Store sample values (limit to avoid memory issues)
                            if len(metadata_samples[key]) < 20:
                                metadata_samples[key].append(value)

                            # For string values, collect unique values (up to a limit)
                            if isinstance(value, (str, int, float, bool)):
                                if len(metadata_values[key]) < 50:  # Limit unique values stored
                                    metadata_values[key].add(str(value))

                collected_samples += len(vector_ids)

                # Check for pagination
                if hasattr(result.pagination, 'next') and result.pagination.next:
                    pagination_token = result.pagination.next
                else:
                    break

            except Exception as e:
                print(f"Error during pagination: {e}")
                break

    except Exception as e:
        print(f"Error exploring metadata: {e}")
        return

    # Display results
    print(f"Analyzed {vector_count} vectors with metadata")
    print(f"Found {len(metadata_keys)} unique metadata keys\n")

    if not metadata_keys:
        print("No metadata found in sampled vectors.")
        return

    # Display metadata key analysis
    print("=== METADATA KEYS AND TYPES ===")
    for key in sorted(metadata_keys):
        types = list(metadata_types[key])
        print(f"\n📋 Key: '{key}'")
        print(f"   Types: {', '.join(types)}")
        print(f"   Unique values found: {len(metadata_values[key])}")

        # Show sample values
        samples = metadata_samples[key][:10]  # Show first 10 samples
        print("   Sample values:")
        for i, sample in enumerate(samples, 1):
            # Truncate long strings
            if isinstance(sample, str) and len(sample) > 100:
                sample_str = f"{sample[:100]}..."
            else:
                sample_str = str(sample)
            print(f"     {i}. {sample_str}")

    # Display unique values for categorical fields
    print("\n\n=== CATEGORICAL METADATA VALUES ===")
    for key in sorted(metadata_keys):
        if len(metadata_values[key]) <= 20 and len(metadata_values[key]) > 1:
            print(f"\n🏷️  '{key}' values:")
            for value in sorted(metadata_values[key]):
                print(f"   - {value}")

    # Show fields with many unique values (likely text fields)
    text_fields = []
    for key in metadata_keys:
        if len(metadata_values[key]) > 20:
            text_fields.append(key)

    if text_fields:
        print(f"\n\n=== TEXT/HIGH-CARDINALITY FIELDS ===")
        print("These fields have many unique values (likely text content):")
        for field in text_fields:
            print(f"   - '{field}' ({len(metadata_values[field])}+ unique values)")


def get_specific_metadata_values(metadata_key, namespace="__default__", limit=100):
    """
    Get all unique values for a specific metadata key.
    Useful for exploring categorical fields in detail.
    """
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("semantic-search-demo")

    print(f"=== VALUES FOR METADATA KEY: '{metadata_key}' ===\n")

    values = set()
    value_counts = Counter()

    try:
        # Sample vectors to find values for the specific key
        pagination_token = None
        processed = 0

        while processed < limit:
            try:
                if pagination_token:
                    result = index.list_paginated(
                        namespace=namespace,
                        limit=min(50, limit - processed),
                        pagination_token=pagination_token
                    )
                else:
                    result = index.list_paginated(
                        namespace=namespace,
                        limit=min(50, limit - processed)
                    )

                if not result.vectors:
                    break

                vector_ids = [v.id for v in result.vectors]
                fetch_result = index.fetch(ids=vector_ids, namespace=namespace)

                for vector_id, vector_data in fetch_result.vectors.items():
                    if vector_data.metadata and metadata_key in vector_data.metadata:
                        value = vector_data.metadata[metadata_key]
                        values.add(str(value))
                        value_counts[str(value)] += 1

                processed += len(vector_ids)

                if hasattr(result.pagination, 'next') and result.pagination.next:
                    pagination_token = result.pagination.next
                else:
                    break

            except Exception as e:
                print(f"Error during pagination: {e}")
                break

    except Exception as e:
        print(f"Error getting values for key '{metadata_key}': {e}")
        return

    if values:
        print(f"Found {len(values)} unique values for '{metadata_key}':")
        print(f"(Showing values from {processed} vectors)\n")

        # Sort by frequency
        sorted_values = value_counts.most_common()
        for value, count in sorted_values:
            # Truncate long values
            display_value = value[:80] + "..." if len(value) > 80 else value
            print(f"   {count:3d}x | {display_value}")
    else:
        print(f"No values found for metadata key '{metadata_key}'")


if __name__ == "__main__":
    # Full metadata exploration
    explore_metadata_values(sample_size=500)

    # Example: Get specific values for a particular field
    # Uncomment and modify based on fields found in exploration above
    print("\n" + "="*60)
    print("DETAILED EXPLORATION OF SPECIFIC FIELD")
    print("="*60)

    # Replace 'category' with an actual field name from your metadata
    get_specific_metadata_values('category', limit=200)