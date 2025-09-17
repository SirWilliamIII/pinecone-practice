#!/usr/bin/env python3
"""
Interactive query testing for your Obsidian knowledge base
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pinecone_search import search_obsidian_knowledge_base


def interactive_query_session():
    """Run interactive query session"""
    # Setup
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = "multimodal-search-v2"
    MODEL = os.getenv("TRANSFORMER")
    # Initialize components
    print("🔄 Loading model and connecting to Pinecone...")
    model = SentenceTransformer(MODEL)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Get index stats
    stats = index.describe_index_stats()
    print(f"✅ Connected to index with {stats.total_vector_count} documents")

    print("\n" + "=" * 60)
    print("INTERACTIVE OBSIDIAN SEARCH")
    print("=" * 60)
    print("Tips for testing different query types:")
    print("• Try exact topics from your notes")
    print("• Try related concepts that might not use exact words")
    print("• Try questions about topics you wrote about")
    print("• Try technical terms vs. plain language")
    print("• Type 'quit' to exit")
    print("-" * 60)

    while True:
        try:
            # Get query
            query = input("\n🔍 Search query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not query:
                print("Please enter a search query")
                continue

            # Get number of results
            try:
                num_results = input("Number of results (default 5): ").strip()
                top_k = int(num_results) if num_results else 5
                top_k = max(1, min(top_k, 20))  # Limit 1-20
            except ValueError:
                top_k = 5

            # Perform search
            print(f"\n🔍 Searching for: '{query}' (top {top_k})")
            print("=" * 50)

            results = search_obsidian_knowledge_base(query, model, index, top_k)

            # Analyze results
            if results["matches"]:
                scores = [match["score"] for match in results["matches"]]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)

                print(f"\n📊 Score Analysis:")
                print(f"   Best match: {max_score:.4f}")
                print(f"   Average: {avg_score:.4f}")

                if max_score > 0.7:
                    print("   🎯 Excellent relevance!")
                elif max_score > 0.4:
                    print("   ✅ Good semantic match")
                elif max_score > 0.2:
                    print("   🤔 Somewhat related")
                else:
                    print("   ❓ Weak connections")

            else:
                print("❌ No results found")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def test_predefined_queries():
    """Test a set of predefined queries to understand search behavior"""

    # Setup (same as above)
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = "semantic-search-demo"
    MODEL = os.getenv("TRANSFORMER")
    model = SentenceTransformer(MODEL)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Predefined test queries
    test_queries = ["HTTPie"]

    print("\n" + "=" * 60)
    print("BATCH QUERY TESTING")
    print("=" * 60)
    print("Testing predefined queries to understand search patterns...")

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)

        results = search_obsidian_knowledge_base(query, model, index, top_k=3)
        for j, match in enumerate(results["matches"], 1):
            print(f"  {j}. Score: {match['score']:.4f}")
            print(f"     File: {match['metadata'].get('source', 'Unknown')}")
            print(f"     Content: {match['metadata'].get('text', 'No text')[:200]}...")
            print()

        if results["matches"]:
            best_score = results["matches"][0]["score"]
            print(f"📊 Best match score: {best_score:.4f}")
        else:
            print("❌ No results")


if __name__ == "__main__":
    interactive_query_session()
