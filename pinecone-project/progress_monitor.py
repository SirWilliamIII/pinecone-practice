#!/usr/bin/env python3
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

def monitor_progress():
    """Monitor indexing progress in real-time."""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index_name = "multimodal-search-v2"

    print(f"🔍 Monitoring progress for index: {index_name}")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        index = pc.Index(index_name)
        last_count = 0

        while True:
            try:
                stats = index.describe_index_stats()
                current_count = stats.total_vector_count
                dimension = stats.dimension

                # Calculate rate if we have previous data
                if last_count > 0:
                    rate = current_count - last_count
                    rate_text = f"(+{rate}/10s)" if rate > 0 else "(no change)"
                else:
                    rate_text = ""

                # Show progress
                print(f"📊 Vectors: {current_count:,} {rate_text}")
                print(f"🔧 Dimension: {dimension}")
                print(f"⏰ {time.strftime('%H:%M:%S')}")
                print("-" * 30)

                last_count = current_count
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"❌ Error checking stats: {e}")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    except Exception as e:
        print(f"❌ Failed to connect to index: {e}")

if __name__ == "__main__":
    monitor_progress()