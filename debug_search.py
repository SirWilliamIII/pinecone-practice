#!/usr/bin/env python3
"""Debug the search manager initialization."""

import logging
from full_script import Config, SemanticSearchManager

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def debug_search_manager():
    print("🔄 Testing SemanticSearchManager initialization...")
    
    try:
        # Load config
        config = Config.from_env()
        print(f"✅ Config loaded: index={config.index_name}")
        
        # Initialize search manager
        search_manager = SemanticSearchManager(config)
        print("✅ Search manager created")
        
        # Check Pinecone connection
        print(f"Pinecone client: {search_manager.pinecone_manager.pc}")
        print(f"Index object: {search_manager.pinecone_manager.index}")
        
        # Try to ensure index exists (this might be where it fails)
        print("\n🔄 Testing index connection...")
        index = search_manager.pinecone_manager.ensure_index_exists()
        print(f"✅ Index connected: {index}")
        
        # Try a simple search
        print("\n🔄 Testing search...")
        results = search_manager.search("test", top_k=1)
        print(f"✅ Search works! Found {len(results)} results")
        
        if results:
            print(f"Sample result: {results[0]['filename']}")
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_search_manager()