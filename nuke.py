from pinecone import Pinecone
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
# Initialize pinecone
pc = Pinecone(api_key=api_key)

# Delete the entire index (if it exists)
try:
    pc.delete_index("semantic-search-demo")
    print("Index deleted successfully")
except Exception as e:
    print(f"Index doesn't exist or already deleted: {e}")

# Create new index with same name (or different dimensions if needed)


pc.create_index(
    name="semantic-search-demo",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-west-2"),
)
