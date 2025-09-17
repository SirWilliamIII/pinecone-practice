import os
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pinecone_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    
    # Required settings
    pinecone_api_key: str
    
    # Index settings
    index_name: str = "semantic-search-demo"
    dimension: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-west-2"
    
    # Document processing
    vault_path: str = "~/Markdown"
    max_text_length: int = 3000
    chunk_overlap: int = 200
    
    # Performance settings
    batch_size: int = 50  # Increased for better performance
    max_workers: int = 4
    upload_delay: float = 0.5  # Reduced delay between batches
    
    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")
        
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        if self.batch_size <= 0 or self.batch_size > 100:
            raise ValueError("Batch size must be between 1 and 100")
        
        self.vault_path = os.path.expanduser(self.vault_path)
        if not os.path.exists(self.vault_path):
            raise ValueError(f"Vault path does not exist: {self.vault_path}")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        load_dotenv()
        
        return cls(
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            index_name=os.getenv("INDEX_NAME", cls.index_name),
            vault_path=os.getenv("VAULT_PATH", cls.vault_path),
            dimension=int(os.getenv("DIMENSION", cls.dimension)),
            batch_size=int(os.getenv("BATCH_SIZE", cls.batch_size)),
            max_workers=int(os.getenv("MAX_WORKERS", cls.max_workers)),
        )


@dataclass
class Document:
    """Document data structure with metadata."""
    id: str
    text: str
    filename: str
    full_path: str
    category: str = "obsidian_note"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate document hash and add metadata."""
        self.metadata.update({
            'text_hash': self._generate_hash(),
            'text_length': len(self.text),
            'filename': self.filename,
            'category': self.category
        })
    
    def _generate_hash(self) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(self.text.encode()).hexdigest()


class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vault_path = Path(config.vault_path)
        
    def load_documents(self) -> List[Document]:
        """Load and process all markdown documents from vault."""
        logger.info(f"Loading documents from: {self.vault_path}")
        
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault path not found: {self.vault_path}")
        
        md_files = list(self.vault_path.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")
        
        documents = []
        failed_files = []
        
        for i, file_path in enumerate(md_files):
            try:
                doc = self._process_file(file_path, i)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append(str(file_path))
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _process_file(self, file_path: Path, index: int) -> Optional[Document]:
        """Process a single markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().strip()
            
            if not content:
                logger.debug(f"Skipping empty file: {file_path}")
                return None
            
            # Truncate content if too long
            if len(content) > self.config.max_text_length:
                content = content[:self.config.max_text_length]
                logger.debug(f"Truncated content for {file_path}")
            
            relative_path = file_path.relative_to(self.vault_path)
            
            return Document(
                id=f"md_{index + 1}",
                text=content,
                filename=str(relative_path.with_suffix('')),
                full_path=str(file_path),
            )
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            return None


class EmbeddingGenerator:
    """Handles embedding generation with optimizations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embeddings(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for documents with batch processing."""
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        # Extract texts for batch processing
        texts = [doc.text for doc in documents]
        
        try:
            # Generate embeddings in batch for better performance
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Assign embeddings back to documents
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding.tolist()
                logger.debug(f"Generated embedding for {doc.filename}: {len(doc.embedding)} dimensions")
            
            logger.info("Embedding generation complete")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise


class PineconeManager:
    """Manages Pinecone operations with retry logic and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pc = None
        self.index = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Pinecone client."""
        try:
            self.pc = Pinecone(api_key=self.config.pinecone_api_key)
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def ensure_index_exists(self):
        """Ensure the index exists, create if necessary."""
        try:
            if self.pc.has_index(name=self.config.index_name):
                logger.info(f"Using existing index: {self.config.index_name}")
            else:
                logger.info(f"Creating new index: {self.config.index_name}")
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region
                    )
                )
                self._wait_for_index_ready()
            
            self.index = self.pc.Index(self.config.index_name)
            return self.index
            
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise
    
    def _wait_for_index_ready(self, max_wait: int = 300):
        """Wait for index to be ready with polling."""
        logger.info("Waiting for index to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                if self.pc.has_index(name=self.config.index_name):
                    # Try to get index stats to confirm it's fully ready
                    temp_index = self.pc.Index(self.config.index_name)
                    temp_index.describe_index_stats()
                    logger.info("Index is ready!")
                    return
            except Exception:
                pass
            
            time.sleep(5)
        
        raise TimeoutError(f"Index not ready after {max_wait} seconds")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    def upsert_documents(self, documents: List[Document]) -> bool:
        """Upload documents to Pinecone with retry logic."""
        if not self.index:
            raise ValueError("Index not initialized")
        
        vectors = self._prepare_vectors(documents)
        total_batches = len(vectors) // self.config.batch_size + (1 if len(vectors) % self.config.batch_size else 0)
        
        logger.info(f"Uploading {len(vectors)} vectors in {total_batches} batches")
        
        successful_batches = 0
        failed_batches = []
        
        for i in range(0, len(vectors), self.config.batch_size):
            batch = vectors[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            
            try:
                logger.debug(f"Uploading batch {batch_num}/{total_batches}: {len(batch)} vectors")
                self.index.upsert(vectors=batch)
                successful_batches += 1
                
                # Brief delay to avoid rate limiting
                if self.config.upload_delay > 0:
                    time.sleep(self.config.upload_delay)
                    
            except Exception as e:
                logger.error(f"Failed to upload batch {batch_num}: {e}")
                failed_batches.append(batch_num)
        
        if failed_batches:
            logger.error(f"Failed to upload {len(failed_batches)} batches: {failed_batches}")
            return False
        
        logger.info(f"Successfully uploaded {successful_batches} batches")
        return True
    
    def _prepare_vectors(self, documents: List[Document]) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """Prepare vectors for upload."""
        vectors = []
        for doc in documents:
            if not doc.embedding:
                logger.warning(f"Document {doc.id} has no embedding, skipping")
                continue
            
            # Prepare metadata (Pinecone has size limits)
            metadata = {
                "text": doc.text[:1000],  # Limit text size in metadata
                "filename": doc.filename,
                "category": doc.category,
                "text_hash": doc.metadata.get("text_hash", ""),
                "text_length": doc.metadata.get("text_length", 0)
            }
            
            vectors.append((doc.id, doc.embedding, metadata))
        
        return vectors
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.index:
            raise ValueError("Index not initialized")
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": getattr(stats, 'dimension', self.config.dimension),
                "index_fullness": getattr(stats, 'index_fullness', 0.0)
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise


class SemanticSearchManager:
    """Main manager class orchestrating the entire pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.doc_processor = DocumentProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.pinecone_manager = PineconeManager(config)
    
    def index_documents(self) -> bool:
        """Complete pipeline to index documents."""
        try:
            logger.info("Starting document indexing pipeline")
            
            # Step 1: Load documents
            documents = self.doc_processor.load_documents()
            if not documents:
                logger.warning("No documents to process")
                return False
            
            # Step 2: Generate embeddings
            documents = self.embedding_generator.generate_embeddings(documents)
            
            # Step 3: Ensure Pinecone index exists
            self.pinecone_manager.ensure_index_exists()
            
            # Step 4: Upload to Pinecone
            success = self.pinecone_manager.upsert_documents(documents)
            
            if success:
                # Step 5: Verify upload
                stats = self.pinecone_manager.get_index_stats()
                logger.info(f"Indexing complete! Total vectors: {stats['total_vector_count']}")
                return True
            else:
                logger.error("Failed to upload documents")
                return False
                
        except Exception as e:
            logger.error(f"Indexing pipeline failed: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the indexed documents."""
        try:
            logger.info(f"Searching for: '{query}'")
            
            if not self.pinecone_manager.index:
                raise ValueError("Index not initialized")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.model.encode(query)
            
            # Perform search
            results = self.pinecone_manager.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.get('matches', []):
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'filename': match['metadata'].get('filename', 'Unknown'),
                    'text': match['metadata'].get('text', ''),
                    'category': match['metadata'].get('category', 'unknown')
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

def interactive_search(search_manager: SemanticSearchManager):
    """Interactive search loop for testing queries."""
    print("\n" + "="*60)
    print("INTERACTIVE SEARCH MODE")
    print("="*60)
    print("Enter your search queries (type 'quit' to exit)")
    print("Examples:")
    print("  - 'machine learning performance'")
    print("  - 'project management ideas'") 
    print("  - 'debugging techniques'")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            query = input("\nSearch query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                print("Please enter a search query")
                continue
            
            # Ask for number of results
            try:
                top_k_input = input("Number of results (default 5): ").strip()
                top_k = int(top_k_input) if top_k_input else 5
                top_k = max(1, min(top_k, 20))  # Limit between 1-20
            except ValueError:
                top_k = 5
            
            # Perform search
            print(f"\nSearching for: '{query}' (top {top_k} results)")
            print("-" * 40)
            
            results = search_manager.search(query, top_k=top_k)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['filename']}")
                    print(f"   Score: {result['score']:.4f}")
                    print(f"   Preview: \"{result['text'][:150]}...\"")
                    
                # Show similarity scores explanation
                print(f"\n📊 Score explanation:")
                print(f"   1.0 = perfect match, 0.0 = completely unrelated")
                print(f"   Scores above 0.7 are usually very relevant")
                print(f"   Scores 0.4-0.7 are somewhat related")
                print(f"   Scores below 0.4 might be noise")
                
            else:
                print("❌ No results found")
                print("Try a different query or check if documents are indexed")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Search error: {e}")
            continue
        
def main():
    """Main execution function."""
    try:
        # Load configuration
        config = Config.from_env()
        logger.info("Configuration loaded successfully")
        
        # Initialize search manager
        search_manager = SemanticSearchManager(config)
        
        # Ask user what they want to do
        print("\nWhat do you want to do?")
        print("1. Index documents (first time setup)")
        print("2. Search existing index")
        print("3. Both (index then search)")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            # Index documents
            print("\n🔄 Starting document indexing...")
            success = search_manager.index_documents()
            if not success:
                logger.error("Document indexing failed")
                return 1
            print("✅ Indexing complete!")
        
        if choice in ['2', '3']:
            # Interactive search
            interactive_search(search_manager)
        
        logger.info("Process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        return 1
    
    
def debug_connection():
    """Debug the connection issue."""
    print("🔄 DEBUG: Testing connection step by step...")
    
    try:
        # Step 1: Config
        config = Config.from_env()
        print(f"✅ Config loaded: {config.index_name}")
        
        # Step 2: Initialize components
        search_manager = SemanticSearchManager(config)
        print("✅ Search manager created")
        
        # Step 3: Check Pinecone client
        pc = search_manager.pinecone_manager.pc
        print(f"✅ Pinecone client: {pc}")
        
        # Step 4: Check if index exists
        has_index = pc.has_index(name=config.index_name)
        print(f"✅ Index exists: {has_index}")
        
        # Step 5: Try to connect to index
        if has_index:
            index = pc.Index(config.index_name)
            print(f"✅ Index object created: {index}")
            
            # Step 6: Try to get stats
            stats = index.describe_index_stats()
            print(f"✅ Stats: {stats.total_vector_count} vectors")
            
            # Step 7: Set the index in the manager
            search_manager.pinecone_manager.index = index
            print("✅ Index assigned to manager")
            
            # Step 8: Try search
            results = search_manager.search("test", top_k=1)
            print(f"✅ Search successful: {len(results)} results")
            
        else:
            print("❌ Index doesn't exist")
            
    except Exception as e:
        print(f"❌ Failed at step: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    exit(main())
