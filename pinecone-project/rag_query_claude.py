#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import anthropic

# Configuration
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
claude_api_key = os.getenv("ANTHROPIC_API_KEY")  # Add this to your .env
index_name = "multimodal-search-v2"

# Initialize
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
model = SentenceTransformer("clip-ViT-B-32")

# Set up Claude
client = anthropic.Anthropic(api_key=claude_api_key)

def search_knowledge_base(question, top_k=5):
    """Search your knowledge base for relevant context."""
    print(f"🔍 Searching for: {question}")

    # 1. Embed the question
    query_embedding = model.encode(question).tolist()

    # 2. Search your index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # 3. Extract context from results
    contexts = []
    sources = []

    for match in results.matches:
        contexts.append(match.metadata['text'])
        sources.append({
            'filename': match.metadata.get('filename', 'Unknown'),
            'category': match.metadata.get('category', 'Unknown'),
            'file_type': match.metadata.get('file_type', 'text'),
            'score': round(match.score, 3)
        })

    return contexts, sources

def ask_claude(question, contexts):
    """Send question + context to Claude for final answer."""

    # Build the prompt
    context_text = "\n\n".join(contexts[:3])  # Limit to top 3 for token efficiency

    prompt = f"""You are a helpful assistant answering questions based on the user's personal knowledge base.

Context from knowledge base:
{context_text}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite relevant details from the context
- Keep your answer concise but comprehensive

Answer:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    except Exception as e:
        return f"Error calling Claude: {e}"

def ask_question(question):
    """Complete RAG pipeline: retrieve + generate."""

    # Step 1: Retrieve relevant context
    contexts, sources = search_knowledge_base(question)

    if not contexts:
        return "No relevant information found in your knowledge base."

    # Step 2: Generate answer using Claude
    print(f"📚 Found {len(contexts)} relevant chunks")
    answer = ask_claude(question, contexts)

    # Step 3: Show sources
    print(f"\n🤖 Claude's Answer:\n{answer}")
    print(f"\n📖 Sources:")
    for i, source in enumerate(sources, 1):
        file_type_emoji = "🖼️" if source['file_type'] == 'image' else "📄" if source['file_type'] == 'pdf' else "📝"
        print(f"  {i}. {file_type_emoji} {source['filename']} ({source['category']}) - Score: {source['score']}")

    return answer, sources

def interactive_mode():
    """Interactive Q&A session."""
    print("🧠 RAG System with Claude Ready! Ask questions about your knowledge base.")
    print("Your knowledge base includes:")
    print("  📝 Markdown files (CockroachDB, AWS, Security, etc.)")
    print("  📄 PDF documents")
    print("  🖼️ Images (diagrams, screenshots)")
    print("\nType 'quit' to exit.\n")

    while True:
        question = input("❓ Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print(f"\n{'='*50}")
        try:
            ask_question(question)
        except Exception as e:
            print(f"❌ Error: {e}")
        print(f"{'='*50}\n")

# Sample questions to try:
sample_questions = [
    "What is CockroachDB and how does it work?",
    "How do I set up networking in AWS?",
    "What are some common security frameworks?",
    "Show me Python cheat sheet information",
    "What's in my Spanish learning notes?"
]

if __name__ == "__main__":
    print("🚀 Testing RAG Pipeline with Claude...")
    print("\nSample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")

    print("\n" + "="*60)

    # Start interactive mode
    interactive_mode()