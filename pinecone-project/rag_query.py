#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import openai

# Configuration
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")  # Add this to your .env
index_name = "multimodal-search-v2"

# Initialize
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
model = SentenceTransformer("clip-ViT-B-32")

# Set up OpenAI (or you can use Anthropic's Claude API)
openai.api_key = openai_api_key

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
            'score': round(match.score, 3)
        })

    return contexts, sources

def ask_llm(question, contexts):
    """Send question + context to LLM for final answer."""

    # Build the prompt
    context_text = "\n\n".join(contexts)

    prompt = f"""You are a helpful assistant that answers questions based on the provided context from the user's knowledge base.

Context from knowledge base:
{context_text}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite relevant details from the context
- If you reference information, mention which source it came from

Answer:"""

    try:
        # Using OpenAI GPT (you can switch to Claude API here)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling LLM: {e}"

def ask_question(question):
    """Complete RAG pipeline: retrieve + generate."""

    # Step 1: Retrieve relevant context
    contexts, sources = search_knowledge_base(question)

    if not contexts:
        return "No relevant information found in your knowledge base."

    # Step 2: Generate answer using LLM
    print(f"📚 Found {len(contexts)} relevant chunks")
    answer = ask_llm(question, contexts)

    # Step 3: Show sources
    print(f"\n🤖 Answer:\n{answer}")
    print(f"\n📖 Sources:")
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source['filename']} ({source['category']}) - Score: {source['score']}")

    return answer, sources

def interactive_mode():
    """Interactive Q&A session."""
    print("🧠 RAG System Ready! Ask questions about your knowledge base.")
    print("Type 'quit' to exit.\n")

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

if __name__ == "__main__":
    # Test with a sample question
    sample_question = "What is CockroachDB and how does it work?"

    print("🚀 Testing RAG Pipeline...")
    ask_question(sample_question)

    print("\n" + "="*60)

    # Start interactive mode
    interactive_mode()