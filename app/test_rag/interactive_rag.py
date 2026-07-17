import sys
import os
import io

if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.rag.pipeline import RAGPipeline
from app.services.fallback import FallbackService
from app.core.config import TOP_K


def main():
    print("Loading RAG Pipeline and Embedding model...")
    pipeline = RAGPipeline()
    fallback = FallbackService()
    print("RAG System Ready! Type your question below.")
    print("To exit, type 'exit' or 'quit'.\n")
    
    while True:
        try:
            question = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not question:
            continue
            
        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
            
        print(f"\nProcessing query: '{question}'...")
        
        # 1. Retrieve top_k chunks using cosine similarity
        chunks = pipeline.query(question, top_k=TOP_K)
        if not chunks:
            print("No relevant chunks found in the database.\n")
            continue
            
        # 2. Build context
        context = "\n\n".join([f"Page {c['page']}:\n{c['text']}" for c in chunks])
        
        # 3. Generate humanised answer
        print("Generating humanised answer from LLM...")
        try:
            answer_dict = fallback.generate(question, context=context)
            reply = answer_dict.get("reply", "No reply generated.")
        except Exception as e:
            reply = f"Error generating answer: {e}"
            
        # 4. Print outputs formatted in separate rows
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)
        print("HUMANISED ANSWER:")
        print("-" * 80)
        print(reply)
        print("=" * 80)
        
        print("\nTOP CHUNKS RETRIEVED:")
        print("=" * 80)
        for idx, chunk in enumerate(chunks):
            print(f"PAGE NUMBER: {chunk['page']}")
            print(f"CHUNK ID: {chunk.get('chunk_index', idx)}")
            print(f"SIMILARITY SCORE: {chunk.get('score', 0.0):.4f}")
            print("-" * 80)
            print("CHUNK CONTENT:")
            print(chunk['text'])
            print("=" * 80)
        print("\n")


if __name__ == "__main__":
    main()
