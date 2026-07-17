import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.rag.pipeline import RAGPipeline
from app.core.config import TOP_K

print("Initializing RAGPipeline...")
p = RAGPipeline()
print("RAGPipeline initialized successfully.")

question = "What is NP740VJG?"
print(f"Querying: '{question}'")
results = p.query(question, top_k=TOP_K)

print(f"Found {len(results)} results:")
for r in results:
    print(f"Page: {r['page']}, Score: {r.get('score', 0.0)}")
    print(r['text'][:200])
    print("-" * 50)
