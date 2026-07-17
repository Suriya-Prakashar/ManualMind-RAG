import sys
import io
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("Testing health endpoint...")
response = client.get("/")
print("Health status:", response.status_code, response.json())

# Test normal chat endpoint
print("\nTesting normal POST /chat endpoint...")
question = "What is NP740VJG?"
print(f"Question: '{question}'")
response = client.post("/chat", json={"question": question}, params={"stream": False})
print("Chat status:", response.status_code)
if response.status_code == 200:
    res_data = response.json()
    print("Reply:", res_data["reply"])
    print("Provider (Header):", response.headers.get("X-RAG-Provider"))
    print("Model (Header):", response.headers.get("X-RAG-Model"))
    print("Fallback (Header):", response.headers.get("X-RAG-Fallback"))
    print("Stream (Header):", response.headers.get("X-RAG-Stream"))
    print("Sources found:", len(res_data.get("sources") or []))
    for s in res_data.get("sources", [])[:2]:
        print(f"  - Page {s['page']} (Score: {s['score']:.4f})")
else:
    print("Error response:", response.text)

# Test streaming chat endpoint
print("\nTesting streaming POST /chat endpoint...")
response = client.post("/chat", json={"question": question}, params={"stream": True})
print("Streaming status:", response.status_code)
print("Provider (Header):", response.headers.get("X-RAG-Provider"))
print("Model (Header):", response.headers.get("X-RAG-Model"))
print("Fallback (Header):", response.headers.get("X-RAG-Fallback"))
print("Stream (Header):", response.headers.get("X-RAG-Stream"))
if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            print("Stream line:", json.loads(line))
else:
    print("Error response:", response.text)
