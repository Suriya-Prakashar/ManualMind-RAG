import litellm
import sys
from app.core.config import EMBEDDING_MODEL

class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions

    # --- Asynchronous Methods (User Added) ---

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = await litellm.aembedding(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.extend(item["embedding"] for item in response.data)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        response = await litellm.aembedding(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
            task_type="RETRIEVAL_QUERY",
        )
        return response.data[0]["embedding"]

    # --- Synchronous Methods (for backward compatibility with the codebase) ---

    def embed_text(self, text: str) -> list[float]:
        """Synchronously embeds a single text query."""
        response = litellm.embedding(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
            task_type="RETRIEVAL_QUERY",
        )
        return response.data[0]["embedding"]

    def embed_documents(self, chunks: list) -> list:
        """Synchronously embeds a list of document chunks and attaches the embeddings."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []

        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = litellm.embedding(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.extend(item["embedding"] for item in response.data)

        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                **chunk,
                "embedding": embedding,
            })
        return results

# Alias class name for compatibility with the rest of the codebase
Embedder = EmbeddingModel