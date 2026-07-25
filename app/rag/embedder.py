from sentence_transformers import SentenceTransformer

from app.core.config import EMBEDDING_MODEL


class Embedder:
    """
    Load the embedding model once and reuse it.
    """

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded successfully.")

    def embed_text(self, text: str):
        """
        Generate an embedding for a single text.
        """
        return self.model.encode(
            text,
            normalize_embeddings=True,
        )

    def embed_documents(self, chunks: list):
        """
        Generate embeddings and attach them to chunk metadata.
        """
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        results = []

        for chunk, embedding in zip(chunks, embeddings):
            results.append(
                {
                    **chunk,
                    "embedding": embedding,
                }
            )

        return results