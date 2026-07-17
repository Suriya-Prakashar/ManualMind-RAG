from sentence_transformers import SentenceTransformer
from app.core.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded successfully.")

    def embed_text(self, text):
        """
        Generate an embedding for a single text.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding

    def embed_chunks(self, chunks):
        """
        Generate embeddings for all chunks.
        """
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings