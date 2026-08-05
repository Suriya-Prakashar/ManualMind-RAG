from app.core.config import TOP_K 


class Retriever:

    def __init__(self, embedder, db):
        self.embedder = embedder
        self.db = db

    def search(self, question, top_k=TOP_K):
        """
        Search the most relevant chunks using MongoDB Atlas Vector Search.
        """
        if not self.db or not self.db.is_connected:
            raise RuntimeError("MongoDB is not connected. MongoDB is required for retrieving chunks.")

        # 1. Embed query
        query_vector = self.embedder.embed_text(question)

        # 2. Query MongoDB using Atlas Vector Search
        try:
            results = self.db.vector_search(query_vector, top_k=top_k)
        except Exception as e:
            raise RuntimeError(f"Failed to perform MongoDB vector search: {e}")

        return results

