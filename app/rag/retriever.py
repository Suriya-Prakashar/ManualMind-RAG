import numpy as np


class Retriever:

    def __init__(self, vector_db, embedder, db=None):
        self.vector_db = vector_db
        self.embedder = embedder
        self.db = db

    def search(self, question, top_k=3):
        """
        Search the most relevant chunks.
        """

        # Convert question to embedding
        question_embedding = self.embedder.embed_text(question)

        # FAISS expects shape (1, dimension)
        question_embedding = np.array(
            [question_embedding],
            dtype=np.float32
        )

        # Search
        scores, indices = self.vector_db.index.search(
            question_embedding,
            top_k
        )

        if not self.db or not self.db.is_connected:
            raise RuntimeError("MongoDB is not connected. MongoDB is required for retrieving chunks.")

        # Fetch chunks from MongoDB
        valid_ids = [int(idx) for idx in indices[0] if idx != -1]
        try:
            db_chunks = self.db.get_chunks_by_ids(valid_ids)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch chunks from MongoDB: {e}")

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            index_int = int(index)
            
            # Retrieve from MongoDB
            if index_int in db_chunks:
                chunk = db_chunks[index_int].copy()
                chunk.pop('_id', None)
                chunk["score"] = float(score)
                chunk["chunk_id"] = chunk.get("chunk_id", index_int)
                results.append(chunk)

        return results