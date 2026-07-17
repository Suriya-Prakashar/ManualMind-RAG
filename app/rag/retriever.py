import numpy as np


class Retriever:

    def __init__(self, vector_db, embedder):
        self.vector_db = vector_db
        self.embedder = embedder

    def search(self, question, metadata, top_k=3):
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

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            chunk = metadata[index].copy()

            chunk["score"] = float(score)
            chunk["chunk_id"] = chunk.get("chunk_id", int(index))

            results.append(chunk)

        return results