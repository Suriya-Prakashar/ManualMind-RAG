import faiss
import numpy as np
from app.core.config import VECTORSTORE_DIR


class VectorDB:

    def __init__(self):

        self.index = None

    def create_index(self, embeddings):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(
            np.array(
                embeddings,
                dtype=np.float32
            )
        )

    def save(self):
        index_path = str(VECTORSTORE_DIR / "faiss.index")

        faiss.write_index(
            self.index,
            index_path
        )

        print("FAISS Vector Index Saved Successfully.")

    def load(self):
        index_path = str(VECTORSTORE_DIR / "faiss.index")

        self.index = faiss.read_index(
            index_path
        )

        print("FAISS Vector Index Loaded Successfully.")