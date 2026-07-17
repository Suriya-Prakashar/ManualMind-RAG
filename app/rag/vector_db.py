import faiss
import pickle
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

    def save(self, chunks):
        index_path = str(VECTORSTORE_DIR / "faiss.index")
        metadata_path = str(VECTORSTORE_DIR / "metadata.pkl")

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            metadata_path,
            "wb"
        ) as f:

            pickle.dump(chunks, f)

        print("Vector Database Saved Successfully.")

    def load(self):
        index_path = str(VECTORSTORE_DIR / "faiss.index")
        metadata_path = str(VECTORSTORE_DIR / "metadata.pkl")

        self.index = faiss.read_index(
            index_path
        )

        with open(
            metadata_path,
            "rb"
        ) as f:

            chunks = pickle.load(f)

        print("Vector Database Loaded Successfully.")

        return chunks