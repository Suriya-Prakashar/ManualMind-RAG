from pathlib import Path
import os
import faiss
import numpy as np

from app.core.config import PDF_PATH, VECTORSTORE_DIR, TOP_K
from app.rag.pdf_uploader import PDFLoader
from app.rag.cleaner import process_pages
from app.rag.embedder import Embedder
from app.rag.vector_db import VectorDB
from app.rag.retriever import Retriever
from app.core.database import MongoDatabase
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:

    def __init__(self):
        self.embedder = Embedder()
        self.vector_db = VectorDB()
        self.db = MongoDatabase()
        self.retriever = None

        # Build or load index
        self.initialize_index()

    def initialize_index(self, force_rebuild=False):
        index_path = VECTORSTORE_DIR / "faiss.index"

        # Check if MongoDB is connected and empty to trigger automatic ingestion
        db_empty = False
        if self.db.is_connected:
            try:
                count = self.db.collection.count_documents({})
                if count == 0:
                    logger.info("MongoDB database is connected but has no documents. Forcing ingestion pipeline to inject data...")
                    db_empty = True
            except Exception as e:
                logger.warning(f"Failed to verify MongoDB document count: {e}")

        if not force_rebuild and index_path.exists() and not db_empty:
            logger.info("Loading existing FAISS vector database...")
            self.vector_db.load()
            self.retriever = Retriever(self.vector_db, self.embedder, self.db)
            
            if not self.db.is_connected:
                logger.warning("MongoDB is offline. Dynamic metadata queries will fail.")
        else:
            if db_empty:
                logger.info("Initiating automatic data ingestion to populate MongoDB...")
            else:
                logger.info("Vector database index not found or rebuild requested. Initiating ingestion pipeline...")
            self.build_index()

    def build_index(self):
        if not self.db.is_connected:
            raise RuntimeError("MongoDB is not connected. MongoDB is required to store chunk metadata.")

        # Create output directory if it doesn't exist
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load PDF
        logger.info(f"Loading PDF from {PDF_PATH}...")
        loader = PDFLoader(PDF_PATH)
        pages = loader.load_pdf()

        # 2. Clean and Chunk Text
        logger.info("Cleaning and chunking text...")
        chunks = process_pages(pages)

        # 3. Generate Embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embedded_chunks = self.embedder.embed_documents(chunks)

        # 4. Create and Save Vector DB
        logger.info("Creating FAISS index...")
        embeddings = [ec["embedding"] for ec in embedded_chunks]
        embeddings_arr = np.array(embeddings, dtype=np.float32)
        self.vector_db.create_index(embeddings_arr)
        self.vector_db.save()

        # Save to MongoDB (including the embedding converted to serializable list)
        logger.info("Saving chunks and embeddings to MongoDB...")
        chunks_to_save = []
        for ec in embedded_chunks:
            chunk_copy = {k: v for k, v in ec.items() if k != "embedding"}
            chunk_copy["embedding"] = ec["embedding"].tolist() if hasattr(ec["embedding"], "tolist") else list(ec["embedding"])
            chunks_to_save.append(chunk_copy)

        self.db.save_chunks(chunks_to_save)

        # 5. Initialize retriever
        self.retriever = Retriever(self.vector_db, self.embedder, self.db)
        logger.info("Ingestion pipeline completed successfully.")

    def query(self, question: str, top_k: int = TOP_K) -> list:
        if not self.retriever:
            raise ValueError("Retriever is not initialized. Please build or load the index first.")
        return self.retriever.search(question, top_k=top_k)

