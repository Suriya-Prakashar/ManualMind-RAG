from pathlib import Path
import os

from app.core.config import PDF_PATH, TOP_K
from app.rag.pdf_uploader import PDFLoader
from app.rag.cleaner import process_pages
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.core.database import MongoDatabase
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:

    def __init__(self):
        self.embedder = Embedder()
        self.db = MongoDatabase()
        self.retriever = Retriever(self.embedder, self.db)

        # Build or load index with robust error handling
        try:
            self.initialize_index()
        except Exception as e:
            logger.error(
                f"\n" + "="*80 + "\n"
                f"CRITICAL ERROR: Failed to initialize RAG Pipeline on startup: {e}\n"
                f"The application will start, but queries to the /chat endpoint will fail.\n"
                f"Please ensure MongoDB is online and the PDF manual is placed in '{PDF_PATH}'.\n"
                f"="*80 + "\n"
            )

    def initialize_index(self, force_rebuild=False):
        # Check if MongoDB is connected and empty to trigger automatic ingestion
        db_empty = True
        if self.db.is_connected:
            try:
                count = self.db.collection.count_documents({})
                if count > 0:
                    db_empty = False
            except Exception as e:
                logger.warning(f"Failed to verify MongoDB document count: {e}")

        if not force_rebuild and not db_empty:
            logger.info("MongoDB contains data. RAG Pipeline is ready for queries.")
            if not self.db.is_connected:
                logger.warning("MongoDB is offline. Queries will fail.")
        else:
            if db_empty:
                logger.info("Initiating automatic data ingestion to populate MongoDB...")
            else:
                logger.info("Vector database rebuild requested. Initiating ingestion pipeline...")
            self.build_index()

    def build_index(self):
        if not self.db.is_connected:
            raise RuntimeError("MongoDB is not connected. MongoDB is required to store chunk metadata and embeddings.")

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

        # 4. Save to MongoDB (including the embedding converted to serializable list)
        logger.info("Saving chunks and embeddings to MongoDB...")
        chunks_to_save = []
        for ec in embedded_chunks:
            chunk_copy = {k: v for k, v in ec.items() if k != "embedding"}
            chunk_copy["embedding"] = ec["embedding"].tolist() if hasattr(ec["embedding"], "tolist") else list(ec["embedding"])
            chunks_to_save.append(chunk_copy)

        self.db.save_chunks(chunks_to_save)
        logger.info("Ingestion pipeline completed successfully.")

    def query(self, question: str, top_k: int = TOP_K) -> list:
        if not self.retriever:
            raise ValueError("Retriever is not initialized. Please build or load the index first.")
        return self.retriever.search(question, top_k=top_k)
