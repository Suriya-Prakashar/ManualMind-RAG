from pathlib import Path
import os
import faiss
import numpy as np

from app.core.config import PDF_PATH, VECTORSTORE_DIR, TOP_K
from app.rag.pdf_uploader import PDFLoader
from app.rag.cleaner import TextCleaner, TextChunker
from app.rag.embedder import Embedder
from app.rag.vector_db import VectorDB
from app.rag.retriever import Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:

    def __init__(self):
        self.embedder = Embedder()
        self.vector_db = VectorDB()
        self.metadata = []
        self.retriever = None

        # Build or load index
        self.initialize_index()

    def initialize_index(self, force_rebuild=False):
        index_path = VECTORSTORE_DIR / "faiss.index"
        metadata_path = VECTORSTORE_DIR / "metadata.pkl"

        if not force_rebuild and index_path.exists() and metadata_path.exists():
            logger.info("Loading existing vector database...")
            self.metadata = self.vector_db.load()
            self.retriever = Retriever(self.vector_db, self.embedder)
        else:
            logger.info("Vector database not found or rebuild requested. Initiating ingestion pipeline...")
            self.build_index()

    def build_index(self):
        # Create output directory if it doesn't exist
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load PDF
        logger.info(f"Loading PDF from {PDF_PATH}...")
        loader = PDFLoader(PDF_PATH)
        pages = loader.load_pdf()

        # 2. Clean Text
        logger.info("Cleaning text...")
        cleaner = TextCleaner()
        cleaned_pages = cleaner.clean(pages)

        # 3. Chunk Text
        logger.info("Chunking text...")
        chunker = TextChunker()
        chunks = chunker.chunk(cleaned_pages)
        self.metadata = chunks

        # 4. Generate Embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.embed_chunks(chunks)

        # 5. Create and Save Vector DB
        logger.info("Creating FAISS index...")
        embeddings_arr = np.array(embeddings, dtype=np.float32)
        self.vector_db.create_index(embeddings_arr)
        self.vector_db.save(chunks)

        # 6. Initialize retriever
        self.retriever = Retriever(self.vector_db, self.embedder)
        logger.info("Ingestion pipeline completed successfully.")

    def query(self, question: str, top_k: int = TOP_K) -> list:
        if not self.retriever:
            raise ValueError("Retriever is not initialized. Please build or load the index first.")
        return self.retriever.search(question, self.metadata, top_k=top_k)
