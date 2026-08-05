from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import GEMINI_API_KEY, EMBEDDING_MODEL


class Embedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = 768,
    ):
        # Strip provider prefix if present (e.g. "gemini/gemini-embedding-001" -> "gemini-embedding-001")
        if model_name.startswith("gemini/"):
            model_name = model_name.replace("gemini/", "", 1)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            output_dimensionality=dimension,
        )

    def embed_documents(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of chunk dicts synchronously.
        """
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single query text synchronously.
        """
        return self.embeddings.embed_query(text)