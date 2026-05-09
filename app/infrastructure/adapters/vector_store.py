"""Adapter para ChromaDB como vector store."""
import logging
from pathlib import Path
from typing import Optional

from app.domain.entities import DocumentChunk
from app.domain.ports import IVectorStore

logger = logging.getLogger(__name__)

ChromaDB_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    ChromaDB_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB no instalado: pip install chromadb")

if ChromaDB_AVAILABLE:
    Client = chromadb.PersistentClient


class ChromaDBAdapter(IVectorStore):
    """Adapter para ChromaDB como vector store."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "tutor_inteligente"
    ):
        if not ChromaDB_AVAILABLE:
            raise ImportError("ChromaDB no disponible. Instale con: pip install chromadb")

        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._client = Client(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Colección RAG del Tutor Inteligente"}
        )
        logger.info(f"ChromaDB inicializado: {collection_name}")

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Agrega chunks con sus embeddings al vector store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Número de chunks debe igualar número de embeddings")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "topic": chunk.topic,
                "page": chunk.page or 0
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Agregados {len(chunks)} chunks a ChromaDB")

    def query(
        self, 
        query_embedding: list[float], 
        top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Busca los chunks más similares."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        chunks_with_scores = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                chunk = DocumentChunk(
                    chunk_id=results["ids"][0][i],
                    content=doc,
                    source_file=metadata.get("source_file", ""),
                    topic=metadata.get("topic", "General"),
                    page=metadata.get("page", 0)
                )
                distance = results["distances"][0][i]
                similarity = 1 - distance
                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def get_count(self) -> int:
        """Retorna el número de chunks en la colección."""
        return self._collection.count()

    def clear(self) -> None:
        """Limpia todos los chunks de la colección."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )
        logger.info("Colección ChromaDB limpiada")


class InMemoryVectorStore(IVectorStore):
    """Vector store en memoria para testing sin dependencias externas."""

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def query(
        self, 
        query_embedding: list[float], 
        top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        if not self._embeddings:
            return []

        similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self._embeddings
        ]

        indexed = list(zip(self._chunks, similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()
