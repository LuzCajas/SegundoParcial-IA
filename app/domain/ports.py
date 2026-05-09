"""Puertos (interfaces) del dominio para arquitectura hexagonal."""
from typing import Protocol

from app.domain.entities import DocumentChunk, Query, AnswerResponse


class IIndexer(Protocol):
    """Puerto para indexación de documentos."""

    def index_documents(self, sources_dir: str) -> dict:
        """Indexa todos los documentos del directorio."""
        ...


class IRetriever(Protocol):
    """Puerto para recuperación de contexto."""

    def retrieve(self, query: Query, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Recupera los chunks más relevantes para la query."""
        ...


class ILLMGateway(Protocol):
    """Puerto para comunicación con el LLM."""

    def generate(self, prompt: str) -> str:
        """Genera una respuesta a partir del prompt."""
        ...


class IDocumentLoader(Protocol):
    """Puerto para carga de documentos."""

    def load(self, file_path: str) -> list[DocumentChunk]:
        """Carga y fragmenta un documento."""
        ...


class IVectorStore(Protocol):
    """Puerto para el almacén de vectores."""

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Agrega chunks con sus embeddings."""
        ...

    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[DocumentChunk, float]]:
        """Busca los chunks más similares."""
        ...
