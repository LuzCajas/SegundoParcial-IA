"""Entidades del dominio para el Tutor Inteligente."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DocumentChunk:
    """Representa un fragmento de documento indexado."""
    chunk_id: str
    content: str
    source_file: str
    topic: str
    page: Optional[int] = None


@dataclass(frozen=True)
class Query:
    """Representa una consulta del estudiante."""
    text: str
    student_id: Optional[str] = None


@dataclass
class RetrievedContext:
    """Contexto recuperado del vector store para una query."""
    query: Query
    chunks: list[DocumentChunk]
    scores: list[float]


@dataclass
class AnswerResponse:
    """Respuesta generada por el tutor."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    topic: Optional[str] = None

    @property
    def has_context(self) -> bool:
        return len(self.chunks) > 0 if hasattr(self, 'chunks') else len(self.sources) > 0

    @property
    def is_insufficient(self) -> bool:
        return self.answer == "No tengo información suficiente en mis fuentes para responder esta pregunta."
