"""Casos de uso de aplicación."""
import logging
from typing import Optional

from app.domain.entities import (
    DocumentChunk, Query, AnswerResponse
)
from app.domain.ports import (
    IIndexer, IRetriever, ILLMGateway
)

logger = logging.getLogger(__name__)


class IndexDocumentsUC:
    """Caso de uso para indexar documentos en el repositorio."""

    def __init__(self, indexer: IIndexer):
        self._indexer = indexer

    def execute(self, sources_dir: str = "./fuentes") -> dict:
        """Ejecuta la indexación de documentos."""
        logger.info(f"Iniciando indexación de: {sources_dir}")
        result = self._indexer.index_documents(sources_dir)
        logger.info(f"Indexación completada: {result.get('chunks', 0)} chunks")
        return result


class AskTutorUC:
    """Caso de uso para responder preguntas del estudiante."""

    SYSTEM_PROMPT = """Eres un tutor inteligente de Inteligencia Artificial. 
Tu rol es responder preguntas de estudiantes basándote EXCLUSIVAMENTE en el 
contexto recuperado de tus fuentes documentales.

REGLAS OBLIGATORIAS:
1. Responde SOLO con información del contexto proporcionado
2. Si la información es insuficiente, responde: "No tengo información suficiente en mis fuentes para responder esta pregunta."
3. Cita las fuentes usadas usando el formato [fuente: nombre_archivo, tema: tema]
4. Sé pedagógico y claro en tus explicaciones
5. Si hay información contradictoria en el contexto, señálalo

CONTEXTO RECUPERADO:
{context}
"""

    def __init__(
        self,
        retriever: IRetriever,
        llm: ILLMGateway,
        similarity_threshold: float = 0.5
    ):
        self._retriever = retriever
        self._llm = llm
        self._threshold = similarity_threshold

    def execute(self, query_text: str, student_id: Optional[str] = None) -> AnswerResponse:
        """Procesa una consulta del estudiante."""
        query = Query(text=query_text, student_id=student_id)
        logger.info(f"Procesando query: {query_text[:50]}...")

        retrieved = self._retriever.retrieve(query, top_k=5)
        relevant_chunks = [
            (chunk, score) for chunk, score in retrieved 
            if score >= self._threshold
        ]

        if not relevant_chunks:
            logger.warning("No se encontraron chunks relevantes")
            return AnswerResponse(
                answer="No tengo información suficiente en mis fuentes para responder esta pregunta.",
                sources=[]
            )

        context = self._build_context(relevant_chunks)
        prompt = self._build_prompt(context, query_text)
        answer = self._llm.generate(prompt)

        sources = [
            {
                "filename": chunk.source_file,
                "topic": chunk.topic,
                "excerpt": chunk.content[:200] + "...",
                "relevance_score": round(score, 3)
            }
            for chunk, score in relevant_chunks
        ]

        return AnswerResponse(answer=answer, sources=sources)

    def _build_context(
        self, 
        chunks_with_scores: list[tuple[DocumentChunk, float]]
    ) -> str:
        """Construye el contexto formateado para el prompt."""
        context_parts = []
        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            context_parts.append(
                f"[Fragmento {i}] ({chunk.source_file}, tema: {chunk.topic}, "
                f"relevancia: {score:.2f})\n{chunk.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, context: str, query: str) -> str:
        """Construye el prompt completo con system message."""
        return self.SYSTEM_PROMPT.format(context=context) + f"\n\nPREGUNTA DEL ESTUDIANTE:\n{query}"
