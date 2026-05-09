"""Retrievier RAG: recupera contexto y genera respuestas."""
import logging
import re
from typing import Optional

from app.domain.entities import DocumentChunk, Query, AnswerResponse
from app.domain.ports import IVectorStore, ILLMGateway
from app.infrastructure.adapters.embeddings import EmbeddingsAdapter, MockEmbeddingsAdapter

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Pipeline de recuperación y respuesta RAG."""

    INJECTION_PATTERNS = [
        r"\[/INST\]",
        r"<script>",
        r"</script>",
        r"{{",
        r"}}",
        r"\\[system\\]",
        r"\\[INST\\]",
    ]

    def __init__(
        self,
        vector_store: IVectorStore,
        embeddings: EmbeddingsAdapter,
        llm: ILLMGateway,
        similarity_threshold: float = 0.5
    ):
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._llm = llm
        self._threshold = similarity_threshold

    def sanitize_query(self, query_text: str) -> str:
        """Sanitiza la query contra prompt injection."""
        sanitized = query_text
        for pattern in self.INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

    def retrieve(self, query: Query, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """
        Recupera los chunks más relevantes para una query.
        
        Args:
            query: Objeto Query con el texto.
            top_k: Número de fragmentos a recuperar.
            
        Returns:
            Lista de tuplas (DocumentChunk, score_similitud).
        """
        sanitized = self.sanitize_query(query.text)
        query_embedding = self._embeddings.embed_single(sanitized)
        results = self._vector_store.query(query_embedding, top_k)
        
        logger.info(f"Recuperados {len(results)} chunks para query")
        return results

    def retrieve_and_answer(
        self, 
        query_text: str, 
        top_k: int = 5
    ) -> AnswerResponse:
        """
        Recupera contexto y genera respuesta completa.
        
        Args:
            query_text: Pregunta del estudiante.
            top_k: Número de fragmentos a recuperar.
            
        Returns:
            AnswerResponse con respuesta y fuentes.
        """
        sanitized = self.sanitize_query(query_text)
        query = Query(text=sanitized)

        results = self.retrieve(query, top_k)
        relevant = [
            (chunk, score) for chunk, score in results 
            if score >= self._threshold
        ]

        if not relevant:
            return AnswerResponse(
                answer="No tengo información suficiente en mis fuentes para responder esta pregunta.",
                sources=[]
            )

        prompt = self._build_prompt(relevant, sanitized)
        answer = self._llm.generate(prompt)

        sources = [
            {
                "filename": chunk.source_file,
                "topic": chunk.topic,
                "excerpt": chunk.content[:150] + "...",
                "relevance_score": round(score, 3)
            }
            for chunk, score in relevant
        ]

        return AnswerResponse(answer=answer, sources=sources)

    def _build_prompt(
        self, 
        relevant_chunks: list[tuple[object, float]], 
        query: str
    ) -> str:
        """Construye el prompt aumentado con el contexto recuperado."""
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"[Fragmento {i}] (fuente: {chunk.source_file}, "
                f"tema: {chunk.topic})\n{chunk.content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Eres un tutor inteligente de Inteligencia Artificial.

Responde la pregunta del estudiante basándote EXCLUSIVAMENTE en el contexto
proporcionado a continuación. Si no tienes suficiente información, indica
claramente que no puedes responder.

CONTEXTO:
{context}

PREGUNTA: {query}

Responde de forma pedagógica, cita las fuentes usadas."""
        return prompt

    def get_context_count(self) -> int:
        """Retorna el número de chunks en el vector store."""
        if hasattr(self._vector_store, "get_count"):
            return self._vector_store.get_count()
        return 0


def create_retriever(
    vector_store: Optional[IVectorStore] = None
) -> RAGRetriever:
    """
    Factory function para crear el retriever configurado.
    
    Args:
        vector_store: Vector store a usar (crea ChromaDB por defecto).
        
    Returns:
        Instancia configurada de RAGRetriever.
    """
    if vector_store is None:
        from app.infrastructure.adapters.vector_store import ChromaDBAdapter
        vector_store = ChromaDBAdapter(
            persist_directory="./chroma_db",
            collection_name="tutor_inteligente"
        )

    embeddings = MockEmbeddingsAdapter()
    llm = MockLLMAdapter()

    return RAGRetriever(vector_store, embeddings, llm)


def _is_openai_available() -> bool:
    try:
        from openai import OpenAI
        return True
    except ImportError:
        return False


def _create_openai_llm():
    from app.infrastructure.adapters.llm import OpenAIAdapter
    return OpenAIAdapter(model="gpt-4o")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = create_retriever()
    print(f"Vector store tiene {retriever.get_context_count()} chunks")
