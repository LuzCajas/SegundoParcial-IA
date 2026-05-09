"""Tests para el dominio y casos de uso."""
import pytest
from app.domain.entities import DocumentChunk, Query, AnswerResponse
from app.infrastructure.adapters.vector_store import InMemoryVectorStore
from app.infrastructure.adapters.llm import MockLLMAdapter


class TestEntities:
    """Tests para las entidades del dominio."""

    def test_document_chunk_creation(self):
        chunk = DocumentChunk(
            chunk_id="test_1",
            content="Contenido de prueba",
            source_file="test.pdf",
            topic="Testing"
        )
        assert chunk.chunk_id == "test_1"
        assert chunk.content == "Contenido de prueba"
        assert chunk.topic == "Testing"

    def test_query_creation(self):
        query = Query(text="¿Qué es A*?", student_id="student_123")
        assert query.text == "¿Qué es A*?"
        assert query.student_id == "student_123"

    def test_answer_response_insufficient(self):
        response = AnswerResponse(
            answer="No tengo información suficiente en mis fuentes para responder esta pregunta.",
            sources=[]
        )
        assert response.is_insufficient is True


class TestVectorStore:
    """Tests para el vector store en memoria."""

    def test_add_and_query(self):
        store = InMemoryVectorStore()
        
        chunks = [
            DocumentChunk("c1", "A* es un algoritmo", "test.pdf", "Búsqueda"),
            DocumentChunk("c2", "Prolog usa inferencia", "prolog.pdf", "Prolog"),
        ]
        embeddings = [[1.0, 0.0], [0.5, 0.5]]
        
        store.add(chunks, embeddings)
        
        results = store.query([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][1] > results[1][1]

    def test_clear(self):
        store = InMemoryVectorStore()
        store.add([], [])
        store.clear()
        assert store.query([1.0, 0.0], top_k=5) == []


class TestMockLLM:
    """Tests para el LLM mock."""

    def test_mock_response(self):
        llm = MockLLMAdapter()
        response = llm.generate("¿Qué es A*?")
        assert "A*" in response
        assert llm.call_count == 1

    def test_mock_prolog(self):
        llm = MockLLMAdapter()
        response = llm.generate("¿Qué es Prolog?")
        assert "Prolog" in response


class TestRAGPipeline:
    """Tests de integración para el pipeline RAG."""

    def test_full_pipeline_mock(self):
        from app.infrastructure.retriever import RAGRetriever
        
        store = InMemoryVectorStore()
        chunks = [
            DocumentChunk("c1", "A* combina costo real más heurística", "guia.pdf", "Búsqueda"),
            DocumentChunk("c2", "Una heurística es admisible si nunca sobreestima", "guia.pdf", "Búsqueda"),
        ]
        embeddings = [[0.9, 0.1], [0.1, 0.9]]
        store.add(chunks, embeddings)
        
        class SimpleEmbedder:
            def embed_single(self, text):
                if "heurística" in text or "A*" in text:
                    return [0.9, 0.1]
                return [0.5, 0.5]
        
        llm = MockLLMAdapter()
        retriever = RAGRetriever(store, SimpleEmbedder(), llm, 0.3)
        
        response = retriever.retrieve_and_answer("¿Qué es A*?")
        assert response.answer is not None
        assert len(response.sources) > 0
