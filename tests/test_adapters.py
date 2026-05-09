"""Tests para los adapters de infraestructura."""
import pytest
from pathlib import Path
import tempfile
import os

from app.domain.entities import DocumentChunk
from app.infrastructure.adapters.document_loader import PyMuPDFAdapter


class TestDocumentLoader:
    """Tests para el loader de documentos."""

    def test_markdown_parsing(self):
        loader = PyMuPDFAdapter()
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            f.write("# Búsqueda Informada\n\n")
            f.write("A* es un algoritmo de búsqueda informada.\n\n")
            f.write("## Heurísticas\n\n")
            f.write("Una heurística admisible nunca sobreestima.\n")
            temp_path = f.name
        
        try:
            chunks = loader.load(temp_path)
            assert len(chunks) > 0
            assert any("A*" in c.content for c in chunks)
            assert any("heurístic" in c.content for c in chunks)
        finally:
            os.unlink(temp_path)

    def test_prolog_parsing(self):
        loader = PyMuPDFAdapter()
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pl', delete=False, encoding='utf-8'
        ) as f:
            f.write("padre(juan, pedro).\n")
            f.write("madre(maria, pedro).\n")
            f.write("progenitor(X, Y) :- padre(X, Y).\n")
            f.write("progenitor(X, Y) :- madre(X, Y).\n")
            temp_path = f.name
        
        try:
            chunks = loader.load(temp_path)
            assert len(chunks) >= 2
            assert any("padre" in c.content for c in chunks)
        finally:
            os.unlink(temp_path)

    def test_topic_extraction(self):
        loader = PyMuPDFAdapter()
        
        prolog_path = str(Path(tempfile.gettempdir()) / "test_prolog.pl")
        with open(prolog_path, 'w', encoding='utf-8') as f:
            f.write("hecho(a, b).\n")
        
        try:
            chunks = loader.load(prolog_path)
            assert all(c.topic == "Prolog" for c in chunks)
        finally:
            os.unlink(prolog_path)


class TestEmbeddingsAdapter:
    """Tests para el adapter de embeddings."""

    def test_in_memory_similarity(self):
        from app.infrastructure.adapters.vector_store import InMemoryVectorStore
        
        store = InMemoryVectorStore()
        chunks = [
            DocumentChunk("c1", "A* es búsqueda informada", "test.pdf", "AI"),
            DocumentChunk("c2", "BFS es búsqueda no informada", "test.pdf", "AI"),
        ]
        
        def simple_embed(text):
            if "A*" in text or "informada" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]
        
        store.add(chunks, [simple_embed(c.content) for c in chunks])
        
        results = store.query([1.0, 0.0], top_k=1)
        assert results[0][0].chunk_id == "c1"
