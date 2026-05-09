"""Indexador RAG: carga documentos, crea embeddings y los almacena en ChromaDB."""
import logging
from pathlib import Path
from typing import Optional, Union

from app.domain.entities import DocumentChunk
from app.domain.ports import IVectorStore
from app.infrastructure.adapters.document_loader import PyMuPDFAdapter
from app.infrastructure.adapters.embeddings import EmbeddingsAdapter, MockEmbeddingsAdapter

logger = logging.getLogger(__name__)


class RAGIndexer:
    """Pipeline de indexación RAG completo."""

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".pl", ".txt"}

    def __init__(
        self,
        document_loader: PyMuPDFAdapter,
        embeddings: EmbeddingsAdapter,
        vector_store: IVectorStore
    ):
        self._loader = document_loader
        self._embeddings = embeddings
        self._vector_store = vector_store

    def index_documents(self, sources_dir: str) -> dict:
        """
        Indexa todos los documentos del directorio fuente.
        
        Args:
            sources_dir: Ruta al directorio con documentos.
            
        Returns:
            Dict con estadísticas de la indexación.
        """
        sources_path = Path(sources_dir)
        
        if not sources_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {sources_dir}")

        all_chunks: list[DocumentChunk] = []
        topics: set[str] = set()
        processed_files = 0

        for file_path in sources_path.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.debug(f"Ignorando archivo: {file_path.name}")
                continue

            logger.info(f"Procesando: {file_path.name}")
            try:
                chunks = self._loader.load(str(file_path))
                all_chunks.extend(chunks)
                topics.update(chunk.topic for chunk in chunks)
                processed_files += 1
            except Exception as e:
                logger.error(f"Error procesando {file_path.name}: {e}")
                continue

        if not all_chunks:
            logger.warning("No se encontraron chunks para indexar")
            return {
                "status": "empty",
                "chunks": 0,
                "topics": [],
                "files": 0
            }

        logger.info(f"Generando embeddings para {len(all_chunks)} chunks...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embeddings.embed(texts)

        logger.info("Almacenando en vector store...")
        self._vector_store.add(all_chunks, embeddings)

        result = {
            "status": "indexed",
            "chunks": len(all_chunks),
            "topics": sorted(topics),
            "files": processed_files
        }
        logger.info(f"Indexación completada: {result}")
        return result


def create_indexer(
    sources_dir: str = "./fuentes",
    vector_store: Optional[IVectorStore] = None
) -> RAGIndexer:
    """
    Factory function para crear el indexer configurado.
    
    Args:
        sources_dir: Directorio de documentos fuente.
        vector_store: Vector store a usar (crea ChromaDB por defecto).
        
    Returns:
        Instancia configurada de RAGIndexer.
    """
    if vector_store is None:
        from app.infrastructure.adapters.vector_store import ChromaDBAdapter
        vector_store = ChromaDBAdapter(
            persist_directory="./chroma_db",
            collection_name="tutor_inteligente"
        )

    document_loader = PyMuPDFAdapter(chunk_size=512, chunk_overlap=64)
    embeddings = MockEmbeddingsAdapter()
    
    indexer = RAGIndexer(document_loader, embeddings, vector_store)
    return indexer


def run_indexing(sources_dir: str = "./fuentes") -> dict:
    """Función de entrada para indexación directa."""
    indexer = create_indexer(sources_dir)
    return indexer.index_documents(sources_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_indexing()
    print(f"Indexación completada: {result}")
