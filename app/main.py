"""API FastAPI para el Tutor Inteligente."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.application.use_cases import AskTutorUC, IndexDocumentsUC
from app.infrastructure.adapters.document_loader import PyMuPDFAdapter
from app.infrastructure.adapters.embeddings import EmbeddingsAdapter, MockEmbeddingsAdapter
from app.infrastructure.adapters.llm import OpenAIAdapter, MockLLMAdapter
from app.infrastructure.adapters.vector_store import ChromaDBAdapter
from app.infrastructure.indexer import RAGIndexer
from app.infrastructure.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_vector_store: Optional[ChromaDBAdapter] = None
_indexer: Optional[RAGIndexer] = None
_retriever: Optional[RAGRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos al iniciar la app."""
    global _vector_store, _indexer, _retriever
    
    logger.info("Inicializando Tutor Inteligente...")
    
    _vector_store = ChromaDBAdapter(
        persist_directory="./chroma_db",
        collection_name="tutor_inteligente"
    )
    
    embeddings = _resolve_embeddings()
    
    _indexer = RAGIndexer(
        document_loader=PyMuPDFAdapter(chunk_size=512, chunk_overlap=64),
        embeddings=embeddings,
        vector_store=_vector_store
    )
    
    _retriever = RAGRetriever(
        vector_store=_vector_store,
        embeddings=embeddings,
        llm=MockLLMAdapter(),
        similarity_threshold=0.5
    )
    
    logger.info("Tutor Inteligente inicializado")
    yield
    
    logger.info("Cerrando Tutor Inteligente...")


def _resolve_embeddings():
    """Intenta cargar sentence-transformers; fallback a MockEmbeddingsAdapter."""
    try:
        emb = EmbeddingsAdapter(provider="transformers")
        logger.info("Usando sentence-transformers para embeddings")
        return emb
    except Exception:
        logger.warning("sentence-transformers no disponible, usando MockEmbeddingsAdapter")
        return MockEmbeddingsAdapter()


app = FastAPI(
    title="Tutor Inteligente de IA",
    description="API RAG para responder preguntas sobre Inteligencia Artificial",
    version="1.0.0",
    lifespan=lifespan
)


class IndexRequest(BaseModel):
    sources_dir: str = "./fuentes"


class IndexResponse(BaseModel):
    status: str
    chunks: int
    topics: list[str]
    files: int


class AskRequest(BaseModel):
    query: str
    student_id: Optional[str] = None


class SourceInfo(BaseModel):
    filename: str
    topic: str
    excerpt: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


@app.get("/")
async def root():
    """Endpoint de verificación de salud."""
    return {
        "status": "online",
        "service": "Tutor Inteligente de IA",
        "version": "1.0.0"
    }


@app.get("/stats")
async def stats():
    """Retorna estadísticas del sistema."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")
    
    return {
        "chunks_indexed": _vector_store.get_count(),
        "service": "ready"
    }


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """
    Fuerza la re-indexación de documentos.
    
    Lee todos los archivos del directorio ./fuentes/, los fragmenta,
    genera embeddings y los almacena en ChromaDB.
    """
    if _indexer is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")
    
    try:
        logger.info(f"Indexando documentos desde: {request.sources_dir}")
        result = _indexer.index_documents(request.sources_dir)
        
        return IndexResponse(
            status=result.get("status", "unknown"),
            chunks=result.get("chunks", 0),
            topics=result.get("topics", []),
            files=result.get("files", 0)
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en indexación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_tutor(request: AskRequest):
    """
    Responde una pregunta del estudiante usando RAG.
    
    Recupera contexto relevante del vector store y genera una
    respuesta fundada en las fuentes documentales.
    """
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query no puede estar vacía")
    
    try:
        logger.info(f"Procesando query: {request.query[:50]}...")
        response = _retriever.retrieve_and_answer(request.query)
        
        return AskResponse(
            answer=response.answer,
            sources=[
                SourceInfo(**source) for source in response.sources
            ]
        )
    except Exception as e:
        logger.error(f"Error en consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
