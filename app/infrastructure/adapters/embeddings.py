"""Adapter para generación de embeddings."""
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

OpenAI_AVAILABLE = False
try:
    from openai import OpenAI
    OpenAI_AVAILABLE = True
except ImportError:
    pass

SentenceTransformers_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformers_AVAILABLE = True
except ImportError:
    pass


class EmbeddingsAdapter:
    """Adapter para generación de embeddings."""

    def __init__(self, provider: str = "openai"):
        self._provider = provider.lower()

        if self._provider == "openai":
            if not OpenAI_AVAILABLE:
                raise ImportError("Instale openai: pip install openai")
            self._client = OpenAI()
            self._model = "text-embedding-3-small"
        elif self._provider == "transformers":
            if not SentenceTransformers_AVAILABLE:
                raise ImportError("Instale sentence-transformers: pip install sentence-transformers")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Provider no soportado: {provider}")

        logger.info(f"Embeddings adapter inicializado: {self._provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos."""
        if not texts:
            return []

        if self._provider == "openai":
            response = self._client.embeddings.create(
                model=self._model,
                input=texts
            )
            return [item.embedding for item in response.data]
        else:
            return self._model.encode(texts).tolist()

    def embed_single(self, text: str) -> list[float]:
        """Genera embedding para un solo texto."""
        return self.embed([text])[0]


class MockEmbeddingsAdapter:
    """Mock adapter que genera embeddings deterministas sin modelo externo."""

    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings mock basados en hash del texto."""
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            vec = [(seed >> (i * 8)) & 0xFF / 255.0 for i in range(self._dimensions)]
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
