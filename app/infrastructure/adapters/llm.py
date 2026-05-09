"""Adapter para comunicación con LLM (OpenAI)."""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

OpenAI_AVAILABLE = False
try:
    from openai import OpenAI
    OpenAI_AVAILABLE = True
except ImportError:
    pass


class OpenAIAdapter:
    """Adapter para GPT-4o y otros modelos OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ):
        if not OpenAI_AVAILABLE:
            raise ImportError("Instale openai: pip install openai")

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        logger.info(f"OpenAI adapter inicializado: {model}")

    def generate(self, prompt: str) -> str:
        """Genera una respuesta a partir del prompt."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature
            )

            answer = response.choices[0].message.content
            logger.info(f"LLM generó {len(answer)} caracteres")
            return answer

        except Exception as e:
            logger.error(f"Error en generación LLM: {e}")
            raise


class MockLLMAdapter:
    """Adapter mock para testing sin API externa."""

    def __init__(self):
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        """Retorna una respuesta mock basada en el contexto."""
        self._call_count += 1
        logger.info(f"MockLLM llamada #{self._call_count}")

        if "Búsqueda Informada" in prompt or "A*" in prompt:
            return ("A* es un algoritmo de búsqueda informada que combina "
                    "el costo real del camino con una heurística admisible. "
                    "[fuente: guia_ia.pdf, tema: Búsqueda Informada]")
        elif "Prolog" in prompt:
            return ("Prolog es un lenguaje de programación lógica basado "
                    "en cláusulas de Horn. [fuente: reglas_prolog.pl]")
        else:
            return ("Esta es una respuesta mock del tutor. "
                    f"Tu pregunta fue procesada correctamente. "
                    "[mock: llamada #{self._call_count}]")

    @property
    def call_count(self) -> int:
        return self._call_count
