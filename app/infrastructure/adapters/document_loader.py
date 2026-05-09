"""Adapter para carga de documentos usando PyMuPDF y parsers personalizados."""
import logging
from pathlib import Path
from typing import Optional

from app.domain.entities import DocumentChunk
from app.domain.ports import IDocumentLoader

logger = logging.getLogger(__name__)


class PyMuPDFAdapter(IDocumentLoader):
    """Carga y fragmenta documentos usando PyMuPDF."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load(self, file_path: str) -> list[DocumentChunk]:
        """Carga un documento y lo fragmenta en chunks."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".md":
            return self._load_markdown(file_path)
        elif suffix == ".pl":
            return self._load_prolog(file_path)
        else:
            return self._load_text(file_path)

    def _load_pdf(self, file_path: str) -> list[DocumentChunk]:
        """Carga y fragmenta un PDF usando PyMuPDF."""
        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF no instalado: pip install pymupdf")
            raise

        chunks = []
        doc = fitz.open(file_path)
        file_name = Path(file_path).name

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if not text.strip():
                continue

            page_chunks = self._chunk_text(
                text, file_name, page_num + 1
            )
            chunks.extend(page_chunks)

        doc.close()
        logger.info(f"PDF {file_name}: {len(chunks)} chunks extraídos")
        return chunks

    def _chunk_text(
        self, 
        text: str, 
        source_file: str, 
        page: int,
        topic: str = "General"
    ) -> list[DocumentChunk]:
        """Fragmenta texto en chunks con solapamiento."""
        separators = ["\n\n", "\n", ". ", " "]
        chunks = []

        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]

            for sep in separators:
                if sep in chunk_text:
                    last_sep = chunk_text.rfind(sep)
                    chunk_text = chunk_text[:last_sep + len(sep)]
                    break

            if len(chunk_text) < 50:
                start += len(chunk_text)
                continue

            chunks.append(DocumentChunk(
                chunk_id=f"{source_file}_{page}_{chunk_id}",
                content=chunk_text.strip(),
                source_file=source_file,
                topic=topic,
                page=page
            ))
            chunk_id += 1

            start += len(chunk_text) - self._chunk_overlap

        return chunks

    def _load_markdown(self, file_path: str) -> list[DocumentChunk]:
        """Carga y fragmenta un archivo Markdown."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        topic = self._extract_topic_from_path(file_path)
        chunks = self._chunk_text(content, Path(file_path).name, 1, topic)
        logger.info(f"Markdown {Path(file_path).name}: {len(chunks)} chunks")
        return chunks

    def _load_prolog(self, file_path: str) -> list[DocumentChunk]:
        """Carga y fragmenta un archivo Prolog."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rules = self._parse_prolog(content)
        chunks = []

        for i, rule in enumerate(rules):
            chunks.append(DocumentChunk(
                chunk_id=f"prolog_rule_{i}",
                content=rule,
                source_file=Path(file_path).name,
                topic="Prolog"
            ))

        logger.info(f"Prolog {Path(file_path).name}: {len(chunks)} reglas extraídas")
        return chunks

    def _parse_prolog(self, content: str) -> list[str]:
        """Extrae hechos y reglas de Prolog."""
        rules = []
        current_rule = []

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            current_rule.append(line)
            if line.endswith("."):
                rule_text = " ".join(current_rule)
                rules.append(rule_text)
                current_rule = []

        if current_rule:
            rules.append(" ".join(current_rule))

        return rules

    def _load_text(self, file_path: str) -> list[DocumentChunk]:
        """Carga un archivo de texto genérico."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        topic = self._extract_topic_from_path(file_path)
        chunks = self._chunk_text(content, Path(file_path).name, 1, topic)
        logger.info(f"Texto {Path(file_path).name}: {len(chunks)} chunks")
        return chunks

    def _extract_topic_from_path(self, file_path: str) -> str:
        """Extrae el tema del nombre del archivo."""
        name = Path(file_path).stem.lower()
        if "prolog" in name:
            return "Prolog"
        elif "busqueda" in name or "search" in name:
            return "Búsqueda Informada"
        elif "logica" in name or "logic" in name:
            return "Lógica de Predicados"
        elif "aprendizaje" in name or "ml" in name:
            return "Aprendizaje Automático"
        elif "redes" in name or "neural" in name:
            return "Redes Neuronales"
        return "General"
