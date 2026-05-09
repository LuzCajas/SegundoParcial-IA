# Tutor Inteligente de IA - Documentación Técnica
## Producto Mínimo Viable (MVP) - Metodología SDD

---

## 1. Stack Tecnológico

| Capa / Componente | Herramienta | Justificación |
|---|---|---|
| **Lenguaje base** | Python 3.11+ | Tipado estático con type hints, ecosistema maduro para LLM/RAG |
| **Orquestación RAG** | LangChain (v0.3+) | Abstrae pipeline: loaders → splitter → embed → store → retrieve → generate |
| **Modelo Embeddings** | `text-embedding-3-small` (OpenAI) / sentence-transformers | 1536 dims, económico, buena performance en benchmarks MTEB |
| **Vector Store** | ChromaDB (local) | Embeddings persistidos en disco, zero-config, rápido para MVP |
| **LLM Generativo** | GPT-4o (OpenAI) | Reasoning fuerte, contexto 128k, funciona bien con prompts largos |
| **Gestión docs** | PyMuPDF | Extrae texto y tablas de PDFs con layout preservado |
| **Interfaz API** | FastAPI | Endpoints asincrónicos, validación con Pydantic, OpenAPI auto |

### Diagrama de Integración

```
[PDFs / .pl / .md]
    ↓ PyMuPDF / Parser custom
[Document Loaders]
    ↓ RecursiveCharacterTextSplitter
[Chunks → Embeddings]
    ↓
[ChromaDB]
    ↓ similarity_search
[Prompt aumentado + GPT-4o]
    ↓
[Respuesta al estudiante via FastAPI]
```

---

## 2. Fase sdd-init: Ingestión del Repositorio

### Formatos y herramientas

| Formato | Herramienta | Estrategia |
|---|---|---|
| PDF (Guía Didáctica) | `PyMuPDF` | Extracción de texto con preservación de secciones |
| `.pl` (reglas Prolog) | Parser custom con regex | Extracción de hechos y reglas como texto plano |
| `.md` (tablas de verdad) | Parser markdown | Parsing de tablas con encabezados preservados |

### Estrategia de chunking

| Parámetro | Valor | Justificación |
|---|---|---|
| `chunk_size` | 512 tokens | Los fragmentos caben en la ventana de atención del embedding |
| `chunk_overlap` | 64 tokens | Solapamiento del 12.5% evita perder contexto en bordes |

### Prompt de exploración para sdd-init

```
## sdd-init Prompt: Exploración del Repositorio del Curso de IA

Eres un agente de investigación académica. Explora los documentos:
- Guía Didáctica de IA (PDF)
- Reglas Prolog (.pl)
- Tablas de verdad de Lógica de Predicados (.md)

Instrucciones:
1. Lista todos los archivos del directorio ./fuentes/
2. Para cada archivo, resume tema principal, estructura y términos clave
3. Identifica ambigüedades en el contenido
4. Propón una taxonomía de topics para organizar el vector store
5. Estima la cobertura esperada del temario

Justificación: Un modelo con ventana de 128k (ej. kimi-k2.5) puede procesar
la Guía Didáctica completa (~200 páginas) en una sola llamada sin truncamiento.
```

---

## 3. Fases sdd-propose & sdd-spec

### 3A - Arquitectura Limpia (Hexagonal)

```
┌─────────────────────────────────────────────┐
│           INTERFAZ / ENTREGA                │
│   FastAPI (/index, /ask)                    │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           CAPA DE APLICACIÓN               │
│   Use Cases: IndexDocumentsUC, AskTutorUC   │
│   Ports: IIndexer, IRetriever, ILLMGateway  │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│               DOMINIO                      │
│   Entities: DocumentChunk, Query,           │
│             AnswerResponse                  │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│            INFRAESTRUCTURA                 │
│   Adapters: ChromaDBAdapter, OpenAIAdapter,│
│             PyMuPDFAdapter                 │
└─────────────────────────────────────────────┘
```

### 3B - Prompt para sdd-spec

```markdown
# Spec Request: Tutor Inteligente de IA – RAG MVP

## Contexto
Sistema RAG académico que responde preguntas de estudiantes sobre IA,
ingeriendo fuentes locales: Guía Didáctica, reglas Prolog y tablas de verdad.

## Requisitos Funcionales
- RF-01: Indexar documentos de ./fuentes/ en ChromaDB
- RF-02: Dado query, recuperar 5 fragmentos y generar respuesta
- RF-03: Incluir referencias a fuentes en cada respuesta
- RF-04: Si no hay contexto, responder "No tengo información suficiente..."
- RF-05: Exponer POST /index y POST /ask

## Criterios de Aceptación

**Escenario 1:** Dado ./fuentes/ con documentos → POST /index → status "indexed", ≥100 chunks

**Escenario 2:** Dado ChromaDB con chunks → preguntar "¿Qué es A*?" → respuesta menciona A* con fuentes

**Escenario 3:** Dado base vacía → cualquier query → mensaje de información insuficiente (HTTP 200)

## Restricciones No Funcionales
- Latencia máxima: ≤8 segundos (p95)
- Precisión RAG: 90% verificable contra fuentes
- Seguridad: Sanitizar inputs contra prompt injection
```

---

## 4. Fase sdd-apply

### Prompt para generación de código

```markdown
## sdd-apply Prompt: RAG Indexer + Retriever

Genera código Python puro para:

### Módulo 1: indexer.py
- Lee documentos desde: ./fuentes/ (.pdf, .pl, .md)
- Chunking: chunk_size=512, overlap=64
- Embeddings: text-embedding-3-small
- Persiste en: ChromaDB ./chroma_db

### Módulo 2: retriever.py
- Recibe query en lenguaje natural
- Recupera top-5 fragmentos más relevantes
- Construye prompt aumentado + envía a GPT-4o
- Retorna respuesta con fuentes

### Restricciones
- Sin frameworks externos salvo LangChain y ChromaDB
- Type hints completos
- Manejo de errores con try/except
- Cumplimiento SOLID (SRP, DIP)
```

### Justificación del modelo

`codestral-latest` (Mistral) elegido porque:
- Entrenado en 80+ lenguajes de programación
- Genera código Python idiomático
- Maneja mejor APIs de LangChain/ChromaDB

---

## 5. Fase sdd-verify

### Prompt de auditoría

```markdown
## sdd-verify Prompt: Senior Code Reviewer

Audita el código generado con máximo rigor:

### Checklist
1. **Calidad código**: PEP 8, type hints, manejo de errores
2. **Alucinaciones RAG**: ¿El LLM usa solo contexto recuperado?
3. **Prompt Injection**: ¿La query se sanitiza?
4. **SOLID**: ¿SRP y DIP aplicados?
5. **Separación indexer/retriever**: ¿Desacoplados?

### Formato de reporte
```
## HALLAZGO #[N]
- Severidad: CRÍTICO / ADVERTENCIA / SUGERENCIA
- Archivo / línea:
- Descripción:
- Corrección propuesta:
```
```

### Justificación del modelo

`gpt-5.4-thinking` o `claude-3-7-sonnet-thinking` para verificación:
- Chain-of-thought extendido para análisis de consistencia
- Mejor detección de prompt injection
- Razonamiento explícito sobre múltiples artefactos

---

## 6. Aplicación de Principios SOLID

| Principio | Aplicación | Ejemplo Concreto |
|---|---|---|
| **S — Single Responsibility** | indexer.py solo indexa; retriever.py solo recupera | No se consultan entre sí |
| **O — Open/Closed** | Agregar Pinecone sin modificar lógica existente | Crear PineconeAdapter |
| **L — Liskov Substitution** | ChromaDBAdapter y PineconeAdapter intercambiables | Mismo contrato `.query()` |
| **I — Interface Segregation** | IIndexer e IRetriever separados | No interfaz monolítica |
| **D — Dependency Inversion** | Use cases dependen de abstracciones | `AskTutorUC(retriever: IRetriever)` |

### Diagrama de Dependencias (DIP)

```
main.py
   │ inyecta ChromaDBAdapter() en IndexDocumentsUC
   │ inyecta OpenAIAdapter() en AskTutorUC
   ▼
IndexDocumentsUC → IIndexer (interface)
AskTutorUC → IRetriever (interface)
   │              │
   ▼              ▼
ChromaDBAdapter    OpenAIAdapter
(Dominio no conoce implementaciones concretas)
```

---

## 7. Estructura del Proyecto

```
tutor_inteligente/
├── app/
│   ├── domain/
│   │   ├── entities.py      # DocumentChunk, Query, AnswerResponse
│   │   ├── ports.py         # IIndexer, IRetriever, ILLMGateway
│   │   └── __init__.py
│   ├── application/
│   │   ├── use_cases.py     # IndexDocumentsUC, AskTutorUC
│   │   └── __init__.py
│   ├── infrastructure/
│   │   ├── adapters/
│   │   │   ├── document_loader.py  # PyMuPDFAdapter
│   │   │   ├── embeddings.py       # EmbeddingsAdapter
│   │   │   ├── vector_store.py     # ChromaDBAdapter
│   │   │   └── llm.py              # OpenAIAdapter, MockLLMAdapter
│   │   ├── indexer.py      # Pipeline RAG de indexación
│   │   ├── retriever.py    # Pipeline RAG de recuperación
│   │   └── main.py         # API FastAPI
│   └── __init__.py
├── fuentes/
│   ├── guia_didactica.md
│   ├── reglas_prolog.pl
│   └── tablas_verdad.md
├── tests/
│   ├── test_domain.py
│   ├── test_adapters.py
│   └── test_api.py
├── requirements.txt
└── README.md
```

---

## 8. Uso del Sistema

### Instalación

```bash
cd tutor_inteligente
pip install -r requirements.txt
```

### Indexación de documentos

```python
from app.infrastructure.indexer import run_indexing

result = run_indexing("./fuentes")
print(result)  # {'status': 'indexed', 'chunks': 150, 'topics': [...], 'files': 3}
```

### Consulta al tutor

```python
from app.infrastructure.retriever import create_retriever

retriever = create_retriever()
response = retriever.retrieve_and_answer("¿Qué es el algoritmo A*?")
print(response.answer)
print(response.sources)
```

### API FastAPI

```bash
uvicorn app.main:app --reload
```

```bash
# Indexar documentos
curl -X POST http://localhost:8000/index

# Consultar
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Qué es A*?"}'
```

---

## Autores

Desarrollado como parte del parcial de Inteligencia Artificial, Noveno Semestre.
