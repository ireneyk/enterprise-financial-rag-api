# Financial Docs AI (Enterprise RAG Microservice)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini%20%7C%20text--embedding--3--small-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Tests](https://img.shields.io/badge/Tests-72%20Passed-brightgreen.svg)](https://pytest.org)

**Financial Docs AI** is a production-grade, headless backend microservice engineered for high-accuracy financial document ingestion, token-aware semantic chunking, high-dimensional vector search via `pgvector`, and hallucination-free grounded question-answering with OpenAI LLMs.

---

## Architecture Flow

```
┌─────────────────────────┐
│     Client / React      │  (HTTP / JSON & Multipart Uploads)
│        Frontend         │
└────────────┬────────────┘
             │ Bearer JWT Auth & Requests
             ▼
┌─────────────────────────┐
│         FastAPI         │  • JWT Authentication & RBAC
│   Backend Microservice  │  • PDF Extraction (PyMuPDF / pypdf)
│                         │  • Token-Aware Chunking (tiktoken ~500 tokens / 50 overlap)
└──────┬───────────┬──────┘
       │           │
       │ Vectorize │ Cosine Similarity Search
       │ & Query   │
       ▼           ▼
┌──────────────┐  ┌──────────────────────────────────────────────┐
│    OpenAI    │  │             PostgreSQL + pgvector            │
│     APIs     │  │                                              │
│              │  │  • FinancialDocument (Metadata & Status)     │
│  • Embeddings│◄─┼─ • DocumentChunk (Vector[1536] Embeddings)   │
│  • gpt-4o-   │  │  • Cosine Distance Index (<=> operator)      │
│    mini      │  └──────────────────────────────────────────────┘
└──────────────┘
```

### Ingestion & Query Lifecycle

1. **Ingest & Extract (`POST /api/v1/financial-docs/upload`)**:
   - The client uploads an earnings report or SEC filing (PDF).
   - `PyMuPDF` extracts the raw textual data with zero quality loss (with `pypdf` automated fallback).
   - `tiktoken` splits the document into semantic chunks (~500 tokens with 50-token sliding overlap).
   - `text-embedding-3-small` generates 1536-dimensional dense vectors.
   - Text chunks and vector embeddings are stored transactionally in PostgreSQL using `pgvector`.

2. **Retrieve & Ground (`POST /api/v1/financial-docs/query`)**:
   - User submits a financial question, optionally filtered by `document_id`.
   - The query is vectorized and compared against indexed chunks via cosine distance (`DocumentChunk.embedding <=> query_vector`).
   - The top 5 highest-similarity chunks are injected into an adversarial anti-hallucination prompt.
   - `gpt-4o-mini` generates a strictly grounded answer with verifiable citation excerpts.

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **API Framework** | **FastAPI** | High-performance asynchronous REST microservice |
| **Relational Database** | **PostgreSQL 17** | Document metadata, chunks, and relational tables |
| **Vector Search** | **pgvector** | Native vector embeddings & cosine distance operator (`<=>`) |
| **ORM / Data Modeling**| **SQLModel / SQLAlchemy** | Type-safe models with Pydantic and SQLAlchemy 2.0 integration |
| **Database Migrations** | **Alembic** | Automated linear database schema versioning |
| **Package Manager** | **uv** | Ultra-fast Python package and project dependency management |
| **PDF Extraction** | **PyMuPDF (`fitz`) / pypdf** | Robust text and table extraction from financial PDFs |
| **Tokenization** | **tiktoken** | Accurate BPE tokenization matching OpenAI model boundaries |
| **AI / Embeddings** | **OpenAI SDK** | `text-embedding-3-small` (1536 dims) & `gpt-4o-mini` |
| **Testing** | **PyTest** | Unit, service, and API integration test coverage |
| **Containerization** | **Docker & Docker Compose** | Reproducible multi-container microservice infrastructure |

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── alembic/              # Linear database migrations (including pgvector)
│   │   ├── api/
│   │   │   ├── routes/           # Endpoint handlers (financial_docs, login, users)
│   │   │   ├── routers/          # Router exports
│   │   │   ├── deps.py           # JWT Authentication & DB session dependencies
│   │   │   └── main.py           # Aggregated API router
│   │   ├── core/
│   │   │   ├── config.py         # Pydantic Settings & environment variables
│   │   │   ├── db.py             # DB engine, pgvector extension init & seeding
│   │   │   └── security.py       # Password hashing & JWT token creation
│   │   ├── models/
│   │   │   ├── __init__.py       # Re-exported domain models
│   │   │   └── rag.py            # FinancialDocument & DocumentChunk (Vector[1536])
│   │   ├── services/
│   │   │   └── rag_service.py    # Core business logic: PDF, chunk, embed, search
│   │   ├── main.py               # FastAPI application entrypoint
│   │   └── utils.py              # Utility functions
│   ├── pyproject.toml            # Backend dependencies and tools
│   └── tests/                    # 72 automated pytest test cases
├── scripts/
│   └── init-db.sql               # Docker Postgres init: CREATE EXTENSION vector
├── compose.yml                   # Production / baseline Docker Compose configuration
├── compose.override.yml          # Development Compose configuration
├── docker-compose.yml            # Standard Docker Compose alias
├── docker-compose.override.yml   # Standard Docker Compose override alias
├── .env                          # Environment variables configuration
└── README.md
```

---

## Local Setup & Execution

### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- OpenAI API Key

### 2. Configure Environment Variables
Copy or edit `.env` in the project root:

```bash
# General
FASTAPI_ENV=development
PROJECT_NAME="Financial Docs AI"

# Secrets (Change for production)
SECRET_KEY=supersecretkey-financial-docs-rag-2026
FIRST_SUPERUSER=admin@example.com
FIRST_SUPERUSER_PASSWORD=adminpassword123

# OpenAI API Key
OPENAI_API_KEY=sk-proj-your-actual-openai-api-key

# Database
POSTGRES_PASSWORD=postgrespassword
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/app
```

### 3. Run with Docker Compose

Launch the PostgreSQL (with pgvector), Traefik proxy, and FastAPI backend microservice:

```bash
docker compose up -d
```

*(or using legacy syntax)*:
```bash
docker-compose up -d
```

Check container health:
```bash
docker compose ps
```

The API will be available at:
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **Alternative ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/api/v1/utils/health-check/`

---

## API Usage & cURL Examples

### 1. Authenticate & Obtain Bearer Token

```bash
curl -X POST "http://localhost:8000/api/v1/login/access-token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=adminpassword123"
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Save your token:
```bash
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

### 2. Upload and Ingest a Financial Document (PDF)

Upload any financial statement, quarterly earnings, or 10-K filing:

```bash
curl -X POST "http://localhost:8000/api/v1/financial-docs/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@q3_financial_report.pdf;type=application/pdf"
```

**Response**:
```json
{
  "document_id": 1,
  "filename": "q3_financial_report.pdf",
  "status": "completed",
  "chunks_count": 24,
  "message": "Document successfully processed, embedded, and indexed for search."
}
```

---

### 3. Query Financial Documents with Grounded Citations

Query across all financial documents:

```bash
curl -X POST "http://localhost:8000/api/v1/financial-docs/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the total quarterly revenues and operating margins?"
  }'
```

Query scoped to a specific document (`document_id`):

```bash
curl -X POST "http://localhost:8000/api/v1/financial-docs/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the primary revenue drivers according to management?",
    "document_id": 1
  }'
```

**Response**:
```json
{
  "query": "What were the total quarterly revenues and operating margins?",
  "answer": "According to the Q3 Financial Report, total revenue reached $85.8 billion, representing a 5% increase year-over-year. The operating margin was 30.2%, driven by strong performance in the Services division.",
  "citations": [
    "Consolidated Statements of Operations: Total net sales were $85,777 million compared to $81,797 million in the year-ago quarter. Operating income was $25,352 million, yielding an operating margin of 30.2%.",
    "Services segment revenue reached an all-time record of $24.2 billion, up 14% year-over-year, significantly expanding the gross margin profile."
  ]
}
```

---

### 4. List and Manage Uploaded Documents

**List all indexed documents**:
```bash
curl -X GET "http://localhost:8000/api/v1/financial-docs/" \
  -H "Authorization: Bearer $TOKEN"
```

**Delete a document and cascade chunk removal**:
```bash
curl -X DELETE "http://localhost:8000/api/v1/financial-docs/1" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Running Automated Tests

Run the complete test suite using `uv` and `pytest`:

```bash
cd backend
uv run pytest
```

Run test suite with coverage report:
```bash
uv run coverage run -m pytest
uv run coverage report
```

Run code formatting and linting:
```bash
uv run ruff check app tests
uv run ruff format app tests
```

---

## License

This project is licensed under the MIT License.
