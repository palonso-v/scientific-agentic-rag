# Scientific Agentic RAG Assistant

A Python-based scientific literature assistant using retrieval-augmented generation (RAG), semantic search, LLM reranking, grounded answer generation, validated citations, and production-oriented API infrastructure.

The repository includes a FastAPI backend, PostgreSQL persistence, usage analytics, Docker containerization, automated testing, GitHub Actions CI, and cloud deployment on Google Cloud Run.

## Features

- Token-based document chunking and OpenAI embeddings
- Semantic retrieval using normalized vector similarity
- LLM-based passage reranking
- Grounded answer generation with structured outputs
- Citation ID validation
- Insufficient-evidence handling
- FastAPI REST service
- Asynchronous request handling and concurrency control
- Response caching
- PostgreSQL request persistence and usage analytics
- User feedback storage
- Docker containerization
- Automated tests and GitHub Actions CI
- Cloud deployment on Google Cloud Run
- Runtime secret management through Google Secret Manager

## Technology Stack

Python, OpenAI API, FastAPI, Pydantic, NumPy, PostgreSQL, Supabase, psycopg, Docker, pytest, GitHub Actions, Google Cloud Run, Google Secret Manager.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/palonso-v/scientific-agentic-rag.git
cd scientific-agentic-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure:

```text
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_postgresql_connection_string
```

An OpenAI API account with available API credits is required.

A PostgreSQL database is also required for request persistence, usage analytics, and feedback storage.

The deployed version uses Supabase PostgreSQL.

### 5. Build the example document index

The repository includes a synthetic climate-science document at:

```text
app/extracted/paper.json
```

Run:

```bash
python -m app.indexing --source app/extracted/paper.json --paper-id climate-demo
```

This creates a local vector index in:

```text
app/index_v2/climate-demo/
```

### 6. Test semantic retrieval

```bash
python -m app.retrieval
```

Example question:

> Which greenhouse gases are mentioned in the document?

### 7. Run the full RAG assistant

```bash
python -m app.rag
```

Example questions:

- Which greenhouse gases are mentioned in the document?
- What were the carbon dioxide concentrations in 2015 and 2025?
- What was the global average temperature in 2024?

The final question tests insufficient-evidence handling.

### 8. Run the API

```bash
python -m uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Available endpoints include:

- `GET /health`
- `POST /ask`
- `GET /stats`

### 9. Run automated tests

```bash
python -m pytest tests/ -q
```

The automated suite uses mocked LLM calls, so it does not require real OpenAI API requests.

Database tests run against PostgreSQL.

## Architecture

```text
Scientific Document
        |
        v
Token-Based Chunking
        |
        v
OpenAI Embeddings
        |
        v
Local Vector Index
        |
        v
Semantic Retrieval
        |
        v
LLM Passage Reranking
        |
        v
Grounded Answer Generation
        |
        v
Citation Validation
        |
        v
FastAPI
        |
        +----------------------+
        |                      |
        v                      v
API Response            PostgreSQL Persistence
                               |
                               v
                        Requests / Feedback
                        Usage / Cost Analytics
```

## Deployment Architecture

```text
Client
  |
  v
Google Cloud Run
  |
  v
FastAPI Application
  |
  +--------------------+
  |                    |
  v                    v
OpenAI API       Supabase PostgreSQL
  |
  v
RAG Pipeline
```

Runtime secrets such as `OPENAI_API_KEY` and `DATABASE_URL` are provided to Cloud Run through Google Secret Manager.

The application container is built from source and deployed to Google Cloud Run.

## Evaluation and Reliability

The application includes:

- Unit tests
- API integration tests
- PostgreSQL integration tests
- Mocked LLM responses for deterministic testing
- Retrieval and agent evaluation scripts
- Structured error handling
- Upstream API failure handling
- Request latency tracking
- Token usage tracking
- Estimated API cost tracking
- Cache-hit analytics
- Request and feedback persistence
- GitHub Actions CI
- Docker build validation
- Docker API health-check smoke test
- Temporary PostgreSQL service during CI

## Continuous Integration

GitHub Actions automatically:

1. Starts a temporary PostgreSQL service
2. Installs application dependencies
3. Runs the automated test suite
4. Builds the Docker image
5. Starts the application container
6. Verifies the `/health` endpoint

This allows database-backed tests and container startup behavior to be validated without connecting CI to the production database.

## Cloud Deployment

The application is deployed on Google Cloud Run.

The deployed service uses:

- Google Cloud Run for container execution
- Google Secret Manager for runtime credentials
- Supabase PostgreSQL for persistent application data
- OpenAI API for embeddings, reranking, and generation

The application does not rely on local SQLite storage, so request history and analytics persist across Cloud Run container restarts and revisions.

## Limitations

This project is a production-oriented portfolio system rather than a large-scale commercial deployment.

- The example document is small and synthetic.
- Real LLM requests require API credentials and incur API costs.
- The vector index must currently be built before retrieval.
- Automated API tests mock external LLM calls.
- Retrieval quality on large heterogeneous document collections has not yet been established by the example demonstration.
- The current retrieval layer uses semantic vector search rather than hybrid lexical + semantic retrieval.
- The local vector index is not yet backed by a managed vector database.
- The current cloud deployment is intentionally small-scale.
- Authentication and user-level authorization are not yet implemented.
- More advanced agent orchestration is still under development.

## Planned Improvements

Planned next steps include:

- Stateful agent orchestration with LangGraph
- MCP-based tool integration
- Improved tracing and observability
- Hybrid lexical + semantic retrieval
- More formal RAG and agent evaluation
- Stronger deployment automation and production monitoring

## Author

Pedro Alonso Vaquero

[GitHub](https://github.com/palonso-v)
