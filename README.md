
# Scientific RAG Assistant

A Python-based scientific literature assistant using retrieval-augmented
generation (RAG), semantic search, LLM reranking, and grounded answer
generation with validated citations.

The repository includes a FastAPI backend, SQLite request logging,
usage analytics, automated testing, and GitHub Actions continuous
integration.

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
- SQLite request persistence and usage analytics
- Docker containerization
- Automated tests and GitHub Actions CI

## Technology Stack

Python, OpenAI API, FastAPI, Pydantic, NumPy, SQLite,
Docker, pytest, GitHub Actions.

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

### 4. Configure the OpenAI API key

Copy `.env.example` to `.env` and replace the placeholder
with your OpenAI API key.

An OpenAI API account with available API credits is required.

### 5. Build the example document index

The repository includes a synthetic climate-science document
at `app/extracted/paper.json`.

Run:

```bash
python -m app.indexing --source app/extracted/paper.json --paper-id climate-demo
```

This creates a local vector index in `app/index_v2/climate-demo/`.

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

http://127.0.0.1:8000/docs

Available endpoints include:

- `GET /health`
- `POST /ask`
- `GET /stats`

### 9. Run automated tests

```bash
python -m pytest tests/ -q
```

The automated suite uses mocked LLM calls, so it does not
require real API requests.

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
FastAPI Response
        |
        v
SQLite Logging and Analytics
```

## Evaluation and Reliability

The application includes:

- Unit and API integration tests
- Mocked LLM responses for deterministic testing
- Retrieval and agent evaluation scripts
- Structured error handling
- Request latency and token usage tracking
- Cache-hit analytics
- GitHub Actions CI with Docker build and API health-check validation

## Limitations

This project is a portfolio demonstration, not a production
deployment.

- The example document is small and synthetic.
- Real LLM requests require API credentials and incur API costs.
- The vector index must be built locally before using retrieval.
- Automated API tests mock external LLM calls.
- Retrieval quality on large document collections has not been
  established by the example demonstration.
- No cloud deployment is currently provided.

## Author

Pedro Alonso Vaquero

[GitHub](https://github.com/palonso-v)
