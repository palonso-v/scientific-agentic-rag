# Scientific Agentic RAG Assistant

A Python-based scientific literature assistant combining retrieval-augmented generation (RAG), semantic search, LLM reranking, grounded answer generation, stateful agent orchestration, standardized tool integration, and production-oriented API infrastructure.

The project includes a deterministic RAG pipeline alongside a LangGraph-based agent capable of dynamically selecting literature retrieval and calculation tools. Tool access can be provided through Model Context Protocol (MCP), while Langfuse tracing provides visibility into agent execution, tool calls, failures, and RAG usage.

The application also includes a FastAPI backend, PostgreSQL persistence, usage analytics, Docker containerization, automated testing, GitHub Actions CI, and cloud deployment on Google Cloud Run.

## Features

### RAG and Retrieval

- Token-based document chunking
- OpenAI embeddings
- Semantic retrieval using normalized vector similarity
- LLM-based passage reranking
- Grounded answer generation with structured outputs
- Citation ID validation
- Insufficient-evidence handling
- Source-aware answer generation

### Agentic Workflow

- Stateful agent orchestration with LangGraph
- Explicit shared agent state
- Conditional routing between agent nodes
- Literature retrieval, calculation, and final synthesis nodes
- Multi-step decision loop
- Maximum-step termination guard
- Graceful tool-failure handling
- Application-level routing constraints for required calculations
- Separate deterministic RAG and agentic execution paths

### Tool Integration

- Model Context Protocol (MCP) client/server integration
- Calculator exposed as an MCP tool
- LangGraph tool execution through MCP
- Tool failures preserved in agent state
- Graceful degradation when an MCP service is unavailable

### API and Backend

- FastAPI REST service
- Pydantic request and response validation
- Asynchronous request handling
- Blocking RAG and agent execution moved to worker threads
- Concurrency control using `asyncio.Semaphore`
- Response caching
- Structured HTTP error handling
- Request IDs for application-level tracing

### Persistence and Analytics

- PostgreSQL request persistence
- Supabase-hosted PostgreSQL deployment
- Request latency tracking
- Token usage tracking
- Estimated API cost tracking
- Cache-hit analytics
- User feedback storage
- REST-accessible usage statistics

### Observability

- Langfuse tracing for LangGraph agent runs
- RAG spans with token and estimated cost metadata
- MCP tool spans
- Error-level tracing for failed tool calls
- Request IDs correlated between FastAPI logs and agent traces
- Structured application logging

### Infrastructure

- Docker containerization
- Automated tests with pytest
- PostgreSQL integration tests
- GitHub Actions CI
- Temporary PostgreSQL services during CI
- Docker build validation
- API health-check smoke tests
- Google Cloud Run deployment
- Google Secret Manager for runtime credentials

## Technology Stack

Python, OpenAI API, FastAPI, Pydantic, LangGraph, MCP, Langfuse, NumPy, PostgreSQL, Supabase, psycopg, Docker, pytest, GitHub Actions, Google Cloud Run, Google Secret Manager.

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

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

An OpenAI API account with available API credits is required for real LLM requests.

A PostgreSQL database is required for request persistence, analytics, and feedback storage.

The deployed application uses Supabase PostgreSQL.

Langfuse credentials are optional for the core RAG pipeline but required for Langfuse observability and tracing.

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

### 7. Run the deterministic RAG assistant

```bash
python -m app.rag
```

Example questions:

- Which greenhouse gases are mentioned in the document?
- What were the carbon dioxide concentrations in 2015 and 2025?
- What was the global average temperature in 2024?

The final question can be used to test insufficient-evidence handling.

## MCP Tool Service

The agent can access calculation functionality through an MCP server.

Start the MCP server in a separate terminal:

```bash
python -m app.mcp_server
```

The local MCP endpoint runs at:

```text
http://127.0.0.1:8001/mcp
```

The MCP server exposes calculator functionality while reusing the application's existing calculation implementation.

The MCP client can also be tested independently:

```bash
python -m app.mcp_client
```

Example output:

```text
Result: 20.0
```

The calculator is intentionally simple and primarily demonstrates standardized tool access through MCP.

## LangGraph Agent

The LangGraph agent combines literature retrieval, tool execution, and final synthesis through an explicit stateful workflow.

Run it directly with:

```bash
python -m app.langgraph_agent
```

For a question requiring both retrieval and arithmetic, a typical execution trace is:

```text
literature
    |
    v
calculator
    |
    v
finish
```

The agent maintains state including:

- Original question
- Observations
- Proposed actions
- Tool instructions
- Step count
- Final answer
- Tool errors

Important workflow constraints are enforced in application code rather than relying entirely on LLM decisions. For example, when the user explicitly requests a calculation, the agent cannot finish successfully without an explicit calculator result unless the calculator tool has failed.

## Run the API

Start the MCP server first if you want to use calculator-backed agent requests:

```bash
python -m app.mcp_server
```

Then start FastAPI in another terminal:

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
- `POST /agent`
- `GET /stats`

### `/ask`

`POST /ask` runs the deterministic RAG pipeline.

Example request:

```json
{
  "question": "Which greenhouse gases are mentioned in the document?"
}
```

### `/agent`

`POST /agent` runs the LangGraph agent.

Example request:

```json
{
  "question": "According to the document, what were the carbon dioxide concentrations in 2015 and 2025, and calculate the increase?"
}
```

A typical successful result follows:

```text
literature -> calculator -> finish
```

If the MCP calculator is unavailable, the failure is stored in agent state and surfaced gracefully rather than silently bypassing the tool.

## Architecture

```text
                         User
                           |
                           v
                        FastAPI
                           |
                +----------+----------+
                |                     |
                v                     v
          POST /ask              POST /agent
                |                     |
                v                     v
     Deterministic RAG           LangGraph Agent
                |                     |
                |                 Decision Node
                |                     |
                |             +-------+-------+
                |             |               |
                |             v               v
                |        Literature        Calculator
                |             |               |
                |             v               v
                |         RAG Pipeline     MCP Client
                |                             |
                |                             v
                |                         MCP Server
                |                             |
                |                             v
                |                         Calculator
                |                             |
                |             +---------------+
                |             |
                |             v
                |         Final Synthesis
                |             |
                +-------------+
                      |
                      v
                   Response
```

## RAG Pipeline

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
```

## Observability Architecture

```text
FastAPI Request
      |
      v
  Request ID
      |
      v
LangGraph Execution
      |
      +----------------------+
      |                      |
      v                      v
 Literature RAG         MCP Calculator
      |                      |
      v                      v
 token / cost            result / error
 metadata                tool span
      |                      |
      +----------+-----------+
                 |
                 v
             Langfuse
                 |
                 v
       Trace / Span Inspection
```

Langfuse is used to trace agent execution and inspect important operations such as:

- LangGraph node execution
- Literature/RAG calls
- MCP calculator calls
- Tool failures
- Token usage
- Estimated LLM cost
- Request metadata

A shared request ID can be used to correlate application logs with Langfuse traces.

## Persistence Architecture

```text
FastAPI
   |
   v
PostgreSQL
   |
   +-------------------+
   |                   |
   v                   v
Requests            Feedback
   |
   v
Usage / Cost / Latency Analytics
```

The deployed application uses Supabase PostgreSQL so request data persists independently of Cloud Run container instances.

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
  +---------------------+
  |                     |
  v                     v
OpenAI API       Supabase PostgreSQL

Runtime credentials
        |
        v
Google Secret Manager
```

Runtime secrets such as `OPENAI_API_KEY` and `DATABASE_URL` are supplied to Cloud Run through Google Secret Manager.

The primary FastAPI application is containerized and deployed to Google Cloud Run.

The current MCP calculator integration is demonstrated as a separate MCP service. A production deployment would require that MCP service to be deployed and reachable by the agent rather than manually started by the end user.

## Evaluation and Reliability

The project includes:

- Unit tests
- API integration tests
- PostgreSQL integration tests
- LangGraph routing tests
- MCP integration tests using mocked MCP calls
- Tool-failure tests
- Maximum-step termination tests
- Unknown-action fallback tests
- Insufficient-evidence tests
- Mocked LLM responses for deterministic testing
- Retrieval and agent evaluation scripts
- Structured error handling
- Upstream API failure handling
- Request latency tracking
- Token usage tracking
- Estimated API cost tracking
- Cache-hit analytics
- Request and feedback persistence
- Langfuse tracing
- GitHub Actions CI
- Docker build validation
- Docker API health-check smoke testing
- Temporary PostgreSQL services during CI

The tests verify agent behavior without requiring real OpenAI or MCP calls during CI.

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

The FastAPI application is deployed on Google Cloud Run.

The deployed system uses:

- Google Cloud Run for container execution
- Google Secret Manager for runtime credentials
- Supabase PostgreSQL for persistent application data
- OpenAI API for embeddings, reranking, planning, and generation
- Langfuse Cloud for agent observability and tracing

The application does not rely on local SQLite persistence, so request history and analytics survive Cloud Run container restarts and revisions.

## Design Principles

### Deterministic workflows where possible

The project keeps the standard RAG path deterministic rather than using an agent for every request.

Agentic orchestration is used when the next step depends on intermediate results or requires dynamic tool selection.

### Explicit tool boundaries

Each tool has a defined responsibility:

```text
Literature tool -> retrieve grounded evidence
Calculator tool -> perform arithmetic
Final synthesis -> combine existing results
```

The final synthesis step is instructed not to silently reproduce failed tool functionality.

### Hard constraints for critical behavior

LLM decisions are treated as probabilistic.

Important workflow requirements are enforced in code. For example, if a calculation is explicitly requested, the application verifies that a calculation result exists before allowing a normal completion path.

### Frameworks for orchestration, not core logic

Core retrieval, reranking, grounding, validation, and calculation functionality remains implemented directly in Python.

LangGraph is used for higher-level stateful agent orchestration, while MCP provides a standardized tool interface.

## Limitations

This project is a production-oriented portfolio system rather than a large-scale commercial deployment.

- The example corpus is small and synthetic.
- Real LLM requests require API credentials and incur API costs.
- The vector index must currently be built before retrieval.
- Automated API tests mock external LLM calls.
- Retrieval quality on large heterogeneous document collections has not yet been established by the demonstration corpus.
- Retrieval currently uses dense semantic search rather than hybrid lexical and semantic retrieval.
- The vector index is currently local rather than backed by a managed vector database.
- The MCP calculator is deliberately simple and is primarily used to demonstrate MCP architecture.
- The MCP service is not yet independently deployed as part of the cloud architecture.
- The current cloud deployment is intentionally small-scale.
- Authentication and user-level authorization are not yet implemented.
- Agent evaluation is currently limited compared with a production evaluation framework.

## Planned Improvements

Planned next steps include:

- Hybrid lexical + semantic retrieval
- BM25 retrieval and Reciprocal Rank Fusion
- Retrieval benchmarking against semantic-only search
- More formal RAG and agent evaluation
- More meaningful MCP-backed external tools
- Independent cloud deployment of MCP services
- Stronger deployment automation
- Production monitoring and alerting
- Authentication and authorization
- Evaluation-driven agent regression testing

## Author

Pedro Alonso Vaquero

[GitHub](https://github.com/palonso-v)
