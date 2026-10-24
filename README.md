# Scientific Agentic RAG Assistant

A production-oriented scientific literature assistant combining retrieval-augmented generation (RAG), hybrid semantic and lexical retrieval, LLM reranking, grounded answer generation, agent orchestration, standardized tool integration, observability, persistence, and cloud deployment.

The project provides two execution paths:

- a deterministic RAG pipeline for direct literature questions
- a LangGraph agent capable of selecting literature retrieval and calculation tools dynamically

The system supports user-provided scientific PDFs, incremental document indexing, hybrid retrieval with dense embeddings and BM25, Reciprocal Rank Fusion (RRF), LLM reranking, MCP tool integration, Langfuse tracing, PostgreSQL persistence, Docker, automated testing, and deployment to Google Cloud Run.

---

## Features

### RAG and Retrieval

- PDF ingestion with PyMuPDF
- Token-based document chunking
- OpenAI embeddings
- Dense semantic retrieval using normalized vector similarity
- BM25 lexical retrieval
- Hybrid dense and lexical candidate retrieval
- Reciprocal Rank Fusion
- LLM passage reranking
- Grounded answer generation
- Citation validation
- Insufficient-evidence handling
- Source-aware answers
- Incremental document indexing
- Retrieval evaluation using Hit@k and Mean Reciprocal Rank
- Candidate-depth evaluation for reranking quality and cost

### Agentic Workflow

- Stateful orchestration with LangGraph
- Shared agent state
- Conditional routing between nodes
- Literature retrieval, calculation, and final synthesis
- Multi-step decision loop
- Maximum-step termination guard
- Safe handling of tool failures
- Application-level routing rules
- Separate deterministic RAG and agent execution paths

### Tool Integration

- Model Context Protocol client/server integration
- Calculator exposed as an MCP tool
- LangGraph tool execution through MCP
- Tool failures preserved in agent state
- Safe degradation when the MCP service is unavailable

### API and Backend

- FastAPI REST API
- Pydantic request and response validation
- Asynchronous request handling
- Blocking RAG and agent work moved to worker threads
- Concurrency control with `asyncio.Semaphore`
- Response caching
- HTTP error handling
- Structured logging
- Request IDs for trace correlation

### Persistence and Analytics

- PostgreSQL persistence
- Supabase-hosted PostgreSQL
- Request latency tracking
- Token usage tracking
- Estimated API cost tracking
- Cache-hit analytics
- User feedback storage
- REST-accessible usage statistics

### Observability

- Langfuse tracing
- Agent execution traces
- RAG spans
- MCP tool spans
- Tool failure traces
- Token and cost metadata
- Request IDs correlated with application logs

### Infrastructure

- Docker containerization
- pytest automated tests
- PostgreSQL integration tests
- GitHub Actions CI
- Temporary PostgreSQL service during CI
- Docker build validation
- `/health` smoke tests
- Google Cloud Run deployment
- Google Secret Manager

---

## Technology Stack

Python, OpenAI API, FastAPI, Pydantic, LangGraph, MCP, Langfuse, PyMuPDF, NumPy, rank-bm25, PostgreSQL, Supabase, psycopg, Docker, pytest, GitHub Actions, Google Cloud Run, and Google Secret Manager.

---

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

An OpenAI API account with available API credits is required for real embedding and LLM requests.

PostgreSQL is used for request persistence, analytics, and feedback storage.

The deployed application uses Supabase PostgreSQL.

Langfuse credentials are optional for the core RAG pipeline but required for tracing and observability.

---

## Using Your Own PDFs

The application supports indexing your own scientific papers directly from PDF files.

### 1. Create a `papers` directory

From the project root:

```text
scientific-agentic-rag/
├── app/
├── evals/
├── papers/
├── tests/
└── ...
```

Place the PDFs you want to query inside:

```text
papers/
├── paper1.pdf
├── paper2.pdf
└── paper3.pdf
```

The PDFs themselves do not need to be committed to the repository.

### 2. Index the papers

Run:

```bash
python -m app.indexing --pdf-dir papers
```

For each new paper, the ingestion pipeline performs:

```text
PDF
 |
 v
Text Extraction
 |
 v
Token Chunking
 |
 v
OpenAI Embeddings
 |
 v
Per-Paper Index
```

Generated indexes are stored under:

```text
app/index_v2/
```

Each paper receives its own index directory.

Example:

```text
app/index_v2/
├── attention-is-all-you-need/
├── retrieval-augmented-generation/
└── react/
```

### 3. Incremental indexing

The indexing pipeline tracks previously processed documents.

If the command is run again without changing the PDFs:

```bash
python -m app.indexing --pdf-dir papers
```

unchanged documents are skipped.

Example:

```text
SKIP attention-is-all-you-need.pdf (unchanged)
SKIP retrieval-augmented-generation.pdf (unchanged)
```

If a new PDF is added:

```text
papers/
├── attention-is-all-you-need.pdf
├── retrieval-augmented-generation.pdf
└── react.pdf
```

only the new document is processed:

```text
SKIP attention-is-all-you-need.pdf (unchanged)
SKIP retrieval-augmented-generation.pdf (unchanged)
INDEX react.pdf
```

This avoids recomputing chunks and embeddings for documents that have already been indexed.

Modified documents are rebuilt when their stored indexing metadata no longer matches the current source or indexing configuration.

### 4. Start the application

After indexing the papers, start FastAPI:

```bash
python -m uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to test the API directly in the browser.

If documents are indexed while the application is already running, restart the API so the in-memory retrieval collection is reloaded.

---

## Example Document

The repository also includes a small synthetic climate-science document that can be used without downloading external papers:

```text
app/extracted/paper.json
```

Build its index with:

```bash
python -m app.indexing \
  --source app/extracted/paper.json \
  --paper-id climate-demo
```

This creates:

```text
app/index_v2/climate-demo/
```

The JSON indexing path is retained for controlled experiments and evaluation.

---

## Testing Retrieval

Run:

```bash
python -m app.retrieval
```

The retrieval demo compares:

- dense semantic retrieval
- BM25 lexical retrieval
- hybrid retrieval using Reciprocal Rank Fusion

Example question:

```text
What were the carbon dioxide concentrations in 2015 and 2025?
```

---

## Deterministic RAG

Run:

```bash
python -m app.rag
```

The RAG pipeline performs:

```text
Question
 |
 v
Hybrid Retrieval
 |
 v
Candidate Pool
 |
 v
LLM Reranking
 |
 v
Grounded Generation
 |
 v
Citation Validation
```

Example questions for the synthetic climate document include:

```text
Which greenhouse gases are mentioned in the document?
```

```text
What were the carbon dioxide concentrations in 2015 and 2025?
```

```text
What was the global average temperature in 2024?
```

The final example can be used to test insufficient-evidence handling.

---

## Hybrid Retrieval

The first-stage retriever combines two complementary retrieval methods.

### Dense Retrieval

The user question is embedded with the same embedding model used for document chunks.

Normalized vector similarity identifies semantically related passages.

Dense retrieval is useful when the question and relevant passage express the same concept using different wording.

### BM25 Retrieval

BM25 provides lexical retrieval based on exact token overlap while accounting for:

- inverse document frequency
- term frequency
- term-frequency saturation
- document length normalization

This is particularly useful for:

- model names
- acronyms
- numerical values
- identifiers
- uncommon scientific terminology

### Reciprocal Rank Fusion

Dense and BM25 rankings are combined using Reciprocal Rank Fusion rather than directly combining their raw scores.

```text
Question
   |
   +--------------------+
   |                    |
   v                    v
Dense Retrieval     BM25 Retrieval
   |                    |
   +---------+----------+
             |
             v
    Reciprocal Rank Fusion
             |
             v
      Hybrid Candidate Pool
             |
             v
         LLM Reranker
             |
             v
        Grounded Answer
```

RRF operates on ranking positions rather than raw scores, allowing cosine similarity and BM25 scores to be combined without score normalization.

The current system uses equal weighting for dense and BM25 retrieval.

Increasing the BM25 weight did not improve development-set retrieval performance, so equal-weight fusion was retained.

---

## RAG Pipeline

```text
Scientific Documents
        |
        v
PDF / JSON Ingestion
        |
        v
Token-Based Chunking
        |
        +----------------------+
        |                      |
        v                      v
OpenAI Embeddings         Tokenized Corpus
        |                      |
        v                      v
Local Vector Index         BM25 Index
        |                      |
        v                      v
Dense Retrieval           BM25 Retrieval
        |                      |
        +----------+-----------+
                   |
                   v
          Reciprocal Rank Fusion
                   |
                   v
           Hybrid Candidate Pool
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

---

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

The calculator is intentionally simple. Its main purpose is to demonstrate standardized external tool access through MCP.

---

## LangGraph Agent

The LangGraph agent combines literature retrieval, calculation, and final synthesis through an explicit stateful workflow.

Run it directly with:

```bash
python -m app.langgraph_agent
```

A typical execution for a question requiring both retrieval and arithmetic is:

```text
literature
    |
    v
calculator
    |
    v
finish
```

The agent state contains information such as:

- original question
- retrieved observations
- previous actions
- tool instructions
- step count
- final answer
- tool errors

Important workflow rules are enforced in application code rather than relying entirely on model behavior.

For example, when the user explicitly requests a calculation, the agent cannot silently bypass the calculator and perform the arithmetic during final synthesis.

If the calculator fails, the failure is stored in the agent state and the response reports that the requested calculation could not be completed.

---

## Run the API

Start the MCP server first if calculator-backed agent requests are required:

```bash
python -m app.mcp_server
```

Then start FastAPI in another terminal:

```bash
python -m uvicorn app.api:app --reload
```

Open the interactive API documentation:

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

Example:

```json
{
  "question": "How does ReAct combine reasoning and acting?"
}
```

### `/agent`

`POST /agent` runs the LangGraph workflow.

Example:

```json
{
  "question": "According to the document, what were the carbon dioxide concentrations in 2015 and 2025, and calculate the increase?"
}
```

A successful execution typically follows:

```text
literature -> calculator -> finish
```

If the MCP calculator is unavailable, the failure is stored in agent state and surfaced in the final response instead of silently bypassing the tool.

---

## Application Architecture

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
     Deterministic RAG          LangGraph Agent
                |                     |
                |                Decision Node
                |                     |
                |             +-------+-------+
                |             |               |
                |             v               v
                |        Literature       Calculator
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
                |       Final Synthesis
                |             |
                +-------------+
                      |
                      v
                   Response
```

---

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
token / cost             result / error
metadata                  tool span
      |                      |
      +----------+-----------+
                 |
                 v
              Langfuse
                 |
                 v
        Trace / Span Inspection
```

Langfuse is used to inspect:

- LangGraph execution
- RAG operations
- MCP calculator calls
- tool failures
- token usage
- estimated LLM cost
- request metadata

A shared request ID allows application logs to be correlated with Langfuse traces.

---

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
Requests             Feedback
   |
   v
Usage / Cost / Latency Analytics
```

The deployed application uses Supabase PostgreSQL so persisted request and feedback data remain available independently of Cloud Run container instances.

---

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

Runtime secrets such as:

```text
OPENAI_API_KEY
DATABASE_URL
```

are supplied to Cloud Run through Google Secret Manager.

The FastAPI application is containerized and deployed to Google Cloud Run.

The MCP calculator currently runs as a separate MCP service for demonstration. A production deployment would require that service to be deployed independently and reachable by the agent.

Langfuse tracing can be enabled when the corresponding credentials are provided.

---

## Evaluation and Reliability

The repository includes tests for:

- deterministic RAG behavior
- API endpoints
- PostgreSQL integration
- LangGraph routing
- MCP tool calls
- tool failures
- maximum-step termination
- unknown-action fallback
- insufficient-evidence behavior
- structured errors
- upstream API failures
- persistence
- analytics
- mocked LLM behavior
- retrieval evaluation

The test suite avoids real external LLM and MCP calls during CI.

---

## Retrieval Evaluation

The repository includes a synthetic retrieval benchmark containing:

- 52 passages
- 29 questions

The benchmark deliberately includes:

- semantically similar distractors
- near-duplicate concepts
- exact identifiers
- numerical values
- related model names
- scientific terminology
- queries favoring semantic retrieval
- queries favoring lexical retrieval

Dense retrieval, BM25, and hybrid RRF retrieval were evaluated using Hit@k and Mean Reciprocal Rank.

### Full 29-Case Benchmark

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.828 | 1.000 | 1.000 | 0.908 |
| BM25 | 0.862 | 1.000 | 1.000 | 0.931 |
| Hybrid RRF | 0.828 | 1.000 | 1.000 | 0.914 |

The results show complementary behavior between semantic and lexical retrieval.

BM25 performs particularly well for exact identifiers and closely related alternatives, while dense retrieval performs better for some queries requiring semantic matching.

The benchmark does not establish that one method is universally superior.

Hybrid retrieval is retained as a robust candidate-generation strategy that exposes both lexical and semantic signals to the downstream reranker.

---

## Weighted RRF Evaluation

The benchmark was split reproducibly into a development set and a held-out test set.

BM25 fusion weights of:

```text
1.00
1.10
1.25
1.50
```

were evaluated while keeping the dense weight fixed at:

```text
1.0
```

Development results:

| Dense Weight | BM25 Weight | Dev Hit@1 | Dev MRR |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 0.800 | 0.900 |
| 1.0 | 1.1 | 0.750 | 0.875 |
| 1.0 | 1.25 | 0.750 | 0.875 |
| 1.0 | 1.5 | 0.750 | 0.875 |

Increasing the BM25 contribution did not improve development performance.

Equal-weight fusion was therefore retained.

Held-out results:

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.889 | 1.000 | 1.000 | 0.944 |
| BM25 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid RRF | 0.889 | 1.000 | 1.000 | 0.944 |

Because the held-out set contains only nine synthetic queries, the results are treated as diagnostic evidence rather than proof of general superiority.

---

## Candidate-Depth Evaluation

The LLM reranker was evaluated using candidate pools of:

```text
candidate_k = 3
candidate_k = 5
candidate_k = 10
candidate_k = 20
```

On the 20-query development split, every configuration achieved:

```text
Hit@3 = 1.000
MRR   = 1.000
```

Average reranking cost increased with candidate depth:

| candidate_k | Hit@3 | MRR | Avg Input Tokens | Avg Est. Cost |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1.000 | 1.000 | 325.0 | $0.000433 |
| 5 | 1.000 | 1.000 | 433.8 | $0.000506 |
| 10 | 1.000 | 1.000 | 700.7 | $0.000704 |
| 20 | 1.000 | 1.000 | 1238.1 | $0.001111 |

The synthetic development benchmark therefore favored `candidate_k=3`.

On the nine-query held-out split, both `candidate_k=3` and `candidate_k=20` preserved:

```text
Hit@3 = 1.000
MRR   = 1.000
```

Using three candidates reduced:

- reranker input tokens by **73.6%**
- estimated reranking cost by **60.8%**

However, the production system retains:

```text
candidate_k=20
```

Earlier experiments with full scientific papers showed that small candidate pools could miss relevant evidence.

Because the candidate-depth benchmark uses short synthetic passages, its efficiency result is not assumed to generalize directly to long scientific documents.

A larger real-document benchmark would be required before safely reducing the production candidate pool.

---

## Running Retrieval Evaluation

Build the evaluation index:

```powershell
python -m app.indexing `
  --source evals/retrieval_corpus.json `
  --index evals/retrieval_index `
  --paper-id retrieval-eval
```

Run the retrieval benchmark:

```bash
python -m evals.retrieval_eval
```

Run the candidate-depth experiment:

```bash
python -m evals.candidate_k_eval
```

The generated evaluation index is not intended to be committed to source control.

---

## Continuous Integration

GitHub Actions automatically:

1. starts a temporary PostgreSQL service
2. installs application dependencies
3. runs the automated test suite
4. builds the Docker image
5. starts the application container
6. verifies the `/health` endpoint

This validates database-backed application behavior and container startup without connecting CI to the production database.

---

## Cloud Deployment

The FastAPI application is deployed on Google Cloud Run.

The deployed system uses:

- Google Cloud Run for container execution
- Google Secret Manager for runtime credentials
- Supabase PostgreSQL for persistent application data
- OpenAI API for embeddings, reranking, agent decisions, and generation
- Langfuse for optional tracing and observability

The application does not rely on local SQLite persistence, so request history and analytics survive container restarts and new Cloud Run revisions.

---

## Design Principles

### Deterministic Workflows Where Possible

The normal RAG path remains deterministic rather than using an agent for every request.

Agentic orchestration is used only when execution depends on intermediate results or dynamic tool selection.

### Separate Ingestion from Query Serving

Document ingestion and indexing are performed separately from normal application requests.

```text
PDFs
 |
 v
Indexing Pipeline
 |
 v
Persistent Local Index

Question
 |
 v
Retrieval
 |
 v
Reranking
 |
 v
Generation
```

Documents are not reprocessed every time the API starts.

New or modified PDFs are indexed incrementally, while unchanged documents are skipped.

### Explicit Tool Boundaries

Each tool has a defined responsibility:

```text
Literature tool -> retrieve grounded evidence

Calculator tool -> perform arithmetic

Final synthesis -> combine existing results
```

Final synthesis is not allowed to silently reproduce functionality that should have been performed by a failed tool.

### Hard Constraints for Critical Behavior

LLM decisions are treated as probabilistic.

Important workflow requirements are therefore enforced in application code.

For example, if the user explicitly requests a calculation, the application verifies that a calculation result exists before allowing the standard successful completion path.

### Frameworks for Orchestration, Not Core Logic

Core retrieval, reranking, grounding, validation, and calculation logic remains implemented directly in Python.

LangGraph is used for stateful orchestration.

MCP provides a standardized interface for external tools.

Langfuse provides tracing and observability.

### Evaluation Before Optimization

Retrieval and reranking settings are changed only after measuring effects on:

- retrieval quality
- token usage
- estimated cost
- latency

Synthetic benchmarks are treated as controlled experiments rather than substitutes for realistic scientific-document evaluation.

---

## Limitations

This project is a production-oriented portfolio system rather than a large-scale commercial deployment.

Current limitations include:

- real LLM and embedding requests require API credentials and incur API costs
- indexes are stored locally rather than in a managed vector database
- the API must currently be restarted after indexing new documents so the in-memory collection is reloaded
- retrieval evaluation is based primarily on a small synthetic benchmark
- candidate-depth optimization has not yet been validated on a large collection of full scientific papers
- the production reranker therefore retains a conservative candidate pool of 20 passages
- automated API tests mock external LLM calls
- the MCP calculator is intentionally simple
- the MCP service is not yet independently deployed as part of the cloud architecture
- authentication and user-level authorization are not implemented
- agent evaluation remains limited compared with a dedicated production evaluation framework
- the current deployment is intentionally small in scale

---

## Future Improvements

Potential extensions include:

- retrieval evaluation on larger collections of full scientific papers
- formal end-to-end RAG evaluation
- broader agent regression evaluation
- managed vector storage
- automatic index refresh without restarting the API
- asynchronous document ingestion jobs for deployed environments
- additional MCP tools
- independent MCP cloud deployment
- authentication and authorization
- stronger deployment automation
- production monitoring and alerting

---

## Author

Pedro Alonso Vaquero

[GitHub](https://github.com/palonso-v)