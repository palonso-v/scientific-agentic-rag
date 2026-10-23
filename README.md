# Scientific Agentic RAG Assistant

A Python-based scientific literature assistant combining retrieval-augmented generation (RAG), hybrid lexical and semantic retrieval, LLM reranking, grounded answer generation, stateful agent orchestration, standardized tool integration, and production-oriented API infrastructure.

The project includes a deterministic RAG pipeline alongside a LangGraph-based agent capable of dynamically selecting literature retrieval and calculation tools. Retrieval combines dense embedding search with BM25 lexical search using Reciprocal Rank Fusion (RRF). Tool access can be provided through Model Context Protocol (MCP), while Langfuse tracing provides visibility into agent execution, RAG operations, tool calls, and failures.

The application also includes a FastAPI backend, PostgreSQL persistence, usage analytics, Docker containerization, automated testing, GitHub Actions CI, and cloud deployment on Google Cloud Run.

## Features

### RAG and Retrieval

- Token-based document chunking
- OpenAI embeddings
- Dense semantic retrieval using normalized vector similarity
- BM25 lexical retrieval
- Hybrid dense + lexical candidate retrieval
- Reciprocal Rank Fusion (RRF)
- LLM-based passage reranking
- Grounded answer generation with structured outputs
- Citation ID validation
- Insufficient-evidence handling
- Source-aware answer generation
- Retrieval benchmarking using Hit@k and Mean Reciprocal Rank (MRR)
- Candidate-depth evaluation for reranking quality and cost

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

Python, OpenAI API, FastAPI, Pydantic, LangGraph, MCP, Langfuse, NumPy, rank-bm25, PostgreSQL, Supabase, psycopg, Docker, pytest, GitHub Actions, Google Cloud Run, Google Secret Manager.

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

### 6. Test retrieval

```bash
python -m app.retrieval
```

The retrieval demo displays results from:

- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid retrieval using Reciprocal Rank Fusion

Example question:

> What were the carbon dioxide concentrations in 2015 and 2025?

### 7. Run the deterministic RAG assistant

```bash
python -m app.rag
```

Example questions:

- Which greenhouse gases are mentioned in the document?
- What were the carbon dioxide concentrations in 2015 and 2025?
- What was the global average temperature in 2024?

The final question can be used to test insufficient-evidence handling.

## Hybrid Retrieval

The first-stage retriever combines two complementary retrieval strategies.

### Dense retrieval

The user question is embedded using the same embedding model used to index document chunks. Normalized vector similarity is then used to identify semantically related passages.

Dense retrieval can recover relevant passages even when the query and document use different wording.

### BM25 lexical retrieval

BM25 scores chunks according to lexical overlap while accounting for:

- Query-term frequency
- Inverse document frequency
- Term-frequency saturation
- Document-length normalization

Lexical retrieval is particularly useful for exact identifiers, acronyms, model names, numerical values, and uncommon terminology.

### Reciprocal Rank Fusion

The dense and BM25 rankings are combined using Reciprocal Rank Fusion rather than directly combining their raw scores.

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

RRF operates on ranking positions rather than raw retrieval scores, allowing dense cosine similarities and BM25 scores to be combined without requiring score normalization.

The current application uses equal weighting for dense and BM25 retrieval. A development-set experiment with increased BM25 weighting did not improve retrieval quality, so equal-weight fusion was retained.

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

Important workflow constraints are enforced in application code rather than relying entirely on LLM decisions.

For example, when a user explicitly requests a calculation, the agent cannot silently bypass the calculator and perform the arithmetic during final synthesis. If the calculator fails, the failure is preserved in agent state and the final answer reports that the requested calculation could not be completed.

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

A typical successful execution follows:

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
        +----------------------+
        |                      |
        v                      v
OpenAI Embeddings         Tokenized Corpus
        |                      |
        v                      v
Local Vector Index        BM25 Index
        |                      |
        v                      v
Dense Retrieval          Lexical Retrieval
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

Langfuse tracing is integrated into the application and can be enabled when Langfuse credentials are provided.

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
- Retrieval evaluation scripts
- Candidate-depth evaluation
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

The automated test suite verifies application and agent behavior without requiring real OpenAI or MCP calls during CI.

## Retrieval Evaluation

The repository includes a synthetic retrieval benchmark with 52 passages and 29 questions.

The benchmark deliberately includes:

- Semantically similar distractors
- Near-duplicate concepts
- Exact identifiers
- Numerical values
- Related model names
- Scientific terminology
- Queries that favor semantic matching
- Queries that favor lexical matching

Dense retrieval, BM25, and hybrid RRF retrieval were compared using Hit@k and Mean Reciprocal Rank (MRR).

### Full 29-case benchmark

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.828 | 1.000 | 1.000 | 0.908 |
| BM25 | 0.862 | 1.000 | 1.000 | 0.931 |
| Hybrid RRF | 0.828 | 1.000 | 1.000 | 0.914 |

The benchmark shows complementary behavior between semantic and lexical retrieval.

Some queries are ranked better by BM25, especially cases involving exact identifiers or closely related alternatives. Other queries are ranked better by dense retrieval when semantic similarity is more important than exact word overlap.

Hybrid RRF preserves both signals in a common candidate-ranking pipeline.

### Weighted RRF Evaluation

The evaluation cases were also split reproducibly into a development set and held-out test set.

BM25 fusion weights of:

```text
1.00
1.10
1.25
1.50
```

were evaluated while keeping the dense weight fixed at `1.0`.

Increasing the BM25 contribution did not improve development performance:

| Dense Weight | BM25 Weight | Dev Hit@1 | Dev MRR |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 0.800 | 0.900 |
| 1.0 | 1.1 | 0.750 | 0.875 |
| 1.0 | 1.25 | 0.750 | 0.875 |
| 1.0 | 1.5 | 0.750 | 0.875 |

Equal-weight RRF was therefore retained.

On the nine held-out queries:

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.889 | 1.000 | 1.000 | 0.944 |
| BM25 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid RRF | 0.889 | 1.000 | 1.000 | 0.944 |

Because this is a small synthetic benchmark, these results are treated as diagnostic evidence rather than proof that one retrieval method is universally superior.

## Candidate-Depth Evaluation

The LLM reranker was evaluated using hybrid candidate pools of:

```text
candidate_k = 3
candidate_k = 5
candidate_k = 10
candidate_k = 20
```

On the 20-query development split, all four configurations achieved:

```text
Hit@3 = 1.000
MRR   = 1.000
```

Average reranking cost increased as the candidate pool grew:

| candidate_k | Hit@3 | MRR | Avg Input Tokens | Avg Est. Cost |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1.000 | 1.000 | 325.0 | $0.000433 |
| 5 | 1.000 | 1.000 | 433.8 | $0.000506 |
| 10 | 1.000 | 1.000 | 700.7 | $0.000704 |
| 20 | 1.000 | 1.000 | 1238.1 | $0.001111 |

The smallest candidate pool was therefore selected by the development-set optimization rule.

On the nine-query held-out split, `candidate_k=3` and `candidate_k=20` both preserved:

```text
Hit@3 = 1.000
MRR   = 1.000
```

while `candidate_k=3` reduced:

- Reranker input tokens by **73.6%**
- Estimated reranking cost by **60.8%**

However, the application retains a conservative production default of `candidate_k=20`.

Earlier experiments on full scientific papers showed that small semantic-retrieval candidate pools could miss relevant evidence. Because the current candidate-depth benchmark uses short synthetic passages, the observed efficiency improvement is not assumed to generalize to long, heterogeneous scientific documents.

A future real-document benchmark can determine whether hybrid retrieval allows the production candidate pool to be reduced safely.

## Running Retrieval Evaluation

Build the synthetic retrieval evaluation index:

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

Run the reranker candidate-depth experiment:

```bash
python -m evals.candidate_k_eval
```

The generated evaluation index is not intended to be committed to source control.

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

The application does not rely on local SQLite persistence, so request history and analytics survive Cloud Run container restarts and revisions.

Langfuse can be configured separately for tracing and agent observability.

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

### Evaluation before optimization

Retrieval and reranking parameters are changed only after measuring their effect on quality, token usage, cost, and latency.

Synthetic benchmarks are treated as controlled experiments rather than substitutes for evaluation on realistic scientific-document collections.

## Limitations

This project is a production-oriented portfolio system rather than a large-scale commercial deployment.

- The demonstration corpus is small and synthetic.
- Real LLM requests require API credentials and incur API costs.
- Document indexes must currently be built before retrieval.
- Automated API tests mock external LLM calls.
- Retrieval uses hybrid dense semantic search and BM25 lexical search with Reciprocal Rank Fusion.
- The current retrieval benchmark is synthetic and relatively small.
- Retrieval quality has not yet been systematically evaluated on a large heterogeneous collection of full scientific papers.
- Candidate-depth optimization has not yet been validated on a larger real-document corpus.
- The production reranker therefore retains a conservative candidate pool of 20 passages.
- The vector index is currently local rather than backed by a managed vector database.
- The MCP calculator is deliberately simple and is primarily used to demonstrate MCP architecture.
- The MCP service is not yet independently deployed as part of the cloud architecture.
- The current cloud deployment is intentionally small-scale.
- Authentication and user-level authorization are not yet implemented.
- Agent evaluation remains limited compared with a production evaluation framework.

## Future Improvements

Potential future extensions include:

- Retrieval and candidate-depth evaluation on larger collections of full scientific papers
- More formal end-to-end RAG and agent evaluation
- More substantial MCP-backed external tools
- Independent cloud deployment of MCP services
- Managed vector storage for larger document collections
- Authentication and user-level authorization
- Stronger deployment automation
- Production monitoring and alerting
- Evaluation-driven agent regression testing

## Author

Pedro Alonso Vaquero

[GitHub](https://github.com/palonso-v)