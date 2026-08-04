# CodeGraphAgent

CodeGraphAgent is an LLM-guided repository-level code understanding agent for
analyzing large Python and C/C++ codebases. It builds a lightweight symbol graph,
uses an OpenAI-compatible LLM to plan tools and compose grounded answers, runs
deterministic code-intelligence tools for factual analysis, and records agent
traces for evaluation.

The project focuses on engineering-scale code context rather than single-file
completion. It is designed around questions such as:

- Which functions and modules are affected if this symbol changes?
- What is the cross-file call chain from this entry point?
- Where is a business rule implemented in the repository?
- Which tests should protect this change?
- Can the code-understanding workflow be evaluated offline?

## Core Capabilities

- Python AST parser for modules, classes, functions, imports, inheritance and
  call sites.
- Lightweight C/C++ parser for includes, function definitions and function
  calls.
- Symbol graph with `defines`, `calls`, `imports` and `inherits` edges.
- BM25-like code retriever over signatures, docstrings and source snippets.
- LLM planner and answer composer with deterministic fallback for offline tests.
- Multi-round tool loop with lightweight session memory and error guardrails.
- OpenAI-compatible Function Calling schemas for each code-intelligence tool.
- Tool-using Agent workflow for repository summary, search, symbol explanation,
  call-chain tracing, impact analysis and test recommendation.
- PR diff, coverage and runtime-trace evidence tools that complement the static
  symbol graph for test-impact analysis.
- Agent trace output for every run.
- Offline evaluation tasks for code retrieval, dependency reasoning and
  recommendation quality.
- CLI, optional FastAPI endpoint, unit tests and GitHub Actions workflow.

## Architecture

```text
Repository
    -> RepositoryIndexer
       -> PythonParser / CppParser
       -> Symbol Graph
       -> CodeRetriever
    -> CodeGraphAgent
       -> load session memory
       -> LLM planner selects tools from Function Calling schemas
       -> call code-intelligence tools with guardrails
       -> optionally merge PR diff / coverage / runtime trace evidence
       -> LLM composer writes an evidence-grounded answer
       -> repeat for another round if evidence is weak or uncertain
       -> save Agent Trace and session memory
    -> Eval Runner
```

## Quick Start

### Isolated Environment

Use `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Or use Conda:

```bash
conda env create -f environment.yml
conda activate codegraph-agent
```

Index the sample repository:

```bash
python -m codegraph_agent.cli --repo examples/mini_repo --index-out runs/mini_repo_index.json
```

Ask a code-understanding question:

```bash
python -m codegraph_agent.cli --repo examples/mini_repo --query "If calculate_total changes, which code may be affected?"
```

Ask for PR/test evidence:

```bash
python -m codegraph_agent.cli --repo examples/mini_repo --query "Analyze this PR diff and recommend tests" --no-llm
python -m codegraph_agent.cli --repo examples/mini_repo --query "Find coverage gaps for calculate_total" --no-llm
python -m codegraph_agent.cli --repo examples/mini_repo --query "Use runtime trace evidence for calculate_total" --no-llm
```

Configure an OpenAI-compatible LLM for planner/composer mode:

```bash
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
CODEGRAPH_LLM_MODEL=gpt-4o-mini
```

The normal CLI/API path requires the LLM planner and composer to succeed. If the
API key or endpoint is missing, the run fails instead of silently pretending to
be an LLM Agent.

For CI or offline demos, use deterministic fallback:

```bash
python -m codegraph_agent.cli --repo examples/mini_repo --query "If calculate_total changes, which code may be affected?" --no-llm
```

Run offline evaluation:

```bash
python -m codegraph_agent.eval_runner --repo examples/mini_repo --tasks eval/tasks.json
```

Runtime artifacts such as traces and evaluation reports are written to a
user-writable runtime directory by default. Set `CODEGRAPH_RUNTIME_DIR` to
override it:

```bash
CODEGRAPH_RUNTIME_DIR=./runtime
python -m codegraph_agent.eval_runner --repo examples/mini_repo --tasks eval/tasks.json
```

Run tests:

```bash
python -m unittest discover tests
```

Optional API:

```bash
pip install -r requirements.txt
uvicorn codegraph_agent.api:app --reload --port 8000
```

The API caches the built `CodeGraph` per resolved `repo_path`, so repeated
questions against the same repository reuse the existing symbol graph instead
of rescanning the repository on every request. Pass `refresh_index: true` when
the repository has changed and the graph should be rebuilt.

The request body also accepts `session_id`, so repeated questions from the same
conversation reuse lightweight session memory and can trigger a second tool
round when the first pass is uncertain.

Docker:

```bash
docker build -t codegraph-agent .
docker run -p 8000:8000 codegraph-agent
```

Then call:

```json
POST /analyze
{
  "repo_path": "examples/mini_repo",
  "query": "Show the call chain from quote_with_coupon",
  "refresh_index": false
}
```

## Example Output

For the query:

```text
If calculate_total changes, which code may be affected?
```

The agent uses:

1. `repository_summary`
2. `impact_analysis`
3. `test_recommendations`

and returns impacted callers, impacted files and focused test suggestions.

The planner sees these function-callable tools: `repository_summary`,
`search_code`, `explain_symbol`, `call_chain`, `impact_analysis`,
`test_recommendations`, `pr_change_analysis`, `coverage_gap_analysis`, and
`runtime_trace_analysis`. The LLM decides which functions to call, while the
symbol graph and external evidence tools produce the factual evidence used in
the final answer.

## Evidence Boundary

This is a lightweight code-intelligence Agent prototype. It does not train a
code model, does not claim full semantic equivalence checking, and does not
replace a compiler, type checker or production static-analysis engine. The LLM
plans tool usage and composes the final explanation, while symbol-graph tools
produce the factual evidence. Python analysis is AST-based; C/C++ analysis is
intentionally lightweight and should be replaced by Tree-sitter, clangd or
libclang for production-grade C++ understanding.

The current agent now also keeps lightweight session memory, retries with a
second tool round when evidence is weak, and records tool/runtime errors inside
trace output instead of failing silently.

The current demo now includes sample `changes.diff`, `coverage.json` and
`runtime_traces.json` files under `examples/mini_repo`. These represent how a
production deployment can combine static source analysis with CI, coverage and
observed runtime traces. Static analysis is broad and works before code runs;
runtime evidence is more factual but only covers executed paths.

The current design keeps clear extension points for:

- Tree-sitter based multi-language parsing
- embedding-based code retrieval
- richer LLM tool-calling schemas
- learned reranking
- sandboxed test execution
- large-repository sharding and cache invalidation
- PR-diff ingestion from GitHub/GitLab CI
- coverage ingestion from pytest/coverage.py reports
- runtime trace ingestion from OpenTelemetry or service-mesh traces

## Open Source Safety

Do not commit private repositories, proprietary code, secrets, local model
weights, prompt logs or trace files containing private code. The repository
includes `.gitignore` rules for local runs and generated artifacts.
