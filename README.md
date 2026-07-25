# CodeGraphAgent

CodeGraphAgent is a repository-level code understanding agent for analyzing
large Python and C/C++ codebases. It builds a lightweight symbol graph, retrieves
relevant code context, runs deterministic code-intelligence tools, and records
agent traces for evaluation.

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
- Tool-using Agent workflow for repository summary, search, symbol explanation,
  call-chain tracing, impact analysis and test recommendation.
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
       -> plan query
       -> call code-intelligence tools
       -> compose answer with evidence
       -> save Agent Trace
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
  "query": "Show the call chain from quote_with_coupon"
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

## Evidence Boundary

This is a lightweight code-intelligence prototype. It does not train a code
model, does not claim full semantic equivalence checking, and does not replace a
compiler, type checker or production static-analysis engine. Python analysis is
AST-based; C/C++ analysis is intentionally lightweight and should be replaced by
Tree-sitter, clangd or libclang for production-grade C++ understanding. The
current planner is deterministic and tool-oriented; an LLM planner can be added
behind the same tool interface when model-backed planning is required.

The current design keeps clear extension points for:

- Tree-sitter based multi-language parsing
- embedding-based code retrieval
- LLM planner integration
- learned reranking
- sandboxed test execution
- large-repository sharding and cache invalidation

## Open Source Safety

Do not commit private repositories, proprietary code, secrets, local model
weights, prompt logs or trace files containing private code. The repository
includes `.gitignore` rules for local runs and generated artifacts.
