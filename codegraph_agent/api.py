from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import Lock

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None
    BaseModel = object

from .agent import CodeGraphAgent
from .graph_store import CodeGraph
from .repository_indexer import RepositoryIndexer


if FastAPI is not None:
    app = FastAPI(title="CodeGraphAgent")
else:
    app = None


class QueryRequest(BaseModel):
    repo_path: str
    query: str
    use_llm: bool = True
    require_llm: bool = True
    refresh_index: bool = False


_GRAPH_CACHE: dict[str, CodeGraph] = {}
_GRAPH_CACHE_LOCK = Lock()


def _repo_key(repo_path: str) -> str:
    return str(Path(repo_path).expanduser().resolve())


def get_or_build_graph(repo_path: str, refresh: bool = False) -> CodeGraph:
    """Cache repository graphs so API calls do not rescan unchanged repos."""
    key = _repo_key(repo_path)
    with _GRAPH_CACHE_LOCK:
        if refresh or key not in _GRAPH_CACHE:
            _GRAPH_CACHE[key] = RepositoryIndexer().build(key)
        return _GRAPH_CACHE[key]


if app is not None:

    @app.post("/analyze")
    def analyze(request: QueryRequest):
        graph = get_or_build_graph(request.repo_path, refresh=request.refresh_index)
        agent = CodeGraphAgent(
            request.repo_path,
            use_llm=request.use_llm,
            require_llm=request.require_llm,
            graph=graph,
        )
        return asdict(agent.answer(request.query))
