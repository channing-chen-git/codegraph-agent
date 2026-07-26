from __future__ import annotations

from dataclasses import asdict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None
    BaseModel = object

from .agent import CodeGraphAgent


if FastAPI is not None:
    app = FastAPI(title="CodeGraphAgent")
else:
    app = None


class QueryRequest(BaseModel):
    repo_path: str
    query: str
    use_llm: bool = True
    require_llm: bool = True


if app is not None:

    @app.post("/analyze")
    def analyze(request: QueryRequest):
        agent = CodeGraphAgent(
            request.repo_path,
            use_llm=request.use_llm,
            require_llm=request.require_llm,
        )
        return asdict(agent.answer(request.query))
