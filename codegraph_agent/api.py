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


if app is not None:

    @app.post("/analyze")
    def analyze(request: QueryRequest):
        agent = CodeGraphAgent(request.repo_path)
        return asdict(agent.answer(request.query))
