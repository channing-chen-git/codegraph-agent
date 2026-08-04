from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List

from ..models import AgentTraceStep
from .config import resolve_runtime_path


class TraceRecorder:
    def __init__(self, trace_dir: str | Path = "runs/traces"):
        self.trace_dir = resolve_runtime_path(trace_dir)
        self.trace_id = uuid.uuid4().hex[:12]
        self.steps: List[AgentTraceStep] = []
        self.started_at = time.time()

    def add(
        self,
        step: str,
        tool: str,
        query: str,
        summary: str,
        confidence: float,
        round_index: int = 1,
        status: str = "ok",
        error: str = "",
    ) -> None:
        self.steps.append(
            AgentTraceStep(
                step=step,
                tool=tool,
                query=query,
                summary=summary,
                confidence=confidence,
                round_index=round_index,
                status=status,
                error=error,
            )
        )

    def save(self, answer: str) -> str:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{self.trace_id}.json"
        payload = {
            "trace_id": self.trace_id,
            "duration_ms": int((time.time() - self.started_at) * 1000),
            "answer": answer,
            "steps": [asdict(step) for step in self.steps],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
