from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LLMResult:
    content: str
    used_llm: bool
    provider: str
    model: str
    error: str = ""


class LLMClient:
    """Small OpenAI-compatible chat client with deterministic offline fallback."""

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider or os.getenv("CODEGRAPH_LLM_PROVIDER", "openai")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("CODEGRAPH_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))

    @property
    def enabled(self) -> bool:
        return self.provider != "mock" and bool(self.api_key)

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> LLMResult:
        if not self.enabled:
            return LLMResult(
                content="",
                used_llm=False,
                provider=self.provider,
                model=self.model,
                error="LLM disabled or API key missing",
            )

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return LLMResult(content=content, used_llm=True, provider=self.provider, model=self.model)
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return LLMResult(
                content="",
                used_llm=False,
                provider=self.provider,
                model=self.model,
                error=str(exc),
            )
