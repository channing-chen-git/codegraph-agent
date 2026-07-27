from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()


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
        self.base_url = (
            base_url
            or os.getenv("CODEGRAPH_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
        )
        self.model = model or os.getenv(
            "CODEGRAPH_LLM_MODEL",
            os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")),
        )
        self.timeout = float(os.getenv("CODEGRAPH_LLM_TIMEOUT", "60"))
        self.max_retries = max(0, int(os.getenv("CODEGRAPH_LLM_RETRIES", "2")))

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

        try:
            import requests
        except ImportError as exc:
            return LLMResult(
                content="",
                used_llm=False,
                provider=self.provider,
                model=self.model,
                error=f"requests is required for LLM calls: {exc}",
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        endpoint = f"{self.base_url}/chat/completions"
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return LLMResult(
                    content=content,
                    used_llm=True,
                    provider=self.provider,
                    model=self.model,
                )
            except requests.HTTPError as exc:
                body = response.text[:500].replace("\n", " ")
                last_error = f"HTTP {response.status_code}: {body or exc}"
                break
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        return LLMResult(
            content="",
            used_llm=False,
            provider=self.provider,
            model=self.model,
            error=f"LLM request failed at {endpoint}: {last_error}",
        )
